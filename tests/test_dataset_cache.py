"""
Tests for ecg_cnn.data.dataset_cache

Goals
-----
1) Basic roundtrip and parameter validation
2) Cache behavior: RAM, disk, and force-reload
3) Path canonicalization and cache-key changes
4) Validation error paths for _validate_cfg and _validate_loaded
"""

import numpy as np
import pandas as pd
import pytest

from ecg_cnn.config.config_loader import TrainConfig
from ecg_cnn.data.dataset_cache import (
    get_dataset_cached,
    _cache_key,
    _RAM_CACHE,
    _disk_cache_path,
    _drop_unknowns,
)


def make_cfg(**kw):
    base = dict(
        model="ECGResNet",
        lr=1e-3,
        batch_size=64,
        weight_decay=0.0,
        n_epochs=1,
        save_best=True,
        sample_only=True,  # keep fast in CI
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        n_folds=2,
        verbose=False,
        # plotting keys tolerated by TrainConfig in your repo
        plots_enable_ovr=False,
        plots_ovr_classes=[],
    )
    base.update(kw)
    return TrainConfig(**base)


def _fake_loaded(n=4, unknown=False):
    """Small synthetic (X, y, meta) helper for caching tests."""
    X = np.random.randn(n, 12, 100).astype(np.float32)
    y = ["NORM"] * n
    if unknown:
        y[0] = "Unknown"
    meta = pd.DataFrame({"id": list(range(n))})
    return X, y, meta


# ======================================================================
# A. Basic roundtrip & simple parameter validation
# ======================================================================


def test_cached_roundtrip(tmp_path, monkeypatch):
    # Keep cache files under tmp so we don't pollute the repo
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    # Tiny fake dataset so no real I/O happens
    def fake_sample(**kwargs):
        X = np.random.randn(4, 12, 100).astype(np.float32)
        y = ["NORM", "NORM", "MI", "STTC"]
        meta = pd.DataFrame({"id": [0, 1, 2, 3]})
        return X, y, meta

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )

    cfg = make_cfg(sample_only=True)

    # first call: force reload to bypass any RAM/disk cache
    X1, y1, m1 = get_dataset_cached(cfg, use_disk_cache=True, force_reload=True)
    # second call: should come from cache, identical content
    X2, y2, m2 = get_dataset_cached(cfg, use_disk_cache=True, force_reload=False)

    assert (
        isinstance(X1, np.ndarray)
        and isinstance(y1, list)
        and isinstance(m1, pd.DataFrame)
    )
    assert len(X1) == len(y1) == len(m1)
    assert np.array_equal(X1, X2)
    assert y1 == y2
    assert m1.equals(m2)


def test_invalid_sampling_rate(tmp_path, monkeypatch):
    # Invalid sampling_rate value is rejected early
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    cfg = make_cfg(sampling_rate=123)
    with pytest.raises(ValueError, match=r"cfg\.sampling_rate must be 100 or 500"):
        get_dataset_cached(cfg)


def test_subsample_bounds(tmp_path, monkeypatch):
    # subsample_frac must be within (0, 1]
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    cfg = make_cfg(subsample_frac=0.0)
    with pytest.raises(ValueError, match=r"cfg\.subsample_frac must be in \(0, 1\]"):
        get_dataset_cached(cfg)


# ======================================================================
# B. _validate_cfg branches (type/value checks)
# ======================================================================


def test_get_dataset_cached_rejects_nontrainconfig_typeerror(tmp_path, monkeypatch):
    # Non-TrainConfig object is rejected
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)
    with pytest.raises(TypeError, match=r"cfg must be TrainConfig"):
        get_dataset_cached(cfg="not-a-config")  # type: ignore[arg-type]


def test_get_dataset_cached_validate_cfg_bad_sample_only_type(tmp_path, monkeypatch):
    # sample_only must be bool
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)
    bad = make_cfg(sample_only="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"cfg\.sample_only must be bool"):
        get_dataset_cached(bad)


def test_get_dataset_cached_validate_cfg_subsample_out_of_range(tmp_path, monkeypatch):
    # subsample_frac outside (0, 1] raises
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)
    bad = make_cfg(subsample_frac=0.0)
    with pytest.raises(ValueError, match=r"cfg\.subsample_frac must be in \(0, 1\]"):
        get_dataset_cached(bad)


def test_get_dataset_cached_validate_cfg_bad_sampling_rate_value(tmp_path, monkeypatch):
    # sampling_rate must be 100 or 500
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)
    bad = make_cfg(sampling_rate=250)
    with pytest.raises(ValueError, match=r"cfg\.sampling_rate must be 100 or 500"):
        get_dataset_cached(bad)


def test_get_dataset_cached_validate_cfg_nonexistent_dirs(tmp_path, monkeypatch):
    # Provided data_dir/sample_dir must exist
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)
    bad1 = make_cfg(data_dir=tmp_path / "missingA")
    with pytest.raises(ValueError, match=r"cfg\.data_dir does not exist"):
        get_dataset_cached(bad1)

    bad2 = make_cfg(sample_dir=tmp_path / "missingB")
    with pytest.raises(ValueError, match=r"cfg\.sample_dir does not exist"):
        get_dataset_cached(bad2)


# ======================================================================
# C. Cache behavior: RAM, disk, force-reload; sample vs full loaders
# ======================================================================


def test_get_dataset_cached_uses_ram_cache_on_second_call(tmp_path, monkeypatch):
    # Second call returns from RAM cache; loader called only once
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    calls = {"n": 0}

    def fake_sample(**kwargs):
        calls["n"] += 1
        return _fake_loaded(n=3)

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )

    cfg = make_cfg(sample_only=True)
    _RAM_CACHE.clear()

    # 1) populate RAM cache
    X1, y1, m1 = get_dataset_cached(cfg, use_disk_cache=False, force_reload=False)
    assert calls["n"] == 1

    # 2) second call should return from RAM
    X2, y2, m2 = get_dataset_cached(cfg, use_disk_cache=False, force_reload=False)
    assert calls["n"] == 1
    assert X1.shape == X2.shape and y1 == y2 and len(m1) == len(m2)


def test_get_dataset_cached_reads_from_disk_cache_when_present(tmp_path, monkeypatch):
    # Reads from on-disk .npz when RAM cache is empty and file exists
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    calls = {"n": 0}

    def fake_sample(**kwargs):
        calls["n"] += 1
        return _fake_loaded(n=5)

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )

    cfg = make_cfg(sample_only=True)
    _RAM_CACHE.clear()
    cache_file = _disk_cache_path(_cache_key(cfg))
    cache_file.unlink(missing_ok=True)

    # prime disk cache
    X1, y1, m1 = get_dataset_cached(cfg, use_disk_cache=True, force_reload=True)
    assert calls["n"] == 1
    assert cache_file.exists()

    # clear RAM and load from disk
    _RAM_CACHE.clear()
    X2, y2, m2 = get_dataset_cached(cfg, use_disk_cache=True, force_reload=False)
    assert calls["n"] == 1
    assert X1.shape == X2.shape and y1 == y2 and len(m1) == len(m2)


def test_get_dataset_cached_force_reload_bypasses_caches(tmp_path, monkeypatch):
    # force_reload=True bypasses both RAM and disk caches
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    calls = {"n": 0}

    def fake_sample(**kwargs):
        calls["n"] += 1
        return _fake_loaded(n=2)

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )

    cfg = make_cfg(sample_only=True)
    import ecg_cnn.data.dataset_cache as dataset_cache

    dataset_cache._RAM_CACHE.clear()
    key = _cache_key(cfg)
    cache_file = dataset_cache._disk_cache_path(key)
    cache_file.unlink(missing_ok=True)

    _ = get_dataset_cached(cfg, use_disk_cache=True, force_reload=False)
    assert calls["n"] == 1

    _ = get_dataset_cached(cfg, use_disk_cache=True, force_reload=True)
    assert calls["n"] == 2


def test_get_dataset_cached_sample_only_calls_sample_loader_and_drops_unknowns(
    tmp_path, monkeypatch
):
    # sample_only path loads sample set and filters out 'Unknown' labels
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    import ecg_cnn.data.dataset_cache as dataset_cache

    dataset_cache._RAM_CACHE.clear()

    def fake_sample(**kwargs):
        return _fake_loaded(n=4, unknown=True)

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )

    cfg = make_cfg(sample_only=True)
    X, y, meta = get_dataset_cached(cfg, use_disk_cache=False, force_reload=False)
    assert len(y) == 3 and "Unknown" not in y


def test_get_dataset_cached_full_calls_full_loader(tmp_path, monkeypatch):
    # full-data path uses load_ptbxl_full
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    calls = {"sample": 0, "full": 0}

    def fake_sample(**kwargs):
        calls["sample"] += 1
        return _fake_loaded(n=3)

    def fake_full(**kwargs):
        calls["full"] += 1
        return _fake_loaded(n=6)

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )
    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_full", fake_full, raising=True
    )

    cfg = make_cfg(sample_only=False)
    X, y, meta = get_dataset_cached(cfg, use_disk_cache=False)
    assert calls["full"] == 1 and calls["sample"] == 0


# ======================================================================
# D. _canon_paths and cache-key changes when paths differ
# ======================================================================


def test_get_dataset_cached_cache_key_differs_when_paths_change(tmp_path, monkeypatch):
    # Cache key changes when canonicalized data/sample paths differ
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    cfg1 = make_cfg(data_dir=None, sample_dir=None)
    key1 = _cache_key(cfg1)

    ddir = tmp_path / "ptb_data_root"
    sdir = tmp_path / "sample_root"
    ddir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    cfg2 = make_cfg(data_dir=str(ddir), sample_dir=str(sdir))
    key2 = _cache_key(cfg2)

    assert key1 != key2


# ======================================================================
# E. Additional _validate_cfg and _validate_loaded error paths
# ======================================================================


def test_validate_cfg_subsample_frac_not_floatlike(tmp_path, monkeypatch):
    # subsample_frac must be float-like
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)
    bad = make_cfg(subsample_frac="nope")  # not float-like
    with pytest.raises(TypeError, match=r"cfg\.subsample_frac must be float-like"):
        get_dataset_cached(bad)


def test_validate_cfg_sampling_rate_wrong_type(tmp_path, monkeypatch):
    # sampling_rate must be int
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)
    bad = make_cfg(sampling_rate="500")  # wrong type (string)
    with pytest.raises(TypeError, match=r"cfg\.sampling_rate must be int"):
        get_dataset_cached(bad)


def test_validate_loaded_raises_on_bad_ndim(tmp_path, monkeypatch):
    # X with ndim < 2 is invalid
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    def fake_sample(**kwargs):
        X = np.random.randn(5).astype(np.float32)
        y = ["NORM"] * 5
        meta = pd.DataFrame({"id": range(5)})
        return X, y, meta

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )
    cfg = make_cfg(sample_only=True)
    with pytest.raises(ValueError, match=r"X must be a numpy array with ndim>=2"):
        get_dataset_cached(cfg, use_disk_cache=False, force_reload=True)


def test_validate_loaded_raises_on_bad_meta_type(tmp_path, monkeypatch):
    # meta must be a pandas DataFrame
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)
    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache._drop_unknowns",
        lambda X, y, meta: (X, y, meta),
        raising=True,
    )

    def fake_sample(**kwargs):
        X = np.random.randn(4, 12, 100).astype(np.float32)
        y = ["NORM"] * 4
        meta = {"id": [0, 1, 2, 3]}  # not a DataFrame
        return X, y, meta

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )

    cfg = make_cfg(sample_only=True)
    with pytest.raises(ValueError, match=r"meta must be a pandas DataFrame"):
        get_dataset_cached(cfg, use_disk_cache=False, force_reload=True)


def test_validate_loaded_raises_on_empty_dataset(tmp_path, monkeypatch):
    # Empty X is rejected
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)

    def fake_sample(**kwargs):
        X = np.empty((0, 12, 100), dtype=np.float32)
        y = []
        meta = pd.DataFrame({"id": []})
        return X, y, meta

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )
    cfg = make_cfg(sample_only=True)
    with pytest.raises(ValueError, match=r"Loaded dataset is empty\."):
        get_dataset_cached(cfg, use_disk_cache=False, force_reload=True)


def test_validate_loaded_raises_on_length_mismatch(tmp_path, monkeypatch):
    # len(X), len(y), len(meta) must match
    monkeypatch.setattr("ecg_cnn.data.dataset_cache.OUTPUT_DIR", tmp_path, raising=True)
    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache._drop_unknowns",
        lambda X, y, meta: (X, y, meta),
        raising=True,
    )

    def fake_sample(**kwargs):
        X = np.random.randn(4, 12, 100).astype(np.float32)
        y = ["NORM"] * 3
        meta = pd.DataFrame({"id": [0, 1, 2, 3]})
        return X, y, meta

    monkeypatch.setattr(
        "ecg_cnn.data.dataset_cache.load_ptbxl_sample", fake_sample, raising=True
    )

    cfg = make_cfg(sample_only=True)
    with pytest.raises(
        ValueError, match=r"Length mismatch: len\(X\)=4, len\(y\)=3, len\(meta\)=4"
    ):
        get_dataset_cached(cfg, use_disk_cache=False, force_reload=True)


# ======================================================================
# f. _drop_unknons error paths
# ======================================================================


def test_drop_unknowns_raises_when_meta_not_dataframe():
    X = np.random.randn(3, 12, 100).astype(np.float32)
    y = ["NORM", "NORM", "NORM"]
    meta = {"id": [0, 1, 2]}  # dict, not DataFrame
    with pytest.raises(ValueError, match=r"meta must be a pandas DataFrame"):
        _drop_unknowns(X, y, meta)  # type: ignore[arg-type]


def test_drop_unknowns_raises_on_length_mismatch_precheck():
    X = np.random.randn(3, 12, 100).astype(np.float32)
    y = ["NORM", "NORM"]  # shorter than X
    meta = pd.DataFrame({"id": [0, 1, 2]})
    with pytest.raises(
        ValueError,
        match=r"Length mismatch before drop: len\(X\)=3, len\(y\)=2, len\(meta\)=3",
    ):
        _drop_unknowns(X, y, meta)
