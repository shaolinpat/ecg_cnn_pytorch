# tests/test_datset_cache.py

"""
Tests for ecg_cnn.data.dataset_cache.

Goals
-----
    1) Basic roundtrip and parameter validation
    2) Cache behavior: RAM, disk, and force-reload
    3) Path canonicalization and _cache_key changes
    4) Validation error paths for _validate_cfg and _validate_loaded
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ecg_cnn.data.dataset_cache as dc


# ------------------------------------------------------------------------------
# Local fixtures (scoped to this file)
# ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_ram_cache():
    """Ensure the module-level RAM cache never leaks between tests."""
    dc._RAM_CACHE.clear()
    try:
        yield
    finally:
        dc._RAM_CACHE.clear()


@pytest.fixture
def patch_dataset_cache_paths(patch_paths, monkeypatch):
    """
    Patch imported constants *inside* ecg_cnn.data.dataset_cache so tests
    that rely on the conftest temp dirs hit the right paths.

    Returns
    -------
    (output_dir, ptbxl_dir)
        The patched output and PTB-XL directories.
    """
    _, _, _, output_dir, _, _, ptbxl_dir = patch_paths
    # Patch where they're *used* (dataset_cache), not just where defined.
    monkeypatch.setattr(dc, "OUTPUT_DIR", output_dir, raising=True)
    monkeypatch.setattr(dc, "PTBXL_DATA_DIR", ptbxl_dir, raising=True)
    return output_dir, ptbxl_dir


def _fake_loaded(n: int = 4, unknown: bool = False):
    """Small synthetic (X, y, meta) helper for caching tests."""
    X = np.random.randn(n, 12, 100).astype(np.float32)
    y = ["NORM"] * n
    if unknown:
        y[0] = "Unknown"
    meta = pd.DataFrame({"id": list(range(n))})
    return X, y, meta


# ------------------------------------------------------------------------------
# A. Basic roundtrip & simple parameter validation
# ------------------------------------------------------------------------------


def test_cached_roundtrip(patch_paths, make_train_config, monkeypatch):
    # Tiny fake dataset so no real I/O happens.
    def fake_sample(**kwargs):
        X = np.random.randn(4, 12, 100).astype(np.float32)
        y = ["NORM", "NORM", "MI", "STTC"]
        meta = pd.DataFrame({"id": [0, 1, 2, 3]})
        return X, y, meta

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)

    cfg = make_train_config(sample_only=True)

    # first call: force reload to bypass any RAM/disk cache
    X1, y1, m1 = dc.get_dataset_cached(cfg, use_disk_cache=True, force_reload=True)
    # second call: should come from cache, identical content
    X2, y2, m2 = dc.get_dataset_cached(cfg, use_disk_cache=True, force_reload=False)

    assert (
        isinstance(X1, np.ndarray)
        and isinstance(y1, list)
        and isinstance(m1, pd.DataFrame)
    )
    assert len(X1) == len(y1) == len(m1)
    np.testing.assert_array_equal(X1, X2)
    assert y1 == y2
    pd.testing.assert_frame_equal(m1, m2)


def test_invalid_sampling_rate(patch_paths, make_train_config):
    # Invalid sampling_rate value is rejected early.
    cfg = make_train_config(sampling_rate=123)
    with pytest.raises(ValueError, match=r"^cfg\.sampling_rate must be 100 or 500"):
        dc.get_dataset_cached(cfg)


def test_subsample_bounds(patch_paths, make_train_config):
    # subsample_frac must be within (0, 1].
    cfg = make_train_config(subsample_frac=0.0)
    with pytest.raises(ValueError, match=r"^cfg\.subsample_frac must be in \(0, 1\]"):
        dc.get_dataset_cached(cfg)


# ------------------------------------------------------------------------------
# B. _validate_cfg branches (type/value checks)
# ------------------------------------------------------------------------------


def test_get_dataset_cached_rejects_nontrainconfig_typeerror(patch_paths):
    with pytest.raises(TypeError, match=r"^cfg must be TrainConfig"):
        dc.get_dataset_cached(cfg="not-a-config")  # type: ignore[arg-type]


def test_get_dataset_cached_validate_cfg_bad_sample_only_type(
    patch_paths, make_train_config
):
    bad = make_train_config(sample_only="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"^cfg\.sample_only must be bool"):
        dc.get_dataset_cached(bad)


def test_get_dataset_cached_validate_cfg_subsample_out_of_range(
    patch_paths, make_train_config
):
    bad = make_train_config(subsample_frac=0.0)
    with pytest.raises(ValueError, match=r"^cfg\.subsample_frac must be in \(0, 1\]"):
        dc.get_dataset_cached(bad)


def test_get_dataset_cached_validate_cfg_bad_sampling_rate_value(
    patch_paths, make_train_config
):
    bad = make_train_config(sampling_rate=250)
    with pytest.raises(ValueError, match=r"^cfg\.sampling_rate must be 100 or 500"):
        dc.get_dataset_cached(bad)


def test_get_dataset_cached_validate_cfg_nonexistent_dirs(
    tmp_path, patch_paths, make_train_config
):
    bad1 = make_train_config(data_dir=tmp_path / "missingA")
    with pytest.raises(ValueError, match=r"^cfg\.data_dir does not exist"):
        dc.get_dataset_cached(bad1)

    bad2 = make_train_config(sample_dir=tmp_path / "missingB")
    with pytest.raises(ValueError, match=r"^cfg\.sample_dir does not exist"):
        dc.get_dataset_cached(bad2)


# ------------------------------------------------------------------------------
# C. Cache behavior: RAM, disk, force-reload; sample vs full loaders
# ------------------------------------------------------------------------------


def test_get_dataset_cached_uses_ram_cache_on_second_call(
    patch_paths, make_train_config, monkeypatch
):
    # Second call returns from RAM cache; loader called only once.
    calls = {"n": 0}

    def fake_sample(**kwargs):
        calls["n"] += 1
        return _fake_loaded(n=3)

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)

    cfg = make_train_config(sample_only=True)

    # 1) populate RAM cache
    X1, y1, m1 = dc.get_dataset_cached(cfg, use_disk_cache=False, force_reload=False)
    assert calls["n"] == 1

    # 2) second call should return from RAM (no new load)
    X2, y2, m2 = dc.get_dataset_cached(cfg, use_disk_cache=False, force_reload=False)
    assert calls["n"] == 1
    assert X1.shape == X2.shape and y1 == y2 and len(m1) == len(m2)


def test_get_dataset_cached_reads_from_disk_cache_when_present(
    patch_dataset_cache_paths, make_train_config, monkeypatch
):
    # Reads from on-disk .npz when RAM cache is empty and file exists.
    output_dir, _ = patch_dataset_cache_paths
    calls = {"n": 0}

    def fake_sample(**kwargs):
        calls["n"] += 1
        return _fake_loaded(n=5)

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)

    cfg = make_train_config(sample_only=True)

    cache_file = dc._disk_cache_path(dc._cache_key(cfg))
    cache_file.unlink(missing_ok=True)

    # 1) Prime: write .npz to disk via fresh load
    X1, y1, m1 = dc.get_dataset_cached(cfg, use_disk_cache=True, force_reload=True)
    assert calls["n"] == 1
    assert cache_file.exists()
    assert cache_file.parent == (output_dir / "cache")

    # filename should be derived from the cache key (allow module prefix)
    assert cache_file.stem.endswith(dc._cache_key(cfg))

    # 2) Critical: clear RAM so the next call must hit the DISK READ branch
    dc._RAM_CACHE.clear()

    # 3) Read from disk (no new loader call)
    X2, y2, m2 = dc.get_dataset_cached(cfg, use_disk_cache=True, force_reload=False)
    assert calls["n"] == 1
    np.testing.assert_array_equal(X1, X2)
    assert y1 == y2
    pd.testing.assert_frame_equal(m1, m2)


def test_get_dataset_cached_force_reload_bypasses_caches(
    patch_dataset_cache_paths, make_train_config, monkeypatch
):
    calls = {"n": 0}

    def fake_sample(**kwargs):
        calls["n"] += 1
        return _fake_loaded(n=2)

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)

    cfg = make_train_config(sample_only=True)

    key = dc._cache_key(cfg)
    cache_file = dc._disk_cache_path(key)
    cache_file.unlink(missing_ok=True)

    _ = dc.get_dataset_cached(cfg, use_disk_cache=True, force_reload=False)
    assert calls["n"] == 1

    _ = dc.get_dataset_cached(cfg, use_disk_cache=True, force_reload=True)
    assert calls["n"] == 2


def test_get_dataset_cached_sample_only_calls_sample_loader_and_drops_unknowns(
    patch_paths, make_train_config, monkeypatch
):
    # sample_only path loads sample set and filters out 'Unknown' labels.
    def fake_sample(**kwargs):
        return _fake_loaded(n=4, unknown=True)

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)

    cfg = make_train_config(sample_only=True)
    X, y, meta = dc.get_dataset_cached(cfg, use_disk_cache=False, force_reload=False)
    assert len(y) == 3 and "Unknown" not in y


def test_get_dataset_cached_full_calls_full_loader(
    patch_paths, make_train_config, monkeypatch
):
    # full-data path uses load_ptbxl_full.
    calls = {"sample": 0, "full": 0}

    def fake_sample(**kwargs):
        calls["sample"] += 1
        return _fake_loaded(n=3)

    def fake_full(**kwargs):
        calls["full"] += 1
        return _fake_loaded(n=6)

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)
    monkeypatch.setattr(dc, "load_ptbxl_full", fake_full, raising=True)

    cfg = make_train_config(sample_only=False)
    X, y, meta = dc.get_dataset_cached(cfg, use_disk_cache=False)
    assert calls["full"] == 1 and calls["sample"] == 0


def test_get_dataset_cached_raises_when_all_labels_unknown(
    patch_paths, make_train_config, monkeypatch
):
    """If all labels are 'Unknown', post-filter validation should fail."""

    def fake_sample(**kwargs):
        X = np.random.randn(2, 12, 100).astype(np.float32)
        y = ["Unknown", "Unknown"]
        meta = pd.DataFrame({"id": [0, 1]})
        return X, y, meta

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)
    cfg = make_train_config(sample_only=True)

    with pytest.raises(ValueError, match=r"^Loaded dataset is empty\."):
        dc.get_dataset_cached(cfg, use_disk_cache=False, force_reload=True)


# ------------------------------------------------------------------------------
# D. _canon_paths and _cache_key when paths differ
# ------------------------------------------------------------------------------


def test_get_dataset_cached_cache_key_differs_when_paths_change(
    tmp_path, patch_dataset_cache_paths, make_train_config
):
    # Cache key changes when canonicalized data/sample paths differ.
    _, ptbxl_dir = patch_dataset_cache_paths

    cfg1 = make_train_config(
        data_dir=None, sample_dir=None
    )  # resolves to patched PTBXL_DATA_DIR
    key1 = dc._cache_key(cfg1)

    alt_data = tmp_path / "alt_ptbxl"
    alt_data.mkdir(parents=True, exist_ok=True)
    sdir = tmp_path / "sample_root"
    sdir.mkdir(parents=True, exist_ok=True)

    cfg2 = make_train_config(data_dir=str(alt_data), sample_dir=str(sdir))
    key2 = dc._cache_key(cfg2)

    assert key1 != key2


def test_cache_key_uses_patched_default_ptbxl_dir(
    patch_dataset_cache_paths, make_train_config
):
    """
    When data_dir is None, _canon_paths should resolve to the patched PTBXL_DATA_DIR.
    Since dataset_cache imported PTBXL_DATA_DIR by value, we patch that module's constant.
    """
    _, ptbxl_dir = patch_dataset_cache_paths

    cfg_none = make_train_config(data_dir=None, sample_dir=None)
    cfg_explicit = make_train_config(data_dir=str(ptbxl_dir), sample_dir=None)

    assert dc._cache_key(cfg_none) == dc._cache_key(cfg_explicit)


# ------------------------------------------------------------------------------
# E. Additional _validate_cfg and _validate_loaded error paths
# ------------------------------------------------------------------------------


def test_validate_cfg_subsample_frac_not_floatlike(patch_paths, make_train_config):
    # subsample_frac must be float-like
    bad = make_train_config(subsample_frac="nope")  # not float-like
    with pytest.raises(TypeError, match=r"^cfg\.subsample_frac must be float-like"):
        dc.get_dataset_cached(bad)


def test_validate_cfg_sampling_rate_wrong_type(patch_paths, make_train_config):
    # sampling rate must be an int
    bad = make_train_config(sampling_rate="500")  # wrong type (string)
    with pytest.raises(TypeError, match=r"^cfg\.sampling_rate must be int"):
        dc.get_dataset_cached(bad)


def test_validate_loaded_raises_on_bad_ndim(
    patch_paths, make_train_config, monkeypatch
):
    # X with ndim < 2 is invalid.
    def fake_sample(**kwargs):
        X = np.random.randn(5).astype(np.float32)
        y = ["NORM"] * 5
        meta = pd.DataFrame({"id": range(5)})
        return X, y, meta

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)
    cfg = make_train_config(sample_only=True)
    with pytest.raises(ValueError, match=r"^X must be a numpy array with ndim>=2"):
        dc.get_dataset_cached(cfg, use_disk_cache=False, force_reload=True)


def test_validate_loaded_raises_on_bad_meta_type(
    patch_paths, make_train_config, monkeypatch
):
    # meta must be a pandas DataFrame.
    def fake_sample(**kwargs):
        X = np.random.randn(4, 12, 100).astype(np.float32)
        y = ["NORM"] * 4
        meta = {"id": [0, 1, 2, 3]}  # not a DataFrame
        return X, y, meta

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)

    cfg = make_train_config(sample_only=True)
    with pytest.raises(ValueError, match=r"^meta must be a pandas DataFrame"):
        dc.get_dataset_cached(cfg, use_disk_cache=False, force_reload=True)


def test_validate_loaded_raises_on_empty_dataset(
    patch_paths, make_train_config, monkeypatch
):
    # Empty X is rejected.
    def fake_sample(**kwargs):
        X = np.empty((0, 12, 100), dtype=np.float32)
        y = []
        meta = pd.DataFrame({"id": []})
        return X, y, meta

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)
    cfg = make_train_config(sample_only=True)
    with pytest.raises(ValueError, match=r"^Loaded dataset is empty\."):
        dc.get_dataset_cached(cfg, use_disk_cache=False, force_reload=True)


def test_validate_loaded_raises_on_length_mismatch_pre_drop_path(
    patch_paths, make_train_config, monkeypatch
):
    # len(X), len(y), len(meta) mismatch caught in _drop_unknowns "before drop" path.
    def fake_sample(**kwargs):
        X = np.random.randn(4, 12, 100).astype(np.float32)
        y = ["NORM"] * 3
        meta = pd.DataFrame({"id": [0, 1, 2, 3]})
        return X, y, meta

    monkeypatch.setattr(dc, "load_ptbxl_sample", fake_sample, raising=True)

    cfg = make_train_config(sample_only=True)
    with pytest.raises(
        ValueError,
        match=r"^Length mismatch before drop: len\(X\)=4, len\(y\)=3, len\(meta\)=4",
    ):
        dc.get_dataset_cached(cfg, use_disk_cache=False, force_reload=True)


def test_validate_loaded_meta_type_error_hits():
    """Directly exercise _validate_loaded meta type check."""
    X = np.random.randn(2, 12, 100).astype(np.float32)
    y = ["NORM", "MI"]
    meta = {"id": [0, 1]}  # not a DataFrame
    with pytest.raises(ValueError, match=r"^meta must be a pandas DataFrame"):
        dc._validate_loaded(X, y, meta)  # type: ignore[arg-type]


def test_validate_loaded_length_mismatch():
    """Directly exercise _validate_loaded length mismatch path."""
    X = np.random.randn(4, 12, 100).astype(np.float32)
    y = ["NORM"] * 3  # mismatch
    meta = pd.DataFrame({"id": [0, 1, 2, 3]})
    with pytest.raises(
        ValueError, match=r"^Length mismatch: len\(X\)=4, len\(y\)=3, len\(meta\)=4"
    ):
        dc._validate_loaded(X, y, meta)


# ------------------------------------------------------------------------------
# F. _drop_unknowns error paths
# ------------------------------------------------------------------------------


def test_drop_unknowns_raises_when_meta_not_dataframe():
    X = np.random.randn(3, 12, 100).astype(np.float32)
    y = ["NORM", "NORM", "NORM"]
    meta = {"id": [0, 1, 2]}  # dict, not DataFrame
    with pytest.raises(ValueError, match=r"^meta must be a pandas DataFrame"):
        dc._drop_unknowns(X, y, meta)  # type: ignore[arg-type]


def test_drop_unknowns_raises_on_length_mismatch_precheck():
    X = np.random.randn(3, 12, 100).astype(np.float32)
    y = ["NORM", "NORM"]  # shorter than X
    meta = pd.DataFrame({"id": [0, 1, 2]})
    with pytest.raises(
        ValueError,
        match=r"^Length mismatch before drop: len\(X\)=3, len\(y\)=2, len\(meta\)=3",
    ):
        dc._drop_unknowns(X, y, meta)


def test_drop_unknowns_noop_when_no_unknowns():
    """If there are no 'Unknown' labels, the function should be a no-op."""
    X = np.random.randn(3, 12, 100).astype(np.float32)
    y = ["NORM", "MI", "STTC"]
    meta = pd.DataFrame({"id": [0, 1, 2]})

    X2, y2, meta2 = dc._drop_unknowns(X, y, meta)
    assert X2.shape == X.shape
    assert y2 == y
    pd.testing.assert_frame_equal(meta2, meta)
