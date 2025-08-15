# data/dataset_cache.py

"""
Dataset caching utilities for PTB-XL.

Purpose
-------
Avoid re-loading / re-preprocessing the multi-GB PTB-XL data for every fold/run.
Provides:
- In-process (RAM) cache for the current Python process
- Optional on-disk cache (.npz under outputs/cache/) for fast cold starts

Public API
----------
    X, y, meta = get_dataset_cached(cfg, use_disk_cache=True, force_reload=False)

Notes
-----
Cache key includes only data-affecting fields:
(sample_only, subsample_frac, sampling_rate, data_dir, sample_dir)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ecg_cnn.config.config_loader import TrainConfig
from ecg_cnn.data.data_utils import load_ptbxl_full, load_ptbxl_sample
from ecg_cnn.paths import OUTPUT_DIR, PTBXL_DATA_DIR

# -----------------------------
# Module-level in-process cache
# -----------------------------
_RAM_CACHE: dict[str, tuple[np.ndarray, list[str], pd.DataFrame]] = {}


def _validate_cfg(cfg: TrainConfig) -> None:
    """Validate the subset of config fields that influence data loading."""
    if not isinstance(cfg, TrainConfig):
        raise TypeError(f"cfg must be TrainConfig, got {type(cfg)}")
    if not isinstance(cfg.sample_only, bool):
        raise TypeError("cfg.sample_only must be bool")
    try:
        frac = float(cfg.subsample_frac)
    except Exception as e:
        raise TypeError("cfg.subsample_frac must be float-like") from e
    if not (0.0 < frac <= 1.0):
        raise ValueError("cfg.subsample_frac must be in (0, 1]")
    if not isinstance(cfg.sampling_rate, int):
        raise TypeError("cfg.sampling_rate must be int")
    if cfg.sampling_rate not in (100, 500):
        raise ValueError("cfg.sampling_rate must be 100 or 500")
    if cfg.data_dir is not None and not Path(cfg.data_dir).exists():
        raise ValueError(f"cfg.data_dir does not exist: {cfg.data_dir}")
    if cfg.sample_dir is not None and not Path(cfg.sample_dir).exists():
        raise ValueError(f"cfg.sample_dir does not exist: {cfg.sample_dir}")


def _canon_paths(cfg: TrainConfig) -> tuple[str, str]:
    """Return canonicalized (data_dir, sample_dir) strings for stable cache keys."""
    data_dir = str((Path(cfg.data_dir) if cfg.data_dir else PTBXL_DATA_DIR).resolve())
    sample_dir = str(Path(cfg.sample_dir).resolve()) if cfg.sample_dir else ""
    return data_dir, sample_dir


def _cache_key(cfg: TrainConfig) -> str:
    """Stable short key for this dataset variant."""
    data_dir, sample_dir = _canon_paths(cfg)
    payload = f"{bool(cfg.sample_only)}|{float(cfg.subsample_frac):.6f}|{int(cfg.sampling_rate)}|{data_dir}|{sample_dir}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _disk_cache_path(key: str) -> Path:
    """Return path to the on-disk .npz cache file for a given key."""
    return OUTPUT_DIR / "cache" / f"ptbxl_{key}.npz"


def _drop_unknowns(
    X: np.ndarray, y: Iterable[str], meta: pd.DataFrame
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Remove rows with label 'Unknown' across X, y, meta."""
    if not isinstance(meta, pd.DataFrame):
        raise ValueError("meta must be a pandas DataFrame")
    y_list = list(y)
    if len(X) != len(y_list) or len(meta) != len(y_list):
        raise ValueError(
            f"Length mismatch before drop: len(X)={len(X)}, len(y)={len(y_list)}, len(meta)={len(meta)}"
        )
    keep = np.asarray([lbl != "Unknown" for lbl in y_list], dtype=bool)
    X2 = X[keep]
    y2 = [lbl for i, lbl in enumerate(y_list) if keep[i]]
    meta2 = meta.loc[keep].reset_index(drop=True)
    return X2, y2, meta2


def _validate_loaded(X: np.ndarray, y: Iterable[str], meta: pd.DataFrame) -> None:
    if not isinstance(X, np.ndarray) or X.ndim < 2:
        raise ValueError(
            f"X must be a numpy array with ndim>=2; got {type(X)} shape={getattr(X,'shape',None)}"
        )
    if not isinstance(meta, pd.DataFrame):
        raise ValueError("meta must be a pandas DataFrame")
    y_list = list(y)
    if len(X) == 0:
        raise ValueError("Loaded dataset is empty.")
    if len(X) != len(y_list) or len(X) != len(meta):
        raise ValueError(
            f"Length mismatch: len(X)={len(X)}, len(y)={len(y_list)}, len(meta)={len(meta)}"
        )


def get_dataset_cached(
    cfg: TrainConfig,
    *,
    use_disk_cache: bool = True,
    force_reload: bool = False,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Return PTB-XL arrays (X, y, meta) using in-process and optional on-disk caching.

    Parameters
    ----------
    cfg : TrainConfig
        Configuration containing data-related fields.
    use_disk_cache : bool, default=True
        If True, read/write a compressed .npz to outputs/cache/.
    force_reload : bool, default=False
        If True, bypass caches and reload from source, then refresh caches.

    Returns
    -------
    (X, y, meta)
        X: np.ndarray [N, ...]
        y: list[str] of length N
        meta: pandas.DataFrame with N rows
    """
    _validate_cfg(cfg)
    key = _cache_key(cfg)
    cache_file = _disk_cache_path(key)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # RAM cache
    if not force_reload and key in _RAM_CACHE:
        return _RAM_CACHE[key]

    # Disk cache
    if use_disk_cache and not force_reload and cache_file.exists():
        npz = np.load(cache_file, allow_pickle=True)
        X = npz["X"]
        y = npz["y"].tolist()
        meta = pd.DataFrame(npz["meta"].tolist())
        _validate_loaded(X, y, meta)
        _RAM_CACHE[key] = (X, y, meta)
        return _RAM_CACHE[key]

    # Load fresh
    data_dir, _ = _canon_paths(cfg)
    if cfg.sample_only:
        X, y, meta = load_ptbxl_sample(
            sample_dir=cfg.sample_dir,
            ptb_path=Path(data_dir),
        )
    else:
        X, y, meta = load_ptbxl_full(
            data_dir=Path(data_dir),
            subsample_frac=float(cfg.subsample_frac),
            sampling_rate=int(cfg.sampling_rate),
        )

    X, y, meta = _drop_unknowns(X, y, meta)
    _validate_loaded(X, y, meta)

    # Save caches
    _RAM_CACHE[key] = (X, y, meta)
    if use_disk_cache:
        np.savez_compressed(
            cache_file,
            X=X,
            y=np.array(y, dtype=object),
            meta=np.array(meta.to_dict(orient="records"), dtype=object),
        )

    return X, y, meta
