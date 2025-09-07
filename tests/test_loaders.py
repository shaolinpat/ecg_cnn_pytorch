# tests/test_loaders.py

"""
Integration smoke tests for data loaders.

This runs only when the expected PTB-XL directories exist locally.
No environment variables, no custom pytest markers.
"""

import numpy as np
import pytest

from ecg_cnn.data.data_utils import load_ptbxl_full, load_ptbxl_sample
from ecg_cnn.paths import PTBXL_DATA_DIR, PROJECT_ROOT

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def test_loaders_integration_smoke():
    """
    Integration smoke test:
    - If PTB-XL full dataset is available, run on a 1% subsample.
    - Always run the sample loader.
    """

    db_csv = PTBXL_DATA_DIR / "ptbxl_database.csv"
    scp_csv = PTBXL_DATA_DIR / "scp_statements.csv"

    if not (db_csv.is_file() and scp_csv.is_file() and SAMPLE_DIR.is_dir()):
        # Skip gracefully if required files are missing
        print("Skipping full loader integration smoke: PTB-XL dataset not available")
        return

    # 1) Full loader on a small subsample (1%)
    Xf, yf, meta_f = load_ptbxl_full(
        data_dir=PTBXL_DATA_DIR,
        subsample_frac=0.01,
        sampling_rate=100,
    )
    yf = [str(lbl) for lbl in yf]

    assert isinstance(Xf, np.ndarray)
    assert Xf.ndim == 3
    assert len(yf) == len(meta_f) == len(Xf)
    assert np.isfinite(Xf).all()
    assert all(isinstance(lbl, str) for lbl in yf)

    print("=== Full loader (1% subsample) ===")
    print("X.shape:", Xf.shape)
    print("Raw unique labels:", sorted(set(yf)))

    keep_f = np.array([lbl != "Unknown" for lbl in yf], dtype=bool)
    Xf2 = Xf[keep_f]
    yf2 = [lbl for i, lbl in enumerate(yf) if keep_f[i]]
    meta_f2 = meta_f.loc[keep_f].reset_index(drop=True)

    assert len(Xf2) == len(yf2) == len(meta_f2)
    print("After dropping 'Unknown':")
    print("  X.shape:", Xf2.shape)
    print("  Remaining classes:", sorted(set(yf2)))
    print()

    # 2) Sample loader on the 100-record subset
    Xs, ys, meta_s = load_ptbxl_sample(sample_dir=SAMPLE_DIR, ptb_path=PTBXL_DATA_DIR)
    ys = [str(lbl) for lbl in ys]

    assert isinstance(Xs, np.ndarray)
    assert Xs.ndim == 3
    assert len(ys) == len(meta_s) == len(Xs)
    assert np.isfinite(Xs).all()
    assert all(isinstance(lbl, str) for lbl in ys)

    print("=== Sample loader (100 records) ===")
    print("X.shape:", Xs.shape)
    print("Raw unique labels:", sorted(set(ys)))

    keep_s = np.array([lbl != "Unknown" for lbl in ys], dtype=bool)
    Xs2 = Xs[keep_s]
    ys2 = [lbl for i, lbl in enumerate(ys) if keep_s[i]]
    meta_s2 = meta_s.loc[keep_s].reset_index(drop=True)

    assert len(Xs2) == len(ys2) == len(meta_s2)
    print("After dropping 'Unknown':")
    print("  X.shape:", Xs2.shape)
    print("  Remaining classes:", sorted(set(ys2)))
