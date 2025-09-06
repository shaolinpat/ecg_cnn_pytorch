# tests/test_loaders.py

"""
Integration smoke tests for data loaders.

Purpose
-------
Light sanity checks that the end-to-end data loading paths work on *real* PTB-XL
data on a developer machine. These tests are intentionally skipped in CI unless
explicitly enabled.

Enable locally with:
    RUN_INTEGRATION=1 pytest -m "integration and slow" -s tests/test_loaders.py

They remain conservative: minimal assertions, no file writes, no monkeypatching,
and no reliance on conftest fixtures beyond global constants.
"""

import numpy as np
import os
import pytest

from ecg_cnn.data.data_utils import load_ptbxl_full, load_ptbxl_sample
from ecg_cnn.paths import PTBXL_DATA_DIR, PROJECT_ROOT

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

INTEGRATION_ENABLED = os.getenv("RUN_INTEGRATION") == "1"

skip_reason = (
    "Integration disabled (set RUN_INTEGRATION=1) or required data dirs missing"
)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not INTEGRATION_ENABLED or not PTBXL_DATA_DIR.is_dir() or not SAMPLE_DIR.is_dir(),
    reason=skip_reason,
)
def test_loaders_integration_smoke():
    # 1) Full loader on a small subsample (1%) to keep runtime reasonable
    Xf, yf, meta_f = load_ptbxl_full(
        data_dir=PTBXL_DATA_DIR,
        subsample_frac=0.01,
        sampling_rate=100,
    )

    # Minimal sanity assertions (non-brittle)
    assert isinstance(Xf, np.ndarray)
    assert Xf.ndim == 3
    assert len(yf) == len(meta_f) == len(Xf)
    assert np.isfinite(Xf).all()
    assert all(isinstance(lbl, str) for lbl in yf)

    # Optional summaries for manual inspection (visible with -s)
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


if __name__ == "__main__":
    if not (INTEGRATION_ENABLED and PTBXL_DATA_DIR.is_dir() and SAMPLE_DIR.is_dir()):
        raise SystemExit(skip_reason)
    test_loaders_integration_smoke()
