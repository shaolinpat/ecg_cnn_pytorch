# tests/test_dataset.py

"""
Tests for ecg_cnn.data.dataset.PTBXLFullDataset.

Covers
------
    - End-to-end shape/type check via __getitem__
    - __len__ correctness with a single mocked record
"""

import numpy as np
import os
import pandas as pd
import pytest

# Optional deps: skip cleanly if missing
wfdb = pytest.importorskip("wfdb", reason="wfdb not installed")
torch = pytest.importorskip("torch", reason="torch not installed")

from pathlib import Path

from ecg_cnn.data.dataset import PTBXLFullDataset

# ------------------------------------------------------------------------------
# Helper: write a minimal WFDB record (lr) under the PTB-XL-like structure
# ------------------------------------------------------------------------------


def _write_mock_ptb_record(root: Path, ecg_id: int) -> str:
    """
    Creates
    -------
    A tiny WFDB record like 'records100/00000/00123_lr' under root.

    Returns
    -------
    str:
        The relative path (without extension) suitable for filename_lr.
    """
    bucket = f"{(ecg_id // 1000) * 1000:05d}"
    rec_dir = root / "records100" / bucket
    rec_dir.mkdir(parents=True, exist_ok=True)

    rec_name = f"{ecg_id:05d}_lr"
    signal = np.random.normal(0.0, 1e-3, size=(1000, 12))  # (N, leads)

    # wfdb.wrsamp writes to CWD; switch in/out safely
    prev = os.getcwd()
    os.chdir(rec_dir)
    try:
        wfdb.wrsamp(
            rec_name,
            fs=100,
            units=["mV"] * 12,
            sig_name=[f"lead{i}" for i in range(12)],
            p_signal=signal,
        )
    finally:
        os.chdir(prev)

    return f"records100/{bucket}/{rec_name}"


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------


def test_ptbxl_full_dataset_shape(tmp_path: Path):
    """
    End-to-end smoke test: one-row meta + one mock WFDB record.
    Ensures __len__ and __getitem__ work and tensor shape matches (12, 1000).
    """
    # Arrange: PTB-XL root with a single mock record
    ptb_root = tmp_path / "ptbxl"
    ecg_id = 15215
    rel_path = _write_mock_ptb_record(
        ptb_root, ecg_id
    )  # e.g., records100/15000/15215_lr

    # Meta CSV (index = ecg_id) with filename_lr + minimal scp_codes
    meta_df = pd.DataFrame(
        {"ecg_id": [ecg_id], "filename_lr": [rel_path], "scp_codes": [["NORM"]]}
    ).set_index("ecg_id")

    meta_csv = tmp_path / "meta.csv"
    meta_df.to_csv(meta_csv)

    # Minimal SCP CSV (index = code) - content shape is enough for loader paths
    scp_df = pd.DataFrame({"description": ["Normal"]}, index=["NORM"])
    scp_csv = tmp_path / "scp_statements.csv"
    scp_df.to_csv(scp_csv)

    # Act: instantiate dataset and fetch the only sample
    dataset = PTBXLFullDataset(meta_csv=meta_csv, scp_csv=scp_csv, ptb_path=ptb_root)

    # Assert: __len__ and sample shape/types
    assert len(dataset) == 1
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (12, 1000)
    assert isinstance(y, int)
