import pytest
import torch
from pathlib import Path

from ecg_cnn.data.dataset import PTBXLFullDataset


def test_ptbxl_full_dataset_shape(tmp_path):
    # Minimal meta CSV with 1 known-good ID
    meta_csv = tmp_path / "meta.csv"
    meta_csv.write_text(
        "ecg_id,filename_lr,scp_codes\n"
        "15215,records100/15000/15215_lr,\"['NORM']\"\n"
    )

    scp_csv = tmp_path / "scp.csv"
    scp_csv.write_text("NORM,Normal sinus rhythm\n")

    ptb_dir = Path("./data/ptbxl/physionet.org/files/ptb-xl/1.0.3")
    if not ptb_dir.exists():
        pytest.skip("PTB-XL data directory not found")

    dataset = PTBXLFullDataset(
        meta_csv=meta_csv,
        scp_csv=scp_csv,
        ptb_path=ptb_dir,
    )

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (12, 1000)
    assert isinstance(y, int)
