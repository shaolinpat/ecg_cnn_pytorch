import sys
import pytest
from pathlib import Path
from ecg_cnn.config import PTBXL_DATA_DIR
from ecg_cnn.training.cli_args import parse_args


@pytest.mark.parametrize(
    "cli_input,expected",
    [
        (
            [],
            {
                "sample_dir": "data/larger_sample",
                "sample_only": False,
                "data_dir": PTBXL_DATA_DIR,
                "subsample_frac": 1.0,
                "batch_size": 32,
                "kernel_sizes": [16, 3, 3],
                "conv_dropout": 0.3,
                "fc_dropout": 0.5,
            },
        ),
        (
            ["--sample-only"],
            {
                "sample_only": True,
            },
        ),
        (
            ["--sample-dir", "samples/test"],
            {
                "sample_dir": "samples/test",
            },
        ),
        (
            ["--data-dir", "/mnt/ptbxl"],
            {
                "data_dir": "/mnt/ptbxl",
            },
        ),
        (
            ["--subsample-frac", "0.2"],
            {
                "subsample_frac": 0.2,
            },
        ),
        (
            ["--batch-size", "64"],
            {
                "batch_size": 64,
            },
        ),
        (
            ["--kernel-sizes", "9", "5", "3"],
            {
                "kernel_sizes": [9, 5, 3],
            },
        ),
        (
            ["--conv-dropout", "0.1"],
            {
                "conv_dropout": 0.1,
            },
        ),
        (
            ["--fc-dropout", "0.4"],
            {
                "fc_dropout": 0.4,
            },
        ),
    ],
)
def test_parse_args(monkeypatch, cli_input, expected):
    monkeypatch.setattr(sys, "argv", ["train.py"] + cli_input)
    args = parse_args()
    for key, val in expected.items():
        actual = getattr(args, key)
        if isinstance(actual, Path) and isinstance(val, (str, Path)):
            assert actual.resolve() == Path(val).resolve()
        else:
            assert actual == val


@pytest.mark.parametrize(
    "invalid_kernels",
    [
        ["--kernel-sizes", "3", "5"],  # too few
        ["--kernel-sizes", "3", "5", "7", "9"],  # too many
        ["--kernel-sizes", "3", "-5", "7"],  # negative value
        ["--kernel-sizes", "3", "abc", "7"],  # non-integer string
        ["--kernel-sizes", "3", "0", "7"],  # zero not allowed
    ],
)
def test_invalid_kernel_sizes_count_or_type(monkeypatch, invalid_kernels):
    monkeypatch.setattr(sys, "argv", ["train.py"] + invalid_kernels)
    with pytest.raises(SystemExit):
        parse_args()


def test_valid_kernel_sizes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--kernel-sizes", "3", "5", "7"])
    args = parse_args()
    assert args.kernel_sizes == [3, 5, 7]
