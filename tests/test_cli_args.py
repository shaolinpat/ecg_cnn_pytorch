import argparse
import sys
import pytest
from ecg_cnn.training.cli_args import parse_args


@pytest.mark.parametrize(
    "cli_input,expected",
    [
        (
            [],
            {
                "sample_dir": "data/larger_sample",
                "sample_only": False,
                "data_dir": "../data/ptbxl/physionet.org/files/ptb-xl/1.0.3",
                "subsample_frac": 1.0,
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
    ],
)
def test_parse_args(monkeypatch, cli_input, expected):
    monkeypatch.setattr(sys, "argv", ["train.py"] + cli_input)
    args = parse_args()
    for key, val in expected.items():
        assert getattr(args, key) == val
