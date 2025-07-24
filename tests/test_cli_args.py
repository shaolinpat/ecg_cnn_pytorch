import argparse
import pytest
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from ecg_cnn.config.config_loader import TrainConfig
from ecg_cnn.paths import PTBXL_DATA_DIR
from ecg_cnn.training.cli_args import (
    parse_args,
    _positive_int,
    override_config_with_args,
)


# ------------------------------------------------------------------------------
# def parse_args():
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# def _positive_int(value):
# ------------------------------------------------------------------------------


def test_positive_int_valid():
    assert _positive_int("5") == 5


def test_positive_int_negative():
    with pytest.raises(argparse.ArgumentTypeError, match="not a positive integer"):
        _positive_int("-3")


def test_positive_int_non_numeric():
    with pytest.raises(argparse.ArgumentTypeError, match="not an integer"):
        _positive_int("five")


# ------------------------------------------------------------------------------
# def override_config_with_args(config: TrainConfig, args: Namespace)
#       -> TrainConfig:
# ------------------------------------------------------------------------------


def make_min_config(**overrides):
    defaults = dict(
        model="ECGConvNet",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir="/data",
        sample_dir="/samples",
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_override_config_with_valid_args():
    config = make_min_config()
    args = Namespace(
        model="ECGConvNetPlus",
        lr=0.01,
        batch_size=128,
        weight_decay=0.001,
        epochs=20,
        save_best=False,
        sample_only=True,
        subsample_frac=0.5,
        sampling_rate=500,
        data_dir="/new/data",
        sample_dir="/new/samples",
    )
    updated = override_config_with_args(config, args)
    assert updated.model == "ECGConvNetPlus"
    assert updated.lr == 0.01
    assert updated.batch_size == 128
    assert updated.weight_decay == 0.001
    assert updated.epochs == 20
    assert updated.save_best is False
    assert updated.sample_only is True
    assert updated.subsample_frac == 0.5
    assert updated.sampling_rate == 500
    assert updated.data_dir == "/new/data"
    assert updated.sample_dir == "/new/samples"


def test_override_config_ignores_none_values():
    config = make_min_config(batch_size=64)
    args = Namespace(batch_size=None)
    updated = override_config_with_args(config, args)
    assert updated.batch_size == 64


def test_override_config_invalid_subsample_frac():
    config = make_min_config()
    args = Namespace(subsample_frac=1.5)
    with pytest.raises(ValueError, match="subsample_frac must be in"):
        override_config_with_args(config, args)


def test_override_config_invalid_sampling_rate():
    config = make_min_config()
    args = Namespace(sampling_rate=250)
    with pytest.raises(ValueError, match="sampling_rate must be 100 or 500"):
        override_config_with_args(config, args)


def test_override_config_rejects_nonstring_model():
    config = TrainConfig(
        model=["ECGConvNet"],  # Invalid: not a string
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir="/data",
        sample_dir="/sample",
    )
    dummy_args = argparse.Namespace(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
    )
    with pytest.raises(ValueError, match="model must be a string"):
        override_config_with_args(config, dummy_args)


def test_override_config_accepts_none_data_dir_and_sample_dir():
    config = TrainConfig(
        model="ECGConvNet",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
    )
    dummy_args = argparse.Namespace(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
    )
    updated = override_config_with_args(config, dummy_args)
    assert updated.model == "ECGConvNet"
    assert updated.lr == 0.001
    assert updated.batch_size == 64
    assert updated.weight_decay == 0.0
    assert updated.epochs == 10
    assert updated.save_best is True
    assert updated.sample_only is False
    assert updated.subsample_frac == 1.0
    assert updated.sampling_rate == 100
    assert updated.data_dir == None
    assert updated.sample_dir == None


def test_override_config_rejects_nonstring_data_dir():
    config = TrainConfig(
        model="ECGConvNet",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=0.22,  # Invalid: not a string
        sample_dir="/sample",
    )
    dummy_args = argparse.Namespace(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
    )
    with pytest.raises(ValueError, match="data_dir must be a string or None"):
        override_config_with_args(config, dummy_args)


def test_override_config_rejects_nonstring_sample_dir():
    config = TrainConfig(
        model="ECGConvNet",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir="/data",
        sample_dir=123,  # Invalid: not a string
    )
    dummy_args = argparse.Namespace(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
    )
    with pytest.raises(ValueError, match="sample_dir must be a string or None"):
        override_config_with_args(config, dummy_args)
