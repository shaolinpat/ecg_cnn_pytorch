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
    "cli_input, expected",
    [
        # === Defaults (when no CLI arguments are passed) ===
        (
            [],
            {
                "sample_dir": None,
                "sample_only": None,
                "data_dir": None,
                "subsample_frac": None,
                "batch_size": None,
                "kernel_sizes": None,
                "conv_dropout": None,
                "fc_dropout": None,
                "lr": None,
                "weight_decay": None,
                "epochs": None,
                "model": None,
                "save_best": None,
                "verbose": None,
            },
        ),
        # === Individual Overrides ===
        (["--sample-only"], {"sample_only": True}),
        (["--sample-dir", "samples/test"], {"sample_dir": "samples/test"}),
        (["--data-dir", "/mnt/ptbxl"], {"data_dir": "/mnt/ptbxl"}),
        (["--subsample-frac", "0.2"], {"subsample_frac": 0.2}),
        (["--batch-size", "64"], {"batch_size": 64}),
        (["--kernel-sizes", "9", "5", "3"], {"kernel_sizes": [9, 5, 3]}),
        (["--conv-dropout", "0.1"], {"conv_dropout": 0.1}),
        (["--fc-dropout", "0.4"], {"fc_dropout": 0.4}),
        (["--lr", "0.0005"], {"lr": 0.0005}),
        (["--weight-decay", "0.001"], {"weight_decay": 0.001}),
        (["--epochs", "20"], {"epochs": 20}),
        (["--model", "ECGConvNetV2"], {"model": "ECGConvNetV2"}),
        (["--save-best"], {"save_best": True}),
        (["--verbose"], {"verbose": True}),
    ],
)
def test_parse_args(monkeypatch, cli_input, expected):
    monkeypatch.setattr(sys, "argv", ["train.py"] + cli_input)
    args = parse_args()

    for key, expected_val in expected.items():
        actual_val = getattr(args, key)

        # Normalize Paths for comparison
        if isinstance(actual_val, Path) or isinstance(expected_val, Path):
            assert Path(actual_val) == Path(
                expected_val
            ), f"{key} path mismatch: expected {expected_val}, got {actual_val}"
        else:
            assert (
                actual_val == expected_val
            ), f"Expected {expected_val!r} for {key}, got {actual_val!r}"


def test_override_config_updates_fields():
    # Step 1: Create a baseline TrainConfig (matches typical baseline.yaml)
    base_config = TrainConfig(
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
        verbose=False,
    )

    # Step 2: Create a mock CLI args object (as if parsed by argparse)
    cli_args = Namespace(
        model="ECGConvNetV2",
        lr=0.005,
        batch_size=128,
        weight_decay=0.01,
        epochs=25,
        save_best=False,
        sample_only=True,
        subsample_frac=0.2,
        sampling_rate=500,
        data_dir="/mnt/data",
        sample_dir="samples/override",
        verbose=True,
    )

    # Step 3: Apply overrides
    updated_config = override_config_with_args(base_config, cli_args)

    # Step 4: Assert that overrides took effect
    assert updated_config.model == "ECGConvNetV2"
    assert updated_config.lr == 0.005
    assert updated_config.batch_size == 128
    assert updated_config.weight_decay == 0.01
    assert updated_config.epochs == 25
    assert updated_config.save_best is False
    assert updated_config.sample_only is True
    assert updated_config.subsample_frac == 0.2
    assert updated_config.sampling_rate == 500
    assert str(updated_config.data_dir) == "/mnt/data"
    assert str(updated_config.sample_dir) == "samples/override"
    assert updated_config.verbose is True


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


def test_override_config_with_args_with_valid_args():
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


def test_override_config_with_args_ignores_none_values():
    config = make_min_config(batch_size=64)
    args = Namespace(batch_size=None)
    updated = override_config_with_args(config, args)
    assert updated.batch_size == 64


def test_override_config_with_args_invalid_subsample_frac():
    config = make_min_config()
    args = Namespace(subsample_frac=1.5)
    with pytest.raises(ValueError, match="subsample_frac must be in"):
        override_config_with_args(config, args)


def test_override_config_with_args_invalid_sampling_rate():
    config = make_min_config()
    args = Namespace(sampling_rate=250)
    with pytest.raises(ValueError, match="sampling_rate must be 100 or 500"):
        override_config_with_args(config, args)


def test_override_config_with_args_rejects_nonstring_model():
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
    with pytest.raises(ValueError, match="Model must be a non-empty string"):
        override_config_with_args(config, dummy_args)


def test_override_config_with_args_accepts_none_data_dir_and_sample_dir():
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


def test_override_config_with_args_rejects_nonstring_data_dir():
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
    with pytest.raises(ValueError, match="data_dir must be a string, Path, or None"):
        override_config_with_args(config, dummy_args)


def test_override_config_with_args_rejects_nonstring_sample_dir():
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
        verbose=None,
    )
    with pytest.raises(ValueError, match="sample_dir must be a string, Path, or None"):
        override_config_with_args(config, dummy_args)


def test_override_confi_with_args_rejects_non_boolean_verbose():
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
        data_dir="test",
        sample_dir="/sample",
        verbose="true",
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
        verbose=None,
    )
    with pytest.raises(ValueError, match="verbose must be a boolean or None"):
        override_config_with_args(config, dummy_args)


@pytest.mark.parametrize(
    "cli_args, expected_overrides",
    [
        (
            Namespace(batch_size=128, lr=0.001, verbose=True),
            {"batch_size": 128, "lr": 0.001, "verbose": True},
        ),
        (Namespace(model="ECGConvNetV2"), {"model": "ECGConvNetV2"}),
        (
            Namespace(subsample_frac=0.2, save_best=False),
            {"subsample_frac": 0.2, "save_best": False},
        ),
        (
            Namespace(sample_only=True, data_dir="/mnt/data", sample_dir="sample/path"),
            {
                "sample_only": True,
                "data_dir": "/mnt/data",
                "sample_dir": "sample/path",
            },
        ),
        (
            Namespace(epochs=50, weight_decay=0.0001),
            {"epochs": 50, "weight_decay": 0.0001},
        ),
    ],
)
def test_override_config_with_args_applies_correctly(cli_args, expected_overrides):
    # Start from a YAML-style default config
    base_config = TrainConfig(
        model="ECGConvNet",
        lr=0.01,
        batch_size=64,
        weight_decay=0.0,
        epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
    )

    # Fill in unspecified CLI args with None to simulate realistic Namespace
    for field in [
        "model",
        "lr",
        "batch_size",
        "weight_decay",
        "epochs",
        "save_best",
        "sample_only",
        "subsample_frac",
        "sampling_rate",
        "data_dir",
        "sample_dir",
        "verbose",
    ]:
        if not hasattr(cli_args, field):
            setattr(cli_args, field, None)

    # Apply CLI overrides
    updated = override_config_with_args(base_config, cli_args)

    # Verify overridden fields
    for key, expected_value in expected_overrides.items():
        actual_value = getattr(updated, key)
        assert (
            actual_value == expected_value
        ), f"{key}: expected {expected_value!r}, got {actual_value!r}"

    # Ensure other fields remain unchanged
    for key in base_config.__dataclass_fields__:
        if key not in expected_overrides:
            assert getattr(updated, key) == getattr(
                base_config, key
            ), f"{key} unexpectedly changed"


def test_override_config_with_args_rejects_empty_model():
    config = TrainConfig(
        model="",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir="test",
        sample_dir="/sample",
        verbose="true",
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
        verbose=None,
    )
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        override_config_with_args(config, dummy_args)


def test_override_config_with_args_rejects_model_with_blankd_name():
    config = TrainConfig(
        model="  ",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir="test",
        sample_dir="/sample",
        verbose="true",
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
        verbose=None,
    )
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        override_config_with_args(config, dummy_args)
