# tests/test_cli_args.py

"""
Tests for ecg_cnn.training.cli_args.

Covers
------
    - parse_training_args(): defaults and per-flag overrides
    - parse_evaluate_args(): defaults and per-flag overrides
    - _positive_int(): value parsing/validation
    - override_config_with_args(): happy path + validation/error paths
"""

import argparse
import sys

import pytest

from ecg_cnn.training.cli_args import (
    parse_training_args,
    _positive_int,
    override_config_with_args,
)


# ------------------------------------------------------------------------------
# def parse_training_args():
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
                "n_epochs": None,
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
        (["--n_epochs", "20"], {"n_epochs": 20}),
        (["--model", "ECGConvNetV2"], {"model": "ECGConvNetV2"}),
        (["--save-best"], {"save_best": True}),
        (["--verbose"], {"verbose": True}),
    ],
)
def test_parse_training_args(monkeypatch, cli_input, expected):
    monkeypatch.setattr(sys, "argv", ["train.py"] + cli_input)
    args = parse_training_args()

    for key, expected_val in expected.items():
        actual_val = getattr(args, key)
        assert (
            actual_val == expected_val
        ), f"Expected {expected_val!r} for {key}, got {actual_val!r}"


def test_override_config_updates_fields(make_train_config, make_args):
    # Step 1: Create a baseline TrainConfig (matches typical baseline.yaml)
    base_config = make_train_config(
        model="ECGConvNet", lr=0.001, batch_size=64, verbose=False
    )

    # Step 2: mock CLI args (as if parsed by argparse)
    cli_args = make_args(
        model="ECGConvNetV2",
        lr=0.005,
        batch_size=128,
        weight_decay=0.01,
        n_epochs=25,
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
    assert updated_config.n_epochs == 25
    assert updated_config.n_folds == 2
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
def test_parse_training_args_invalid_kernel_sizes_count_or_type(
    monkeypatch, invalid_kernels
):
    monkeypatch.setattr(sys, "argv", ["train.py"] + invalid_kernels)
    with pytest.raises(SystemExit):
        parse_training_args()


def test_parse_training_args_valid_kernel_sizes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--kernel-sizes", "3", "5", "7"])
    args = parse_training_args()
    assert args.kernel_sizes == [3, 5, 7]


# ------------------------------------------------------------------------------
# def _positive_int():
# ------------------------------------------------------------------------------


def test_positive_int_valid():
    assert _positive_int("5") == 5


def test_positive_int_negative():
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"^-3 is not a positive integer"
    ):
        _positive_int("-3")


def test_positive_int_non_numeric():
    with pytest.raises(argparse.ArgumentTypeError, match=r"^five is not an integer"):
        _positive_int("five")


# ------------------------------------------------------------------------------
# override_config_with_args()
# ------------------------------------------------------------------------------


# 1) valid args case
def test_override_config_with_args_with_valid_args(make_train_config, make_args):
    base_config = make_train_config()
    args = make_args(
        lr=0.0005,
        batch_size=16,
        weight_decay=0.01,
        n_epochs=5,
        n_folds=1,
        sample_only=False,
        subsample_frac=0.5,
        sampling_rate=100,
        data_dir="/tmp/data",
        sample_dir="/tmp/sample",
        verbose=True,
        model="ECGConvNet",
    )
    new_cfg = override_config_with_args(base_config, args)
    assert new_cfg.lr == 0.0005
    assert new_cfg.batch_size == 16
    assert new_cfg.weight_decay == 0.01
    assert new_cfg.n_epochs == 5
    assert new_cfg.sampling_rate == 100
    assert new_cfg.subsample_frac == 0.5
    assert str(new_cfg.data_dir) == "/tmp/data"
    assert str(new_cfg.sample_dir) == "/tmp/sample"
    assert new_cfg.verbose is True


# 2) ignores None
def test_override_config_with_args_ignores_none_values(make_train_config, make_args):
    base_config = make_train_config(lr=0.001, batch_size=8)
    args = make_args(
        lr=None,
        batch_size=None,
        weight_decay=None,
        n_epochs=None,
        n_folds=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
        verbose=None,
        model=None,
    )
    new_cfg = override_config_with_args(base_config, args)
    assert new_cfg.lr == 0.001
    assert new_cfg.batch_size == 8


# 3) invalid subsample
def test_override_config_with_args_invalid_subsample_frac(make_train_config, make_args):
    base_config = make_train_config()
    args = make_args(subsample_frac=1.5)  # invalid
    with pytest.raises(ValueError, match=r"^subsample_frac must be in \(0.0, 1.0\]"):
        override_config_with_args(base_config, args)


def test_override_config_with_args_invalid_sampling_rate(make_train_config, make_args):
    base_config = make_train_config()
    args = make_args(sampling_rate=0)  # invalid
    with pytest.raises(ValueError, match=r"^sampling_rate must be 100 or 500"):
        override_config_with_args(base_config, args)


def test_override_config_with_args_rejects_nonstring_model(
    make_train_config, make_args
):
    config = make_train_config(model=["ECGConvNet"])  # Invalid: not a string
    dummy_args = make_args(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        n_epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
    )
    with pytest.raises(ValueError, match=r"^Model must be a non-empty string"):
        override_config_with_args(config, dummy_args)


def test_override_config_with_args_accepts_none_data_dir_and_sample_dir(
    make_train_config, make_args
):
    config = make_train_config(data_dir=None, sample_dir=None)
    dummy_args = make_args(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        n_epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
    )
    updated = override_config_with_args(config, dummy_args)
    assert updated.model == config.model
    assert updated.lr == config.lr
    assert updated.batch_size == config.batch_size
    assert updated.weight_decay == config.weight_decay
    assert updated.n_epochs == config.n_epochs
    assert updated.save_best is config.save_best
    assert updated.sample_only is config.sample_only
    assert updated.subsample_frac == config.subsample_frac
    assert updated.sampling_rate == config.sampling_rate
    assert updated.data_dir is None
    assert updated.sample_dir is None


def test_override_config_with_args_rejects_nonstring_data_dir(
    make_train_config, make_args
):
    config = make_train_config(data_dir=0.22)  # Invalid: not a string/path/None
    dummy_args = make_args(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        n_epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
    )
    with pytest.raises(ValueError, match=r"^data_dir must be a string, Path, or None"):
        override_config_with_args(config, dummy_args)


def test_override_config_with_args_rejects_nonstring_sample_dir(
    make_train_config, make_args
):
    config = make_train_config(sample_dir=123)  # Invalid: not a string/path/None
    dummy_args = make_args(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        n_epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
        verbose=None,
    )
    with pytest.raises(
        ValueError, match=r"^sample_dir must be a string, Path, or None"
    ):
        override_config_with_args(config, dummy_args)


def test_override_config_with_args_rejects_non_boolean_verbose(
    make_train_config, make_args
):
    config = make_train_config(verbose="true")  # Invalid type
    dummy_args = make_args(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        n_epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
        verbose=None,
    )
    with pytest.raises(ValueError, match=r"^verbose must be a boolean or None"):
        override_config_with_args(config, dummy_args)


@pytest.mark.parametrize(
    "cli_args_dict, expected_overrides",
    [
        (
            {"batch_size": 128, "lr": 0.001, "verbose": True},
            {"batch_size": 128, "lr": 0.001, "verbose": True},
        ),
        ({"model": "ECGConvNetV2"}, {"model": "ECGConvNetV2"}),
        (
            {"subsample_frac": 0.2, "save_best": False},
            {"subsample_frac": 0.2, "save_best": False},
        ),
        (
            {"sample_only": True, "data_dir": "/mnt/data", "sample_dir": "sample/path"},
            {"sample_only": True, "data_dir": "/mnt/data", "sample_dir": "sample/path"},
        ),
        (
            {"n_epochs": 50, "weight_decay": 0.0001},
            {"n_epochs": 50, "weight_decay": 0.0001},
        ),
    ],
)
def test_override_config_with_args_applies_correctly(
    make_train_config, make_args, cli_args_dict, expected_overrides
):
    # Start from a YAML-style default config
    base_config = make_train_config(
        model="ECGConvNet",
        lr=0.01,
        batch_size=64,
        weight_decay=0.0,
        n_epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
        n_folds=2,
    )

    # Build a Namespace-like object via fixture and fill unspecified with None
    fields = [
        "model",
        "lr",
        "batch_size",
        "weight_decay",
        "n_epochs",
        "save_best",
        "sample_only",
        "subsample_frac",
        "sampling_rate",
        "data_dir",
        "sample_dir",
        "verbose",
    ]
    filled = {k: cli_args_dict.get(k, None) for k in fields}
    cli_args = make_args(**filled)

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


def test_override_config_with_args_rejects_empty_model(make_train_config, make_args):
    config = make_train_config(model="")
    dummy_args = make_args(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        n_epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
        verbose=None,
    )
    with pytest.raises(ValueError, match=r"^Model must be a non-empty string."):
        override_config_with_args(config, dummy_args)


def test_override_config_with_args_rejects_model_with_blanked_name(
    make_train_config, make_args
):
    config = make_train_config(model="  ")
    dummy_args = make_args(
        model=None,
        lr=None,
        batch_size=None,
        weight_decay=None,
        n_epochs=None,
        save_best=None,
        sample_only=None,
        subsample_frac=None,
        sampling_rate=None,
        data_dir=None,
        sample_dir=None,
        verbose=None,
    )
    with pytest.raises(ValueError, match=r"^Model must be a non-empty string"):
        override_config_with_args(config, dummy_args)
