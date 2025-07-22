# import numpy as np
# import os
# import pandas as pd
import pytest

# import torch
# import wfdb

# from pathlib import Path
# from unittest.mock import patch
# from ecg_cnn.data.dataset import PTBXLFullDataset

from ecg_cnn.utils.plot_utils import (
    _validate_hparams,
    _build_plot_title,
    format_hparams,
)


# ------------------------------------------------------------------------------
# def _validate_hparams(
#     lr: float,
#     bs: int,
#     wd: float,
#     fold: int,
#     epochs: int,
#     prefix: str,
#     fname_metric: str = "",
# ) -> None:
# ------------------------------------------------------------------------------


def test_validate_hparams_all_valid_minimal():
    # Should not raise
    _validate_hparams(0.001, 32, 0.01, 0, 10, "test")


def test_validate_hparams_all_valid_full():
    # Should not raise
    _validate_hparams(0.001, 32, 0.01, 1, 100, "test", fname_metric="loss")


def test_validate_hparams_lr_not_int_or_float():
    with pytest.raises(ValueError, match="Learning rate must be positive int or float"):
        _validate_hparams(
            lr="0.001",
            bs=32,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_lr_negative():
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        _validate_hparams(
            lr=-0.001,
            bs=32,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_lr_too_small():
    with pytest.raises(
        ValueError,
        match=r"Learning rate must be positive int or float in range \[1e-6, 1\.0\]",
    ):
        _validate_hparams(
            lr=0.0,
            bs=32,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_lr_too_large():
    with pytest.raises(
        ValueError,
        match=r"Learning rate must be positive int or float in range \[1e-6, 1\.0\]",
    ):
        _validate_hparams(
            lr=10.001,
            bs=32,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_bs_not_int():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        _validate_hparams(
            lr=0.001,
            bs=32.0,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_bs_too_small():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        _validate_hparams(
            lr=0.001,
            bs=0,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_bs_too_large():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        _validate_hparams(
            lr=0.001,
            bs=4097,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_wd_not_int_or_float():
    with pytest.raises(
        ValueError,
        match="Weight decay must be int or float",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd="0.0001",
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_wd_not_too_small():
    with pytest.raises(
        ValueError,
        match=r"Weight decay must be int or float in range \[0.0, 1.0\]",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd=-0.1,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_wd_not_too_large():
    with pytest.raises(
        ValueError,
        match=r"Weight decay must be int or float in range \[0.0, 1.0\]",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd=2,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_fold_not_int_but_string():
    with pytest.raises(
        ValueError,
        match="Fold number must be a non-negative integer",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd=0.01,
            fold="2",
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_fold_not_int_but_float():
    with pytest.raises(
        ValueError,
        match="Fold number must be a non-negative integer",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2.0,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_fold_too_small():
    with pytest.raises(
        ValueError,
        match="Fold number must be a non-negative integer",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=-1,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_epochs_not_int_but_string():
    with pytest.raises(
        ValueError,
        match="Epochs must be an integer",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epochs="3",
            prefix="bad",
        )


def test_validate_hparams_epochs_not_int_but_float():
    with pytest.raises(
        ValueError,
        match="Epochs must be an integer",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epochs=3.0,
            prefix="bad",
        )


def test_validate_hparams_epochs_too_small():
    with pytest.raises(
        ValueError,
        match=r"Epochs must be an integer in range \[1, 1000\]",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epochs=0,
            prefix="bad",
        )


def test_validate_hparams_epochs_too_large():
    with pytest.raises(
        ValueError,
        match=r"Epochs must be an integer in range \[1, 1000\]",
    ):
        _validate_hparams(
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epochs=1001,
            prefix="bad",
        )


def test_validate_hparams_prefix_not_string():
    with pytest.raises(ValueError, match="Prefix must be a non-empty string"):
        _validate_hparams(lr=0.001, bs=32, wd=0.001, fold=0, epochs=10, prefix=3)


def test_validate_hparams_prefix_empty_string():
    with pytest.raises(ValueError, match="Prefix must be a non-empty string"):
        _validate_hparams(lr=0.001, bs=32, wd=0.001, fold=0, epochs=10, prefix="")


def test_validate_hparams_fname_metric_not_string():
    with pytest.raises(ValueError, match="Metric name must be a string"):
        _validate_hparams(
            lr=0.001, bs=32, wd=0.001, fold=0, epochs=10, prefix="bad", fname_metric=3
        )


# ------------------------------------------------------------------------------
# def _build_plot_title(metric: str, lr: float, bs: int, wd: float, fold: int)
#     -> str:
# ------------------------------------------------------------------------------


def test_build_plot_title_valid():
    result = _build_plot_title(metric="Accuracy", lr=0.001, bs=32, wd=0.01, fold=2)
    assert result == "Accuracy by Epoch\nLR=0.001, BS=32, WD=0.01, Fold=2"


def test_build_plot_title_invalid_lr():
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        _build_plot_title(metric="Accuracy", lr=-0.001, bs=32, wd=0.01, fold=2)


def test_build_plot_title_invalid_metric_type():
    with pytest.raises(ValueError, match="Metric name must be a string"):
        _build_plot_title(metric=123, lr=0.001, bs=32, wd=0.01, fold=2)


# ------------------------------------------------------------------------------
# def format_hparams(
#     lr: float,
#     bs: int,
#     wd: float,
#     fold: int,
#     epochs: int,
#     prefix: str,
#     fname_metric: str = "",
# ) -> str:
# ------------------------------------------------------------------------------
def test_format_hparams_lr_many_decimal_places():
    result = format_hparams(
        lr=0.0001234567, bs=32, wd=0.0009876543, fold=2, epochs=3, prefix="final"
    )
    assert "lr000123" in result and "wd000988" in result


def test_format_hparams_basic():
    result = format_hparams(
        lr=0.001,
        bs=32,
        wd=0.0001,
        fold=2,
        epochs=3,
        prefix="final",
        fname_metric="accuracy",
    )
    assert result == "final_accuracy_lr001_bs32_wd0001_fold2_epoch3"


def test_format_hparams_no_metric():
    result = format_hparams(
        lr=0.001,
        bs=32,
        wd=0.0001,
        fold=2,
        epochs=3,
        prefix="best",
    )
    assert result == "best_lr001_bs32_wd0001_fold2_epoch3"


def test_format_hparams_trim_zeros():
    result = format_hparams(
        lr=0.00100,
        bs=32,
        wd=0.000100,
        fold=2,
        epochs=3,
        prefix="best",
    )
    assert result == "best_lr001_bs32_wd0001_fold2_epoch3"


def test_format_hparams_lr_bottom_range_ok():
    result = format_hparams(
        lr=0.000001, bs=128, wd=1, fold=1, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_lr000001_" in result


def test_format_hparams_lr_top_range_ok_int():
    result = format_hparams(
        lr=1, bs=128, wd=1, fold=1, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_lr1_" in result


def test_format_hparams_lr_top_range_ok_float():
    result = format_hparams(
        lr=1.0, bs=128, wd=1, fold=1, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_lr1_" in result


def test_format_hparams_bs_bottom_range():
    result = format_hparams(
        lr=0.001, bs=1, wd=0.0, fold=1, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_bs1_" in result


def test_format_hparams_bs_top_range():
    result = format_hparams(
        lr=0.001, bs=4096, wd=0.0, fold=1, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_bs4096_" in result


def test_format_hparams_wd_zero_ok():
    result = format_hparams(
        lr=0.001, bs=128, wd=0.0, fold=1, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_wd0_" in result


def test_format_hparams_wd_one_ok():
    result = format_hparams(
        lr=0.001, bs=128, wd=1.0, fold=1, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_wd1_" in result


def test_format_hparams_wd_zero_int_ok():
    result = format_hparams(
        lr=0.001, bs=128, wd=0, fold=1, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_wd0_" in result


def test_format_hparams_wd_one_int_ok():
    result = format_hparams(
        lr=0.001, bs=128, wd=1, fold=1, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_wd1_" in result


def test_format_hparams_fold_zero():
    result = format_hparams(
        lr=0.001, bs=128, wd=0.1, fold=0, epochs=5, prefix="test", fname_metric="loss"
    )
    assert "_fold0_" in result


def test_format_hparams_epochs_bottom_of_range():
    result = format_hparams(
        lr=0.001, bs=128, wd=0.1, fold=0, epochs=1, prefix="test", fname_metric="loss"
    )
    assert "_epoch1" in result


def test_format_hparams_epochs_top_of_range():
    result = format_hparams(
        lr=0.001,
        bs=128,
        wd=0.1,
        fold=0,
        epochs=1000,
        prefix="test",
        fname_metric="loss",
    )
    assert "_epoch1000" in result


def test_format_hparams_prefix_lower_cased():
    result = format_hparams(
        lr=0.001, bs=128, wd=0.1, fold=0, epochs=1, prefix="TEST", fname_metric="loss"
    )
    assert "test_loss" in result


def test_format_hparams_fname_metric_empty_string_ok():
    result = format_hparams(
        lr=0.001, bs=128, wd=0.1, fold=0, epochs=1, prefix="test", fname_metric=""
    )
    assert "test_lr" in result


def test_format_hparams_fname_metric_missing_ok():
    result = format_hparams(lr=0.001, bs=128, wd=0.1, fold=0, epochs=1, prefix="test")
    assert "test_lr" in result


def test_format_hparams_fname_metric_lower_cased():
    result = format_hparams(
        lr=0.001, bs=128, wd=0.1, fold=0, epochs=1, prefix="test", fname_metric="LOSS"
    )
    assert "test_loss" in result


# ------------------------------------------------------------------------------
# def save_plot_curves(
#     x_vals: list[float],
#     y_vals: list[float],
#     x_label: str,
#     y_label: str,
#     title_metric: str,
#     fname_metric: str,
#     lr: float,
#     bs: int,
#     wd: float,
#     fold: int,
#     epochs: int,
#     out_folder: str | Path,
#     prefix: str,
# ):
# ------------------------------------------------------------------------------
