import numpy as np

# import os
# import pandas as pd
import matplotlib.pyplot as plt
import pytest
import tempfile

# import torch
# import wfdb

from pathlib import Path

# from unittest.mock import patch
# from ecg_cnn.data.dataset import PTBXLFullDataset

from ecg_cnn.utils.plot_utils import (
    _build_plot_title,
    format_hparams,
    save_plot_curves,
    save_confusion_matrix,
    save_pr_threshold_curve,
    save_classification_report,
    evaluate_and_plot,
)
from ecg_cnn.utils.validate import validate_hparams


# ------------------------------------------------------------------------------
# Helper function
# ------------------------------------------------------------------------------
def default_hparam_kwargs():
    return dict(
        lr=0.001,
        bs=64,
        wd=0.0,
        fold=1,
        epochs=10,
        prefix="test",
        fname_metric="some_metric",
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
#     out_folder: str | Path,
#     lr: float,
#     bs: int,
#     wd: float,
#     fold: int,
#     epochs: int,
#     prefix: str,
#     fname_metric: str,
# ):
# ------------------------------------------------------------------------------


def test_save_plot_curves_valid():
    x_vals = list(range(10))
    y_vals = [v * 0.9 for v in x_vals]
    out_dir = Path(tempfile.mkdtemp())
    default_hparams = default_hparam_kwargs()

    save_plot_curves(
        x_vals=x_vals,
        y_vals=y_vals,
        x_label="Epoch",
        y_label="Accuracy",
        title_metric="Test Accuracy",
        out_folder=out_dir,
        **default_hparams,
    )

    # Check that a PNG file was created
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 1
    assert "test" in pngs[0].name
    assert "some_metric" in pngs[0].name


@pytest.mark.parametrize(
    "bad_input",
    [
        ("x_vals", "not_a_list"),
        ("y_vals", {"bad": "dict"}),
        ("x_vals", [1, 2, "three"]),
        ("y_vals", [1.0, None, 2.0]),
    ],
)
def test_save_plot_curves_invalid_x_y_inputs(bad_input):
    key, bad_value = bad_input
    kwargs = dict(
        x_vals=list(range(5)),
        y_vals=[v * 0.9 for v in range(5)],
        x_label="Epoch",
        y_label="Loss",
        title_metric="Loss Curve",
        out_folder=tempfile.mkdtemp(),
        **default_hparam_kwargs(),
    )
    kwargs[key] = bad_value
    with pytest.raises(ValueError, match=f"{key}.*must"):
        save_plot_curves(**kwargs)


def test_save_plot_curves_length_mismatch_x_y():
    with pytest.raises(ValueError, match="same length"):
        save_plot_curves(
            x_vals=[0, 1, 2],
            y_vals=[0.5, 0.4],
            x_label="Epoch",
            y_label="Loss",
            title_metric="Loss",
            out_folder=tempfile.mkdtemp(),
            **default_hparam_kwargs(),
        )


@pytest.mark.parametrize(
    "param", ["x_label", "y_label", "title_metric", "fname_metric", "prefix"]
)
def test_save_plot_curves_bad_string_inputs(param):
    kwargs = dict(
        x_vals=[1, 2, 3],
        y_vals=[1, 2, 3],
        x_label="Epoch",
        y_label="Accuracy",
        title_metric="Model Accuracy",
        out_folder=tempfile.mkdtemp(),
        **default_hparam_kwargs(),
    )
    kwargs[param] = "  "  # Empty-ish string
    with pytest.raises(ValueError, match=f"{param}.*non-empty string"):
        save_plot_curves(**kwargs)


def test_save_plot_curves_path_vs_str_for_output_folder():
    # Should work with either Path or str
    tmp_dir = tempfile.mkdtemp()
    for out in [tmp_dir, Path(tmp_dir)]:
        save_plot_curves(
            x_vals=list(range(5)),
            y_vals=list(range(5)),
            x_label="Epoch",
            y_label="Accuracy",
            title_metric="Acc",
            out_folder=out,
            **default_hparam_kwargs(),
        )


def test_save_plot_curves_non_arraylike_x_vals_raises():
    # Triggers: if not hasattr(arr, "__len__")
    with pytest.raises(ValueError, match="x_vals must be array-like"):
        save_plot_curves(
            x_vals=object(),  # not array-like
            y_vals=[0.1, 0.2, 0.3],
            x_label="Epoch",
            y_label="Accuracy",
            title_metric="Accuracy",
            out_folder=tempfile.mkdtemp(),
            **default_hparam_kwargs(),
        )


def test_save_plot_curves_invalid_out_folder_type_raises():
    # Triggers: out_folder must be string or Path
    with pytest.raises(ValueError, match="out_folder must be a string or pathlib.Path"):
        save_plot_curves(
            x_vals=[1, 2, 3],
            y_vals=[1, 2, 3],
            x_label="Epoch",
            y_label="Accuracy",
            title_metric="Accuracy",
            out_folder=42,  # not str or Path
            **default_hparam_kwargs(),
        )


# ------------------------------------------------------------------------------
# def save_confusion_matrix(
#     y_true: list[int],
#     y_pred: list[int],
#     class_names: list[str],
#     out_folder: str | Path,
#     lr: float,
#     bs: int,
#     wd: float,
#     fold: int,
#     epochs: int,
#     prefix: str,
#     fname_metric: str,
#     normalize: bool = True,
# ):
# ------------------------------------------------------------------------------


def test_save_confusion_matrix_valid():
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 2, 1, 2, 1]
    class_names = ["A", "B", "C"]
    out_dir = Path(tempfile.mkdtemp())
    default_hparams = default_hparam_kwargs()

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_folder=out_dir,
        **default_hparams,
        normalize=True,
    )

    files = list(out_dir.glob("*.png"))
    assert len(files) == 1
    assert "some_metric" in files[0].name


@pytest.mark.parametrize(
    "bad_y_true",
    [
        "123",  # not a list
        [0, 1, "2"],  # not all ints
        [1.0, 2.0, 3.0],  # floats instead of ints
    ],
)
def test_save_confusion_matrix_invalid_y_true(bad_y_true):
    with pytest.raises(ValueError, match="y_true must be a list of integers"):
        save_confusion_matrix(
            y_true=bad_y_true,
            y_pred=[0, 1, 2],
            class_names=["A", "B", "C"],
            out_folder=tempfile.mkdtemp(),
            **default_hparam_kwargs(),
        )


@pytest.mark.parametrize(
    "bad_y_pred",
    [
        None,
        [0, 1, None],
        [True, False, 1],
    ],
)
def test_save_confusion_matrix_invalid_y_pred(bad_y_pred):
    with pytest.raises(ValueError, match="y_pred must be a list of integers"):
        save_confusion_matrix(
            y_true=[0, 1, 2],
            y_pred=bad_y_pred,
            class_names=["A", "B", "C"],
            out_folder=tempfile.mkdtemp(),
            **default_hparam_kwargs(),
        )


def test_save_confusion_matrix_mismatched_lengths():
    with pytest.raises(ValueError, match="must be the same length"):
        save_confusion_matrix(
            y_true=[0, 1],
            y_pred=[0, 1, 2],
            class_names=["A", "B", "C"],
            out_folder=tempfile.mkdtemp(),
            **default_hparam_kwargs(),
        )


def test_save_confusion_matrix_invalid_class_names():
    with pytest.raises(ValueError, match="class_names must be a list of strings"):
        save_confusion_matrix(
            y_true=[0, 1, 2],
            y_pred=[0, 1, 2],
            class_names="A,B,C",  # not a list
            out_folder=tempfile.mkdtemp(),
            **default_hparam_kwargs(),
        )


def test_save_confusion_matrix_invalid_out_folder():
    with pytest.raises(ValueError, match="out_folder must be a string or pathlib.Path"):
        save_confusion_matrix(
            y_true=[0, 1, 2],
            y_pred=[0, 1, 2],
            class_names=["A", "B", "C"],
            out_folder=3.16,  # invalid
            **default_hparam_kwargs(),
        )


# ------------------------------------------------------------------------------
# def save_pr_threshold_curve(
#     y_true: list[int] | np.ndarray,
#     y_probs: list[float] | np.ndarray,
#     out_folder: str | Path,
#     lr: float,
#     bs: int,
#     wd: float,
#     fold: int,
#     epochs: int,
#     prefix: str,
#     fname_metric: str,
#     title: str = "Precision & Recall vs Threshold",
# ):
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("good_y_true", [[0, 1, 0, 1, 1], np.array([0, 1, 0, 1, 1])])
@pytest.mark.parametrize(
    "good_y_probs", [[0.2, 0.1, 0.3, 0.1, 0.1], np.array([0.2, 0.1, 0.3, 0.1, 0.1])]
)
def test_save_pr_threshold_curve_success(good_y_true, good_y_probs, tmp_path):
    """
    Test that the PR-threshold curve is successfully saved with valid input.
    """

    out_folder = tmp_path / "plots"
    title = "Test PR Threshold Curve"
    default_hparams = default_hparam_kwargs()

    # Call the function
    save_pr_threshold_curve(
        y_true=good_y_true,
        y_probs=good_y_probs,
        out_folder=out_folder,
        title=title,
        **default_hparams,
    )

    # Construct expected output path
    expected_fname = format_hparams(**default_hparams) + ".png"

    out_path = out_folder / expected_fname

    assert out_path.exists(), f"Expected output file not found: {out_path}"
    assert out_path.is_file(), "Output path is not a file"


@pytest.mark.parametrize("bad_input", [123, "abc", {"a": 1}, (1, 2, 3)])
def test_save_pr_threshold_curve_invalid_y_true_types(bad_input, tmp_path):
    y_probs = [0.1, 0.2, 0.3]

    out_folder = tmp_path / "plots"
    default_hparams = default_hparam_kwargs()
    title = "Test PR Threshold Curve"

    with pytest.raises(ValueError, match="y_true must be list or ndarray"):
        save_pr_threshold_curve(
            y_true=bad_input,
            y_probs=y_probs,
            out_folder=out_folder,
            **default_hparams,
            title=title,
        )


@pytest.mark.parametrize("bad_input", [[123, False], [1, 2, 3.16]])
def test_save_pr_threshold_curve_y_true_contains_non_ints(bad_input, tmp_path):
    y_probs = [0.1, 0.2, 0.3]

    out_folder = tmp_path / "plots"
    default_hparams = default_hparam_kwargs()
    title = "Test PR Threshold Curve"

    with pytest.raises(
        ValueError, match=r"y_true must contain only integers \(not bools\)"
    ):
        save_pr_threshold_curve(
            y_true=bad_input,
            y_probs=y_probs,
            out_folder=out_folder,
            **default_hparams,
            title=title,
        )


@pytest.mark.parametrize("bad_input", [0.123, "abc", {"a": 0.2}, (0.1, 0.2, 0.3)])
def test_save_pr_threshold_curve_invalid_y_probs_types(bad_input, tmp_path):
    y_true = [1, 2, 3]

    out_folder = tmp_path / "plots"
    default_hparams = default_hparam_kwargs()
    title = "Test PR Threshold Curve"

    with pytest.raises(ValueError, match="y_probs must be list or ndarray"):
        save_pr_threshold_curve(
            y_true=y_true,
            y_probs=bad_input,
            out_folder=out_folder,
            **default_hparams,
            title=title,
        )


@pytest.mark.parametrize("bad_input", [[123, False], [1, 2, 3.16]])
def test_save_pr_threshold_curve_y_probs_contains_non_floats(bad_input, tmp_path):
    y_true = [1, 0, 1]

    out_folder = tmp_path / "plots"
    default_hparams = default_hparam_kwargs()
    title = "Test PR Threshold Curve"

    with pytest.raises(ValueError, match="y_probs must contain only float values"):
        save_pr_threshold_curve(
            y_true=y_true,
            y_probs=bad_input,
            out_folder=out_folder,
            **default_hparams,
            title=title,
        )


@pytest.mark.parametrize("bad_input", [[0.1, 1.2], [-1.2, 0.16]])
def test_save_pr_threshold_curve_y_probs_contains_values_out_of_range(
    bad_input, tmp_path
):
    y_true = [1, 0]

    out_folder = tmp_path / "plots"
    default_hparams = default_hparam_kwargs()
    title = "Test PR Threshold Curve"

    with pytest.raises(
        ValueError, match=r"y_probs must contain probabilities in \[0.0, 1.0\]"
    ):
        save_pr_threshold_curve(
            y_true=y_true,
            y_probs=bad_input,
            out_folder=out_folder,
            **default_hparams,
            title=title,
        )


def test_save_pr_threshold_curve_y_true_andy_probs_different_lengths(tmp_path):
    y_true = [1, 2, 3]
    y_probs = [0.1, 0.3]

    out_folder = tmp_path / "plots"
    default_hparams = default_hparam_kwargs()
    title = "Test PR Threshold Curve"

    with pytest.raises(ValueError, match="y_true and y_probs must be the same length"):
        save_pr_threshold_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_folder,
            **default_hparams,
            title=title,
        )


def test_save_pr_threshold_curve_invalid_outfolder(tmp_path):
    y_true = [1, 1, 0]
    y_probs = [0.1, 0.3, 0.3]

    out_folder = 22
    default_hparams = default_hparam_kwargs()
    title = "Test PR Threshold Curve"

    with pytest.raises(ValueError, match="out_folder must be a string or pathlib.Path"):
        save_pr_threshold_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_folder,
            **default_hparams,
            title=title,
        )


# ------------------------------------------------------------------------------
# def save_classification_report(
#     y_true: list[int] | np.ndarray,
#     y_pred: list[int] | np.ndarray,
#     class_names: list[str],
#     out_folder: str | Path,
#     lr: float,
#     bs: int,
#     wd: float,
#     fold: int,
#     epochs: int,
#     prefix: str,
#     fname_metric: str,
#     title: str = "Classification Report",
# ):
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Valid case
# ------------------------------------------------------------------------------
def test_save_classification_report_success(tmp_path):
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    class_names = ["NORM", "MI"]
    hparams = default_hparam_kwargs()

    save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_folder=tmp_path,
        **hparams,
    )

    filename = format_hparams(**hparams)
    out_path_txt = tmp_path / f"{filename}.txt"
    out_path_png = tmp_path / f"{filename}.png"

    assert out_path_txt.exists(), "Missing .txt report"
    assert out_path_png.exists(), "Missing heatmap .png"


# ------------------------------------------------------------------------------
# Invalid y_true
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_y_true", [123, "abc", {"a": 1}, (1, 2)])
def test_save_classification_report_invalid_y_true(bad_y_true, tmp_path):
    y_pred = [0, 1, 1]
    class_names = ["A", "B"]
    hparams = default_hparam_kwargs()

    with pytest.raises(ValueError, match="y_true must be list or ndarray"):
        save_classification_report(
            y_true=bad_y_true,
            y_pred=y_pred,
            class_names=class_names,
            out_folder=tmp_path,
            **hparams,
        )


# ------------------------------------------------------------------------------
# Invalid y_pred
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_y_pred", [None, {"x": 1}, [True, False], 3.14])
def test_save_classification_report_invalid_y_pred(bad_y_pred, tmp_path):
    y_true = [0, 1, 0]
    class_names = ["A", "B"]
    hparams = default_hparam_kwargs()

    expected_message = (
        "y_pred must be list or ndarray"
        if bad_y_pred is None or isinstance(bad_y_pred, (dict, float))
        else "y_pred must contain only integer values"
    )

    with pytest.raises(ValueError, match=expected_message):
        save_classification_report(
            y_true=y_true,
            y_pred=bad_y_pred,
            class_names=class_names,
            out_folder=tmp_path,
            **hparams,
        )


def test_save_classification_report_y_true_with_bools(tmp_path):
    hparams = default_hparam_kwargs()
    with pytest.raises(ValueError, match="y_true must contain only integer values"):
        save_classification_report(
            y_true=[0, 1, True],  # Bool value should trigger rejection
            y_pred=[0, 1, 0],
            class_names=["A", "B"],
            out_folder=tmp_path,
            **hparams,
        )


# ------------------------------------------------------------------------------
# Mismatched lengths
# ------------------------------------------------------------------------------
def test_save_classification_report_mismatched_lengths(tmp_path):
    y_true = [0, 1]
    y_pred = [0, 1, 1]
    class_names = ["A", "B"]
    hparams = default_hparam_kwargs()

    with pytest.raises(ValueError, match="y_true and y_pred must be the same length"):
        save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            out_folder=tmp_path,
            **hparams,
        )


# ------------------------------------------------------------------------------
# Bad class names
# ------------------------------------------------------------------------------
def test_save_classification_report_class_names_not_a_list(tmp_path):
    y_true = [0, 1, 1]
    y_pred = [0, 1, 1]
    class_names = "A"
    hparams = default_hparam_kwargs()

    with pytest.raises(ValueError, match="class_names must be a list of strings"):
        save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            out_folder=tmp_path,
            **hparams,
        )


def test_save_classification_report_class_names_not_a_list_of_strings(tmp_path):
    y_true = [0, 1, 1]
    y_pred = [0, 1, 1]
    class_names = ["A", 3]
    hparams = default_hparam_kwargs()

    with pytest.raises(ValueError, match="class_names must be a list of strings"):
        save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            out_folder=tmp_path,
            **hparams,
        )


# ------------------------------------------------------------------------------
# Invalid out_folder
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_out", [123, None, 3.14])
def test_save_classification_report_invalid_out_folder(bad_out):
    y_true = [0, 1]
    y_pred = [0, 1]
    class_names = ["A", "B"]
    hparams = default_hparam_kwargs()

    with pytest.raises(ValueError, match="out_folder must be a string or pathlib.Path"):
        save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            out_folder=bad_out,
            **hparams,
        )


# ------------------------------------------------------------------------------
# def evaluate_and_plot(
#     y_true: list[int] | np.ndarray,
#     y_pred: list[int] | np.ndarray,
#     train_accs: list[float],
#     val_accs: list[float],
#     train_losses: list[float],
#     val_losses: list[float],
#     lr: float,
#     bs: int,
#     wd: float,
#     fold: int,
#     epochs: int,
#     prefix: str,
#     fname_metric: str,
#     out_folder: str | Path,
#     class_names: list[str],
# ) -> None:
# ------------------------------------------------------------------------------


dummy_accs = [0.8, 0.85, 0.9]
dummy_losses = [0.5, 0.4, 0.3]
class_names = ["Normal", "Abnormal"]


def test_evaluate_and_plot_success(tmp_path):
    y_true = [0, 1, 0]
    y_pred = [0, 1, 1]
    hparams = default_hparam_kwargs()

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=dummy_accs,
        val_accs=dummy_accs,
        train_losses=dummy_losses,
        val_losses=dummy_losses,
        out_folder=tmp_path,
        class_names=class_names,
        **hparams,
    )

    # evaluate_and_plot() hard-codes fname_metric="classification_report" for the report
    report_fname = format_hparams(
        lr=hparams["lr"],
        bs=hparams["bs"],
        wd=hparams["wd"],
        fold=hparams["fold"],
        epochs=hparams["epochs"],
        prefix=hparams["prefix"],
        fname_metric="classification_report",
    )

    # The report (txt + heatmap) is saved under reports/
    assert (tmp_path / "reports" / f"{report_fname}.txt").exists()
    assert (tmp_path / "reports" / f"{report_fname}.png").exists()


@pytest.mark.parametrize(
    "bad_val, err_msg",
    [
        (None, "y_true must be list or ndarray"),
        (123, "y_true must be list or ndarray"),
        ([0, "a", 2], "y_true must contain only integer values"),
        ([True, False], "y_true must contain only integer values"),
        ([0, 1, 2], "class_names length must cover all class indices"),
    ],
)
def test_evaluate_and_plot_invalid_y_true(bad_val, err_msg, tmp_path):
    hparams = default_hparam_kwargs()

    # Make y_pred the same length as y_true *when y_true is a list* so we
    # don’t trip the length check before the intended validation.
    y_pred = [0, 1, 0]
    if isinstance(bad_val, list):
        y_pred = y_pred[: len(bad_val)]

    with pytest.raises(ValueError, match=err_msg):
        evaluate_and_plot(
            y_true=bad_val,
            y_pred=y_pred,
            train_accs=dummy_accs,
            val_accs=dummy_accs,
            train_losses=dummy_losses,
            val_losses=dummy_losses,
            out_folder=tmp_path,
            class_names=class_names,
            **hparams,
        )


@pytest.mark.parametrize(
    "bad_val, err_msg",
    [
        (None, "y_pred must be list or ndarray"),
        ({"a": 1}, "y_pred must be list or ndarray"),
        ([0, 1, "x"], "y_pred must contain only integer values"),
        ([3.14, 2.71], "y_pred must contain only integer values"),
        ([0, 1], "y_true and y_pred must be the same length"),
    ],
)
def test_evaluate_and_plot_invalid_y_pred(bad_val, err_msg, tmp_path):
    hparams = default_hparam_kwargs()
    y_true = [0, 1, 0]

    # Only align lengths when we're *not* testing for the mismatch error
    if (
        isinstance(bad_val, list)
        and len(bad_val) != len(y_true)
        and err_msg != "y_true and y_pred must be the same length"
    ):
        y_true = y_true[: len(bad_val)]

    with pytest.raises(ValueError, match=err_msg):
        evaluate_and_plot(
            y_true=y_true,
            y_pred=bad_val,
            train_accs=dummy_accs,
            val_accs=dummy_accs,
            train_losses=dummy_losses,
            val_losses=dummy_losses,
            out_folder=tmp_path,
            class_names=class_names,
            **hparams,
        )


@pytest.mark.parametrize(
    "bad_val, err_msg",
    [
        (None, "class_names must be a list of strings"),
        (123, "class_names must be a list of strings"),
        ([], "class_names cannot be empty"),
        ([0, 1], "class_names must be a list of strings"),
    ],
)
def test_evaluate_and_plot_invalid_class_names(bad_val, err_msg, tmp_path):
    hparams = default_hparam_kwargs()
    with pytest.raises(ValueError, match=err_msg):
        evaluate_and_plot(
            y_true=[0, 1, 0],
            y_pred=[0, 1, 0],
            train_accs=dummy_accs,
            val_accs=dummy_accs,
            train_losses=dummy_losses,
            val_losses=dummy_losses,
            out_folder=tmp_path,
            class_names=bad_val,
            **hparams,
        )


@pytest.mark.parametrize("bad_val", [None, 123, 3.14])
def test_evaluate_and_plot_invalid_out_folder(bad_val):
    hparams = default_hparam_kwargs()
    with pytest.raises(ValueError, match="out_folder must be a string or Path"):
        evaluate_and_plot(
            y_true=[0, 1, 0],
            y_pred=[0, 1, 0],
            train_accs=dummy_accs,
            val_accs=dummy_accs,
            train_losses=dummy_losses,
            val_losses=dummy_losses,
            out_folder=bad_val,
            class_names=class_names,
            **hparams,
        )


def test_evaluate_and_plot_mismatched_metrics(tmp_path):
    hparams = default_hparam_kwargs()

    # Deliberately make val_accs shorter
    bad_val_accs = [0.8, 0.85]  # should be 3 long

    with pytest.raises(ValueError, match="must be equal in length"):
        evaluate_and_plot(
            y_true=[0, 1, 0],
            y_pred=[0, 1, 1],
            train_accs=dummy_accs,
            val_accs=bad_val_accs,
            train_losses=dummy_losses,
            val_losses=dummy_losses,
            out_folder=tmp_path,
            class_names=class_names,
            **hparams,
        )
