# tests/test_plot_utils.py

"""
Tests for ecg_cnn.utils.plot_utils

Covers
------
    - _build_plot_title(): formatting with/without metric, fold/epoch
    - format_hparams(): filename tokenization and float trimming
    - save_plot_curves(): validation and successful save
    - save_confusion_matrix(): validation and successful save
    - save_pr_threshold_curve(): validation and successful save
    - save_pr_curve(): binary and OvR multiclass save paths
    - save_roc_curve(): binary and OvR multiclass save paths
    - save_classification_report(): validation and heatmap save
    - evaluate_and_plot(): end-to-end artifact generation (binary & multiclass),
      env-override parameters, and error branches

Notes
-----
    - Relies on test-level tmp_path isolation; no artifacts persist.
    - Seeding is handled in conftest.py (no per-file seeds here).
    - All regex match= assertions are anchored with ^ (no trailing $).
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest

from contextlib import redirect_stdout
from pathlib import Path
from sklearn.metrics import brier_score_loss

from ecg_cnn.utils.plot_utils import (
    _build_plot_title,
    format_hparams,
    save_plot_curves,
    save_confusion_matrix,
    save_pr_threshold_curve,
    save_pr_curve,
    save_roc_curve,
    save_classification_report,
    save_calibration_curve,
    save_threshold_sweep_table,
    save_error_tables,
    save_confidence_histogram,
    save_confidence_histogram_split,
    evaluate_and_plot,
)

# ------------------------------------------------------------------------------
# helper methods
# ------------------------------------------------------------------------------


def _simple_pred_from_probs(y_probs: np.ndarray) -> np.ndarray:
    """Helper: argmax for multiclass or threshold for binary probabilities."""
    if y_probs.ndim == 1:
        return (y_probs >= 0.5).astype(int)
    return np.argmax(y_probs, axis=1)


# ------------------------------------------------------------------------------
# _build_plot_title
# ------------------------------------------------------------------------------


def test_build_plot_title_valid():
    title = _build_plot_title(
        model="ECGConvNet",
        lr=1e-3,
        bs=32,
        wd=0.0,
        prefix="final",
        metric="Loss",
        fold=1,
        epoch=5,
    )
    # Contains header and the key tokens
    assert "ECGConvNet: Loss" in title
    assert "LR=0.001, BS=32, WD=0.0, Fold=1, Epoch=5" in title


def test_build_plot_title_valid_metric_is_none():
    title = _build_plot_title(
        model="ECGConvNet",
        lr=5e-4,
        bs=64,
        wd=0.01,
        prefix="best",
        metric=None,  # should default to "Metric"
        fold=None,
        epoch=None,
    )
    assert "ECGConvNet: Metric" in title
    assert "LR=0.0005, BS=64, WD=0.01" in title
    assert "Fold=" not in title and "Epoch=" not in title


# ------------------------------------------------------------------------------
# format_hparams
# ------------------------------------------------------------------------------


def test_format_hparams_filename_tokens_and_trim():
    fname = format_hparams(
        model="ECGConvNet",
        lr=0.001,  # -> "001"
        bs=32,
        wd=0.0,  # -> "0"
        prefix="Final",
        fname_metric="Loss",
        fold=1,
        epoch=5,
    )
    # Expected tokens present
    assert "final" in fname  # prefix lowercased
    assert "loss" in fname  # metric lowercased
    assert "ECGConvNet" in fname
    assert "lr001" in fname
    assert "bs32" in fname
    assert "wd0" in fname
    assert "fold1" in fname
    assert "epoch5" in fname


def test_format_hparams_float_trimming_various_values():
    # Focus on trimming behavior; presence of tokens is sufficient
    fname = format_hparams(
        model="M",
        lr=0.0001234,  # -> 0.000123 -> "000123"
        bs=1,
        wd=0.01,  # -> "01"
        prefix="p",
        fname_metric="m",
    )
    assert "lr000123" in fname
    assert "wd01" in fname


# ------------------------------------------------------------------------------
# save_plot_curves
# ------------------------------------------------------------------------------


def test_save_plot_curves_happy_path(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_plot_curves(
        x_vals=[0.1, 0.2, 0.3],
        y_vals=[0.15, 0.25, 0.35],
        x_label="Epoch",
        y_label="Accuracy",
        title_metric="Accuracy",
        out_folder=out_dir,
        model="ECGConvNet",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="final",
        fname_metric="accuracy",
        fold=2,
        epoch=10,
    )
    # Expect a single .png in out_dir
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 1


def test_save_plot_curves_non_numeric_raises(tmp_path: Path):
    out_dir = tmp_path / "plots2"
    out_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match=r"^x_vals must contain only numeric values"):
        save_plot_curves(
            x_vals=[0.1, "oops", 0.3],
            y_vals=[0.1, 0.2, 0.3],
            x_label="Epoch",
            y_label="Accuracy",
            title_metric="Accuracy",
            out_folder=tmp_path,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="final",
            fname_metric="acc",
        )


def test_save_plot_curves_length_mismatch_raises(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(
        ValueError, match=r"^x_vals and y_vals must have the same length"
    ):
        save_plot_curves(
            x_vals=[1, 2, 3],
            y_vals=[1, 2],
            x_label="Epoch",
            y_label="Loss",
            title_metric="Loss",
            out_folder=tmp_path,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="final",
            fname_metric="loss",
        )


def test_save_plot_curves_bad_out_folder_type_raises():
    with pytest.raises(
        ValueError, match=r"^out_folder must be a string or pathlib\.Path"
    ):
        save_plot_curves(
            x_vals=[1, 2],
            y_vals=[1, 2],
            x_label="x",
            y_label="y",
            title_metric="t",
            out_folder=123,  # bad
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="p",
            fname_metric="m",
        )


@pytest.mark.parametrize(
    "field, value, pattern",
    [
        ("x_label", "", r"^x_label must be a non-empty string"),
        ("y_label", "", r"^y_label must be a non-empty string"),
        ("prefix", "", r"^prefix must be a non-empty string"),
        ("title_metric", "", r"^title_metric must be a non-empty string"),
    ],
)
def test_save_plot_curves_rejects_empty_required_strings(
    tmp_path: Path, field, value, pattern
):
    out_dir = tmp_path / "plots2"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        x_vals=[0, 1, 2],
        y_vals=[1, 2, 3],
        out_folder=out_dir,
        fname_metric="acc",
        title_metric="Accuracy",
        x_label="x",
        y_label="y",
        prefix="prefix",
        model="M",
        lr=0.001,
        bs=8,
        wd=0.0,
    )
    kwargs[field] = value

    with pytest.raises(ValueError, match=pattern):
        save_plot_curves(**kwargs)


def test_save_plot_curves_rejects_non_sized_x_vals(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # x_vals is a generator (no __len__), y_vals is OK
    x_vals = (i for i in range(5))
    y_vals = [0, 1, 2, 3, 4]

    # plot_utils.py raises ValueError with message "...array-like."
    with pytest.raises(ValueError, match=r"^x_vals must be array-like"):
        save_plot_curves(
            x_vals=x_vals,
            y_vals=y_vals,
            out_folder=out_dir,
            fname_metric="acc",
            title_metric="Accuracy",
            x_label="x",
            y_label="y",
            prefix="prefix",
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
        )


# ------------------------------------------------------------------------------
# save_confusion_matrix
# ------------------------------------------------------------------------------


def test_save_confusion_matrix_happy_path(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    class_names = ["NEG", "POS"]
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_folder=out_dir,
        model="ECGConvNet",
        lr=1e-3,
        bs=16,
        wd=0.0,
        prefix="final",
        fname_metric="confusion_matrix",
        normalize=True,
        fold=1,
        epoch=2,
    )
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 1


def test_save_confusion_matrix_bad_class_coverage_raises(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [0, 1, 2]  # class index 2 is out of bounds for 2 names below
    y_pred = [0, 1, 1]
    class_names = ["A", "B"]
    with pytest.raises(
        ValueError,
        match=r"^class_names must cover all class indices in y_true and y_pred",
    ):
        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            out_folder=tmp_path,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="p",
            fname_metric="cm",
        )


def test_save_confusion_matrix_bad_out_folder_type_raises():
    with pytest.raises(
        ValueError, match=r"^out_folder must be a string or pathlib\.Path"
    ):
        save_confusion_matrix(
            y_true=[0, 1],
            y_pred=[0, 1],
            class_names=["A", "B"],
            out_folder=object(),  # bad
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="p",
            fname_metric="cm",
        )


def test_save_confusion_matrix_rejects_nonstring_class_names(tmp_path: Path):
    out_dir = tmp_path / "cm"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 1])

    # Implementation message: "class_names must be a list of strings."
    with pytest.raises(ValueError, match=r"^class_names must be a list of strings"):
        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            out_folder=out_dir,
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
            epoch=None,
            prefix="prefix",
            fname_metric="confmat",
            class_names=[1, 2, 3],  # not strings
        )


# ------------------------------------------------------------------------------
# save_pr_threshold_curve
# ------------------------------------------------------------------------------


def test_save_pr_threshold_curve_happy_path(tmp_path: Path):
    out_dir = tmp_path / "prth"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [0, 1, 0, 1, 1]
    y_probs = [0.1, 0.9, 0.4, 0.8, 0.65]

    save_pr_threshold_curve(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=4,
        wd=0.0,
        epoch=7,
        prefix="final",
        fname_metric="pr_threshold",
        title="Precision & Recall vs Threshold",
        fold=None,
    )
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 1


def test_save_pr_threshold_curve_length_mismatch_raises(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(
        ValueError, match=r"^y_true and y_probs must be the same length"
    ):
        save_pr_threshold_curve(
            y_true=[0, 1, 0],
            y_probs=[0.1, 0.2],
            out_folder=out_dir,
            model="M",
            lr=1e-3,
            bs=4,
            wd=0.0,
            epoch=1,
            prefix="p",
            fname_metric="m",
        )


def test_save_pr_threshold_curve_bad_out_folder_type_raises():
    with pytest.raises(
        ValueError,
        match=r"^out_folder must be a string or pathlib\.Path, got ",
    ):
        save_pr_threshold_curve(
            y_true=[0, 1],
            y_probs=[0.1, 0.9],
            out_folder=123,  # bad
            model="M",
            lr=1e-3,
            bs=1,
            wd=0.0,
            epoch=1,
            prefix="p",
            fname_metric="m",
        )


def test_save_pr_threshold_curve_title_must_be_string(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(
        ValueError,
        match=r"^title must be a string, got ",
    ):
        save_pr_threshold_curve(
            y_true=[0, 1],
            y_probs=[0.1, 0.9],
            out_folder=tmp_path,
            model="M",
            lr=1e-3,
            bs=1,
            wd=0.0,
            epoch=1,
            prefix="p",
            fname_metric="m",
            title=123,  # needs to be a string
        )


# ------------------------------------------------------------------------------
# save_pr_curve
# ------------------------------------------------------------------------------


def test_save_pr_curve_binary_path(tmp_path: Path):
    out_dir = tmp_path / "pr_bin"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [0, 1, 0, 1]
    y_probs = [0.1, 0.9, 0.2, 0.7]
    save_pr_curve(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="final",
        fname_metric="pr_curve",
    )
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 1


def test_save_pr_curve_multiclass_ovr(tmp_path: Path):
    out_dir = tmp_path / "pr_ovr"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 3 classes, OvR curves for i=0..2
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.5, 0.3],
            [0.7, 0.2, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    save_pr_curve(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="final",
        fname_metric="pr_curve",
    )
    # Should create one file per class with suffix embedded in prefix
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 3


def test_save_pr_curve_skips_on_mismatched_2d_probs_shape(tmp_path: Path):
    out_dir = tmp_path / "pr2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # y_probs 2D with mismatched first dimension vs len(y_true)
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.random.rand(3, 2)  # mismatched -> skip branch

    buf = io.StringIO()
    with redirect_stdout(buf):
        save_pr_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_dir,
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
            epoch=None,
            prefix="prefix",
            fname_metric="pr",
            title="PR",
        )
    out = buf.getvalue()
    assert "Skipping PR curve — y_probs shape does not match expected format." in out


def test_save_pr_curve_rejects_non_string_title(tmp_path: Path):
    out_dir = tmp_path / "pr1"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.2, 0.8, 0.3, 0.7])  # binary, 1D OK

    # Implementation message: "title must be a string."
    with pytest.raises(ValueError, match=r"^title must be a string"):
        save_pr_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_dir,
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
            epoch=None,
            prefix="prefix",
            fname_metric="pr",
            title=123,  # wrong type
        )


def test_save_pr_curve_out_folder_must_be_a_string_or_path():
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.2, 0.8, 0.3, 0.7])  # binary, 1D OK

    with pytest.raises(
        ValueError, match=r"^out_folder must be a string or pathlib\.Path\."
    ):
        save_pr_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=["out_dir"],  # wrong type
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
            epoch=None,
            prefix="prefix",
            fname_metric="pr",
            title="Plot",
        )


# ------------------------------------------------------------------------------
# save_roc_curve
# ------------------------------------------------------------------------------


def test_save_roc_curve_binary_path(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [0, 1, 0, 1]
    y_probs = [0.1, 0.9, 0.3, 0.8]
    save_roc_curve(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="final",
        fname_metric="roc_curve",
    )
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 1


def test_save_roc_curve_multiclass_ovr(tmp_path: Path):
    out_dir = tmp_path / "roc_ovr"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.5, 0.3],
            [0.7, 0.2, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    save_roc_curve(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="final",
        fname_metric="roc_curve",
    )
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 3


def test_save_roc_curve_rejects_non_string_title(tmp_path: Path):
    out_dir = tmp_path / "roc1"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.2, 0.8, 0.3, 0.7])  # binary, 1D OK

    with pytest.raises(ValueError, match=r"^title must be a string"):
        save_roc_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_dir,
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
            epoch=None,
            prefix="prefix",
            fname_metric="roc",
            title=object(),  # wrong type
        )


def test_save_roc_curve_out_folder_must_be_string_or_path():
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.2, 0.8, 0.3, 0.7])  # binary, 1D OK

    with pytest.raises(
        ValueError, match=r"^out_folder must be a string or pathlib\.Path\."
    ):
        save_roc_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=["out_dir"],  # wrong type
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
            epoch=None,
            prefix="prefix",
            fname_metric="roc",
            title="Title",
        )


def test_roc_curve_skips_on_bad_shape(capsys):
    # Shape that cannot match either branch:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_probs = np.random.rand(3, 2)  # 2-D but wrong first dim (3 != 6)

    save_roc_curve(
        y_true=y_true,
        y_probs=y_probs,
        out_folder="out_dir",
        model="M",
        lr=0.001,
        bs=32,
        wd=0.0,
        epoch=None,
        prefix="prefix",
        fname_metric="roc",
        title="Title",
    )

    out = capsys.readouterr().out
    assert "Skipping ROC curve — y_probs shape does not match expected format." in out


# ------------------------------------------------------------------------------
# save_classification_report
# ------------------------------------------------------------------------------


def test_save_classification_report_happy_path(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    class_names = ["NEG", "POS"]
    save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_folder=out_dir,
        model="ECGConvNet",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="final",
        fname_metric="classification_report",
        fold=1,
        epoch=2,
        title="Classification Report",
    )
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) == 1


def test_save_classification_report_bad_class_names_type_raises(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match=r"^class_names must be a list of strings"):
        save_classification_report(
            y_true=[0, 1],
            y_pred=[0, 1],
            class_names="A,B",  # bad
            out_folder=tmp_path,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="p",
            fname_metric="r",
        )


def test_save_classification_report_bad_out_folder_type_raises():
    with pytest.raises(
        ValueError,
        match=r"^out_folder must be a string or pathlib\.Path, got ",
    ):
        save_classification_report(
            y_true=[0, 1],
            y_pred=[0, 1],
            class_names=["A", "B"],
            out_folder=123,  # bad
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="p",
            fname_metric="r",
        )


def test_save_classification_report_rejects_non_string_title(tmp_path: Path):
    out_dir = tmp_path / "rep"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    with pytest.raises(ValueError, match=r"^title must be a string"):
        save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            class_names=["A", "B"],
            out_folder=out_dir,
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
            epoch=None,
            prefix="prefix",
            fname_metric="report",
            title=3.14,  # wrong type
        )


# ------------------------------------------------------------------------------
# save_calibration_curve
# ------------------------------------------------------------------------------


def test_save_calibration_curve_bad_out_folder():
    y_true = [0, 1, 0, 1, 1, 0]
    y_probs = [0.2, 0.8, 0.3, 0.7, 0.9, 0.1]
    n_bins = 2.0
    with pytest.raises(
        ValueError, match=r"^out_folder must be a string or pathlib.Path."
    ):
        save_calibration_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=23,  # bad
            model="ECGConvNet",
            lr=0.001,
            bs=64,
            wd=0.0005,
            prefix="final",
            fname_metric="metric",
            n_bins=n_bins,
            fold=2,
            epoch=3,
            title="Calibration Curve",
        )


def test_save_calibration_curve_n_bins_not_integer(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [0, 1, 0, 1, 1, 0]
    y_probs = [0.2, 0.8, 0.3, 0.7, 0.9, 0.1]
    n_bins = 2.0
    with pytest.raises(ValueError, match=r"^n_bins must be an integer >= 2."):
        save_calibration_curve(
            y_true,
            y_probs,
            out_dir,
            "ECGConvNet",
            0.001,
            64,
            0.0005,
            "final",
            "metric",
            n_bins=n_bins,
        )


def test_save_calibration_curve_n_bins_not_integer(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [0, 1, 0, 1, 1, 0]
    y_probs = [0.2, 0.8, 0.3, 0.7, 0.9, 0.1]
    n_bins = 2.0
    with pytest.raises(ValueError, match=r"^n_bins must be an integer >= 2."):
        save_calibration_curve(
            y_true,
            y_probs,
            out_dir,
            "ECGConvNet",
            0.001,
            64,
            0.0005,
            "final",
            "metric",
            n_bins=n_bins,
        )


def test_save_calibration_curve_n_bins_too_small(tmp_path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = [0, 1, 0, 1, 1, 0]
    y_probs = [0.2, 0.8, 0.3, 0.7, 0.9, 0.1]
    n_bins = 1
    with pytest.raises(ValueError, match=r"^n_bins must be an integer >= 2."):
        save_calibration_curve(
            y_true,
            y_probs,
            out_dir,
            "ECGConvNet",
            0.001,
            64,
            0.0005,
            "final",
            "metric",
            n_bins=n_bins,
        )


def test_save_calibration_curve_multiclass_confidence(tmp_path: Path):
    """
    Verifies the multiclass branch:
      - y_probs is 2D with K>1 classes
      - reduction uses argmax==y_true as targets and max prob as confidence
      - returned Brier score matches the reduction
      - PNG is written to disk
    """
    # Multiclass ground truth and probabilities (N=6, K=3)
    y_true = np.array([0, 1, 2, 1, 0, 2], dtype=int)
    y_probs = np.array(
        [
            [0.90, 0.05, 0.05],  # correct, conf=0.90
            [0.10, 0.80, 0.10],  # correct, conf=0.80
            [0.20, 0.20, 0.60],  # correct, conf=0.60
            [0.20, 0.50, 0.30],  # correct, conf=0.50 (ties not present here)
            [
                0.40,
                0.40,
                0.20,
            ],  # incorrect (argmax=0 or 1 tie avoided; we keep unique argmax rows)
            [0.30, 0.20, 0.50],  # correct, conf=0.50
        ],
        dtype=float,
    )

    # Compute expected reduction exactly as in save_calibration_curve
    y_pred = y_probs.argmax(axis=1)
    conf = y_probs.max(axis=1)
    targets = (y_pred == y_true).astype(int)
    expected_brier = brier_score_loss(targets, conf)

    # Call the function under test (uses the multiclass reduction branch)
    png_path, brier = save_calibration_curve(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=tmp_path,
        model="ECGConvNet",
        lr=0.001,
        bs=64,
        wd=0.0005,
        prefix="final",
        fname_metric="calibration",
        n_bins=5,  # any >=2; not part of the Brier computation
        fold=None,
        epoch=3,
        title="Calibration Curve",
    )

    # Assertions
    assert png_path.exists(), "Calibration PNG was not created."
    # Use approx to avoid floating noise; the values should match exactly with same inputs
    assert brier == pytest.approx(expected_brier, rel=0, abs=1e-12)
    # Sanity range
    assert 0.0 <= brier <= 1.0


def test_save_calibration_curve_binary(tmp_path: Path):
    y_true = [0, 1, 0, 1, 1, 0]
    y_probs = [0.2, 0.8, 0.3, 0.7, 0.9, 0.1]
    p, b = save_calibration_curve(
        y_true, y_probs, tmp_path, "ECGConvNet", 0.001, 64, 0.0005, "final", "metric"
    )
    assert p.exists()
    assert 0.0 <= b <= 1.0


# ------------------------------------------------------------------------------
# save_threshold_sweep_table
# ------------------------------------------------------------------------------


def test_save_threshold_sweep_table_binary_1d_probs(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    # one probability per sample for positive class
    y_probs = np.array([0.10, 0.80, 0.25, 0.70, 0.95, 0.30], dtype=float)

    out_csv = save_threshold_sweep_table(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        thresholds=[0.50, 0.60, 0.70],
        average="binary",
        fold=1,
        epoch=1,
    )

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert set(df.columns) == {"threshold", "precision", "recall", "f1", "accuracy"}
    assert df.shape[0] == 3
    # sanity: all metrics within [0,1]
    for col in ["precision", "recall", "f1", "accuracy"]:
        assert (df[col].between(0.0, 1.0)).all()


def test_save_threshold_sweep_table_binary_2d_probs_uses_positive_column(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    # probs[:, 0] = P(neg), probs[:, 1] = P(pos)
    y_probs = np.array(
        [
            [0.80, 0.20],  # true 0
            [0.30, 0.70],  # true 1
            [0.90, 0.10],  # true 0
            [0.40, 0.60],  # true 1
        ],
        dtype=float,
    )

    out_csv = save_threshold_sweep_table(
        y_true=y_true,
        y_probs=y_probs,  # function should slice [:, 1]
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        thresholds=[0.60],  # with col 1 this yields predictions [0,1,0,1]
        average="binary",
        fold=1,
        epoch=1,
    )

    df = pd.read_csv(out_csv)
    assert set(df.columns) == {"threshold", "precision", "recall", "f1", "accuracy"}
    assert df.shape[0] == 1
    # At t=0.60, predicted = [0,1,0,1] -> perfect metrics
    row = df.iloc[0]
    assert row["threshold"] == 0.60
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0
    assert row["accuracy"] == 1.0


def test_save_threshold_sweep_table_multiclass_macro_invariant_over_thresholds(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 2, 1, 0, 2], dtype=int)
    y_probs = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.20, 0.60, 0.20],
            [0.65, 0.25, 0.10],
            [0.10, 0.20, 0.70],
        ],
        dtype=float,
    )

    thresholds = [0.10, 0.50, 0.90]  # macro path ignores threshold (uses argmax)
    out_csv = save_threshold_sweep_table(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        thresholds=thresholds,
        average="macro",
        fold=1,
        epoch=1,
    )

    df = pd.read_csv(out_csv)
    assert set(df.columns) == {"threshold", "precision", "recall", "f1", "accuracy"}
    assert list(df["threshold"]) == thresholds
    # Metrics should be identical across rows since macro path uses argmax
    for col in ["precision", "recall", "f1", "accuracy"]:
        assert df[col].nunique() == 1
        assert (df[col].between(0.0, 1.0)).all()


def test_save_threshold_sweep_average_not_binary_or_macro(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 2, 1, 0, 2], dtype=int)
    y_probs = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.20, 0.60, 0.20],
            [0.65, 0.25, 0.10],
            [0.10, 0.20, 0.70],
        ],
        dtype=float,
    )

    thresholds = [0.10, 0.50, 0.90]  # macro path ignores threshold (uses argmax)

    with pytest.raises(
        ValueError,
        match=r'^average must be "binary" or "macro".',
    ):
        save_threshold_sweep_table(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_dir,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="run",
            thresholds=thresholds,
            average="soup",
            fold=1,
            epoch=1,
        )


def test_save_threshold_sweep_table_default_grid_0_to_1_step_001(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    y_probs = np.array([0.10, 0.80, 0.25, 0.70, 0.95, 0.30], dtype=float)

    out_csv = save_threshold_sweep_table(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        thresholds=None,
        average="binary",
        fold=1,
        epoch=1,
    )

    df = pd.read_csv(out_csv)
    vals = df["threshold"].to_numpy()
    assert len(vals) == 101
    assert np.isclose(vals[0], 0.00)
    assert np.isclose(vals[-1], 1.00)
    diffs = np.diff(vals)
    assert np.allclose(diffs, 0.01)


def test_save_threshold_sweep_table_rejects_non_iterable(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=float)

    with pytest.raises(
        ValueError, match=r"^thresholds must be an iterable of floats in \[0.0, 1.0\]."
    ):
        save_threshold_sweep_table(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_dir,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="run",
            thresholds=0.5,  # not iterable
            average="binary",
            fold=1,
            epoch=1,
        )


def test_save_threshold_sweep_table_rejects_non_numeric_entry(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=float)

    with pytest.raises(
        ValueError, match=r"^All thresholds must be numeric values in \[0.0, 1.0\]."
    ):
        save_threshold_sweep_table(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_dir,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="run",
            thresholds=["a", 0.5],  # bad entry
            average="binary",
            fold=1,
            epoch=1,
        )


def test_save_threshold_sweep_table_rejects_out_of_range_values(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=float)

    with pytest.raises(ValueError, match=r"^All thresholds must be in \[0.0, 1.0\]."):
        save_threshold_sweep_table(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_dir,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="run",
            thresholds=[-0.1, 0.5],  # too low
            average="binary",
            fold=1,
            epoch=1,
        )

    with pytest.raises(ValueError, match=r"^All thresholds must be in \[0.0, 1.0\]."):
        save_threshold_sweep_table(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_dir,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="run",
            thresholds=[0.5, 1.1],  # too high
            average="binary",
            fold=1,
            epoch=1,
        )


def test_save_threshold_sweep_table_accepts_mixed_numeric_and_casts_to_float(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=float)

    given = [0, 0.25, np.float64(0.5), 0.75, 1]
    out_csv = save_threshold_sweep_table(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        thresholds=given,
        average="binary",
        fold=1,
        epoch=1,
    )

    df = pd.read_csv(out_csv)
    got = df["threshold"].tolist()
    assert got == [float(x) for x in given]


def test_save_threshold_sweep_table_binary_path_runs_for_true_binary(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # True binary: labels {0,1}, 1-D probabilities
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    out_csv = save_threshold_sweep_table(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=4,
        wd=0.0,
        prefix="bin",
        thresholds=[0.5],
        average="binary",
    )

    df = pd.read_csv(out_csv)
    assert df.shape[0] == 1
    # With threshold 0.5, y_pred = [0,1,0,1] so perfect metrics
    row = df.iloc[0]
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0
    assert row["accuracy"] == 1.0


def test_save_threshold_sweep_table_ovr_branch_for_multiclass_binary_average(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Multiclass labels {0,1,2}, 2-D probabilities for 3 classes
    y_true = np.array([0, 1, 2, 1], dtype=int)
    y_probs = np.array(
        [
            [0.7, 0.2, 0.1],  # true 0
            [0.1, 0.8, 0.1],  # true 1
            [0.2, 0.2, 0.6],  # true 2
            [0.2, 0.7, 0.1],  # true 1
        ],
        dtype=float,
    )

    # average="binary" triggers OvR branch, treating class 1 as positive
    out_csv = save_threshold_sweep_table(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=4,
        wd=0.0,
        prefix="ovr",
        thresholds=[0.5],
        average="binary",
    )

    df = pd.read_csv(out_csv)
    assert df.shape[0] == 1
    # With t=0.5, for class 1 vs rest, predictions for [0,1,2,1] are [0,1,0,1] -> perfect
    row = df.iloc[0]
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0
    assert row["accuracy"] == 1.0


def test_save_threshold_sweep_table_rejects_non_path_out_folder(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=float)

    # out_folder is an int here, should raise
    with pytest.raises(
        ValueError, match=r"^out_folder must be a string or pathlib.Path."
    ):
        save_threshold_sweep_table(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=123,  # invalid type
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="bad",
            thresholds=[0.5],
            average="binary",
            fold=1,
            epoch=1,
        )


def test_save_threshold_sweep_table_rejects_non_1d_probs_in_binary_path(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Binary labels
    y_true = np.array([0, 1, 0, 1], dtype=int)
    # Bad: 3-column probs (cannot be auto-sliced to 1-D)
    y_probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.3, 0.6, 0.1],
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match=r"^Binary path expects 1-D y_probs"):
        save_threshold_sweep_table(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=out_dir,
            model="M",
            lr=1e-3,
            bs=4,
            wd=0.0,
            prefix="badbin",
            thresholds=[0.5],
            average="binary",  # forces binary path even though probs are 2-D
            fold=1,
            epoch=1,
        )


# ------------------------------------------------------------------------------
# save_error_tables
# ------------------------------------------------------------------------------


def test_save_error_tables_binary_1d_outputs_and_sorted_by_margin(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Binary labels and 1-D probs
    y_true = np.array([0, 1, 1, 0, 1], dtype=int)
    # Threshold 0.5 -> preds = [0,1,0,1,1]
    # FN at idx=2 (true=1, prob=0.4); FP at idx=3 (true=0, prob=0.7)
    y_probs = np.array([0.1, 0.8, 0.4, 0.7, 0.9], dtype=float)

    fn_path, fp_path = save_error_tables(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        threshold=0.5,
        top_k=50,
        fold=1,
        epoch=1,
    )

    assert fn_path is not None and fp_path is not None
    assert fn_path.exists() and fp_path.exists()

    df_fn = pd.read_csv(fn_path)
    df_fp = pd.read_csv(fp_path)

    # Column schema
    assert list(df_fn.columns) == ["index", "y_true", "y_prob_pos", "margin"]
    assert list(df_fp.columns) == ["index", "y_true", "y_prob_pos", "margin"]

    # Margins sorted ascending
    assert (df_fn["margin"].values == np.sort(df_fn["margin"].values)).all()
    assert (df_fp["margin"].values == np.sort(df_fp["margin"].values)).all()

    # Expected one FN (idx=2) and one FP (idx=3)
    assert set(df_fn["index"].tolist()) == {2}
    assert set(df_fp["index"].tolist()) == {3}

    # Sanity on values
    assert float(df_fn.loc[df_fn["index"] == 2, "y_prob_pos"].iloc[0]) == 0.4
    assert float(df_fp.loc[df_fp["index"] == 3, "y_prob_pos"].iloc[0]) == 0.7


def test_save_error_tables_binary_2d_probs_autoslices_positive_column(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 1, 0], dtype=int)
    # probs[:,1] is positive-class prob; threshold 0.5 -> preds [0,1,0,1]
    y_probs_2d = np.array(
        [
            [0.9, 0.1],  # true 0 -> TN
            [0.2, 0.8],  # true 1 -> TP
            [0.6, 0.4],  # true 1 -> FN (pos prob < 0.5)
            [0.3, 0.7],  # true 0 -> FP (pos prob >= 0.5)
        ],
        dtype=float,
    )

    fn_path, fp_path = save_error_tables(
        y_true=y_true,
        y_probs=y_probs_2d,  # should auto-slice [:, 1]
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        threshold=0.5,
        top_k=10,
    )

    df_fn = pd.read_csv(fn_path)
    df_fp = pd.read_csv(fp_path)

    # FN should include index 2; FP should include index 3
    assert 2 in df_fn["index"].tolist()
    assert 3 in df_fp["index"].tolist()


def test_save_error_tables_skips_on_multiclass_labels(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Multiclass labels (three classes)
    y_true = np.array([0, 1, 2, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.7], dtype=float)

    fn_path, fp_path = save_error_tables(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        threshold=0.5,
        top_k=10,
    )

    out = capsys.readouterr().out
    assert "Skipping error tables — expected binary labels." in out
    assert fn_path is None and fp_path is None


def test_save_error_tables_rejects_bad_out_folder_type(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=float)

    with pytest.raises(
        ValueError, match=r"^out_folder must be a string or pathlib.Path."
    ):
        save_error_tables(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=123,  # invalid
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="run",
        )


def test_save_error_tables_rejects_bad_threshold_and_topk(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=float)

    for bad_thr in [-0.1, 1.1, "x"]:
        with pytest.raises(
            ValueError, match=r"^threshold must be a numeric value in \[0.0, 1.0\]"
        ):
            save_error_tables(
                y_true=y_true,
                y_probs=y_probs,
                out_folder=out_dir,
                model="M",
                lr=1e-3,
                bs=8,
                wd=0.0,
                prefix="run",
                threshold=bad_thr,  # invalid
            )

    for bad_k in [0, -1, 1.5]:
        with pytest.raises(ValueError, match=r"^top_k must be a positive integer."):
            save_error_tables(
                y_true=y_true,
                y_probs=y_probs,
                out_folder=out_dir,
                model="M",
                lr=1e-3,
                bs=8,
                wd=0.0,
                prefix="run",
                top_k=bad_k,  # invalid
            )


def test_save_error_tables_topk_limiting_and_margin_order(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construct multiple FNs/FPs to test top_k limiting and order
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=int)
    y_probs = np.array([0.49, 0.2, 0.8, 0.51, 0.6, 0.1, 0.45, 0.55], dtype=float)
    # thr=0.5 -> FNs at idx 0 (margin .01), 1 (.3), 6 (.05)  → sorted by margin: 0,6,1
    #            FPs at idx 3 (.01), 4 (.1), 7 (.05)         → sorted by margin: 3,7,4

    fn_path, fp_path = save_error_tables(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        threshold=0.5,
        top_k=2,  # limit to top-2 closest cases
    )

    df_fn = pd.read_csv(fn_path)
    df_fp = pd.read_csv(fp_path)

    # Check limiting
    assert len(df_fn) == 2
    assert len(df_fp) == 2

    # Check the order by smallest margin first
    assert df_fn["index"].tolist() == [0, 6]  # margins .01, .05
    assert df_fp["index"].tolist() == [3, 7]  # margins .01, .05


def test_save_error_tables_skips_when_probs_not_1d(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Valid binary labels
    y_true = np.array([0, 1, 0, 1], dtype=int)
    # Bad probs: shape (N,3) → not 1D, not 2-col binary
    y_probs = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.3, 0.3, 0.4],
            [0.6, 0.2, 0.2],
        ],
        dtype=float,
    )

    fn_path, fp_path = save_error_tables(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=out_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        threshold=0.5,
        top_k=5,
    )

    out = capsys.readouterr().out
    assert "Skipping error tables — expected 1D probabilities for binary." in out
    assert fn_path is None and fp_path is None


# ------------------------------------------------------------------------------
# save_confidence_histogram
# ------------------------------------------------------------------------------


def test_save_confidence_histogram_binary_saves_png(tmp_path: Path) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Binary labels; function only needs y_probs to be 1-D, but keep labels consistent
    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    y_probs = np.array([0.05, 0.90, 0.40, 0.75, 0.20, 0.10], dtype=float)

    out_path = save_confidence_histogram(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=plot_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        bins=10,
        fold=1,
        epoch=1,
        title="Confidence Histogram",
    )

    assert out_path is not None
    assert out_path.exists()
    # File should be non-empty
    assert out_path.stat().st_size > 0


def test_save_confidence_histogram_skips_on_non_1d_probs(
    tmp_path: Path, capsys
) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Valid binary labels but 2-D probs -> function should skip and return None
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.3, 0.7],
        ],
        dtype=float,
    )

    out_path = save_confidence_histogram(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=plot_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        bins=20,
        fold=1,
        epoch=1,
        title="Confidence Histogram",
    )

    out = capsys.readouterr().out
    assert (
        "Skipping confidence histogram — expected 1D probabilities for binary." in out
    )
    assert out_path is None


def test_save_confidence_histogram_rejects_bad_out_folder_type(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    with pytest.raises(ValueError, match=r"^out_folder must be a string or Path."):
        save_confidence_histogram(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=123,  # invalid type
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="run",
        )


# ------------------------------------------------------------------------------
# save_confidence_histogram_split
# ------------------------------------------------------------------------------


def test_save_confidence_histogram_split_binary_saves_png(tmp_path: Path) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    y_probs = np.array([0.05, 0.90, 0.40, 0.75, 0.20, 0.60], dtype=float)

    out_path = save_confidence_histogram_split(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=plot_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        threshold=0.5,
        bins=10,
        fold=1,
        epoch=1,
        title="Confidence Histogram (Correct vs Incorrect)",
    )

    assert out_path is not None
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_confidence_histogram_split_skips_on_non_1d_probs(
    tmp_path: Path, capsys
) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    # 3 columns -> not auto-sliceable; should skip
    y_probs = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.3, 0.3, 0.4],
            [0.6, 0.2, 0.2],
        ],
        dtype=float,
    )

    out_path = save_confidence_histogram_split(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=plot_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        threshold=0.5,
        bins=20,
    )

    out = capsys.readouterr().out
    assert (
        "Skipping confidence histogram split — expected 1D probabilities for binary."
        in out
    )
    assert out_path is None


def test_save_confidence_histogram_split_skips_labels_not_binary(
    tmp_path: Path, capsys
) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 2], dtype=int)  # bad expect binary labels
    y_probs = np.array([0.05, 0.90, 0.40, 0.75, 0.20, 0.60], dtype=float)

    out_path = save_confidence_histogram_split(
        y_true=y_true,
        y_probs=y_probs,
        out_folder=plot_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        threshold=0.5,
        bins=20,
    )

    out = capsys.readouterr().out
    assert "Skipping confidence histogram split — expected binary labels." in out
    assert out_path is None


def test_save_confidence_histogram_split_autoslices_2col_binary_probs(
    tmp_path: Path,
) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 1, 0], dtype=int)
    # Should auto-slice [:,1]
    y_probs_2d = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],  # incorrect for positive at thr=0.5
            [0.3, 0.7],  # incorrect for negative at thr=0.5
        ],
        dtype=float,
    )

    out_path = save_confidence_histogram_split(
        y_true=y_true,
        y_probs=y_probs_2d,
        out_folder=plot_dir,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        threshold=0.5,
        bins=15,
    )

    assert out_path is not None
    assert out_path.exists()


def test_save_confidence_histogram_split_rejects_bad_threshold(tmp_path: Path) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=float)

    with pytest.raises(
        ValueError, match=r"^threshold must be a numeric value in \[0.0, 1.0\]"
    ):
        save_confidence_histogram_split(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=plot_dir,
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="run",
            threshold=1.5,  # bad
        )


def test_save_confidence_histogram_split_rejects_bad_out_folder() -> None:
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=float)

    with pytest.raises(ValueError, match=r"^out_folder must be a string or Path."):
        save_confidence_histogram_split(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=123,  # bad
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="run",
            threshold=0.5,
        )


# def test_save_threshold_sweep_table_binary(tmp_path):
#     y_true = [0, 1, 0, 1]
#     y_probs = [0.1, 0.9, 0.2, 0.8]
#     csv_p = save_threshold_sweep_table(
#         y_true, y_probs, tmp_path, "ECGConvNet", 0.001, 64, 0.0005, "final"
#     )
#     assert csv_p.exists()
#     import pandas as pd
#     df = pd.read_csv(csv_p)
#     assert set(["threshold", "precision", "recall", "f1", "accuracy"]).issubset(df.columns)

# def test_save_confidence_histogram(tmp_path):
#     y_probs = [0.1, 0.2, 0.8, 0.9]
#     p = save_confidence_histogram(
#         y_probs, tmp_path, "ECGConvNet", 0.001, 64, 0.0005, "final"
#     )
#     assert p.exists()

# def test_save_det_curve_binary(tmp_path):
#     y_true = [0, 1, 0, 1, 1, 0, 0, 1]
#     y_probs = [0.2, 0.9, 0.3, 0.8, 0.7, 0.1, 0.4, 0.6]
#     p = save_det_curve(
#         y_true, y_probs, tmp_path, "ECGConvNet", 0.001, 64, 0.0005, "final"
#     )
#     assert p is not None and p.exists()

# def test_save_lift_gain_curves_binary(tmp_path):
#     y_true = [0, 1, 0, 1, 1, 0, 0, 1]
#     y_probs = [0.2, 0.9, 0.3, 0.8, 0.7, 0.1, 0.4, 0.6]
#     gain_p, lift_p = save_lift_gain_curves(
#         y_true, y_probs, tmp_path, "ECGConvNet", 0.001, 64, 0.0005, "final"
#     )
#     assert gain_p is not None and gain_p.exists()
#     assert lift_p is not None and lift_p.exists()

# def test_save_error_tables_binary(tmp_path):
#     y_true = [0, 1, 0, 1, 1, 0, 0, 1]
#     y_probs = [0.2, 0.9, 0.3, 0.8, 0.51, 0.49, 0.52, 0.48]
#     fn_p, fp_p = save_error_tables(
#         y_true, y_probs, tmp_path, "ECGConvNet", 0.001, 64, 0.0005, "final", threshold=0.5, top_k=3
#     )
#     assert fn_p is not None and fn_p.exists()
#     assert fp_p is not None and fp_p.exists()


# ------------------------------------------------------------------------------
# evaluate_and_plot (integration-style)
# ------------------------------------------------------------------------------


def test_evaluate_and_plot_binary_end_to_end(tmp_path: Path, monkeypatch):
    out_root = tmp_path / "out"
    out_root.mkdir(parents=True, exist_ok=True)

    # Minimal binary case (also exercises PR/ROC/threshold single-curve paths)
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 0]
    y_probs = [0.2, 0.85, 0.3, 0.9, 0.55]
    class_names = ["NEG", "POS"]

    # Train/val curves
    train_accs = [0.6, 0.7, 0.8]
    val_accs = [0.55, 0.68, 0.74]
    train_losses = [1.0, 0.7, 0.5]
    val_losses = [1.1, 0.8, 0.6]

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="ECGConvNet",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="final",
        fname_metric="eval",
        out_folder=out_root,
        class_names=class_names,
        y_probs=y_probs,  # 1D -> binary curves
        fold=1,
        epoch=3,
        enable_ovr=True,  # explicit overrides (independent of env)
        ovr_classes=None,  # not used in binary
    )

    # Check artifacts
    plot_dir = out_root / "plots"
    report_dir = out_root / "reports"
    assert plot_dir.is_dir() and report_dir.is_dir()

    existing = [p.name for p in plot_dir.glob("*.png")]
    assert any("accuracy" in n for n in existing)
    assert any("loss" in n for n in existing)
    assert any("confusion_matrix" in n for n in existing)
    assert any("pr_curve" in n for n in existing)
    assert any("pr_threshold" in n for n in existing)
    assert any("roc_curve" in n for n in existing)

    # Classification report heatmap
    rep_pngs = list(report_dir.glob("*.png"))
    assert rep_pngs, "Expected classification report heatmap PNG"


def test_evaluate_and_plot_multiclass_ovr_enabled_subset(tmp_path: Path):
    out_root = tmp_path / "out_mc"
    out_root.mkdir(parents=True, exist_ok=True)

    # 3-class OvR with subset selection
    class_names = ["NORM", "MI", "STTC"]
    y_true = np.array([0, 1, 2, 1, 0, 2, 1])
    # n x 3 probs
    y_probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.1, 0.7, 0.2],
            [0.8, 0.1, 0.1],
            [0.2, 0.2, 0.6],
            [0.1, 0.8, 0.1],
        ]
    )
    y_pred = y_probs.argmax(axis=1).tolist()

    train_accs = [0.5, 0.6]
    val_accs = [0.45, 0.55]
    train_losses = [1.2, 0.9]
    val_losses = [1.3, 1.0]

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="ECGConvNet",
        lr=1e-3,
        bs=16,
        wd=0.0,
        prefix="final",
        fname_metric="eval",
        out_folder=out_root,
        class_names=class_names,
        y_probs=y_probs,  # 2D -> OvR curves
        fold=2,
        epoch=2,
        enable_ovr=True,
        ovr_classes={"MI"},  # only MI curves should be produced
    )

    plot_dir = out_root / "plots"
    existing = [p.name for p in plot_dir.glob("*.png")]

    # Should still have base plots
    assert any("accuracy" in n for n in existing)
    assert any("loss" in n for n in existing)
    assert any("confusion_matrix" in n for n in existing)

    # OvR per-class curves only for MI (lowercased slug in prefix)
    assert any("_ovr_mi_" in n for n in existing), "Expected MI OvR curves"
    # And *not* for norm/sttc when subset provided
    assert not any("_ovr_norm_" in n for n in existing)
    assert not any("_ovr_sttc_" in n for n in existing)


# ------------------------------------------------------------------------------
# evaluate_and_plot error branches
# ------------------------------------------------------------------------------


def test_evaluate_and_plot_bad_lengths_raises(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # train/val arrays mismatch
    with pytest.raises(
        ValueError,
        match=r"^Training and validation metric lists must be equal in length",
    ):
        evaluate_and_plot(
            y_true=[0, 1],
            y_pred=[0, 1],
            train_accs=[0.5, 0.6],
            val_accs=[0.5],  # mismatch
            train_losses=[1.0, 0.8],
            val_losses=[1.0, 0.9],
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="p",
            fname_metric="m",
            out_folder=out_dir,
            class_names=["A", "B"],
            y_probs=[0.1, 0.9],
            fold=1,
            epoch=2,
        )


def test_evaluate_and_plot_bad_fold_raises(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(
        ValueError, match=r"^fold must be a positive integer if provided"
    ):
        evaluate_and_plot(
            y_true=[0, 1],
            y_pred=[0, 1],
            train_accs=[0.5],
            val_accs=[0.5],
            train_losses=[1.0],
            val_losses=[1.0],
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="p",
            fname_metric="m",
            out_folder=out_dir,
            class_names=["A", "B"],
            y_probs=[0.1, 0.9],
            fold=0,  # invalid per implementation
            epoch=1,
        )


def test_evaluate_and_plot_empty_class_names_raises(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match=r"^class_names cannot be empty"):
        evaluate_and_plot(
            y_true=[0, 1],
            y_pred=[0, 1],
            train_accs=[0.5],
            val_accs=[0.5],
            train_losses=[1.0],
            val_losses=[1.0],
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="p",
            fname_metric="m",
            out_folder=out_dir,
            class_names=[],  # empty -> error
            y_probs=[0.1, 0.9],
            fold=1,
            epoch=1,
        )


def test_evaluate_and_plot_bad_out_folder_type_raises():
    with pytest.raises(ValueError, match=r"^out_folder must be a string or Path, got "):
        evaluate_and_plot(
            y_true=[0, 1],
            y_pred=[0, 1],
            train_accs=[0.5],
            val_accs=[0.5],
            train_losses=[1.0],
            val_losses=[1.0],
            model="M",
            lr=1e-3,
            bs=8,
            wd=0.0,
            prefix="p",
            fname_metric="m",
            out_folder=123,  # bad
            class_names=["A", "B"],
            y_probs=[0.1, 0.9],
            fold=1,
            epoch=1,
        )


def test_evaluate_and_plot_rejects_bad_class_names_length(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.2, 0.8, 0.3, 0.7])  # binary case
    y_pred = _simple_pred_from_probs(y_probs)

    with pytest.raises(
        ValueError,
        match=r"^class_names must cover all class indices in y_true and y_pred.",
    ):
        evaluate_and_plot(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            train_accs=[0.1, 0.2],
            val_accs=[0.09, 0.19],
            train_losses=[1.0, 0.9],
            val_losses=[1.1, 1.0],
            out_folder=out_dir,
            prefix="prefix",
            fname_metric="eval",
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
            class_names=["A"],  # not enough class strings
        )


def test_evaluate_and_plot_rejects_bad_class_names_type(tmp_path: Path):
    out_dir = tmp_path / "eval_bad_type"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.2, 0.8, 0.3, 0.7])  # binary case
    y_pred = _simple_pred_from_probs(y_probs)

    with pytest.raises(
        ValueError, match=r"^class_names must be a non-empty list of strings"
    ):
        evaluate_and_plot(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            train_accs=[0.1, 0.2],
            val_accs=[0.09, 0.19],
            train_losses=[1.0, 0.9],
            val_losses=[1.1, 1.0],
            out_folder=out_dir,
            prefix="prefix",
            fname_metric="eval",
            model="M",
            lr=0.001,
            bs=32,
            wd=0.0,
            class_names=[1, 2],  # not strings
        )


# PR OvR with no matches -> fallback to all classes
def test_evaluate_and_plot_pr_ovr_no_match_falls_back_all(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])

    train_accs = [0.5]
    val_accs = [0.6]
    train_losses = [1.0]
    val_losses = [0.9]

    # 2D probs with 3 classes, rows match n_samples
    z = np.linspace(0.1, 0.9, y_true.size)
    y_probs = np.vstack([z, 1 - z, np.clip(0.5 + 0.1 * z, 0, 1)]).T

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=out_dir,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
        enable_ovr=True,
        ovr_classes={"Z"},  # no overlap with class_names
    )
    files = [p.name for p in (out_dir / "plots").glob("*pr_curve*.png")]

    # Must have exactly one OvR PR curve per class (fallback hit)
    assert sum("_ovr_a" in n for n in files) == 1
    assert sum("_ovr_b" in n for n in files) == 1
    assert sum("_ovr_c" in n for n in files) == 1

    # And no non-OvR PR file slipped through (ensures we were in the OvR branch)
    assert not any("_ovr_" not in n for n in files)


# PR OvR disabled -> prints message
def test_evaluate_and_plot_pr_ovr_disabled_message(tmp_path: Path, capsys):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])

    train_accs = [0.5]
    val_accs = [0.6]
    train_losses = [1.0]
    val_losses = [0.9]

    z = np.linspace(0.1, 0.9, y_true.size)
    y_probs = np.vstack([z, 1 - z, np.clip(0.5 + 0.1 * z, 0, 1)]).T

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=out_dir,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
        enable_ovr=False,  # disabled
        ovr_classes=None,
    )

    out = capsys.readouterr().out
    assert "PR OvR disabled (ECG_PLOTS_ENABLE_OVR is False)" in out


# PR bad shape -> skip message
def test_evaluate_and_plot_pr_bad_shape_skips(tmp_path: Path, capsys):
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])

    train_accs = [0.5]
    val_accs = [0.6]
    train_losses = [1.0]
    val_losses = [0.9]

    # wrong columns: 2 (should be len(class_names)=3)
    y_probs = np.random.RandomState(0).rand(len(y_true), 2)

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=tmp_path,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
        enable_ovr=True,
        ovr_classes=None,
    )

    out = capsys.readouterr().out
    assert "Skipping PR curve" in out  # robust to Unicode dash


# PR-threshold OvR disabled -> prints message
def test_evaluate_and_plot_pr_threshold_ovr_disabled_message(tmp_path: Path, capsys):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])

    train_accs = [0.5]
    val_accs = [0.6]
    train_losses = [1.0]
    val_losses = [0.9]

    z = np.linspace(0.1, 0.9, y_true.size)
    y_probs = np.vstack([z, 1 - z, np.clip(0.5 + 0.1 * z, 0, 1)]).T

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=out_dir,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
        enable_ovr=False,  # disabled triggers PR-threshold message
        ovr_classes=None,
    )

    out = capsys.readouterr().out
    assert "PR-threshold OvR disabled (ECG_PLOTS_ENABLE_OVR is False)" in out


def test_evaluate_and_plot_hits_ovr_fallback(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])
    train_accs, val_accs = [0.5], [0.6]
    train_losses, val_losses = [1.0], [0.9]

    z = np.linspace(0.1, 0.9, y_true.size)
    y_probs = np.vstack([z, 1 - z, np.clip(0.5 + 0.1 * z, 0, 1)]).T  # (n, 3)

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=out_dir,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
        enable_ovr=True,
        ovr_classes={"Z"},  # non-empty, no overlap => triggers fallback line
    )

    names = [p.name for p in (out_dir / "plots").glob("*pr_curve*.png")]
    # Exactly one OvR PR plot per class => loop used fallback range(...)
    assert sum("_ovr_a" in n for n in names) == 1
    assert sum("_ovr_b" in n for n in names) == 1
    assert sum("_ovr_c" in n for n in names) == 1
    assert all("_ovr_" in n for n in names)


def test_evaluate_and_plot_hits_pr_skip(tmp_path: Path, capsys):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])
    train_accs, val_accs = [0.5], [0.6]
    train_losses, val_losses = [1.0], [0.9]

    # Wrong columns: (n, 2) while len(class_names)=3
    y_probs = np.random.RandomState(0).rand(len(y_true), 2)

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=out_dir,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
        enable_ovr=True,
        ovr_classes=None,
    )

    out = capsys.readouterr().out
    assert "Skipping PR curve" in out


def test_evaluate_and_plot_with_empty_ovr(tmp_path: Path):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])

    # minimal, equal-length histories
    train_accs = [0.5]
    val_accs = [0.6]
    train_losses = [1.0]
    val_losses = [0.9]

    # 2D probs with (n_samples, n_classes) to enter all OvR branches
    z = np.linspace(0.1, 0.9, y_true.size)
    y_probs = np.vstack([z, 1 - z, np.clip(0.5 + 0.1 * z, 0, 1)]).T

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=out_dir,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
        enable_ovr=True,  # must be True to enter OvR branches
        ovr_classes=set(),  # EMPTY -> ovr_classes_effective becomes None -> hits the else: idxs = range(...)
    )

    names = [p.name for p in (out_dir / "plots").glob("*.png")]

    # Evidence we looped over ALL classes in each OvR section:
    # PR curves
    assert sum("_ovr_a" in n and "pr_curve" in n for n in names) == 1
    assert sum("_ovr_b" in n and "pr_curve" in n for n in names) == 1
    assert sum("_ovr_c" in n and "pr_curve" in n for n in names) == 1
    # PR-threshold curves
    assert sum("_ovr_a" in n and "pr_threshold" in n for n in names) == 1
    assert sum("_ovr_b" in n and "pr_threshold" in n for n in names) == 1
    assert sum("_ovr_c" in n and "pr_threshold" in n for n in names) == 1
    # ROC curves
    assert sum("_ovr_a" in n and "roc_curve" in n for n in names) == 1
    assert sum("_ovr_b" in n and "roc_curve" in n for n in names) == 1
    assert sum("_ovr_c" in n and "roc_curve" in n for n in names) == 1


def test_evaluate_and_plot_hits_pr_threshold_skip(tmp_path: Path, capsys):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 1, 0, 2])  # multiclass => not binary
    y_pred = np.array([0, 1, 2, 1, 0, 2])
    train_accs, val_accs = [0.5], [0.6]
    train_losses, val_losses = [1.0], [0.9]

    # 1-D vector (length can be n or not; either way it's not binary so PR-threshold hits else)
    y_probs = np.random.RandomState(0).rand(len(y_true))

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=out_dir,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
        enable_ovr=True,
        ovr_classes=None,
    )

    out = capsys.readouterr().out
    assert "Skipping PR curve" in out


# ROC OvR disabled -> prints message
def test_evaluate_and_plot_roc_ovr_disabled_message(tmp_path: Path, capsys):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])

    train_accs = [0.5]
    val_accs = [0.6]
    train_losses = [1.0]
    val_losses = [0.9]

    z = np.linspace(0.1, 0.9, y_true.size)
    y_probs = np.vstack([z, 1 - z, np.clip(0.5 + 0.1 * z, 0, 1)]).T

    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=tmp_path,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
        enable_ovr=False,  # disabled triggers ROC message
        ovr_classes=None,
    )

    out = capsys.readouterr().out
    assert "ROC OvR disabled (ECG_PLOTS_ENABLE_OVR is False)" in out


def test_evaluate_and_plot_binary_with_2d_probs(tmp_path, capsys):
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["neg", "pos"]

    # Binary labels
    y_true = np.array([0, 1, 0, 1])

    # 2D probs with 2 columns (simulate softmax output)
    # Column 0 = P(neg), Column 1 = P(pos)
    y_probs = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.4, 0.6],
        ]
    )

    y_pred = np.argmax(y_probs, axis=1)

    train_accs, val_accs = [0.5], [0.6]
    train_losses, val_losses = [1.0], [0.9]

    # Run evaluate_and_plot — should hit the branch that slices [:,1]
    evaluate_and_plot(
        y_true=y_true,
        y_pred=y_pred,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_losses,
        val_losses=val_losses,
        model="M",
        lr=1e-3,
        bs=8,
        wd=0.0,
        prefix="run",
        fname_metric="metric",
        out_folder=out_dir,
        class_names=class_names,
        y_probs=y_probs,
        fold=1,
        epoch=1,
    )

    out = capsys.readouterr().out
    # Sanity check: calibration output was printed (meaning slice branch executed)
    assert "Calibration Curve" in out or "cal_path" in out
