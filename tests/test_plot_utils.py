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

import numpy as np
import io
import pytest

from pathlib import Path
from contextlib import redirect_stdout

from ecg_cnn.utils.plot_utils import (
    _build_plot_title,
    format_hparams,
    save_plot_curves,
    save_confusion_matrix,
    save_pr_threshold_curve,
    save_pr_curve,
    save_roc_curve,
    save_classification_report,
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


def test_evaluate_and_plot_hits_1097_1149_1202_with_empty_ovr(tmp_path: Path):
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
        out_folder=tmp_path,
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
