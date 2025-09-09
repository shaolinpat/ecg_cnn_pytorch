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
    - shap_sample_background,
    - shap_compute_values,
    - shap_save_channel_summary,
    - save_classification_report_csv,
    - save_fold_summary_csv,

Notes
-----
    - Relies on test-level tmp_path isolation; no artifacts persist.
    - Seeding is handled in conftest.py (no per-file seeds here).
    - All regex match= assertions are anchored with ^ (no trailing $).
"""

from __future__ import annotations

import csv
import io
import json
import numpy as np
import pandas as pd
import pytest
import shap
import torch

from contextlib import redirect_stdout
from pathlib import Path
from sklearn.metrics import brier_score_loss
from types import SimpleNamespace

from ecg_cnn.utils.plot_utils import (
    _build_plot_title,
    _validate_array_3d,
    _to_tensor,
    _to_numpy,
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
    shap_sample_background,
    shap_compute_values,
    shap_save_channel_summary,
    save_classification_report_csv,
    save_fold_summary_csv,
)

# ------------------------------------------------------------------------------
# globals
# ------------------------------------------------------------------------------

shap = pytest.importorskip("shap", reason="SHAP not installed")

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------


def _simple_pred_from_probs(y_probs: np.ndarray) -> np.ndarray:
    """Helper: argmax for multiclass or threshold for binary probabilities."""
    if y_probs.ndim == 1:
        return (y_probs >= 0.5).astype(int)
    return np.argmax(y_probs, axis=1)


# Tiny 1D model for SHAP tests (kept deliberately simple & fast)
class _Tiny1D(torch.nn.Module):
    def __init__(self, in_ch: int, n_class: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(in_ch, 4, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(4, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.fc = torch.nn.Linear(4, n_class)

    def forward(self, x):
        # x: (N, C, T)
        z = self.net(x).squeeze(-1)  # (N, 4)
        return self.fc(z)  # (N, n_class)


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


def test_format_hparams_includes_model_token():
    """
    True branch: non-empty model is included in the filename.
    """
    out = format_hparams(
        model="ECGConvNet",
        lr=0.001,
        bs=64,
        wd=0.0,
        prefix="final",
        fname_metric="loss",
        fold=3,
        epoch=12,
    )
    parts = out.split("_")
    assert parts[0] == "final"
    assert parts[1] == "loss"
    assert "ECGConvNet" in parts  # model token present
    assert any(p.startswith("lr") for p in parts)
    assert any(p.startswith("bs") for p in parts)
    assert any(p.startswith("wd") for p in parts)
    assert "fold3" in parts
    assert "epoch12" in parts


def test_format_hparams_skips_model_when_empty(monkeypatch):
    """
    False branch: empty model string should be skipped.
    Bypass validation so model="" is allowed.
    """
    # Disable validation inside format_hparams without importing the module
    monkeypatch.setitem(
        format_hparams.__globals__, "validate_hparams_formatting", lambda **kwargs: None
    )

    out = format_hparams(
        model="",  # empty -> should NOT add a blank token
        lr=0.001,
        bs=64,
        wd=0.0,
        prefix="best",
        fname_metric="accuracy",
        fold=None,
        epoch=None,
    )
    parts = out.split("_")
    assert parts[0] == "best"
    assert parts[1] == "accuracy"
    # No empty tokens or phantom slot for model
    assert "" not in parts
    # Usual tokens still present
    assert any(p.startswith("lr") for p in parts)
    assert any(p.startswith("bs") for p in parts)
    assert any(p.startswith("wd") for p in parts)
    # Fold/epoch were None, so absent
    assert all(not p.startswith("fold") for p in parts)
    assert all(not p.startswith("epoch") for p in parts)


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


# ------------------------------------------------------------------------------
# SHAP stuff
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# SHAP helper tests
# ------------------------------------------------------------------------------


class Tiny1D(torch.nn.Module):
    def __init__(self, in_ch=1, n_class=2):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_ch, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(4, n_class)

    def forward(self, x):
        z = torch.relu(self.conv(x))
        z = self.pool(z).squeeze(-1)
        return self.fc(z)


def _rand(n, c, t):
    return torch.randn(n, c, t)


def test_shap_background_limits_and_validates():
    X = _rand(50, 2, 128)
    bg = shap_sample_background(X, max_background=16, seed=22)
    assert isinstance(bg, torch.Tensor)
    assert bg.shape == (16, 2, 128)
    X2 = _rand(8, 1, 64)
    bg2 = shap_sample_background(X2, max_background=16, seed=22)
    assert bg2.shape == (8, 1, 64)
    with pytest.raises(ValueError, match=r"^data must have shape \(N, C, T\)"):
        shap_sample_background(torch.randn(8, 64), max_background=4)
    with pytest.raises(ValueError, match=r"^max_background must be >=1"):
        shap_sample_background(X, max_background=0)


def test_shap_compute_values_binary_numpy_shape():
    m = Tiny1D(in_ch=1, n_class=2).eval()
    X = _rand(12, 1, 200)
    bg = shap_sample_background(X, 6, 22)
    sv = shap_compute_values(m, X, bg)
    assert isinstance(sv, np.ndarray)
    assert sv.shape == (12, 1, 200)


def test_shap_compute_values_multiclass_list_shapes():
    m = Tiny1D(in_ch=3, n_class=5).eval()
    X = _rand(10, 3, 160)
    bg = shap_sample_background(X, 5, 22)
    sv_list = shap_compute_values(m, X, bg)
    assert isinstance(sv_list, list)
    assert len(sv_list) == 5
    for sv in sv_list:
        assert isinstance(sv, np.ndarray)
        assert sv.shape == (10, 3, 160)


def test_shap_save_channel_summary_writes_file(tmp_path: Path):
    m = Tiny1D(in_ch=2, n_class=2).eval()
    X = _rand(16, 2, 128)
    bg = shap_sample_background(X, 8, 22)
    sv = shap_compute_values(m, X, bg)
    out = shap_save_channel_summary(sv, X, tmp_path, "shap_summary_test.png")
    assert out.exists()
    assert out.name == "shap_summary_test.png"


def test_shap_save_channel_summary_validates(tmp_path: Path):
    with pytest.raises(ValueError, match=r"^shap_values must have shape \(N, C, T\)"):
        shap_save_channel_summary(
            np.zeros((8, 2)), np.zeros((8, 2, 64)), tmp_path, "a.png"
        )
    with pytest.raises(ValueError, match=r"^X must have shape \(N, C, T\)"):
        shap_save_channel_summary(
            np.zeros((8, 2, 64)), np.zeros((8, 2)), tmp_path, "b.png"
        )


# Tiny 1D model for SHAP tests (kept deliberately simple & fast)
class _Tiny1D(torch.nn.Module):
    def __init__(self, in_ch: int, n_class: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(in_ch, 4, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(4, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.fc = torch.nn.Linear(4, n_class)

    def forward(self, x):
        # x: (N, C, T)
        z = self.net(x).squeeze(-1)  # (N, 4)
        return self.fc(z)  # (N, n_class)


# ------------------------------------------------------------------------------
# _validate_array_3d()
# ------------------------------------------------------------------------------


def test_validate_array_3d_happy_path():
    x = np.zeros((2, 3, 5), dtype=float)
    n, c, t = _validate_array_3d("x", x)
    assert (n, c, t) == (2, 3, 5)


def test_validate_array_3d_wrong_type():
    with pytest.raises(TypeError, match=r"^x must be np.ndarray or torch.Tensor"):
        _validate_array_3d("x", 123)  # not ndarray/tensor


def test_validate_array_3d_wrong_ndim():
    with pytest.raises(ValueError, match=r"^x must have shape \(N, C, T\)"):
        _validate_array_3d("x", np.zeros((2, 3)))  # 2D, not 3D


def test_validate_array_3d_nonpositive_dim():
    with pytest.raises(ValueError, match=r"^x must have positive dimensions"):
        _validate_array_3d("x", np.zeros((1, 0, 4)))


# ------------------------------------------------------------------------------
# _to_tensor()
# ------------------------------------------------------------------------------


def test_to_tensor_from_numpy_and_passthrough():
    x_np = np.random.randn(4, 2, 10).astype(np.float32)
    t1 = _to_tensor(x_np, "x")
    assert isinstance(t1, torch.Tensor)
    x_t = torch.randn(3, 1, 8)
    t2 = _to_tensor(x_t, "x")
    assert t2 is x_t  # passthrough


def test_to_tensor_type_error():
    with pytest.raises(TypeError, match=r"^x must be np.ndarray or torch.Tensor"):
        _to_tensor("bad", "x")


# ------------------------------------------------------------------------------
# _to_numpy()
# ------------------------------------------------------------------------------


def test_to_numpy_from_tensor_and_passthrough():
    x_t = torch.randn(5, 2, 7)
    a1 = _to_numpy(x_t, "x")
    assert isinstance(a1, np.ndarray)
    x_np = np.zeros((2, 2, 2), dtype=float)
    a2 = _to_numpy(x_np, "x")
    assert a2 is x_np  # passthrough


def test_to_numpy_type_error():
    with pytest.raises(TypeError, match=r"^x must be np.ndarray or torch.Tensor"):
        _to_numpy(object(), "x")


# ------------------------------------------------------------------------------
# shap_sample_background()
# ------------------------------------------------------------------------------


def test_shap_sample_background_no_subsample():
    X = np.random.randn(16, 2, 64).astype(np.float32)
    bg = shap_sample_background(X, max_background=32, seed=22)
    # n <= max_background => identity
    assert bg.shape == (16, 2, 64)
    assert np.allclose(bg.numpy(), X)


def test_shap_sample_background_subsample_deterministic():
    X = np.random.randn(64, 2, 64).astype(np.float32)
    bg1 = shap_sample_background(X, max_background=16, seed=22)
    bg2 = shap_sample_background(X, max_background=16, seed=22)
    assert bg1.shape == (16, 2, 64)
    assert np.allclose(bg1.numpy(), bg2.numpy())
    # Different seed likely different sample
    bg3 = shap_sample_background(X, max_background=16, seed=23)
    assert not np.allclose(bg1.numpy(), bg3.numpy())


# ------------------------------------------------------------------------------
# shap_compute_values()
# ------------------------------------------------------------------------------


def test_shap_compute_values_restores_training_mode_and_binary_shape():
    model = _Tiny1D(in_ch=1, n_class=2)
    model.train()  # ensure we start in train mode
    X = np.random.randn(12, 1, 128).astype(np.float32)
    bg = shap_sample_background(X, max_background=6, seed=22)

    sv = shap_compute_values(model, X, bg)  # check_additivity=False inside
    # After call, original training mode restored
    assert model.training is True
    # Binary returns single array (N, C, T)
    assert isinstance(sv, np.ndarray)
    assert sv.shape == (12, 1, 128)


def test_shap_compute_values_multiclass_shapes_list():
    model = _Tiny1D(in_ch=3, n_class=5).eval()
    X = np.random.randn(10, 3, 96).astype(np.float32)
    bg = shap_sample_background(X, max_background=5, seed=22)

    sv_list = shap_compute_values(model, X, bg)
    assert isinstance(sv_list, list) and len(sv_list) == 5
    for sv in sv_list:
        assert isinstance(sv, np.ndarray)
        assert sv.shape == (10, 3, 96)


# ------------------------------------------------------------------------------
# shap_save_channel_summary()
# ------------------------------------------------------------------------------


def test_shap_save_channel_summary_raises_on_empty_list(tmp_path: Path):
    X = np.random.randn(4, 2, 16).astype(np.float32)
    with pytest.raises(
        ValueError, match=r"^Empty shap_values list for multiclass case."
    ):
        shap_save_channel_summary([], X, tmp_path, "empty.png")


def test_shap_save_channel_summary_validates_shape(tmp_path: Path):
    # C == 0 triggers validation error
    sv = np.zeros((5, 0, 10), dtype=np.float32)
    X = np.zeros((5, 0, 10), dtype=np.float32)
    with pytest.raises(ValueError, match=r"^shap_values must have positive dimensions"):
        shap_save_channel_summary(sv, X, tmp_path, "bad.png")


def test_shap_save_channel_summary_writes_file_binary(tmp_path: Path):
    # Simple binary shap: (N, C, T)
    sv = np.random.randn(8, 2, 20).astype(np.float32)
    X = np.random.randn(8, 2, 20).astype(np.float32)
    out = shap_save_channel_summary(sv, X, tmp_path, "chan_imp.png")
    assert out.exists() and out.suffix == ".png" and out.stat().st_size > 0


def test_shap_save_channel_summary_multiclass_picks_max(tmp_path: Path):
    # Two classes; class 1 has larger magnitude
    sv0 = np.random.randn(6, 3, 12).astype(np.float32) * 0.01
    sv1 = np.random.randn(6, 3, 12).astype(np.float32) * 1.50
    X = np.random.randn(6, 3, 12).astype(np.float32)
    out = shap_save_channel_summary([sv0, sv1], X, tmp_path, "mc_imp.png")
    assert out.exists() and out.stat().st_size > 0


# ------------------------------------------------------------------------------
# shap_compute_values()
# ------------------------------------------------------------------------------


def test_shap_compute_values_multiclass_branch_via_stub(monkeypatch):
    class DummyExplainer:
        def __init__(self, model, bg):
            pass

        def shap_values(self, X, check_additivity=False):
            # Force "list" branch: return K=3 class maps (N,C,T)
            n, c, t = X.shape
            return [np.zeros((n, c, t), dtype=np.float32) + k for k in range(3)]

    monkeypatch.setattr(shap, "DeepExplainer", lambda m, bg: DummyExplainer(m, bg))

    # Param-less model that outputs logits (N, K)
    class TinyLogits(torch.nn.Module):
        def forward(self, x):
            # x: (N, C, T) -> logits: (N, 3)
            # Use simple reductions to avoid parameters
            # (any (N,K) is fine for the shape check)
            m = x.mean(dim=2)  # (N, C)
            if m.shape[1] == 1:  # expand to 3 "classes"
                m = m.repeat(1, 3)  # (N, 3)
            else:
                m = m[:, :3]  # (N, 3)
            return m

    model = TinyLogits().eval()

    X = np.random.randn(2, 1, 8).astype(np.float32)
    bg = X[:1]

    out = shap_compute_values(model, X, bg, device=torch.device("cpu"))

    assert isinstance(out, list) and len(out) == 3
    for k, sv in enumerate(out):
        assert sv.shape == (2, 1, 8)


def test_shap_compute_values_raises_when_model_has_no_params_and_no_device():
    """Covers plot_utils.py:2837 — model has no parameters and device=None -> ValueError."""

    class _NoParamModel(torch.nn.Module):
        def forward(self, x):
            # Won’t be reached; the error is raised before any forward pass.
            return torch.zeros(1)

    # Valid (N, C, T) arrays to get past shape checks quickly
    X = np.zeros((1, 1, 8), dtype=np.float32)
    bg = np.zeros((1, 1, 8), dtype=np.float32)

    with pytest.raises(
        ValueError, match=r"^model has no parameters; cannot infer device or run SHAP"
    ) as ei:
        shap_compute_values(_NoParamModel(), X, bg, device=None)

    msg = str(ei.value)
    assert "model has no parameters" in msg
    assert "cannot infer device" in msg


def test_shap_compute_values_invalid_logits_shape_raises():
    """Covers plot_utils.py: invalid model output shape triggers ValueError."""

    class _BadShapeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))  # ensure params exist

        def forward(self, x):
            # Invalid: 2D but first dim != 1, and not 1D either
            return torch.zeros((5, 3))

    X = np.zeros((2, 1, 4), dtype=np.float32)
    bg = np.zeros((1, 1, 4), dtype=np.float32)

    with pytest.raises(
        ValueError, match=r"^Expected model logits of shape \(N,K\) or \(N,\)"
    ) as ei:
        shap_compute_values(_BadShapeModel(), X, bg, device=torch.device("cpu"))
    assert "Expected model logits" in str(ei.value)


def test_shap_compute_values_binary_path_returns_array(monkeypatch):
    """Covers plot_utils.py:single SHAP map is returned directly."""

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 1)  # at least one parameter

        def forward(self, x):
            return torch.zeros(1)  # valid: 1D logits

    # Fake SHAP: DeepExplainer returns a single (N,C,T) ndarray
    fake_out = np.ones((1, 1, 4), dtype=np.float32)

    class _FakeExplainer:
        def __init__(self, model, bg):
            pass

        def shap_values(self, X, **kw):
            return fake_out

    # Patch the *correct* module path (your file is ecg_cnn/utils/plot_utils.py)
    monkeypatch.setattr(
        "ecg_cnn.utils.plot_utils.shap.DeepExplainer",
        lambda m, b: _FakeExplainer(m, b),
        raising=True,
    )

    X = np.zeros((1, 1, 4), dtype=np.float32)
    bg = np.zeros((1, 1, 4), dtype=np.float32)
    out = shap_compute_values(_TinyModel(), X, bg, device=torch.device("cpu"))

    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 1, 4)
    assert (out == 1).all()


def test_shap_compute_values_falls_back_to_gradient_on_tf_import(monkeypatch):
    """Covers plot_utils.py: DeepExplainer raises TF import error; fallback uses GradientExplainer."""

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 1)  # ensure parameters exist

        def forward(self, x):
            return torch.zeros(1)  # valid 1D logits

    # Simulate DeepExplainer failing due to TensorFlow import
    class _DeepFail:
        def __init__(self, model, bg):
            raise ImportError("No module named 'tensorflow'")

    # GradientExplainer succeeds and returns a single (N,C,T) map
    fake_out = np.full((1, 1, 4), 7, dtype=np.float32)

    class _GradOK:
        def __init__(self, model, bg):
            pass

        def shap_values(self, X):
            return fake_out

    monkeypatch.setattr(
        "ecg_cnn.utils.plot_utils.shap.DeepExplainer", _DeepFail, raising=True
    )
    monkeypatch.setattr(
        "ecg_cnn.utils.plot_utils.shap.GradientExplainer", _GradOK, raising=True
    )

    X = np.zeros((1, 1, 4), dtype=np.float32)
    bg = np.zeros((1, 1, 4), dtype=np.float32)
    out = shap_compute_values(_TinyModel(), X, bg, device=torch.device("cpu"))

    # Single-map normalization path -> returns array directly
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 1, 4)
    assert (out == 7).all()


def test_shap_compute_values_tf_import_then_gradient_fails_raises_runtime(monkeypatch):
    """Covers plot_utils.py: DeepExplainer TF error; GradientExplainer also fails => RuntimeError."""

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 1)

        def forward(self, x):
            return torch.zeros(1)

    class _DeepFail:
        def __init__(self, model, bg):
            raise ImportError("No module named 'tensorflow'")

    class _GradFail:
        def __init__(self, model, bg):
            pass

        def shap_values(self, X):
            raise RuntimeError("gradient boom")

    monkeypatch.setattr(
        "ecg_cnn.utils.plot_utils.shap.DeepExplainer", _DeepFail, raising=True
    )
    monkeypatch.setattr(
        "ecg_cnn.utils.plot_utils.shap.GradientExplainer", _GradFail, raising=True
    )

    X = np.zeros((1, 1, 4), dtype=np.float32)
    bg = np.zeros((1, 1, 4), dtype=np.float32)
    with pytest.raises(
        RuntimeError,
        match=r"^SHAP DeepExplainer failed: No module named \'tensorflow\'",
    ) as ei:
        shap_compute_values(_TinyModel(), X, bg, device=torch.device("cpu"))
    msg = str(ei.value)
    assert "SHAP DeepExplainer failed:" in msg
    assert "tensorflow" in msg.lower()


def test_shap_compute_values_deep_fails_non_tf_raises_runtime(monkeypatch):
    """Covers plot_utils.py: DeepExplainer raises non-TF error => RuntimeError."""

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 1)

        def forward(self, x):
            return torch.zeros(1)

    class _DeepFail:
        def __init__(self, model, bg):
            # Not a TF import message; should go straight to RuntimeError
            raise ValueError("weird failure in deep explainer")

    monkeypatch.setattr(
        "ecg_cnn.utils.plot_utils.shap.DeepExplainer", _DeepFail, raising=True
    )

    X = np.zeros((1, 1, 4), dtype=np.float32)
    bg = np.zeros((1, 1, 4), dtype=np.float32)
    with pytest.raises(
        RuntimeError,
        match=r"^SHAP DeepExplainer failed: weird failure in deep explainer",
    ) as ei:
        shap_compute_values(_TinyModel(), X, bg, device=torch.device("cpu"))
    assert "SHAP DeepExplainer failed:" in str(ei.value)


def test_shap_invalid_logits_shape_raises_and_restores_mode(monkeypatch):
    # model forward returns invalid shape -> triggers
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return torch.zeros((1, 1, 8))  # invalid (3D)

    m = M().train()
    X = np.zeros((2, 1, 8), dtype=np.float32)
    bg = np.zeros((1, 1, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        shap_compute_values(m, X, bg)
    assert m.training is True  # training mode restored on error


def test_shap_compute_values_tf_fallback_then_fail_raises_runtimeerror(monkeypatch):
    # DeepExplainer raises TF import error -> GradientExplainer also fails
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return torch.zeros((1, 5))

    m = M()
    X = np.zeros((2, 1, 8), dtype=np.float32)
    bg = np.zeros((1, 1, 8), dtype=np.float32)

    def _deep(*a, **k):
        raise Exception("No module named 'tensorflow'")

    def _grad(*a, **k):
        raise Exception("gradient explainer boom")

    monkeypatch.setattr(shap, "DeepExplainer", _deep, raising=True)
    monkeypatch.setattr(shap, "GradientExplainer", _grad, raising=True)

    with pytest.raises(RuntimeError):
        shap_compute_values(m, X, bg)


def test_shap_compute_values_deep_fail_non_tf_raises_runtimeerror(monkeypatch):
    # DeepExplainer raises non-TF error
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return torch.zeros((1, 3))

    m = M()
    X = np.zeros((2, 1, 8), dtype=np.float32)
    bg = np.zeros((1, 1, 8), dtype=np.float32)

    def _deep(*a, **k):
        raise Exception("some other backend error")

    monkeypatch.setattr(shap, "DeepExplainer", _deep, raising=True)

    with pytest.raises(RuntimeError):
        shap_compute_values(m, X, bg)


def test_shap_compute_values_invalid_logits_shape_restores_mode_false_branch():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return torch.zeros((1, 1, 8))  # invalid (3D)

    m = M().eval()
    X = np.zeros((2, 1, 8), dtype=np.float32)
    bg = np.zeros((1, 1, 8), dtype=np.float32)

    with pytest.raises(ValueError):
        shap_compute_values(m, X, bg)
    assert m.training is False


@pytest.mark.parametrize(
    "deep_err, expect_tf_fallback",
    [
        ("No module named 'tensorflow'", True),
        ("some other backend error", False),
    ],
)
def test_shapc_compute_values_deep_fails(monkeypatch, deep_err, expect_tf_fallback):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return torch.zeros((1, 3))  # valid logits

    m = M()
    X = np.zeros((2, 1, 8), dtype=np.float32)
    bg = np.zeros((1, 1, 8), dtype=np.float32)

    class _Deep:
        def __init__(self, *a, **k):
            raise Exception(deep_err)

    if expect_tf_fallback:

        class _Grad:
            def __init__(self, *a, **k):
                raise Exception("gradient fail")

        shap_ns = SimpleNamespace(DeepExplainer=_Deep, GradientExplainer=_Grad)
    else:
        shap_ns = SimpleNamespace(DeepExplainer=_Deep)

    monkeypatch.setitem(shap_compute_values.__globals__, "shap", shap_ns)

    with pytest.raises(RuntimeError):
        shap_compute_values(m, X, bg)


@pytest.mark.parametrize(
    "deep_err, patch_grad",
    [
        (
            "No module named 'tensorflow'",
            True,
        ),  # false branch of model_was_training
        (
            "some other backend error",
            False,
        ),  # false branch of model_was_training
    ],
)
def test_shap_compute_values_deep_fail_false_branch(monkeypatch, deep_err, patch_grad):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return torch.zeros((1, 3))

    m = M().eval()  # <<< must be eval() to skip the restore call

    X = np.zeros((2, 1, 8), dtype=np.float32)
    bg = np.zeros((1, 1, 8), dtype=np.float32)

    class _Deep:
        def __init__(self, *a, **k):
            raise Exception(deep_err)

    if patch_grad:

        class _Grad:
            def __init__(self, *a, **k):
                raise Exception("grad fail")

        shap_ns = SimpleNamespace(DeepExplainer=_Deep, GradientExplainer=_Grad)
    else:
        shap_ns = SimpleNamespace(DeepExplainer=_Deep)

    monkeypatch.setitem(shap_compute_values.__globals__, "shap", shap_ns)

    with pytest.raises(RuntimeError):
        shap_compute_values(m, X, bg)
    assert m.training is False


# ------------------------------------------------------------------------------
# save_classification_report_csv()
# ------------------------------------------------------------------------------


def test_save_classification_report_csv_raises_on_non_positive_fold_id(tmp_path):
    # Minimal valid y arrays
    y_true = [0, 1]
    y_pred = [0, 1]
    with pytest.raises(ValueError, match=r"^fold_id must be a positive integer") as ei:
        save_classification_report_csv(y_true, y_pred, tmp_path, "t", 0)
    assert "fold_id must be a positive integer" in str(ei.value)


def test_save_cr_raises_on_empty_or_mismatch_vectors(tmp_path):
    # Empty
    with pytest.raises(
        ValueError, match=r"^y_true and y_pred must be same non-zero length"
    ) as ei1:
        save_classification_report_csv([], [], tmp_path, "t", 1)
    assert "y_true and y_pred must be same non-zero length" in str(ei1.value)

    # Mismatch
    with pytest.raises(
        ValueError, match=r"^y_true and y_pred must be same non-zero length"
    ) as ei2:
        save_classification_report_csv([0, 1], [0], tmp_path, "t", 1)
    assert "y_true and y_pred must be same non-zero length" in str(ei2.value)


# ------------------------------------------------------------------------------
# save_fold_summary_csv()
# ------------------------------------------------------------------------------


def test_save_fold_summary_csv_invalid_tag_prints_and_returns_none(tmp_path, capsys):

    out = save_fold_summary_csv(tmp_path / "reports", "")  # invalid tag
    printed = capsys.readouterr().out
    assert out is None
    assert "save_fold_summary_csv: invalid tag provided" in printed


def test_save_fold_summary_csv_requires_two_folds(tmp_path, capsys):
    # Layout expected by save_fold_summary_csv:
    # reports_dir parent has history/ and artifacts/ siblings
    root = tmp_path / "outputs"
    reports = root / "reports"
    history = root / "history"
    artifacts = root / "artifacts"
    for p in (reports, history, artifacts):
        p.mkdir(parents=True, exist_ok=True)

    tag = "ecg_ECGResNet_lr001_bs8_wd0"

    # Only ONE history file -> should print and return None
    (history / f"history_{tag}_fold1.json").write_text(
        '{"best_epoch": 0, "val_acc": [0.8], "val_loss": [0.3]}'
    )

    out = save_fold_summary_csv(reports, tag)
    printed = capsys.readouterr().out

    assert out is None
    assert f"save_fold_summary_csv: found only 1 folds for tag '{tag}'" in printed


def test_save_fold_summary_csv_aggregates_and_writes(tmp_path, capsys):

    # Directory layout
    root = tmp_path / "outputs"
    reports = root / "reports"
    history = root / "history"
    artifacts = root / "artifacts"
    for p in (reports, history, artifacts):
        p.mkdir(parents=True, exist_ok=True)

    tag = "ecg_ECGResNet_lr001_bs8_wd0"

    # Two folds' history files (with explicit best_epoch)
    (history / f"history_{tag}_fold1.json").write_text(
        '{"best_epoch": 1, "val_acc": [0.7, 0.8], "val_loss": [0.4, 0.3]}'
    )
    (history / f"history_{tag}_fold2.json").write_text(
        '{"best_epoch": 0, "val_acc": [0.9], "val_loss": [0.2]}'
    )

    # Matching per-fold classification report CSVs under artifacts/
    # Header must include at least label and f1-score; include a macro avg row
    (artifacts / f"classification_report_{tag}_fold1.csv").write_text(
        "label,precision,recall,f1-score,support\n"
        "CD,0,0,0,0\n"
        "macro avg,0.5,0.5,0.55,2\n"
    )
    (artifacts / f"classification_report_{tag}_fold2.csv").write_text(
        "label,precision,recall,f1-score,support\n"
        "CD,0,0,0,0\n"
        "macro avg,0.5,0.5,0.65,2\n"
    )

    out_csv = save_fold_summary_csv(reports, tag)
    printed = capsys.readouterr().out

    # It should succeed and print the path
    assert out_csv is not None
    assert out_csv.exists()
    assert f"Saved fold summary CSV to {out_csv}" in printed

    # Validate a few key rows/values were written
    content = out_csv.read_text().strip().splitlines()

    # Fold rows are written before the blank line; find them explicitly
    row1 = next(r for r in content if r.startswith("1,"))
    row2 = next(r for r in content if r.startswith("2,"))

    c1 = row1.split(",")
    c2 = row2.split(",")

    # CSV schema from implementation:
    # ["fold","best_epoch","val_acc","val_loss","macro_f1","history_path","cr_csv"]
    assert len(c1) >= 7 and len(c2) >= 7

    # Fold 1 expectations
    assert c1[0] == "1"
    assert c1[2] == "0.8"  # val_acc at best_epoch=1
    assert c1[4] == "0.55"  # macro_f1 from CR CSV

    # Fold 2 expectations
    assert c2[0] == "2"
    assert c2[2] == "0.9"  # val_acc at best_epoch=0
    assert c2[4] == "0.65"  # macro_f1 from CR CSV

    # Check mean lines are present (exact values depend on computation)
    assert any(line.startswith("mean_val_acc,") for line in content)
    assert any(line.startswith("std_val_acc,") for line in content)
    assert any(line.startswith("mean_macro_f1,") for line in content)
    assert any(line.startswith("std_macro_f1,") for line in content)


def test_save_fold_summary_csv_handles_bad_history_json(tmp_path, capsys):
    reports = tmp_path / "reports"
    history = reports.parent / "history"
    artifacts = reports.parent / "artifacts"
    for d in (reports, history, artifacts):
        d.mkdir(parents=True, exist_ok=True)

    tag = "ecg_ECGResNet_lr001_bs8_wd0"

    # One unreadable + one valid so we pass the "<2 folds" check
    (history / f"history_{tag}_fold1.json").write_text("{not json")
    (history / f"history_{tag}_fold2.json").write_text(
        '{"best_epoch":0,"val_acc":[0.9],"val_loss":[0.2]}'
    )

    out = save_fold_summary_csv(reports, tag)
    printed = capsys.readouterr().out

    assert out is not None or out is None  # We only care about the error message
    assert "save_fold_summary_csv: failed to read" in printed


def test_save_fold_summary_csv_handles_val_loss_non_numeric(tmp_path, capsys):
    reports = tmp_path / "reports"
    history = reports.parent / "history"
    artifacts = reports.parent / "artifacts"
    for d in (reports, history, artifacts):
        d.mkdir(parents=True, exist_ok=True)

    tag = "ecg_ECGResNet_lr001_bs8_wd0"

    # Unorderable val_loss so min(..., key=...) throws -> caught -> be=None
    (history / f"history_{tag}_fold1.json").write_text(
        '{"val_loss": [{"a":1},{"b":2}], "val_acc": []}'
    )
    # A second valid fold so we don't trip the "<2 folds" early exit
    (history / f"history_{tag}_fold2.json").write_text(
        '{"best_epoch": 0, "val_acc": [0.9], "val_loss": [0.1]}'
    )

    out = save_fold_summary_csv(reports, tag)
    printed = capsys.readouterr().out

    assert out is not None  # aggregation still succeeds thanks to fold2
    assert "failed to resolve best_epoch" in printed


def test_save_fold_summary_csv_handles_bad_macro_f1_value(tmp_path, capsys):
    reports = tmp_path / "reports"
    history = reports.parent / "history"
    artifacts = reports.parent / "artifacts"
    for d in (reports, history, artifacts):
        d.mkdir(parents=True, exist_ok=True)

    tag = "ecg_ECGResNet_lr001_bs8_wd0"
    (history / f"history_{tag}_fold1.json").write_text(
        '{"best_epoch": 0, "val_acc": [0.5], "val_loss": [0.5]}'
    )
    (history / f"history_{tag}_fold2.json").write_text(
        '{"best_epoch": 0, "val_acc": [0.6], "val_loss": [0.4]}'
    )
    # Corrupt CSV with non-numeric macro avg f1
    (artifacts / f"classification_report_{tag}_fold1.csv").write_text(
        "label,f1-score\nmacro avg,not_a_number\n"
    )
    (artifacts / f"classification_report_{tag}_fold2.csv").write_text(
        "label,f1-score\nmacro avg,0.9\n"
    )

    save_fold_summary_csv(reports, tag)
    printed = capsys.readouterr().out
    assert "bad macro_f1 value" in printed


def test_save_fold_summary_csv_handles_failed_csv_parse(tmp_path, capsys):
    reports = tmp_path / "reports"
    history = reports.parent / "history"
    artifacts = reports.parent / "artifacts"
    for d in (reports, history, artifacts):
        d.mkdir(parents=True, exist_ok=True)

    tag = "ecg_ECGResNet_lr001_bs8_wd0"
    (history / f"history_{tag}_fold1.json").write_text(
        '{"best_epoch":0,"val_acc":[0.5],"val_loss":[0.5]}'
    )
    (history / f"history_{tag}_fold2.json").write_text(
        '{"best_epoch":0,"val_acc":[0.6],"val_loss":[0.4]}'
    )

    # Make CR path a directory -> .open(...) fails -> triggers "failed to parse"
    (artifacts / f"classification_report_{tag}_fold1.csv").mkdir(
        parents=True, exist_ok=True
    )

    out = save_fold_summary_csv(reports, tag)
    printed = capsys.readouterr().out

    assert out is not None  # still writes the summary
    assert "failed to parse" in printed


def test_save_fold_summary_csv_mean_std_empty_lists(tmp_path):
    reports = tmp_path / "reports"
    history = reports.parent / "history"
    artifacts = reports.parent / "artifacts"
    for d in (reports, history, artifacts):
        d.mkdir(parents=True, exist_ok=True)

    tag = "ecg_ECGResNet_lr001_bs8_wd0"
    # Two folds but with empty metrics
    (history / f"history_{tag}_fold1.json").write_text("{}")
    (history / f"history_{tag}_fold2.json").write_text("{}")

    out = save_fold_summary_csv(reports, tag)
    assert (
        out is None or out.exists()
    )  # safe: mean/std branches just return (None, None)


def test_save_fold_summary_csv_handles_write_failure(tmp_path, capsys):
    reports = tmp_path / "reports"
    history = reports.parent / "history"
    artifacts = reports.parent / "artifacts"
    for d in (reports, history, artifacts):
        d.mkdir(parents=True, exist_ok=True)

    tag = "ecg_ECGResNet_lr001_bs8_wd0"
    (history / f"history_{tag}_fold1.json").write_text(
        '{"best_epoch":0,"val_acc":[0.5],"val_loss":[0.5]}'
    )
    (history / f"history_{tag}_fold2.json").write_text(
        '{"best_epoch":0,"val_acc":[0.6],"val_loss":[0.4]}'
    )

    # Pre-create a DIRECTORY where the function will try to write the CSV
    bad_target = reports / f"fold_summary_{tag}.csv"
    bad_target.mkdir(parents=True, exist_ok=True)

    out = save_fold_summary_csv(reports, tag)
    printed = capsys.readouterr().out

    assert out is None
    assert "failed to write" in printed


def test_save_fold_summary_csv_header_missing_label_uses_col0_for_label(tmp_path):
    # Layout
    reports = tmp_path / "reports"
    history = reports.parent / "history"
    artifacts = reports.parent / "artifacts"
    for d in (reports, history, artifacts):
        d.mkdir(parents=True, exist_ok=True)
    tag = "ecg_ECGResNet_lr001_bs8_wd0"

    # Two folds so we pass the "<2 folds" guard
    (history / f"history_{tag}_fold1.json").write_text(
        '{"best_epoch":0,"val_acc":[0.8],"val_loss":[0.3]}'
    )
    (history / f"history_{tag}_fold2.json").write_text(
        '{"best_epoch":0,"val_acc":[0.9],"val_loss":[0.2]}'
    )

    # CSV header *without* "label" → _col("label") returns None,
    # code falls back to label_i = 0. Put "macro avg" in col 0.
    (artifacts / f"classification_report_{tag}_fold1.csv").write_text(
        "L,precision,recall,f1-score,support\n"
        "CD,0,0,0,0\n"
        "macro avg,0.1,0.1,0.55,2\n"
    )
    # Second fold can be minimal/ignored by macro-f1 assertion
    (artifacts / f"classification_report_{tag}_fold2.csv").write_text(
        "label,precision,recall,f1-score,support\n" "macro avg,0.1,0.1,0.65,2\n"
    )

    out_csv = save_fold_summary_csv(reports, tag)
    content = out_csv.read_text().splitlines()
    row1 = next(r for r in content if r.startswith("1,"))
    # macro_f1 is column 5 (index 4)
    assert row1.split(",")[4] == "0.55"


def test_save_fold_summary_csv_header_missing_f1score_uses_index3(tmp_path):
    reports = tmp_path / "reports"
    history = reports.parent / "history"
    artifacts = reports.parent / "artifacts"
    for d in (reports, history, artifacts):
        d.mkdir(parents=True, exist_ok=True)
    tag = "ecg_ECGResNet_lr001_bs8_wd0"

    (history / f"history_{tag}_fold1.json").write_text(
        '{"best_epoch":0,"val_acc":[0.8],"val_loss":[0.3]}'
    )
    (history / f"history_{tag}_fold2.json").write_text(
        '{"best_epoch":0,"val_acc":[0.9],"val_loss":[0.2]}'
    )

    # Header has at least 4 columns but *no* "f1-score".
    # The code sets f1_i = 3, so put macro f1 value at column index 3.
    (artifacts / f"classification_report_{tag}_fold1.csv").write_text(
        "label,precision,recall,f1,S\n" "CD,0,0,0,0\n" "macro avg,0.1,0.1,0.77,2\n"
    )
    (artifacts / f"classification_report_{tag}_fold2.csv").write_text(
        "label,precision,recall,f1,S\n" "macro avg,0.1,0.1,0.22,2\n"
    )

    out_csv = save_fold_summary_csv(reports, tag)
    content = out_csv.read_text().splitlines()
    row1 = next(r for r in content if r.startswith("1,"))
    assert row1.split(",")[4] == "0.77"  # macro_f1 extracted via fallback f1_i=3


def test_save_fold_summary_csv_skips_too_short_rows_then_reads_macro(tmp_path):
    reports = tmp_path / "reports"
    history = reports.parent / "history"
    artifacts = reports.parent / "artifacts"
    for d in (reports, history, artifacts):
        d.mkdir(parents=True, exist_ok=True)
    tag = "ecg_ECGResNet_lr001_bs8_wd0"

    (history / f"history_{tag}_fold1.json").write_text(
        '{"best_epoch":0,"val_acc":[0.8],"val_loss":[0.3]}'
    )
    (history / f"history_{tag}_fold2.json").write_text(
        '{"best_epoch":0,"val_acc":[0.9],"val_loss":[0.2]}'
    )

    # First data row is too short (len(row) <= max(label_i,f1_i)) triggers 'continue'.
    # Second row is valid macro avg and will be parsed.
    (artifacts / f"classification_report_{tag}_fold1.csv").write_text(
        "label,precision,recall,f1-score\n"
        "macro avg,0.5\n"  # too short -> skipped
        "macro avg,0.1,0.1,0.42\n"  # valid -> used
    )
    (artifacts / f"classification_report_{tag}_fold2.csv").write_text(
        "label,precision,recall,f1-score\n" "macro avg,0.1,0.1,0.11\n"
    )

    out_csv = save_fold_summary_csv(reports, tag)
    content = out_csv.read_text().splitlines()
    row1 = next(r for r in content if r.startswith("1,"))
    assert row1.split(",")[4] == "0.42"  # ensured the short row was skipped


def test_save_fold_summary_csv_no_f1_column_skips_macro(tmp_path):
    # hits: header too short -> f1_i stays None -> loop continues with no macro_f1
    tag = "t2"
    reports = tmp_path / "outputs" / "reports"
    artifacts = tmp_path / "outputs" / "artifacts"
    history = tmp_path / "outputs" / "history"
    artifacts.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    # Two folds
    (history / f"history_{tag}_fold1.json").write_text(
        json.dumps({"val_acc": [0.5], "val_loss": [0.4], "best_epoch": 0})
    )
    (history / f"history_{tag}_fold2.json").write_text(
        json.dumps({"val_acc": [0.55], "val_loss": [0.45], "best_epoch": 0})
    )

    # Fold 1 CSV with header <4 cols -> f1_i remains None; no macro_f1 parsed
    with (artifacts / f"classification_report_{tag}_fold1.csv").open(
        "w", newline=""
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b"])  # too short
        w.writerow(["macro avg", "x"])

    out = save_fold_summary_csv(reports, tag)
    assert out is not None and out.exists()
    txt = out.read_text()
    # file is written; macro_f1 column exists but may be empty for the affected fold
    assert "mean_val_acc" in txt and "mean_val_loss" in txt


def test_save_fold_summary_csv_rows_csv_empty(tmp_path):
    tag = "t"
    reports = tmp_path / "outputs" / "reports"
    artifacts = tmp_path / "outputs" / "artifacts"
    history = tmp_path / "outputs" / "history"
    artifacts.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    (history / f"history_{tag}_fold1.json").write_text(
        json.dumps({"val_acc": [0.7], "val_loss": [0.5], "best_epoch": 0})
    )
    (history / f"history_{tag}_fold2.json").write_text(
        json.dumps({"val_acc": [0.6], "val_loss": [0.6], "best_epoch": 0})
    )

    # Empty file => rows_csv == [] => hits skip inner parsing
    (artifacts / f"classification_report_{tag}_fold1.csv").write_text("")

    out = save_fold_summary_csv(reports, tag)
    assert out is not None and out.exists()
