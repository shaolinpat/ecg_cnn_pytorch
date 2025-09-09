# ecg_cnn/utils/plot_utils.py

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import shap
import statistics
import torch

from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)
from typing import Any, List, Optional, Tuple, Union

from ecg_cnn.utils.validate import (
    validate_hparams_formatting,
    validate_y_probs,
    validate_y_true_pred,
)


# ------------------------------------------------------------------------------
# Toggle how many plots get generated
# ------------------------------------------------------------------------------
_ENV_ENABLE = os.getenv("ECG_PLOTS_ENABLE_OVR")  # None => default ON
ENV_ENABLE_OVR = (
    True
    if _ENV_ENABLE is None
    else _ENV_ENABLE.strip().lower() in {"1", "true", "yes", "on"}
)

_OVR_CLASSES_ENV = os.getenv("ECG_PLOTS_OVR_CLASSES", "").strip()
ENV_OVR_CLASSES = {c.strip() for c in _OVR_CLASSES_ENV.split(",") if c.strip()} or None
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Function to standardize plot titles
# ------------------------------------------------------------------------------


def _build_plot_title(
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    metric: str | None = None,
    fold: int | None = None,
    epoch: int | None = None,
) -> str:
    """
    Construct a standardized plot title including key hyperparameters.

    Format:
        "<model>: <metric>\nLR=<lr>, BS=<bs>, WD=<wd>, Fold=<fold>, Epoch=<epoch>"

    Parameters
    ----------
    model : str
        Model string indicates model used (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    prefix : str
        String prefix for context (e.g., "final", "best"). Must be non-empty.
    fold : int or None, optional
        Fold number used in cross-validation. Must be a non-negative integer or None.
    epochs : int or None, optional
        Number of training epochs. Included in title if provided.
    metric : str or None, optional
        Metric name (e.g., "loss", "accuracy") to include in title.
        Defaults to "Metric" if not provided.

    Returns
    -------
    str
        A formatted title string for use in plots.

    Raises
    ------
    ValueError
        If any input fails validation checks.
    """
    if metric is None:
        metric = "Metric"

    validate_hparams_formatting(
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric=metric,
        fold=fold,
        epoch=epoch,
    )

    title = f"{model}: {metric}"

    plot_title = (
        f"{title}\nLR={lr}, BS={bs}, WD={wd}"
        + (f", Fold={fold}" if fold is not None else "")
        + (f", Epoch={epoch}" if epoch is not None else "")
    )
    return plot_title


# ------------------------------------------------------------------------------
# Helper function to standardize filenames for plots (used by save_plot_curves,
# save_confusion_matrix, etc.)
# ------------------------------------------------------------------------------
def format_hparams(
    *,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str = "",
    fold: int | None = None,
    epoch: int | None = None,
) -> str:
    """
    Construct a standardized filename string for saved plots or models.

    Format:
        "<prefix>_<metric>_lr<lr_str>_bs<bs>_wd<wd_str>_fold<fold>_epoch<epoch>"

    If fname_metric is omitted, the "_<metric>" part is skipped.

    Parameters
    ----------
    model : str
        Model string indicates model used (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    prefix : str
        Prefix string indicating model phase or purpose (e.g., "final", "best").
    fname_metric : str, optional
        Metric name (e.g., "loss", "accuracy") to include in filename.
    fold : int or None, optional
        Fold number used in cross-validation. Must be a non-negative integer or None.
    epoch : int
        Epoch that generated the result. Must be in range [1, 1000].

    Returns
    -------
    str
        A formatted string of the form:
            "<prefix>_<metric>_lr<lr>_bs<bs>_wd<wd>_fold<fold>_epoch<epoch>"

    Notes
    -----
    - input validation done by validate_hparams_formatting(lr, bs, wd, prefix,fname_metric, fold, epoch)
    - Floats are truncated to at most 6 decimal places and then stripped of leading "0." and trailing zeros.
    - Floats smaller than 1e-6 are disallowed to avoid loss of precision or misleading filenames.
    """
    # will raise an error if the params are out of spec
    validate_hparams_formatting(
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric=fname_metric,
        fold=fold,
        epoch=epoch,
    )

    # --- Helper function for trimming floats ---
    def _trim(val: float) -> str:
        raw = f"{val:.6f}"
        trimmed = raw[2:] if raw.startswith("0.") else raw
        trimmed = trimmed.rstrip("0").rstrip(".")
        return trimmed or "0"

    # --- Build filename ---
    parts = [prefix.lower()]
    if fname_metric:
        parts.append(fname_metric.lower())
    if model:
        parts.append(model)
    parts.append(f"lr{_trim(lr)}")
    parts.append(f"bs{bs}")
    parts.append(f"wd{_trim(wd)}")
    if fold is not None:
        parts.append(f"fold{fold}")
    if epoch is not None:
        parts.append(f"epoch{epoch}")

    return "_".join(parts)


def save_plot_curves(
    x_vals: list[float],
    y_vals: list[float],
    x_label: str,
    y_label: str,
    title_metric: str,
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str,
    fold: int | None = None,
    epoch: int | None = None,
):
    """
    Draws and saves a line plot comparing training and validation curves
    (e.g., accuracy or loss over epochs), using consistent filename formatting.

    Parameters
    ----------
    x_vals : list of floats
        Training metric values per epoch.
    y_vals : list of floats
        Validation metric values per epoch.
    x_label : str
        Label for the x-axis (e.g., "Epoch").
    y_label : str
        Label for the y-axis (e.g., "Accuracy", "Loss").
    title_metric : str
        Human-readable title (e.g., "Model Accuracy").
    out_folder : str or Path
        Folder to save output files.
    model : str
        Model string indicates model used (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    prefix : str
        Prefix string indicating model phase or purpose (e.g., "final", "best").
    fname_metric : str, optional
        Metric name (e.g., "loss", "accuracy") to include in filename.
    fold : int or None, optional
        Fold number used in cross-validation. Must be a non-negative integer or None.
    epoch : int or None, optional
        Epoch number used for this plot (e.g., best epoch).

    Raises
    ------
    ValueError
        If any input is malformed or inconsistent in type or length.
    """
    # Validate curve data
    for name, arr in [("x_vals", x_vals), ("y_vals", y_vals)]:
        if not hasattr(arr, "__len__"):
            raise ValueError(f"{name} must be array-like.")
        if not all(isinstance(x, (int, float, np.integer, np.floating)) for x in arr):
            raise ValueError(f"{name} must contain only numeric values.")

    if len(x_vals) != len(y_vals):
        raise ValueError(
            f"x_vals and y_vals must have the same length, got {len(x_vals)} and {len(y_vals)}."
        )

    # Normalize to list in case caller passed in NumPy arrays
    x_vals = list(x_vals)
    y_vals = list(y_vals)

    # Validate axis labels and title
    for name, val in [
        ("x_label", x_label),
        ("y_label", y_label),
        ("title_metric", title_metric),
        ("prefix", prefix),
    ]:
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    # Validate output folder
    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")

    # Build filename and title
    filename = (
        format_hparams(
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric=fname_metric,
            fold=fold,
            epoch=epoch,
        )
        + ".png"
    )

    # Normalize path
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / filename

    title = _build_plot_title(
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        metric=title_metric,
        fold=fold,
        epoch=epoch,
    )

    # Draw and save
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, label="Train")
    plt.plot(y_vals, label="Val")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(out_path)
    plt.close()
    print(f"Saved {prefix.lower()} {fname_metric.lower()} plot to {out_path}")


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str,
    normalize: bool = True,
    fold: int | None = None,
    epoch: int | None = None,
):
    """
    Builds a confusion matrix (optionally normalized), displays it with labels,
    and saves it to a standardized filename.

    Parameters
    ----------
    y_true : list of int
        True class labels for evaluation.
    y_pred : list of int
        Predicted class labels from the model.
    class_names : list of str
        List of class labels (e.g., ["NORM", "MI", "STTC", ...]).
    out_folder : str or Path
        Folder to save output files.
    model : str
        Model string indicates model used (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    prefix : str
        Prefix string indicating model phase or purpose (e.g., "final", "best").
    fname_metric : str, optional
        Metric name (e.g., "loss", "accuracy") to include in filename.
    normalize : bool, optional
        Whether to normalize the confusion matrix (default is True).
    fold : int or None, optional
        Fold number used in cross-validation. Must be a non-negative integer or None.
    epoch : int or None, optional
        Epoch number used for this plot (e.g., best epoch).

    Raises
    ------
    ValueError
        If any input is malformed or invalid in type or structure.
    """

    # --- Validate labels ---
    validate_y_true_pred(y_true, y_pred)

    indices_used = set(y_true) | set(y_pred)
    if any(i < 0 or i >= len(class_names) for i in indices_used):
        raise ValueError(
            "class_names must cover all class indices in y_true and y_pred."
        )

    if not isinstance(class_names, list) or not all(
        isinstance(s, str) for s in class_names
    ):
        raise ValueError("class_names must be a list of strings.")

    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")

    # Construct filename using format_hparams (which validates the parameters)
    filename = (
        format_hparams(
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric=fname_metric,
            fold=fold,
            epoch=epoch,
        )
        + ".png"
    )

    # Normalize path
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / filename

    # --- Title block ---
    title_text = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    plot_title = _build_plot_title(
        model, lr, bs, wd, prefix, metric=title_text, fold=fold, epoch=epoch
    )

    # Compute confusion matrix
    cm = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize="true" if normalize else None,
        cmap="Blues",
        values_format=".2f" if normalize else "d",
    )

    cm.ax_.set_title(plot_title)

    cm.figure_.savefig(out_path)
    cm.figure_.clf()
    print(f"Saved {prefix.lower()} {fname_metric.lower()} plot to {out_path}")


def save_pr_threshold_curve(
    y_true: list[int] | np.ndarray,
    y_probs: list[float] | np.ndarray,
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    epoch: int,
    prefix: str,
    fname_metric: str,
    title: str = "Precision & Recall vs Threshold",
    fold: int | None = None,
):
    """
    Plot and save a Precision-Recall vs Threshold curve using standardized filename and validated inputs.

    Parameters
    ----------
    y_true : list of int or np.ndarray
        Binary ground truth labels (0 or 1).
    y_probs : list of float or np.ndarray
        Predicted probabilities for the positive class.
    out_folder : str or Path
        Folder to save output files.
    model : str
        Model string indicates model used (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    prefix : str
        Prefix string indicating model phase or purpose (e.g., "final", "best").
    fname_metric : str, optional
        Metric name (e.g., "loss", "accuracy") to include in filename.
    title : str, optional
        Plot title. Default is "Precision & Recall vs Threshold".
    fold : int | None, optional
        Fold number in cross-validation. Must be non-negative or None.
    epoch : int or None, optional
        Epoch number used for this plot (e.g., best epoch).

    Raises
    ------
    ValueError
        If input types or shapes are invalid.
    """

    # --- Validate inputs ---
    validate_y_true_pred(
        y_true, y_true
    )  # second copy is a dummy; only length/format used
    validate_y_probs(y_probs)

    if len(y_true) != len(y_probs):
        raise ValueError(
            f"y_true and y_probs must be the same length, got {len(y_true)} and {len(y_probs)}."
        )

    if not isinstance(out_folder, (str, Path)):
        raise ValueError(
            f"out_folder must be a string or pathlib.Path, got {type(out_folder)}"
        )

    if not isinstance(title, str):
        raise ValueError(f"title must be a string, got {type(title)}")

    # Construct filename using format_hparams (which validates the parameters)
    filename = (
        format_hparams(
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric=fname_metric,
            fold=fold,
            epoch=epoch,
        )
        + ".png"
    )

    # Normalize path
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / filename

    # Prepare Title
    plot_title = _build_plot_title(
        model, lr, bs, wd, prefix, metric=title, fold=fold, epoch=epoch
    )

    # --- Compute curve ---
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    thresholds = np.append(thresholds, 1.0)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision, label="Precision")
    ax.plot(thresholds, recall, label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(plot_title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    fig.savefig(out_path)
    plt.close(fig)

    print(f"Saved PR curve {prefix.lower()} {fname_metric.lower()} plot to {out_path}")


def save_pr_curve(
    y_true: list[int] | np.ndarray,
    y_probs: list[float] | np.ndarray,
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str,
    title: str = "Precision-Recall Curve",
    fold: int | None = None,
    epoch: int | None = None,
):
    """
    Plot and save a standard Precision-Recall curve with AUPRC in the legend/title.
    Accepts binary (1D y_probs) or one-vs-rest multiclass (2D y_probs with shape [n, n_classes]).
    """
    # --- Validate ---
    validate_y_true_pred(y_true, y_true)  # length/format only
    validate_y_probs(y_probs)
    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")
    if not isinstance(title, str):
        raise ValueError("title must be a string.")

    # Normalize path
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    y_probs = np.asarray(y_probs)
    y_true = np.asarray(y_true)

    def _one_curve(y_true_bin, y_probs_bin, suffix: str):
        pr, rc, _ = precision_recall_curve(y_true_bin, y_probs_bin)
        ap = average_precision_score(y_true_bin, y_probs_bin)

        suffix_text = f" ({suffix})" if suffix else ""
        plot_title = _build_plot_title(
            model,
            lr,
            bs,
            wd,
            prefix,
            metric=f"{title}{suffix_text}",
            fold=fold,
            epoch=epoch,
        )
        filename = (
            format_hparams(
                model=model,
                lr=lr,
                bs=bs,
                wd=wd,
                prefix=prefix if suffix == "" else f"{prefix}_{suffix}",
                fname_metric=fname_metric,
                fold=fold,
                epoch=epoch,
            )
            + ".png"
        )
        out_path = out_folder / filename
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rc, pr, label=f"AUPRC={ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(plot_title)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved PR curve to {out_path}")

    # Binary
    if y_probs.ndim == 1:
        _one_curve(y_true, y_probs, suffix="")
        return

    # Multiclass one-vs-rest
    if y_probs.ndim == 2 and y_probs.shape[0] == y_true.shape[0]:
        n_classes = y_probs.shape[1]
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            _one_curve(y_true_bin, y_probs[:, i], suffix=f"ovr_class{i}")
        return

    print("Skipping PR curve — y_probs shape does not match expected format.")


def save_roc_curve(
    y_true: list[int] | np.ndarray,
    y_probs: list[float] | np.ndarray,
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str,
    title: str = "ROC Curve",
    fold: int | None = None,
    epoch: int | None = None,
):
    """
    Plot and save an ROC curve with AUROC in the legend/title.
    Accepts binary (1D y_probs) or one-vs-rest multiclass (2D y_probs with shape [n, n_classes]).
    """
    # --- Validate ---
    validate_y_true_pred(y_true, y_true)  # length/format only
    validate_y_probs(y_probs)
    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")
    if not isinstance(title, str):
        raise ValueError("title must be a string.")

    # Normalize path
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    y_probs = np.asarray(y_probs)
    y_true = np.asarray(y_true)

    def _one_curve(y_true_bin, y_probs_bin, suffix: str):
        fpr, tpr, _ = roc_curve(y_true_bin, y_probs_bin)
        auc_val = roc_auc_score(y_true_bin, y_probs_bin)

        suffix_text = f" ({suffix})" if suffix else ""
        plot_title = _build_plot_title(
            model,
            lr,
            bs,
            wd,
            prefix,
            metric=f"{title}{suffix_text}",
            fold=fold,
            epoch=epoch,
        )
        filename = (
            format_hparams(
                model=model,
                lr=lr,
                bs=bs,
                wd=wd,
                prefix=prefix if suffix == "" else f"{prefix}_{suffix}",
                fname_metric=fname_metric,
                fold=fold,
                epoch=epoch,
            )
            + ".png"
        )
        out_path = out_folder / filename
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fpr, tpr, label=f"AUROC={auc_val:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(plot_title)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved ROC curve to {out_path}")

    # Binary
    if y_probs.ndim == 1:
        _one_curve(y_true, y_probs, suffix="")
        return

    # Multiclass one-vs-rest
    if y_probs.ndim == 2 and y_probs.shape[0] == y_true.shape[0]:
        n_classes = y_probs.shape[1]
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            _one_curve(y_true_bin, y_probs[:, i], suffix=f"ovr_class{i}")
        return

    print("Skipping ROC curve — y_probs shape does not match expected format.")


def save_classification_report(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str,
    fold: int | None = None,
    epoch: int | None = None,
    title: str = "Classification Report",
):
    """
    Save classification report as a text file and heatmap image, using a standardized filename format.

    Parameters
    ----------
    y_true : list[int] or np.ndarray
        True class labels.
    y_pred : list[int] or np.ndarray
        Predicted class labels.
    class_names : list of str
        List of class labels (e.g., ["NORM", "MI", "STTC", ...]).
    out_folder : str or Path
        Folder to save output files.
    model : str
        Model string indicates model used (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    prefix : str
        Prefix string indicating model phase or purpose (e.g., "final", "best").
    fname_metric : str, optional
        Metric name (e.g., "loss", "accuracy") to include in filename.
    fold : int or None, optional
        Fold number used in cross-validation. Must be a non-negative integer or None.
    epoch : int or None, optional
        Epoch number used for this plot (e.g., best epoch).
    title : str, optional
        Plot title. Default is "Classification Report".

    Raises
    ------
    ValueError
        If input types, lengths, or formats are invalid.
    """

    # --- Validate labels and inputs ---
    validate_y_true_pred(y_true, y_pred)

    if not isinstance(class_names, list) or not all(
        isinstance(s, str) for s in class_names
    ):
        raise ValueError("class_names must be a list of strings.")

    if not isinstance(out_folder, (str, Path)):
        raise ValueError(
            f"out_folder must be a string or pathlib.Path, got {type(out_folder)}"
        )

    if not isinstance(title, str):
        raise ValueError(f"title must be a string, got {type(title)}")

    # Construct filename using format_hparams (which validates the parameters)
    filename = (
        format_hparams(
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric=fname_metric,
            fold=fold,
            epoch=epoch,
        )
        + ".png"
    )

    # Normalize path
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / filename

    # Prepare Title
    plot_title = _build_plot_title(
        model, lr, bs, wd, prefix, metric=title, fold=fold, epoch=epoch
    )

    # Generate classification report dict
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    #  Save heatmap
    df = pd.DataFrame(report_dict).transpose()
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title(plot_title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Saved classification report heatmap to {out_path}")
    print(f"Saved {prefix.lower()} {fname_metric.lower()} plot to {out_path}")


def save_calibration_curve(
    y_true: list[int] | np.ndarray,
    y_probs: list[float] | np.ndarray,
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str = "calibration",
    n_bins: int = 10,
    fold: int | None = None,
    epoch: int | None = None,
    title: str = "Calibration Curve",
):
    """
    Plot and save a reliability diagram (calibration curve) and report the Brier score.

    The calibration plot compares predicted probabilities to empirical accuracies.
    For multiclass inputs, this function reduces to "confidence vs correctness":
    the model's maximum class probability per sample against whether argmax == y_true.

    Parameters
    ----------
    y_true : list[int] or np.ndarray
        Ground-truth labels. For binary, values must be 0 or 1. For multiclass, values
        must be integers in the range [0, K-1].
    y_probs : list[float] or np.ndarray
        Predicted probabilities. For binary, shape (N,). For multiclass, shape (N, K).
        Values must lie in the closed interval [0.0, 1.0].
    out_folder : str or Path
        Folder to save the output PNG.
    model : str
        Model identifier used in standardized titles/filenames (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must satisfy the same constraints as validate_hparams_formatting.
    bs : int
        Batch size. Must satisfy the same constraints as validate_hparams_formatting.
    wd : float
        Weight decay. Must satisfy the same constraints as validate_hparams_formatting.
    prefix : str
        Filename prefix (e.g., "final", "best"). Must be non-empty.
    fname_metric : str, optional
        Metric component for the filename (default "calibration").
    n_bins : int, optional
        Number of bins for the calibration curve. Must be >= 2. Default is 10.
    fold : int or None, optional
        Fold index for cross-validation, if applicable. Non-negative integer or None.
    epoch : int or None, optional
        Epoch number for context in filename/title, if applicable.
    title : str, optional
        Human-readable title prefix for the plot. Default "Calibration Curve".

    Returns
    -------
    (Path, float)
        A tuple of (png_path, brier_score), where png_path is the path to the saved
        figure and brier_score is the scalar Brier score computed on the reduced
        targets and probabilities.

    Raises
    ------
    ValueError
        If inputs fail validation (types, shapes, ranges) or if out_folder is not a
        string or Path.
    Notes
    -----
    - Uses validate_y_true_pred and validate_y_probs for input checks.
    - Filenames and titles are produced via format_hparams and _build_plot_title to
      match repository conventions.
    """

    # Basic validation
    validate_y_true_pred(y_true, y_true)
    validate_y_probs(y_probs)

    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")

    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2.")

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    # Reduce to targets/probs for calibration
    if y_probs.ndim == 2 and y_probs.shape[1] > 1:
        y_pred = y_probs.argmax(axis=1)
        conf = y_probs.max(axis=1)
        targets = (y_pred == y_true).astype(int)
        title_suffix = " (multiclass confidence)"
        probs_for_cal = conf
    else:
        targets = y_true.astype(int)
        probs_for_cal = y_probs.astype(float)
        title_suffix = ""

    prob_true, prob_pred = calibration_curve(
        targets, probs_for_cal, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(targets, probs_for_cal)

    filename = (
        format_hparams(
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric=fname_metric,
            fold=fold,
            epoch=epoch,
        )
        + ".png"
    )

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / filename

    plot_title = _build_plot_title(
        model,
        lr,
        bs,
        wd,
        prefix,
        metric=f"{title}{title_suffix}",
        fold=fold,
        epoch=epoch,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(prob_pred, prob_true, marker="o", label=f"Brier={brier:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(plot_title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved calibration curve to {out_path}")
    return out_path, brier


def save_threshold_sweep_table(
    y_true: list[int] | np.ndarray,
    y_probs: list[float] | np.ndarray,
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str = "threshold_sweep",
    thresholds: list[float] | None = None,
    average: str = "binary",
    fold: int | None = None,
    epoch: int | None = None,
):
    """
    Compute and save a CSV of precision/recall/F1/accuracy across probability thresholds.

    For binary inputs, uses the provided positive-class probabilities directly.
    For multiclass inputs, two modes are supported:
      - average="binary": one-vs-rest using class 1 as the positive class
      - average="macro" : macro-averaged metrics using argmax predictions

    Parameters
    ----------
    y_true : list[int] or np.ndarray
        Ground-truth labels (binary or multiclass).
    y_probs : list[float] or np.ndarray
        Predicted probabilities. Binary: shape (N,). Multiclass: shape (N, K).
    out_folder : str or Path
        Folder to save the output CSV.
    model : str
        Model identifier used in standardized filenames.
    lr : float
        Learning rate. Must satisfy validate_hparams_formatting constraints.
    bs : int
        Batch size. Must satisfy validate_hparams_formatting constraints.
    wd : float
        Weight decay. Must satisfy validate_hparams_formatting constraints.
    prefix : str
        Filename prefix (e.g., "final", "best"). Must be non-empty.
    fname_metric : str, optional
        Metric component for the filename (default "threshold_sweep").
    thresholds : list[float] or None, optional
        Threshold grid in [0.0, 1.0]. If None, uses 0.00..1.00 step 0.01.
    average : str, optional
        "binary" or "macro". See description above. Default "binary".
    fold : int or None, optional
        Fold index for cross-validation, if applicable.
    epoch : int or None, optional
        Epoch number for context in filename/title, if applicable.

    Returns
    -------
    Path
        Path to the saved CSV file with columns:
        ["threshold", "precision", "recall", "f1", "accuracy"].

    Raises
    ------
    ValueError
        If inputs fail validation, average is unsupported, out_folder is invalid,
        or thresholds contain values outside [0.0, 1.0].

    Notes
    -----
    - Pair this CSV with your PR-threshold plot to justify the operating threshold.
    - Filenames use format_hparams for consistency with other artifacts.
    """

    # Validation
    validate_y_true_pred(y_true, y_true)
    validate_y_probs(y_probs)

    if average not in {"binary", "macro"}:
        raise ValueError('average must be "binary" or "macro".')

    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    # Threshold grid
    if thresholds is None:
        thresholds = [i / 100.0 for i in range(0, 101)]
    else:
        if not hasattr(thresholds, "__iter__"):
            raise ValueError("thresholds must be an iterable of floats in [0.0, 1.0].")
        thr_clean = []
        for t in thresholds:
            if not isinstance(t, (float, np.floating, int, np.integer)):
                raise ValueError("All thresholds must be numeric values in [0.0, 1.0].")
            t = float(t)
            if not (0.0 <= t <= 1.0):
                raise ValueError("All thresholds must be in [0.0, 1.0].")
            thr_clean.append(t)
        thresholds = thr_clean

    # Detect binary vs multiclass
    unique_labels = np.unique(y_true)
    is_binary_labels = unique_labels.size == 2
    is_multiclass_probs = y_probs.ndim == 2 and y_probs.shape[1] > 1

    # If binary labels but probs are (n,2), use positive column
    if is_binary_labels and y_probs.ndim == 2 and y_probs.shape[1] == 2:
        y_probs = y_probs[:, 1]

    rows: list[dict] = []

    for t in thresholds:
        if (not is_multiclass_probs) or (average == "binary"):
            if is_multiclass_probs and not is_binary_labels:
                # OvR: treat class 1 as positive
                y_true_bin = (y_true == 1).astype(int)
                y_probs_pos = y_probs[:, 1]
                y_pred_bin = (y_probs_pos >= t).astype(int)
            else:
                y_true_bin = y_true.astype(int)
                if y_probs.ndim != 1:
                    raise ValueError("Binary path expects 1-D y_probs.")
                y_pred_bin = (y_probs >= t).astype(int)

            p = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            r = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            a = accuracy_score(y_true_bin, y_pred_bin)

        else:
            # Multiclass macro
            y_pred_cls = y_probs.argmax(axis=1)
            p = precision_score(y_true, y_pred_cls, average="macro", zero_division=0)
            r = recall_score(y_true, y_pred_cls, average="macro", zero_division=0)
            f = f1_score(y_true, y_pred_cls, average="macro", zero_division=0)
            a = accuracy_score(y_true, y_pred_cls)

        rows.append(
            {"threshold": t, "precision": p, "recall": r, "f1": f, "accuracy": a}
        )

    df = pd.DataFrame(rows)

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    filename = (
        format_hparams(
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric=fname_metric,
            fold=fold,
            epoch=epoch,
        )
        + ".csv"
    )
    out_path = out_folder / filename
    df.to_csv(out_path, index=False)
    print(f"Saved threshold sweep table to {out_path}")
    return out_path


def save_error_tables(
    y_true: list[int] | np.ndarray,
    y_probs: list[float] | np.ndarray,
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    threshold: float = 0.5,
    top_k: int = 50,
    fold: int | None = None,
    epoch: int | None = None,
):
    """
    Save CSVs for top-K false negatives and top-K false positives at a given threshold.

    Each CSV contains the sample index, the true label, the predicted positive-class
    probability, and the absolute margin from the threshold. This is useful for fast
    error triage and inspecting borderline cases.

    Parameters
    ----------
    y_true : list[int] or np.ndarray
        Ground-truth labels for a binary task (0 or 1).
    y_probs : list[float] or np.ndarray
        Predicted positive-class probabilities, shape (N,). Values must be in [0.0, 1.0].
    out_folder : str or Path
        Folder to save the output CSV files.
    model : str
        Model identifier used in standardized filenames.
    lr : float
        Learning rate. Must satisfy validate_hparams_formatting constraints.
    bs : int
        Batch size. Must satisfy validate_hparams_formatting constraints.
    wd : float
        Weight decay. Must satisfy validate_hparams_formatting constraints.
    prefix : str
        Filename prefix (e.g., "final", "best"). Must be non-empty.
    threshold : float, optional
        Decision threshold in [0.0, 1.0]. Default is 0.5.
    top_k : int, optional
        Number of examples to keep in each table (FN/FP). Default is 50.
    fold : int or None, optional
        Fold index for cross-validation, if applicable.
    epoch : int or None, optional
        Epoch number for context in filename/title, if applicable.

    Returns
    -------
    (Path | None, Path | None)
        Tuple of (false_negative_csv_path, false_positive_csv_path). Paths may be None
        if the function is skipped due to invalid input dimensionality.

    Raises
    ------
    ValueError
        If inputs fail validation, out_folder is invalid, threshold/top_k are out of
        range, or y_probs is not 1-D for a binary task.

    Notes
    -----
    - This function is intended for binary classification only. For multiclass, run
      one-vs-rest externally and pass the binary arrays to this function.
    - Filenames are standardized via format_hparams with a "errors" metric suffix and
      additional postfixes "_topfn.csv" and "_topfp.csv".
    """

    # Validation
    validate_y_true_pred(y_true, y_true)
    validate_y_probs(y_probs)

    if not (isinstance(threshold, (float, int)) and 0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be a numeric value in [0.0, 1.0].")
    if not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")

    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    # Require binary labels
    unique_labels = np.unique(y_true)
    if unique_labels.size != 2:
        print("Skipping error tables — expected binary labels.")
        return (None, None)

    # If binary but probs are (n,2), take positive column
    if y_probs.ndim == 2 and y_probs.shape[1] == 2:
        y_probs = y_probs[:, 1]

    if y_probs.ndim != 1:
        print("Skipping error tables — expected 1D probabilities for binary.")
        return (None, None)

    thr = float(threshold)
    preds = (y_probs >= thr).astype(int)
    margins = np.abs(y_probs - thr)

    fn_mask = (y_true == 1) & (preds == 0)
    fp_mask = (y_true == 0) & (preds == 1)

    fn_idx = np.where(fn_mask)[0]
    fp_idx = np.where(fp_mask)[0]

    df_fn = (
        pd.DataFrame(
            {
                "index": fn_idx,
                "y_true": y_true[fn_idx],
                "y_prob_pos": y_probs[fn_idx],
                "margin": margins[fn_idx],
            }
        )
        .sort_values("margin")  # smallest margin first (closest to threshold)
        .head(top_k)
    )

    df_fp = (
        pd.DataFrame(
            {
                "index": fp_idx,
                "y_true": y_true[fp_idx],
                "y_prob_pos": y_probs[fp_idx],
                "margin": margins[fp_idx],
            }
        )
        .sort_values("margin")
        .head(top_k)
    )

    base = format_hparams(
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric="errors",
        fold=fold,
        epoch=epoch,
    )

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    fn_path = out_folder / f"{base}_topfn.csv"
    fp_path = out_folder / f"{base}_topfp.csv"
    df_fn.to_csv(fn_path, index=False)
    df_fp.to_csv(fp_path, index=False)

    print(f"Saved error tables to {fn_path} and {fp_path}")
    return (fn_path, fp_path)


def save_confidence_histogram(
    y_true: list[int] | np.ndarray,
    y_probs: list[float] | np.ndarray,
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str = "confidence_hist",
    bins: int = 20,
    fold: int | None = None,
    epoch: int | None = None,
    title: str = "Prediction Confidence Histogram",
):
    """
    Plot and save a histogram of predicted probabilities (confidence scores).

    Parameters
    ----------
    y_true : list[int] or np.ndarray
        Ground-truth binary labels (0 or 1).
    y_probs : list[float] or np.ndarray
        Predicted positive-class probabilities, shape (N,). Values in [0.0, 1.0].
    out_folder : str or Path
        Directory to save the plot.
    model : str
        Model identifier for standardized filenames.
    lr : float
        Learning rate (for filename metadata).
    bs : int
        Batch size (for filename metadata).
    wd : float
        Weight decay (for filename metadata).
    prefix : str
        Filename prefix (e.g., "final", "best").
    fname_metric : str, optional
        Metric name for filename suffix (default "confidence_hist").
    bins : int, optional
        Number of histogram bins (default 20).
    fold : int or None, optional
        Fold index for CV runs.
    epoch : int or None, optional
        Epoch number.
    title : str, optional
        Plot title.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """

    # Validation
    validate_y_true_pred(y_true, y_true)
    validate_y_probs(y_probs)

    y_probs = np.asarray(y_probs)

    if y_probs.ndim != 1:
        print("Skipping confidence histogram — expected 1D probabilities for binary.")
        return None

    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or Path.")

    # Make histogram
    plt.figure(figsize=(6, 4))
    plt.hist(
        y_probs,
        bins=bins,
        range=(0.0, 1.0),
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    plt.xlabel("Predicted Probability (Positive Class)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()

    # Filename
    filename = (
        format_hparams(
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric=fname_metric,
            fold=fold,
            epoch=epoch,
        )
        + ".png"
    )

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    out_path = out_folder / filename
    plt.savefig(out_path)
    plt.close()

    print(f"Saved confidence histogram to {out_path}")
    return out_path


def save_confidence_histogram_split(
    y_true: list[int] | np.ndarray,
    y_probs: list[float] | np.ndarray,
    out_folder: str | Path,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    threshold: float = 0.5,
    bins: int = 20,
    fold: int | None = None,
    epoch: int | None = None,
    fname_metric: str = "confidence_hist_split",
    title: str = "Confidence Histogram (Correct vs Incorrect)",
):
    """
    Plot and save a histogram of predicted probabilities split by correctness
    at a given decision threshold.

    Parameters
    ----------
    y_true : list[int] or np.ndarray
        Ground-truth binary labels (0 or 1).
    y_probs : list[float] or np.ndarray
        Predicted positive-class probabilities, shape (N,). Values in [0.0, 1.0].
    out_folder : str or Path
        Directory to save the plot.
    model : str
        Model identifier for standardized filenames.
    lr : float
        Learning rate (for filename metadata).
    bs : int
        Batch size (for filename metadata).
    wd : float
        Weight decay (for filename metadata).
    prefix : str
        Filename prefix (e.g., "final", "best").
    threshold : float, optional
        Decision threshold in [0.0, 1.0] used to define correctness. Default 0.5.
    bins : int, optional
        Number of histogram bins (default 20).
    fold : int or None, optional
        Fold index for CV runs.
    epoch : int or None, optional
        Epoch number.
    fname_metric : str, optional
        Metric suffix for the filename.
    title : str, optional
        Plot title.

    Returns
    -------
    Path | None
        Path to the saved PNG file, or None if skipped due to invalid input.
    """

    # Validation
    validate_y_true_pred(y_true, y_true)
    validate_y_probs(y_probs)

    if not (isinstance(threshold, (float, int)) and 0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be a numeric value in [0.0, 1.0].")

    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or Path.")

    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    # Only meaningful for binary, 1-D probs
    if y_probs.ndim == 2 and y_probs.shape[1] == 2:
        # auto-slice positive column if a 2-col binary probability array is passed
        y_probs = y_probs[:, 1]
    if y_probs.ndim != 1:
        print(
            "Skipping confidence histogram split — expected 1D probabilities for binary."
        )
        return None
    if np.unique(y_true).size != 2:
        print("Skipping confidence histogram split — expected binary labels.")
        return None

    thr = float(threshold)
    y_pred = (y_probs >= thr).astype(int)
    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask

    # Common bin edges so bars align
    bins_edges = np.linspace(0.0, 1.0, bins + 1)

    plt.figure(figsize=(6, 4))
    # Correct (light)
    plt.hist(
        y_probs[correct_mask],
        bins=bins_edges,
        alpha=0.65,
        label="Correct",
        edgecolor="black",
    )
    # Incorrect (overlay)
    plt.hist(
        y_probs[incorrect_mask],
        bins=bins_edges,
        alpha=0.65,
        label="Incorrect",
        edgecolor="black",
    )
    # Draw threshold line
    plt.axvline(thr, linestyle="--", linewidth=1.25, label=f"Threshold = {thr:.2f}")

    plt.xlabel("Predicted Probability (Positive Class)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    filename = (
        format_hparams(
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric=fname_metric,
            fold=fold,
            epoch=epoch,
        )
        + ".png"
    )

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    out_path = out_folder / filename
    plt.savefig(out_path)
    plt.close()

    print(f"Saved confidence histogram split to {out_path}")
    return out_path


def evaluate_and_plot(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    train_accs: list[float],
    val_accs: list[float],
    train_losses: list[float],
    val_losses: list[float],
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str,
    out_folder: str | Path,
    class_names: list[str],
    y_probs: list[float] | np.ndarray,
    fold: int | None = None,
    epoch: int | None = None,
    enable_ovr: bool | None = None,  # optional override (env is default)
    ovr_classes: set[str] | None = None,  # optional override (env is default)
) -> None:
    """
    Generate evaluation plots and save classification report using standardized formatting.

    Parameters
    ----------
    y_true : list of int or np.ndarray
        Ground-truth class labels.
    y_pred : list of int or np.ndarray
        Predicted class labels.
    train_accs : list of float
        Training accuracy values per epoch.
    val_accs : list of float
        Validation accuracy values per epoch.
    train_losses : list of float
        Training loss values per epoch.
    val_losses : list of float
        Validation loss values per epoch.
    model : str
        Model string indicates model used (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    prefix : str
        Prefix string indicating model phase or purpose (e.g., "final", "best").
    fname_metric : str, optional
        Metric name (e.g., "loss", "accuracy") to include in filename.
    out_folder : str or Path
        Folder to save output files.
    class_names : list of str
        List of class labels (e.g., ["NORM", "MI", "STTC", ...]).
    y_probs : list of float or np.ndarray
        Predicted probabilities for the positive class (binary classification).
    fold : int or None, optional
        Fold number used in cross-validation. Must be a non-negative integer or
        None.
    epoch : int or None, optional
        Epoch number used for this plot (e.g., best epoch).
    enable_ovr : bool or None, optional
        Enables one-versus-rest plots if set to true. Default is None
    ovr_classes : set[str] or None, optional
        Set of class labels to do one-versus-rest plots for (e.g., {"NORM",
        "MI", "STTC"})

    Raises
    ------
    ValueError
        If input types or shapes are invalid.
    """
    # --- Validation ---
    validate_y_true_pred(y_true, y_pred)
    validate_y_probs(y_probs)

    if not isinstance(class_names, list) or not all(
        isinstance(c, str) for c in class_names
    ):
        raise ValueError("class_names must be a non-empty list of strings.")
    if len(class_names) == 0:
        raise ValueError("class_names cannot be empty.")

    indices_used = set(y_true) | set(y_pred)
    if any(i < 0 or i >= len(class_names) for i in indices_used):
        raise ValueError(
            "class_names must cover all class indices in y_true and y_pred."
        )

    if not isinstance(out_folder, (str, Path)):
        raise ValueError(f"out_folder must be a string or Path, got {type(out_folder)}")
    out_folder = Path(out_folder)

    n_epochs = len(train_accs)
    if not (n_epochs == len(val_accs) == len(train_losses) == len(val_losses)):
        raise ValueError("Training and validation metric lists must be equal in length")

    if fold is not None and (not isinstance(fold, (int, np.integer)) or fold < 1):
        raise ValueError("fold must be a positive integer if provided.")

    validate_hparams_formatting(
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric=fname_metric,
        fold=fold,
        epoch=epoch,
    )

    # Effective OvR settings (env defaults unless explicitly overridden)
    enable_ovr_effective = ENV_ENABLE_OVR if enable_ovr is None else bool(enable_ovr)
    ovr_classes_effective = (
        ENV_OVR_CLASSES
        if ovr_classes is None
        else (set(ovr_classes) if ovr_classes else None)
    )

    # --- Prepare output folders ---
    report_dir = out_folder / "reports"
    plot_dir = out_folder / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Save classification report and heatmap ---
    save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_folder=report_dir,
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        fold=fold,
        prefix=prefix,
        fname_metric="classification_report",
        epoch=epoch,
        title="Classification Report",
    )

    # --- Accuracy curve ---
    save_plot_curves(
        x_vals=train_accs,
        y_vals=val_accs,
        x_label="Epoch",
        y_label="Accuracy",
        title_metric="Accuracy",
        out_folder=plot_dir,
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric="accuracy",
        fold=fold,
        epoch=epoch,
    )

    # --- Loss curve ---
    save_plot_curves(
        x_vals=train_losses,
        y_vals=val_losses,
        x_label="Epoch",
        y_label="Loss",
        title_metric="Loss",
        out_folder=plot_dir,
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric="loss",
        fold=fold,
        epoch=epoch,
    )

    # --- Confusion matrix (normalized) ---
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_folder=plot_dir,
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric="confusion_matrix",
        normalize=True,
        fold=fold,
        epoch=epoch,
    )

    # -------- Shared guards (compute once) --------
    if isinstance(y_probs, list):
        y_probs = np.asarray(y_probs)
    y_probs = np.asarray(y_probs)

    unique_labels = np.unique(y_true)
    is_binary_labels = unique_labels.size == 2
    is_probs_1d = y_probs.ndim == 1
    is_probs_2d = y_probs.ndim == 2
    is_probs_multiclass_K = is_probs_2d and y_probs.shape[1] == len(class_names)

    # Decide evaluation mode once
    binary_mode = is_probs_1d and is_binary_labels
    ovr_mode = is_probs_multiclass_K and enable_ovr_effective
    macro_mode = is_probs_multiclass_K and not enable_ovr_effective

    # OvR indices
    if ovr_mode:
        if ovr_classes_effective:
            idxs = [i for i, c in enumerate(class_names) if c in ovr_classes_effective]
            if not idxs:
                idxs = list(range(len(class_names)))
        else:
            idxs = list(range(len(class_names)))
    else:
        idxs = []

    # ========== Standard PR curve (+AUPRC) ==========
    if binary_mode:
        save_pr_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=plot_dir,
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric="pr_curve",
            fold=fold,
            epoch=epoch,
            title="Precision-Recall Curve",
        )
    elif ovr_mode:
        for i in idxs:
            class_label = class_names[i]
            class_slug = str(class_label).lower()
            y_true_bin = [1 if y == i else 0 for y in y_true]
            save_pr_curve(
                y_true=y_true_bin,
                y_probs=y_probs[:, i],
                out_folder=plot_dir,
                model=model,
                lr=lr,
                bs=bs,
                wd=wd,
                prefix=f"{prefix}_ovr_{class_slug}",
                fname_metric="pr_curve",
                fold=fold,
                epoch=epoch,
                title=f"Precision-Recall Curve (OvR: {class_label})",
            )
    elif macro_mode:
        print("PR OvR disabled (ECG_PLOTS_ENABLE_OVR is False)")
    else:
        print("Skipping PR curve — y_probs shape does not match expected format")

    # ========== Precision/Recall vs. Threshold ==========
    if binary_mode:
        save_pr_threshold_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=plot_dir,
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            epoch=epoch,
            prefix=prefix,
            fname_metric="pr_threshold",
            fold=fold,
            title=f"Precision & Recall vs Threshold ({class_names[1]} vs All)",
        )
    elif ovr_mode:
        for i in idxs:
            class_label = class_names[i]
            class_slug = str(class_label).lower()
            y_true_bin = [1 if y == i else 0 for y in y_true]
            y_probs_bin = y_probs[:, i]
            save_pr_threshold_curve(
                y_true=y_true_bin,
                y_probs=y_probs_bin,
                out_folder=plot_dir,
                model=model,
                lr=lr,
                bs=bs,
                wd=wd,
                epoch=epoch,
                prefix=f"{prefix}_ovr_{class_slug}",
                fname_metric="pr_threshold",
                fold=fold,
                title=f"Precision & Recall vs Threshold (OvR: {class_label})",
            )
    elif macro_mode:
        print("PR-threshold OvR disabled (ECG_PLOTS_ENABLE_OVR is False)")
    else:
        print("Skipping PR curve — y_probs shape does not match expected format")

    # ========== ROC curve (+AUROC) ==========
    if binary_mode:
        save_roc_curve(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=plot_dir,
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric="roc_curve",
            fold=fold,
            epoch=epoch,
            title="ROC Curve",
        )
    elif ovr_mode:
        for i in idxs:
            class_label = class_names[i]
            class_slug = str(class_label).lower()
            y_true_bin = [1 if y == i else 0 for y in y_true]
            save_roc_curve(
                y_true=y_true_bin,
                y_probs=y_probs[:, i],
                out_folder=plot_dir,
                model=model,
                lr=lr,
                bs=bs,
                wd=wd,
                prefix=f"{prefix}_ovr_{class_slug}",
                fname_metric="roc_curve",
                fold=fold,
                epoch=epoch,
                title=f"ROC Curve (OvR: {class_label})",
            )
    elif macro_mode:
        print("ROC OvR disabled (ECG_PLOTS_ENABLE_OVR is False)")
    else:
        print("Skipping ROC curve — y_probs shape does not match expected format")

    # ========== Calibration curve ==========
    if binary_mode or (is_binary_labels and is_probs_2d and y_probs.shape[1] == 2):
        # collapse (n,2) to 1-D if necessary
        y_probs_bin = (
            y_probs[:, 1] if (is_probs_2d and y_probs.shape[1] == 2) else y_probs
        )
        n_bins = 10
        cal_path, brier = save_calibration_curve(
            y_true=y_true,
            y_probs=y_probs_bin,
            out_folder=plot_dir,
            model=model,
            lr=lr,
            wd=wd,
            bs=bs,
            prefix="Calibration",
            fname_metric="calibration",
            n_bins=n_bins,
            fold=fold,
            epoch=epoch,
            title="Calibration Curve",
        )
        print(f"cal_path: {cal_path}, brier: {brier}")
    elif ovr_mode:
        n_bins = 10
        for i in idxs:
            class_label = class_names[i]
            class_slug = str(class_label).lower()
            y_true_bin = [1 if y == i else 0 for y in y_true]
            y_probs_bin = y_probs[:, i]
            cal_path, brier = save_calibration_curve(
                y_true=y_true_bin,
                y_probs=y_probs_bin,
                out_folder=plot_dir,
                model=model,
                lr=lr,
                wd=wd,
                bs=bs,
                prefix=f"{prefix}_ovr_{class_slug}",
                fname_metric="calibration",
                n_bins=n_bins,
                fold=fold,
                epoch=epoch,
                title=f"Calibration Curve (OvR: {class_label})",
            )
            print(f"{class_label} cal_path: {cal_path}, brier: {brier}")
    elif macro_mode:
        print("Skipping Calibration (OvR disabled)")
    else:
        print("Skipping Calibration")

    # ========== Threshold sweep table (CSV) ==========
    if binary_mode:
        # Binary sweep over positive-class probability
        save_threshold_sweep_table(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=report_dir,
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric="threshold_sweep",
            thresholds=None,
            average="binary",
            fold=fold,
            epoch=epoch,
        )
    elif ovr_mode:
        # Per-class OvR sweeps (class i vs rest) using column i as positive prob
        for i in idxs:
            class_label = class_names[i]
            class_slug = str(class_label).lower()
            y_probs_ovr = np.column_stack([1.0 - y_probs[:, i], y_probs[:, i]])
            save_threshold_sweep_table(
                y_true=y_true,
                y_probs=y_probs_ovr,
                out_folder=report_dir,
                model=model,
                lr=lr,
                bs=bs,
                wd=wd,
                prefix=f"{prefix}_ovr_{class_slug}",
                fname_metric="threshold_sweep",
                thresholds=None,
                average="binary",
                fold=fold,
                epoch=epoch,
            )
    elif macro_mode:
        # Macro summary across argmax
        save_threshold_sweep_table(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=report_dir,
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fname_metric="threshold_sweep_macro",
            thresholds=None,
            average="macro",
            fold=fold,
            epoch=epoch,
        )
    else:
        print("Skipping threshold sweep — y_probs shape does not match expected format")

    # ========== Error tables (top-K FN/FP near threshold) ==========
    # Pick your operating threshold. 0.5 is a sane default; or choose a tuned value.
    chosen_thr = 0.5

    if binary_mode:
        save_error_tables(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=report_dir,
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            threshold=chosen_thr,
            top_k=50,
            fold=fold,
            epoch=epoch,
        )
    elif ovr_mode:
        for i in idxs:
            class_label = class_names[i]
            class_slug = str(class_label).lower()
            # Build a binary view for class i: positive prob = column i
            y_true_bin = np.array([1 if y == i else 0 for y in y_true], dtype=int)
            y_probs_bin = y_probs[:, i]
            save_error_tables(
                y_true=y_true_bin,
                y_probs=y_probs_bin,
                out_folder=report_dir,
                model=model,
                lr=lr,
                bs=bs,
                wd=wd,
                prefix=f"{prefix}_ovr_{class_slug}",
                threshold=chosen_thr,
                top_k=50,
                fold=fold,
                epoch=epoch,
            )
    else:
        print("Skipping error tables — not applicable for macro mode.")

    # ========== Confidence Histogram ==========
    # Plot and save histogram of predicted probabilities (confidence scores).
    if binary_mode:
        save_confidence_histogram(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=plot_dir,
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            fold=fold,
            epoch=epoch,
            title="Prediction Confidence Histogram",
        )
    elif ovr_mode:
        for i in idxs:
            class_label = class_names[i]
            class_slug = str(class_label).lower()
            y_true_bin = np.array([1 if y == i else 0 for y in y_true])
            y_probs_bin = y_probs[:, i]
            save_confidence_histogram(
                y_true=y_true_bin,
                y_probs=y_probs_bin,
                out_folder=plot_dir,
                model=model,
                lr=lr,
                bs=bs,
                wd=wd,
                prefix=f"{prefix}_ovr_{class_slug}",
                fold=fold,
                epoch=epoch,
                title=f"Confidence Histogram (OvR: {class_label})",
            )

    # --- Confidence histogram split (Correct vs Incorrect) ---
    chosen_thr = 0.5  # or use a tuned threshold per class

    if binary_mode:
        save_confidence_histogram_split(
            y_true=y_true,
            y_probs=y_probs,
            out_folder=plot_dir,
            model=model,
            lr=lr,
            bs=bs,
            wd=wd,
            prefix=prefix,
            threshold=chosen_thr,
            bins=20,
            fold=fold,
            epoch=epoch,
            title="Confidence Histogram (Correct vs Incorrect)",
        )
    elif ovr_mode:
        for i in idxs:
            class_label = class_names[i]
            class_slug = str(class_label).lower()
            y_true_bin = np.array([1 if y == i else 0 for y in y_true], dtype=int)
            y_probs_bin = y_probs[:, i]
            save_confidence_histogram_split(
                y_true=y_true_bin,
                y_probs=y_probs_bin,
                out_folder=plot_dir,
                model=model,
                lr=lr,
                bs=bs,
                wd=wd,
                prefix=f"{prefix}_ovr_{class_slug}",
                threshold=chosen_thr,
                bins=20,
                fold=fold,
                epoch=epoch,
                title=f"Confidence Histogram (OvR: {class_label}, Correct vs Incorrect)",
            )
    else:
        print("Skipping confidence histogram split — not applicable for macro mode.")


# -----------------------------------------------------------------------------
# SHAP helpers (optional, but clean: all imports at top, all validation inline)
# -----------------------------------------------------------------------------


def _validate_array_3d(
    name: str, x: Union[np.ndarray, torch.Tensor]
) -> Tuple[int, int, int]:
    """
    Validate that input is a 3D array/tensor with shape (N, C, T).

    Parameters
    ----------
    name : str
        Variable name for clearer error messages.
    x : np.ndarray or torch.Tensor
        Input to check.

    Returns
    -------
    (N, C, T) : tuple of int

    Raises
    ------
    TypeError
        If input is not ndarray or Tensor.
    ValueError
        If not 3D or has non-positive dimensions.
    """
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(x)}")

    if x.ndim != 3:
        raise ValueError(f"{name} must have shape (N, C, T), got {tuple(x.shape)}")

    n, c, t = map(int, x.shape)
    if n <= 0 or c <= 0 or t <= 0:
        raise ValueError(f"{name} must have positive dimensions, got {n}, {c}, {t}")

    return n, c, t


def _to_tensor(x: Union[np.ndarray, torch.Tensor], name: str) -> torch.Tensor:
    """Convert ndarray → torch.Tensor, leave tensor as-is."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(x)}")


def _to_numpy(x: Union[np.ndarray, torch.Tensor], name: str) -> np.ndarray:
    """Convert tensor → ndarray (CPU), leave ndarray as-is."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(x)}")


def shap_sample_background(
    data: Union[np.ndarray, torch.Tensor],
    max_background: int = 128,
    seed: int = 22,
) -> torch.Tensor:
    """
    Subsample background examples for SHAP DeepExplainer.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor, shape (N, C, T)
        Input signals.
    max_background : int, default=128
        Maximum number of background samples to keep.
    seed : int, default=22
        RNG seed for reproducibility.

    Returns
    -------
    torch.Tensor, shape (B, C, T)
        Background tensor.
    """
    if max_background < 1:
        raise ValueError(f"max_background must be >=1, got {max_background}")

    _validate_array_3d("data", data)
    data_t = _to_tensor(data, "data")
    n = int(data_t.shape[0])

    if n <= max_background:
        return data_t

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_background, replace=False)
    return data_t[idx]


def shap_compute_values(
    model: torch.nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    background: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Compute SHAP values for ECG CNN inputs.

    Parameters
    ----------
    model : torch.nn.Module
        The (trained) PyTorch model. Caller manages .eval()/.train().
    X : np.ndarray or torch.Tensor, shape (N, C, T)
        Inputs to explain.
    background : np.ndarray or torch.Tensor, shape (B, C, T)
        Background distribution for SHAP.
    device : torch.device, optional
        Device to run on. Defaults to the model's device.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Binary: returns (N, C, T)  [positive-class attribution]
        Multiclass: returns list length=K, each (N, C, T)

    Raises
    ------
    TypeError
        If inputs are not the expected types.
    ValueError
        If shapes are invalid or model outputs have unexpected shape.
    RuntimeError
        If SHAP explainer fails for non-recoverable reasons.
    """
    # --- shape/type validation for inputs ---
    _validate_array_3d("X", X)
    _validate_array_3d("background", background)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model)}")

    X_t = _to_tensor(X, "X")
    bg_t = _to_tensor(background, "background")

    # --- choose device (prefer model's params device if available) ---
    try:
        model_param_device = next(model.parameters()).device  # may raise StopIteration
    except StopIteration:
        # No parameters => cannot infer device; require explicit device
        if device is None:
            raise ValueError("model has no parameters; cannot infer device or run SHAP")
        model_param_device = device

    dev = device or model_param_device
    X_t = X_t.to(dev)
    bg_t = bg_t.to(dev)

    # --- validate model forward output shape with a tiny forward pass ---
    model_was_training = model.training
    with torch.no_grad():
        model.eval()
        sample_logits = model(X_t[:1])
    # Accept (1,K) or (K,) (binary/regression-like). Anything else -> error.
    logits_shape = tuple(int(s) for s in sample_logits.shape)
    ok = (sample_logits.ndim == 2 and logits_shape[0] == 1) or (  # (1,K)
        sample_logits.ndim == 1
    )  # (K,)
    if not ok:
        if model_was_training:
            model.train()
        raise ValueError(
            f"Expected model logits of shape (N,K) or (N,), got {logits_shape}"
        )

    # --- build explainer: try DeepExplainer; on TF import/failure, fall back to GradientExplainer ---
    try:
        explainer = shap.DeepExplainer(model, bg_t)
        # Some SHAP versions require disabling additivity check with CNNs
        shap_values = explainer.shap_values(X_t, check_additivity=False)
    except Exception as e:
        # Common case: DeepExplainer tries to import TensorFlow
        msg = str(e).lower()
        if ("tensorflow" in msg) or ("no module named 'tensorflow'" in msg):
            try:
                explainer = shap.GradientExplainer(model, bg_t)
                shap_values = explainer.shap_values(X_t)
            except Exception as ge:
                if model_was_training:
                    model.train()
                raise RuntimeError(f"SHAP DeepExplainer failed: {e}") from ge
        else:
            if model_was_training:
                model.train()
            raise RuntimeError(f"SHAP DeepExplainer failed: {e}") from e
    finally:
        # restore training mode if needed
        if model_was_training:
            model.train()

    # --- normalize outputs to expected shapes ---
    def _to_np(arr: Any) -> np.ndarray:
        return _to_numpy(arr, "shap_values")

    # Case A: SHAP returns a list already (usually per-class)
    if isinstance(shap_values, list):
        out_list = [_to_np(sv) for sv in shap_values]
    else:
        arr = _to_np(shap_values)
        # Some SHAP versions return (N,C,T,K) for multiclass
        if arr.ndim == 4 and arr.shape[-1] > 1:
            out_list = [arr[..., k] for k in range(arr.shape[-1])]
        else:
            # Binary/regression-like single map
            out_list = [arr]

    # Final contract:
    # - If exactly 1 map -> return (N,C,T)
    # - If exactly 2 maps -> return positive-class map (index 1)
    # - Else -> return list length K
    if len(out_list) == 1:
        _validate_array_3d("shap_values", out_list[0])
        return out_list[0]
    if len(out_list) == 2:
        _validate_array_3d("shap_values", out_list[1])
        return out_list[1]
    for m in out_list:
        _validate_array_3d("shap_values", m)
    return out_list


def shap_save_channel_summary(
    shap_values: Union[np.ndarray, List[np.ndarray]],
    X: Union[np.ndarray, torch.Tensor],
    out_dir: Union[str, Path],
    filename: str,
) -> Path:
    """
    Save per-channel SHAP importance (aggregated over N, T).

    Parameters
    ----------
    shap_values : np.ndarray or list of np.ndarray
        Binary: (N, C, T); Multiclass: list length=K, each (N, C, T).
    X : np.ndarray or torch.Tensor, shape (N, C, T)
        Inputs (used for shape validation only).
    out_dir : str or Path
        Output directory.
    filename : str
        Output file name.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("Empty shap_values list for multiclass case.")
        mags = [np.abs(sv).mean() for sv in shap_values]
        shap_mat = shap_values[int(np.argmax(mags))]
    else:
        shap_mat = shap_values

    _validate_array_3d("shap_values", shap_mat)
    _validate_array_3d("X", X)

    N, C, T = shap_mat.shape

    imp = np.abs(shap_mat).mean(axis=(0, 2))  # (C,)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    plt.figure()
    x = np.arange(C)
    plt.bar(x, imp)
    plt.xlabel("Channel")
    plt.ylabel("Mean |SHAP| over time")
    plt.title("Channel importance (SHAP)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


def save_classification_report_csv(
    y_true, y_pred, out_folder: Path, tag: str, fold_id: int
) -> Path | None:
    """
    Save a per-fold classification report as a tidy CSV.

    Generates a machine-readable classification report for one fold of training/
    evaluation, including precision, recall, F1-score, and support for each class,
    plus a row for the macro average. The output is stored under
    `outputs/reports/classification_report_{tag}_fold{fold_id}.csv`.

    Parameters
    ----------
    y_true : list[int] or np.ndarray
        Ground-truth class labels for all samples in this fold.
    y_pred : list[int] or np.ndarray
        Predicted class labels for all samples in this fold.
    out_folder : Path
        Directory where the report CSV should be saved. Must be writable.
    tag : str
        Unique experiment tag identifying the model/configuration
        (e.g., "ECGResNet_lr0001_bs64_wd00005").
    fold_id : int
        Fold index (1-based) for cross-validation. Must be > 0.

    Returns
    -------
    Path or None
        Full path to the saved CSV file, or None if validation fails or writing
        cannot be completed.

    Raises
    ------
    ValueError
        If inputs are invalid: empty/unequal lengths for y_true/y_pred,
        or non-positive fold_id.

    Notes
    -----
    - Assumes labels are among {"CD","HYP","MI","NORM","STTC"}.
    - Always appends a "macro avg" row for consistency across folds.
    - Intended for downstream fold aggregation in evaluate.py.
    """
    if not isinstance(fold_id, int) or fold_id <= 0:
        raise ValueError("fold_id must be a positive integer")
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be same non-zero length")

    out_folder.mkdir(parents=True, exist_ok=True)

    cr_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    labels = ["CD", "HYP", "MI", "NORM", "STTC"]
    header = ["label", "precision", "recall", "f1-score", "support"]
    out_path = out_folder / f"classification_report_{tag}_fold{fold_id}.csv"

    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for lab in labels:
            row = cr_dict.get(lab, {})
            w.writerow(
                [
                    lab,
                    row.get("precision"),
                    row.get("recall"),
                    row.get("f1-score"),
                    row.get("support"),
                ]
            )
        ma = cr_dict.get("macro avg", {})
        w.writerow(
            [
                "macro avg",
                ma.get("precision"),
                ma.get("recall"),
                ma.get("f1-score"),
                ma.get("support"),
            ]
        )

    print(f"Saved classification report CSV to {out_path}")

    return out_path


def save_fold_summary_csv(reports_dir: Path, tag: str) -> Path | None:
    """
    Aggregate metrics across all folds for a given tag and save to CSV.

    Collects validation accuracy, validation loss, and (if available) macro-F1
    from per-fold artifacts and writes a summary CSV:
    `outputs/reports/fold_summary_{tag}.csv`.

    Parameters
    ----------
    reports_dir : Path
        Directory where reports are written (e.g., PROJECT_ROOT/outputs/reports).
    tag : str
        Experiment tag identifying the model/configuration (e.g.,
        "ECGResNet_lr0001_bs64_wd00005").
    """
    if not isinstance(tag, str) or not tag:
        print("save_fold_summary_csv: invalid tag provided")
        return None

    reports_dir = Path(reports_dir)
    artifacts_dir = (
        reports_dir.parent / "artifacts"
    )  # artifacts/ is parallel to reports/
    hist_dir = reports_dir.parent / "history"  # history/ is parallel to reports/
    reports_dir.mkdir(parents=True, exist_ok=True)

    hist_files = sorted(
        hist_dir.glob(f"history_{tag}_fold*.json"), key=lambda p: p.name
    )
    if len(hist_files) < 2:
        print(
            f"save_fold_summary_csv: found only {len(hist_files)} folds for tag '{tag}' — need at least 2"
        )
        return None

    rows = []
    accs, losses, macrof1s = [], [], []

    for hf in hist_files:
        try:
            with hf.open("r") as fh:
                h = json.load(fh)
        except Exception as e:
            print(f"save_fold_summary_csv: failed to read {hf}: {e}")
            continue

        # Your schema uses singular keys
        val_accs = h.get("val_acc") or []
        val_losses = h.get("val_loss") or []
        be = h.get("best_epoch")

        if be is None and val_losses:
            try:
                be = min(range(len(val_losses)), key=lambda i: val_losses[i])
            except Exception as e:
                print(
                    f"save_fold_summary_csv: failed to resolve best_epoch in {hf}: {e}"
                )
                be = None

        acc = (
            float(val_accs[be])
            if isinstance(be, int) and 0 <= be < len(val_accs)
            else None
        )
        loss = (
            float(val_losses[be])
            if isinstance(be, int) and 0 <= be < len(val_losses)
            else None
        )

        m = re.search(r"fold(\d+)", hf.name)
        fold_id = int(m.group(1)) if m else -1

        cr_csv = artifacts_dir / f"classification_report_{tag}_fold{fold_id}.csv"
        macro_f1 = None
        if cr_csv.exists():
            try:
                with cr_csv.open("r", newline="") as fh:
                    rdr = csv.reader(fh)
                    rows_csv = [r for r in rdr]
                if rows_csv:
                    header = rows_csv[0]

                    def _col(name: str):
                        for i, hname in enumerate(header):
                            if (hname or "").strip().lower() == name:
                                return i
                        return None

                    f1_i = _col("f1-score")
                    label_i = _col("label") if _col("label") is not None else 0
                    if f1_i is None and len(header) >= 4:
                        f1_i = 3
                    for r in rows_csv[1:]:
                        if not r or f1_i is None or len(r) <= max(label_i, f1_i):
                            continue
                        lab = (r[label_i] or "").strip().lower()
                        if lab.startswith("macro"):
                            try:
                                macro_f1 = float(r[f1_i])
                            except Exception as e:
                                print(
                                    f"save_fold_summary_csv: bad macro_f1 value in {cr_csv}: {e}"
                                )
                            break
            except Exception as e:
                print(f"save_fold_summary_csv: failed to parse {cr_csv}: {e}")

        rows.append(
            {
                "fold": fold_id,
                "best_epoch": be,
                "val_acc": acc,
                "val_loss": loss,
                "macro_f1": macro_f1,
                "history_path": str(hf),
                "cr_csv": str(cr_csv) if cr_csv.exists() else "",
            }
        )

        if isinstance(acc, float):
            accs.append(acc)
        if isinstance(loss, float):
            losses.append(loss)
        if isinstance(macro_f1, float):
            macrof1s.append(macro_f1)

    def _mean_std(vals):
        xs = [v for v in vals if isinstance(v, float)]
        if not xs:
            return (None, None)
        return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else None)

    mean_acc, std_acc = _mean_std(accs)
    mean_loss, std_loss = _mean_std(losses)
    mean_mf1, std_mf1 = _mean_std(macrof1s)

    out_csv = reports_dir / f"fold_summary_{tag}.csv"
    try:
        with out_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "fold",
                    "best_epoch",
                    "val_acc",
                    "val_loss",
                    "macro_f1",
                    "history_path",
                    "cr_csv",
                ]
            )
            for r in sorted(rows, key=lambda x: x["fold"]):
                w.writerow(
                    [
                        r["fold"],
                        r["best_epoch"],
                        r["val_acc"],
                        r["val_loss"],
                        r["macro_f1"],
                        r["history_path"],
                        r["cr_csv"],
                    ]
                )
            w.writerow([])
            w.writerow(["mean_val_acc", mean_acc])
            w.writerow(["std_val_acc", std_acc])
            w.writerow(["mean_val_loss", mean_loss])
            w.writerow(["std_val_loss", std_loss])
            w.writerow(["mean_macro_f1", mean_mf1])
            w.writerow(["std_macro_f1", std_mf1])
    except Exception as e:
        print(f"save_fold_summary_csv: failed to write {out_csv}: {e}")
        return None

    print(f"Saved fold summary CSV to {out_csv}")
    return out_csv
