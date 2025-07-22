import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wfdb

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)


# ------------------------------------------------------------------------------
# Shared validation for common hyperparameters
# ------------------------------------------------------------------------------
def _validate_hparams(
    lr: float,
    bs: int,
    wd: float,
    fold: int,
    epochs: int,
    prefix: str,
    fname_metric: str = "",
) -> None:
    """
    Validates hyperparameters used for filename formatting or training configuration.

    Parameters
    ----------
    lr : float
        Learning rate (e.g., 0.0005). Must be between 1e-6 and 1.0.
    bs : int
        Batch size. Must be between 1 and 4096.
    wd : float
        Weight decay. Must be between 0.0 and 1.0.
    fold : int
        Fold number in cross-validation. Must be non-negative.
    epochs : int
        Number of training epochs. Must be between 1 and 1000.
    prefix : str
        Prefix string indicating model phase or purpose (e.g., "final", "best").
    fname_metric : str, optional
        Optional metric name (e.g., "loss", "accuracy") to include in filename.

    Returns
    -------
    Nothing is returned.

    Raises
    ------
    ValueError
        If any of the provided values are of incorrect type or out of valid range.
    """
    if not isinstance(lr, (float, int)) or not (1e-6 <= lr <= 1.0):
        raise ValueError(
            f"Learning rate must be positive int or float in range [1e-6, 1.0]. Got: {lr}"
        )
    if not isinstance(bs, int) or not (1 <= bs <= 4096):
        raise ValueError(f"Batch size must be an integer in range [1, 4096]. Got: {bs}")
    if not isinstance(wd, (float, int)) or not (0.0 <= wd <= 1.0):
        raise ValueError(
            f"Weight decay must be int or float in range [0.0, 1.0]. Got: {wd}"
        )
    if not isinstance(fold, int) or fold < 0:
        raise ValueError(f"Fold number must be a non-negative integer. Got: {fold}")
    if not isinstance(epochs, int) or not (1 <= epochs <= 1000):
        raise ValueError(f"Epochs must be an integer in range [1, 1000]. Got: {epochs}")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"Prefix must be a non-empty string. Got: {prefix!r}")
    if fname_metric is not None and not isinstance(fname_metric, str):
        raise ValueError("Metric name must be a string")


# ------------------------------------------------------------------------------
# Function to standardize plot titles
# ------------------------------------------------------------------------------


def _build_plot_title(metric: str, lr: float, bs: int, wd: float, fold: int) -> str:
    """
    Construct a standardized plot title including key hyperparameters.

    Format:
        "<metric> by Epoch\nLR=<lr>, BS=<bs>, WD=<wd>, Fold=<fold>"

    Parameters
    ----------
    metric : str
        Descriptive title for the metric being plotted (e.g., "Accuracy", "Loss").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    fold : int
        Cross-validation fold number. Must be a non-negative integer.

    Returns
    -------
    str
        A formatted title string for use in plots.

    Raises
    ------
    ValueError
        If any input fails validation checks.
    """
    # Reuse centralized validation logic (dummy value for epochs, prefix)
    _validate_hparams(
        lr=lr, bs=bs, wd=wd, fold=fold, epochs=1, prefix="fake", fname_metric=metric
    )
    return f"{metric} by Epoch\nLR={lr}, BS={bs}, WD={wd}, Fold={fold}"


# ------------------------------------------------------------------------------
# Helper function to standardize filenames for plots (used by save_plot_curves,
# save_confusion_matrix, etc.)
# ------------------------------------------------------------------------------
def format_hparams(
    lr: float,
    bs: int,
    wd: float,
    fold: int,
    epochs: int,
    prefix: str,
    fname_metric: str = "",
) -> str:
    """
    Construct a standardized filename string for saved plots or models.

    Format:
        "<prefix>_<metric>_lr<lr_str>_bs<bs>_wd<wd_str>_fold<fold>_epo<epochs>"

    If fname_metric is omitted, the "_<metric>" part is skipped.

    Parameters
    ----------
    lr : float
        Learning rate (e.g., 0.0005). Must be between 1e-6 and 1.0.
    bs : int
        Batch size. Must be between 1 and 4096.
    wd : float
        Weight decay. Must be between 0.0 and 1.0.
    fold : int
        Fold number in cross-validation. Must be non-negative.
    epochs : int
        Number of training epochs. Must be between 1 and 1000.
    prefix : str
        Prefix string indicating model phase or purpose (e.g., "final", "best").
    fname_metric : str, optional
        Optional metric name (e.g., "loss", "accuracy") to include in filename.

    Returns
    -------
    str
        A formatted string of the form:
            "<prefix>_<metric>_lr<lr>_bs<bs>_wd<wd>_fold<fold>_epoch<epochs>"

    Notes
    -----
    - input validation done by _validate_hyperparams(_validate_hparams(lr, bs, wd, fold, epochs, prefix,fname_metric)
    - Floats are truncated to at most 6 decimal places and then stripped of leading "0." and trailing zeros.
    - Floats smaller than 1e-6 are disallowed to avoid loss of precision or misleading filenames.

    """
    # will raise an error if the params are out of spec
    _validate_hparams(lr, bs, wd, fold, epochs, prefix, fname_metric)

    # --- Helper function for trimming floats ---
    def _trim(val: float) -> str:
        raw = f"{val:.6f}"
        trimmed = raw[2:] if raw.startswith("0.") else raw
        trimmed = trimmed.rstrip("0").rstrip(".")
        return trimmed or "0"

    # --- Build filename ---
    lr_str = _trim(lr)
    wd_str = _trim(wd)
    prefix = prefix.lower()
    metric_part = f"_{fname_metric.lower()}" if fname_metric else ""
    return (
        f"{prefix}{metric_part}_lr{lr_str}_bs{bs}_wd{wd_str}_fold{fold}_epoch{epochs}"
    )


def save_plot_curves(
    x_vals: list[float],
    y_vals: list[float],
    x_label: str,
    y_label: str,
    title_metric: str,
    fname_metric: str,
    lr: float,
    bs: int,
    wd: float,
    fold: int,
    epochs: int,
    out_folder: str | Path,
    prefix: str,
):
    """
    Draws and saves a line plot comparing training and validation curves
    (e.g., accuracy or loss over epochs), using consistent filename formatting.

    Parameters
    ----------
    x_vals : list of float
        Training metric values per epoch.
    y_vals : list of float
        Validation metric values per epoch.
    x_label : str
        Label for the x-axis (e.g., "Epoch").
    y_label : str
        Label for the y-axis (e.g., "Accuracy", "Loss").
    title_metric : str
        Human-readable title (e.g., "Model Accuracy").
    fname_metric : str
        File-safe key for the metric (e.g., "accuracy", "loss").
    lr : float
        Learning rate (must match format_hparams constraints).
    bs : int
        Batch size.
    wd : float
        Weight decay.
    fold : int
        Fold number.
    epochs : int
        Total number of training epochs.
    out_folder : str or Path
        Destination folder where the plot will be saved.
    prefix : str
        Filename prefix indicating phase (e.g., "final", "best").

    Raises
    ------
    ValueError
        If any input is malformed or inconsistent in type or length.
    """
    # # --- Validate curve data ---
    # if not isinstance(x_vals, list) or not all(
    #     isinstance(x, (int, float)) for x in x_vals
    # ):
    #     raise ValueError("x_vals must be a list of numeric values.")
    # if not isinstance(y_vals, list) or not all(
    #     isinstance(y, (int, float)) for y in y_vals
    # ):
    #     raise ValueError("y_vals must be a list of numeric values.")
    # if len(x_vals) != len(y_vals):
    #     raise ValueError(
    #         f"x_vals and y_vals must have the same length. Got {len(x_vals)} and {len(y_vals)}."
    #     )

    # --- Validate curve data ---
    for name, arr in [("x_vals", x_vals), ("y_vals", y_vals)]:
        if not hasattr(arr, "__len__"):
            raise ValueError(f"{name} must be array-like.")
        if not all(isinstance(x, (int, float, np.integer, np.floating)) for x in arr):
            raise ValueError(f"{name} must contain only numeric values.")

    if len(x_vals) != len(y_vals):
        raise ValueError(
            f"x_vals and y_vals must have the same length. Got {len(x_vals)} and {len(y_vals)}."
        )

    # Normalize to list in case user passed NumPy arrays
    x_vals = list(x_vals)
    y_vals = list(y_vals)

    # --- Validate output folder ---
    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")

    # --- Validate axis labels and title ---
    for name, val in [
        ("x_label", x_label),
        ("y_label", y_label),
        ("title_metric", title_metric),
        ("fname_metric", fname_metric),
        ("prefix", prefix),
    ]:
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    # --- Reuse existing hyperparameter validation ---
    filename = format_hparams(
        lr=lr,
        bs=bs,
        wd=wd,
        fold=fold,
        epochs=epochs,
        prefix=prefix,
        fname_metric=fname_metric,
    )
    path = Path(out_folder) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    # --- Plot and save ---
    # title = f"{title_metric} by Epoch\nLR={lr}, BS={bs}, WD={wd}, Fold={fold}"
    title = _build_plot_title(title_metric, lr, bs, wd, fold)

    fig, ax = plt.subplots()
    ax.plot(x_vals, label="Training")
    ax.plot(y_vals, label="Validation")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {prefix.lower()} {fname_metric.lower()} plot to {path}")


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    lr: float,
    bs: int,
    wd: float,
    fold: int,
    epochs: int,
    out_folder: str,
    prefix: str,
    normalize: bool = True,
):
    """
    Builds a confusion matrix (optionally normalized), displays it with labels,
    and save it to a file named:
        {prefix}_confmat_<lr>_<bs>_<fold>_<epochs>.png
    """
    # 1) compute raw confusion matrix and display title
    cm = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize="true" if normalize else None,
        cmap="Blues",
        values_format=".2f" if normalize else "d",
    )

    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    plot_title = f"{title}\nLR={lr}, BS={bs}, WD={wd}, Fold={fold}"
    cm.ax_.set_title(plot_title)

    # 2) Build filename and save
    filename = format_hparams(lr, bs, wd, fold, epochs, prefix, "confmat")
    path = os.path.join(out_folder, filename)
    cm.figure_.savefig(path)
    cm.figure_.clf()
    print(f"Saved {prefix.lower()} confusion matrix to {path}")


def evaluate_and_plot(
    y_true,
    y_pred,
    train_accs,
    val_accs,
    train_losses,
    val_losses,
    lr,
    bs,
    wd,
    fold,
    epochs,
    out_folder,
    class_names,  # list of strings, e.g. ['CD','HYP','MI','NORM','STTC']
):
    print(
        f"\n=== Final Evaluation (LR={lr}, BS={bs}, Fold={fold}, Epochs={epochs}) ==="
    )

    report_dir = Path(out_folder) / "reports"
    heatmap_dir = Path(out_folder) / "plots"

    # 1) Print classification report
    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_path_txt=report_dir / "classification_report.txt",
        out_path_png=heatmap_dir / "classification_report.png",
    )

    # 2) Accuracy curve
    save_plot_curves(
        x_vals=train_accs,
        y_vals=val_accs,
        x_label="Epoch",
        y_label="Accuracy",
        title_metric="Accuracy",
        fname_metric="accuracy",
        lr=lr,
        bs=bs,
        wd=wd,
        fold=fold,
        epochs=epochs,
        out_folder=out_folder,
        prefix="final",
    )

    # 3) Loss curve
    save_plot_curves(
        x_vals=train_losses,
        y_vals=val_losses,
        x_label="Epoch",
        y_label="Loss",
        title_metric="Loss",
        fname_metric="loss",
        lr=lr,
        bs=bs,
        wd=wd,
        fold=fold,
        epochs=epochs,
        out_folder=out_folder,
        prefix="final",
    )

    # 4) Confusion matrix (normalized)
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        lr=lr,
        bs=bs,
        wd=wd,
        fold=fold,
        epochs=epochs,
        out_folder=out_folder,
        prefix="final",
        normalize=True,
    )


def save_pr_threshold_curve(
    *,
    y_true,
    y_probs,
    out_path,
    title="Precision & Recall vs Threshold",
):
    """
    Plot and save a precision-recall vs threshold curve.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (0 or 1), e.g. for "NORM" class vs all.
    y_probs : array-like
        Predicted probabilities for the positive class.
    out_path : str or Path
        Path object representing where to save the figure.
    title : str
        Title to display on the plot.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    thresholds = np.append(thresholds, 1.0)  # ensure alignment

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision, label="Precision")
    ax.plot(thresholds, recall, label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved PR threshold plot to {out_path}")


def save_classification_report(
    *,
    y_true,
    y_pred,
    class_names,
    out_path_txt,
    out_path_png,
    zero_division=0,
    title="Classification Report",
):
    """
    Save classification report as .txt and heatmap (.png).
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=zero_division,
    )

    # --- Save text report ---
    out_path_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_txt, "w") as f:
        f.write(f"{title}:\n")
        f.write(
            classification_report(
                y_true, y_pred, target_names=class_names, zero_division=zero_division
            )
        )
    print(f"Saved classification report to {out_path_txt}")

    # --- Save heatmap ---
    df = pd.DataFrame(report).transpose()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png)
    plt.close(fig)
    print(f"Saved classification report heatmap to {out_path_png}")
