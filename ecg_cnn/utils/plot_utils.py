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

from ecg_cnn.utils.validate import validate_hparams_formatting


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
        "<model>: <metric> by Epoch\nLR=<lr>, BS=<bs>, WD=<wd>, Fold=<fold>, Epoch=<epoch>"

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

    if epoch is not None:
        title = f"{model}: {metric} by Epoch"
    else:
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
        "<prefix>_<metric>_lr<lr_str>_bs<bs>_wd<wd_str>_fold<fold>_epo<epoch>"

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

    # Normalize to list in case user passed NumPy arrays
    x_vals = list(x_vals)
    y_vals = list(y_vals)

    # Validate output folder
    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")

    # Validate axis labels and title
    for name, val in [
        ("x_label", x_label),
        ("y_label", y_label),
        ("title_metric", title_metric),
        ("prefix", prefix),
    ]:
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    # Set filename. format_hparams calls validates_hparams which validates
    # these parameters
    filename = format_hparams(
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric=fname_metric,
        epoch=epoch,
        fold=fold,
    )
    path = Path(out_folder) / f"{filename}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    # --- Plot and save ---
    title = _build_plot_title(model, lr, bs, wd, prefix, title_metric, fold, epoch)

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
    INT_TYPES = (int, np.integer)

    # --- Validate labels ---
    if not isinstance(y_true, list) or not all(
        isinstance(i, INT_TYPES) and not isinstance(i, bool) for i in y_true
    ):
        raise ValueError("y_true must be a list of integers (not bools).")

    if not isinstance(y_pred, list) or not all(
        isinstance(i, INT_TYPES) and not isinstance(i, bool) for i in y_pred
    ):
        raise ValueError("y_pred must be a list of integers (not bools).")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length, got {len(y_true)} and {len(y_pred)}."
        )

    if not isinstance(class_names, list) or not all(
        isinstance(s, str) for s in class_names
    ):
        raise ValueError("class_names must be a list of strings.")

    if not isinstance(out_folder, (str, Path)):
        raise ValueError("out_folder must be a string or pathlib.Path.")

    if fold is not None:
        if not isinstance(fold, INT_TYPES) or fold < 1:
            raise ValueError("fold must be a positive integer if provided.")

    # Validate and normalize path
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # The rest of the params are validated by format_hparams below

    # Compute confusion matrix
    cm = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize="true" if normalize else None,
        cmap="Blues",
        values_format=".2f" if normalize else "d",
    )

    plot_title = (
        f"{model}:  Normalized Confusion Matrix"
        if normalize
        else f"{model}: Confusion Matrix"
    )

    plot_title = (
        f"{plot_title}\nLR={lr}, BS={bs}, WD={wd}"
        + (f", Fold={fold}" if fold is not None else "")
        + (f", Epoch={epoch}" if epoch is not None else "")
    )

    title_text = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    title_text = _build_plot_title(
        model, lr, bs, wd, prefix, metric=title_text, fold=fold, epoch=epoch
    )
    print(f"Title text:  {title_text}")
    print(epoch)

    cm.ax_.set_title(plot_title)

    # Save figure
    filename = format_hparams(
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric=fname_metric,
        fold=fold,
        epoch=epoch,
    )
    path = out_folder / f"{filename}.png"
    cm.figure_.savefig(path)
    cm.figure_.clf()
    print(f"Saved {prefix.lower()} {fname_metric.lower()} plot to {path}")


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
    INT_TYPES = (int, np.integer)
    FLOAT_TYPES = (float, np.floating)

    # --- Validate inputs ---
    if not isinstance(y_true, (list, np.ndarray)):
        raise ValueError(f"y_true must be list or ndarray, got {type(y_true)}")
    if not all(isinstance(i, INT_TYPES) and not isinstance(i, bool) for i in y_true):
        raise ValueError("y_true must contain only integers (not bools)")

    if not isinstance(y_probs, (list, np.ndarray)):
        raise ValueError(f"y_probs must be list or ndarray, got {type(y_probs)}")
    if not all(isinstance(p, FLOAT_TYPES) for p in y_probs):
        raise ValueError("y_probs must contain only float values")

    if not all(0.0 <= p <= 1.0 for p in y_probs):
        raise ValueError("y_probs must contain probabilities in [0.0, 1.0]")

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

    if fold is not None:
        if not isinstance(fold, INT_TYPES) or fold < 1:
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

    # --- Prepare output path ---
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    filename = format_hparams(
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric=fname_metric,
        fold=fold,
        epoch=epoch,
    )
    out_path = out_folder / f"{filename}.png"

    # --- Compute curve ---
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    thresholds = np.append(thresholds, 1.0)

    # Prepate title

    plot_title = f"{model}:  {title}"

    plot_title = (
        f"{plot_title}\nLR={lr}, BS={bs}, WD={wd}"
        + (f", Fold={fold}" if fold is not None else "")
        + (f", Epoch={epoch}" if epoch is not None else "")
    )

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

    print(f"Saved {prefix.lower()} {fname_metric.lower()} plot to {out_path}")


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
    INT_TYPES = (int, np.integer)

    # --- Input validation ---
    if not isinstance(y_true, (list, np.ndarray)):
        raise ValueError("y_true must be list or ndarray")
    if not all(isinstance(i, INT_TYPES) and not isinstance(i, bool) for i in y_true):
        raise ValueError("y_true must contain only integer values")

    if not isinstance(y_pred, (list, np.ndarray)):
        raise ValueError("y_pred must be list or ndarray")
    if not all(isinstance(i, INT_TYPES) and not isinstance(i, bool) for i in y_pred):
        raise ValueError("y_pred must contain only integer values")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length, got {len(y_true)} and {len(y_pred)}."
        )

    if not isinstance(class_names, list) or not all(
        isinstance(c, str) for c in class_names
    ):
        raise ValueError("class_names must be a list of strings")

    if not isinstance(out_folder, (str, Path)):
        raise ValueError(
            f"out_folder must be a string or pathlib.Path, got {type(out_folder)}"
        )

    if not isinstance(title, str):
        raise ValueError(f"title must be a string, got {type(title)}")

    if fold is not None:
        if not isinstance(fold, INT_TYPES) or fold < 1:
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

    # --- Prepare output paths ---
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    fname = format_hparams(
        model=model,
        lr=lr,
        bs=bs,
        wd=wd,
        prefix=prefix,
        fname_metric=fname_metric,
        fold=fold,
        epoch=epoch,
    )
    out_path_txt = out_folder / f"{fname}.txt"
    out_path_png = out_folder / f"{fname}.png"

    # --- Generate classification report ---
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # --- Save report as text ---
    with open(out_path_txt, "w") as f:
        f.write(f"{title}:\n")
        f.write(
            classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0,
            )
        )
    print(f"Saved classification report to {out_path_txt}")

    # --- Save heatmap ---
    df = pd.DataFrame(report_dict).transpose()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path_png)
    plt.close(fig)
    print(f"Saved classification report heatmap to {out_path_png}")


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
    fold: int | None = None,
    epoch: int | None = None,
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
    fold : int or None, optional
        Fold number used in cross-validation. Must be a non-negative integer or None.
    epoch : int or None, optional
        Epoch number used for this plot (e.g., best epoch).

    Raises
    ------
    ValueError
        If input types or shapes are invalid.
    """

    INT_TYPES = (int, np.integer)

    # --- Validate y_true and y_pred ---
    if not isinstance(y_true, (list, np.ndarray)):
        raise ValueError(f"y_true must be list or ndarray, , got {type(y_true)}")
    if not isinstance(y_pred, (list, np.ndarray)):
        raise ValueError(f"y_pred must be list or ndarray, got {type(y_pred)}")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length, got {len(y_true)} and {len(y_pred)}."
        )
    if not all(isinstance(i, INT_TYPES) and not isinstance(i, bool) for i in y_true):
        raise ValueError("y_true must contain only integer values")
    if not all(isinstance(i, INT_TYPES) and not isinstance(i, bool) for i in y_pred):
        raise ValueError("y_pred must contain only integer values")

    # --- Validate class_names ---
    if not isinstance(class_names, list):
        raise ValueError(f"class_names must be a list, got {type(class_names)}")
    if not all(isinstance(c, str) for c in class_names):
        raise ValueError("class_names must be a list of strings")
    if len(class_names) == 0:
        raise ValueError("class_names cannot be empty")
    if max(y_true) >= len(class_names) or max(y_pred) >= len(class_names):
        raise ValueError(
            "class_names length must cover all class indices in y_true and y_pred"
        )

    # --- Validate out_folder ---
    if not isinstance(out_folder, (str, Path)):
        raise ValueError(f"out_folder must be a string or Path, got {type(out_folder)}")
    out_folder = Path(out_folder)

    print(
        f"\n=== Final Evaluation (Model={model}, LR={lr}, BS={bs}, Fold={fold}, Epochs={epoch}) ==="
    )

    # --- Validate train/val metrics ---
    n_epochs = len(train_accs)
    if not (n_epochs == len(val_accs) == len(train_losses) == len(val_losses)):
        raise ValueError("Training and validation metric lists must be equal in length")

    # Validate fold
    if fold is not None:
        if not isinstance(fold, INT_TYPES) or fold < 1:
            raise ValueError("fold must be a positive integer if provided.")

    report_dir = out_folder / "reports"
    plot_dir = out_folder / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Print classification report ---
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
