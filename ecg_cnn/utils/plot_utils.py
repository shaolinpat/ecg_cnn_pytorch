

# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import os
# import wfdb
from sklearn.metrics import (
    # confusion_matrix, 
    classification_report,
    ConfusionMatrixDisplay, 
)



def format_hparams(
    lr: float,
    bs: int,
    wd: float,
    fold: int,
    epochs: int,
    prefix: str,
    fname_metric: str = ""
) -> str:
    """
    Build a filename of the form:
      "<prefix>_<metric>_lr<lr_str>_bs<bs>_wd<wd_str>_fold<fold>_epo<epochs>.png"
    or, if fname_metric is empty (e.g. for model checkpoints):
      "<prefix>_lr<lr_str>_bs<bs>_wd<wd_str>_fold<fold>_epo<epochs>.pt"

    - lr_str: drop leading "0." and trailing zeros from 4-decimal lr.
    - wd_str: same formatting for weight_decay.
    """
    # 1) Convert lr to a trimmed string (e.g. 0.0005 -> "0005", 0.0010 -> "001", 
    #    1.0000 -> "1")
    raw_lr = f"{lr:.4f}"
    if raw_lr.startswith("0."):
        lr_str = raw_lr[2:]
    else:
        lr_str = raw_lr
    lr_str = lr_str.rstrip("0")
    if lr_str.endswith("."):
        lr_str = lr_str[:-1]
    if lr_str == "":
        lr_str = "0"

    # 2) Convert wd in the same way
    raw_wd = f"{wd:.4f}"
    if raw_wd.startswith("0."):
        wd_str = raw_wd[2:]
    else:
        wd_str = raw_wd
    wd_str = wd_str.rstrip("0")
    if wd_str.endswith("."):
        wd_str = wd_str[:-1]
    if wd_str == "":
        wd_str = "0"

    # 3) Lowercase the metric part if provided
    metric_part = ""
    if fname_metric:
        metric_part = "_" + fname_metric.lower()

    # 4) Assemble without extension
    base = (
        prefix
        + metric_part
        + "_lr"
        + lr_str
        + "_bs"
        + str(bs)
        + "_wd"
        + wd_str
        + "_fold"
        + str(fold)
        + "_epo"
        + str(epochs)
    )
    return base


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
    out_folder: str,
    prefix: str,
):
    """
    Draws and saves a two-line plot (train vs val) for a given metric.

    Arguments:
      x_vals:        List of training-metric values (one per epoch)
      y_vals:        List of validation-metric values (one per epoch)
      x_label:       Label for the x-axis (e.g. "Epoch")
      y_label:       Label for the y-axis (e.g. "Model Loss" or "Accuracy")
      title_metric:  The human-readable metric name (e.g. "Model Accuracy" or "Model Loss")
      fname_metric:  The file-safe metric key (e.g. "accuracy" or "loss")
      lr, bs, fold, epochs, out_folder, prefix: as before

    This function builds a title:
       "{title_metric} by Epoch\nLR={lr}, BS={bs}, Fold={fold}, Epochs={epochs}"
    and it saves to:
       "{prefix}_{fname_metric}_{lr_str}_{bs}_{fold}_{epochs}.png"
    """
    title = f"{title_metric} by Epoch\nLR={lr}, BS={bs}, WD={wd}, Fold={fold}"
    filename = format_hparams(
        lr=lr,
        bs=bs,
        wd=wd,
        fold=fold,
        epochs=epochs,
        prefix=prefix,
        fname_metric=fname_metric
    )
    path = os.path.join(out_folder, filename)

    #1) Plot the two curves
    fig, ax = plt.subplots()
    ax.plot(x_vals, label=f"Training")
    ax.plot(y_vals, label=f"Validation")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    # 2) Save and close
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
        normalize: bool=True
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
        values_format=".2f" if normalize else "d"
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
    y_true, y_pred, 
    train_accs, val_accs, 
    train_losses, val_losses, 
    lr, bs, wd, fold, epochs, out_folder, 
    class_names  # list of strings, e.g. ['CD','HYP','MI','NORM','STTC']
):
    print(
        f"\n=== Final Evaluation (LR={lr}, BS={bs}, Fold={fold}, Epochs={epochs}) ==="
    )

    # 1) Print classification report
    print("Classification Report:")
    print(
        classification_report(y_true, y_pred, 
        labels=list(range(len(class_names))), 
        target_names=class_names,
        zero_division=0)
    )

    # 2) Accuracy curve
    save_plot_curves(
        x_vals    = train_accs,
        y_vals    = val_accs,
        x_label   = "Epoch",
        y_label   = "Accuracy",
        title_metric = "Accuracy",
        fname_metric = "accuracy",
        lr        = lr,
        bs        = bs,
        wd        = wd,
        fold      = fold,
        epochs    = epochs,
        out_folder= out_folder,
        prefix    = "final"
    )

    # 3) Loss curve
    save_plot_curves(
        x_vals       = train_losses,
        y_vals       = val_losses,
        x_label      = "Epoch",
        y_label      = "Loss",
        title_metric = "Loss",
        fname_metric = "loss",
        lr           = lr,
        bs           = bs,
        wd           = wd,
        fold         = fold,
        epochs       = epochs,
        out_folder   = out_folder,
        prefix       = "final"
    )

    # 4) Confusion matrix (normalized)
    save_confusion_matrix(
        y_true     = y_true,
        y_pred     = y_pred,
        class_names= class_names,
        lr         = lr,
        bs         = bs,
        wd         = wd,
        fold       = fold,
        epochs     = epochs,
        out_folder = out_folder,
        prefix     = "final",
        normalize  = True
    )

