#!/usr/bin/env python

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from ecg_cnn.config.config_loader import load_training_config, TrainConfig
from ecg_cnn.data.data_utils import load_ptbxl_full, FIVE_SUPERCLASSES
from ecg_cnn.models.model_utils import ECGConvNet
from ecg_cnn.paths import (
    HISTORY_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    OUTPUT_DIR,
    PTBXL_DATA_DIR,
)
from ecg_cnn.utils.plot_utils import (
    save_pr_threshold_curve,
    save_confusion_matrix,
    save_plot_curves,
    evaluate_and_plot,
)

SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    t0 = time.time()

    # Dynamically find the latest saved config YAML
    configs = sorted((RESULTS_DIR).glob("config_*.yaml"), reverse=True)
    if not configs:
        raise FileNotFoundError("No config_*.yaml found in outputs/results/")
    config_path = configs[0]
    print(f"Loading config from: {config_path}")

    # config = load_training_config(config_path)

    # Load raw config and attach extra fields manually
    raw = load_training_config(config_path, strict=False)

    # Separate known and extra fields
    extra = {}
    for key in ("fold", "tag", "config"):
        extra[key] = raw.pop(key, None)

    try:
        config = TrainConfig(**raw)
    except TypeError as e:
        raise ValueError(f"Invalid config structure or missing fields: {e}")

    for key, val in extra.items():
        if val is not None:
            setattr(config, key, val)

    print(f"Config file contents: {config}")
    print(f"extra contents: {extra}")

    # Load data
    print("Loading data for evaluation...")
    X, y, meta = load_ptbxl_full(
        data_dir=PTBXL_DATA_DIR,
        subsample_frac=config.subsample_frac,
        sampling_rate=config.sampling_rate,
    )

    # Filter unknown labels
    keep = np.array([lbl != "Unknown" for lbl in y], dtype=bool)
    X = X[keep]
    y = [lbl for i, lbl in enumerate(y) if keep[i]]
    meta = meta.loc[keep].reset_index(drop=True)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_tensor = torch.tensor(y_encoded).long()
    X_tensor = torch.tensor(X).float()
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Load best model info from summary
    # summary_path = OUTPUT_DIR / "results" / f"summary_{config.model}.json"
    summary_path = OUTPUT_DIR / "results" / f"summary_{extra['tag']}.json"
    with open(summary_path, "r") as f:
        summaries = json.load(f)

    best = min(summaries, key=lambda d: d["loss"])
    best_model_path = Path(best["model_path"])
    best_fold = best.get("fold")
    best_epoch = best.get("best_epoch")

    print(
        f"Evaluating best model from fold {best_fold if best_fold is not None else 'N/A'} (epoch {best_epoch})"
    )
    print(f"Loading model: {best_model_path.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGConvNet(num_classes=len(FIVE_SUPERCLASSES)).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Run evaluation
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * y_batch.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataset)
    print(f"Eval loss: {avg_loss:.4f}")
    print(f"Evaluation completed in {(time.time() - t0) / 60:.2f} minutes.")

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_probs = np.array(all_probs, dtype=np.float32)

    # Try to load history if available
    train_accs, val_accs, train_loss, val_loss = [], [], [], []
    print(f"best_fold:  {best_fold}")
    if best_fold is not None:
        # hist_path = HISTORY_DIR / f"history_fold{best_fold}.json"
        hist_path = HISTORY_DIR / f"history_{extra['tag']}_fold{best_fold}.json"
        print(f"history_path: {hist_path}")
        if hist_path.exists():
            with open(hist_path, "r") as f:
                hist = json.load(f)
            train_accs = hist.get("train_acc", [])
            val_accs = hist.get("val_acc", [])
            train_loss = hist.get("train_loss", [])
            val_loss = hist.get("val_loss", [])
        else:
            print(f"(History not found at {hist_path})")

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=FIVE_SUPERCLASSES, zero_division=0
        )
    )

    # Confusion Matrix
    # save_confusion_matrix(
    #     y_true=y_true.tolist(),
    #     y_pred=y_pred.tolist(),
    #     class_names=FIVE_SUPERCLASSES,
    #     out_folder=OUTPUT_DIR / "plots",
    #     model=config.model,
    #     lr=config.lr,
    #     bs=config.batch_size,
    #     wd=config.weight_decay,
    #     epoch=best_epoch,
    #     prefix="eval",
    #     fname_metric="confusion_matrix",
    #     normalize=True,
    #     fold=(best_fold + 1 if best_fold is not None else None),
    # )

    # # Precision-Recall Curve (NORM vs ALL)
    # norm_class = FIVE_SUPERCLASSES.index("NORM")
    # y_true_binary = (y_true == norm_class).astype(int)
    # y_probs_binary = y_probs[:, norm_class]

    # save_pr_threshold_curve(
    #     y_true=y_true_binary,
    #     y_probs=y_probs_binary,
    #     out_folder=OUTPUT_DIR / "plots",
    #     model=config.model,
    #     lr=config.lr,
    #     bs=config.batch_size,
    #     wd=config.weight_decay,
    #     epoch=best_epoch,
    #     prefix="eval",
    #     fname_metric="pr_threshold_curve",
    #     title="Precision & Recall vs Threshold (NORM vs All)",
    #     fold=(best_fold + 1 if best_fold is not None else None),
    # )

    # if val_loss:
    #     save_plot_curves(
    #         x_vals=list(range(1, len(val_loss) + 1)),
    #         y_vals=val_loss,
    #         x_label="Epoch",
    #         y_label="Loss",
    #         title_metric="Validation Loss Curve",
    #         out_folder=OUTPUT_DIR / "plots",
    #         model=config.model,
    #         lr=config.lr,
    #         bs=config.batch_size,
    #         wd=config.weight_decay,
    #         epoch=best_epoch,
    #         prefix="eval",
    #         fname_metric="loss",
    #         fold=(best_fold + 1 if best_fold is not None else None),
    #     )

    # if val_accs:
    #     save_plot_curves(
    #         x_vals=list(range(1, len(val_loss) + 1)),
    #         y_vals=val_loss,
    #         x_label="Epoch",
    #         y_label="Accuracy",
    #         title_metric="Validation Accuracy Curve",
    #         out_folder=OUTPUT_DIR / "plots",
    #         model=config.model,
    #         lr=config.lr,
    #         bs=config.batch_size,
    #         wd=config.weight_decay,
    #         epoch=best_epoch,
    #         prefix="eval",
    #         fname_metric="loss",
    #         fold=(best_fold + 1 if best_fold is not None else None),
    #     )

    y_true_ep = [int(x) for x in y_true]
    y_pred_ep = [int(x) for x in y_pred]

    print(f"train_accs: {train_accs}")
    print(f"val_accs: {val_accs}")
    print(f"train_loss: {train_loss}")
    print(f"val_loss: {val_loss}")
    print(f"best_fold: {best_fold}")

    evaluate_and_plot(
        y_true=y_true_ep,
        y_pred=y_pred_ep,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_loss,
        val_losses=val_loss,
        model=config.model,
        lr=config.lr,
        bs=config.batch_size,
        wd=config.weight_decay,
        prefix="evaluation",
        fname_metric="eval_summary",
        out_folder=OUTPUT_DIR,
        class_names=FIVE_SUPERCLASSES,
        y_probs=y_probs,
        fold=(best_fold if best_fold is not None else None),
        epoch=best_epoch,
    )

    print(f"Elapsed time: {(time.time() - t0) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
