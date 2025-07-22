#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from ecg_cnn.config import PTBXL_DATA_DIR, MODELS_DIR, OUTPUT_DIR
from ecg_cnn.data.data_utils import load_ptbxl_full, FIVE_SUPERCLASSES
from ecg_cnn.models.model_utils import ECGConvNet
from ecg_cnn.utils.plot_utils import (
    save_pr_threshold_curve,
    save_confusion_matrix,
)

SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    t0 = time.time()

    # Load data
    print("Loading data for evaluation...")
    X, y, meta = load_ptbxl_full(
        data_dir=PTBXL_DATA_DIR, subsample_frac=0.6, sampling_rate=100
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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGConvNet(num_classes=len(FIVE_SUPERCLASSES)).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "ecgconvnet_one_epoch.pth"))
    model.eval()

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

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # === Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=FIVE_SUPERCLASSES, zero_division=0
        )
    )

    # === Confusion Matrix
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=FIVE_SUPERCLASSES,
        lr=0.001,
        bs=64,
        wd=0.0,
        fold=0,
        epochs=1,
        out_folder=OUTPUT_DIR,
        prefix="eval",
        normalize=True,
    )

    # === Precision-Recall Curve (NORM vs ALL)
    norm_class = FIVE_SUPERCLASSES.index("NORM")
    y_true_binary = (y_true == norm_class).astype(int)
    y_probs_binary = y_probs[:, norm_class]

    save_pr_threshold_curve(
        y_true=y_true_binary,
        y_probs=y_probs_binary,
        out_path=OUTPUT_DIR / "plots" / "threshold_pr_norm_vs_all.png",
        title="Precision & Recall vs Threshold (NORM vs Rest)",
    )


if __name__ == "__main__":
    main()
