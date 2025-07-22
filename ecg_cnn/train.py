#!/usr/bin/env python


# ## 0. Imports

import argparse
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from ecg_cnn.config import PTBXL_DATA_DIR, MODELS_DIR
from ecg_cnn.data.data_utils import (
    load_ptbxl_sample,
    load_ptbxl_full,
    FIVE_SUPERCLASSES,
)
from ecg_cnn.models.model_utils import ECGConvNet
from ecg_cnn.training.cli_args import parse_args
from ecg_cnn.training.trainer import train_one_epoch


if __name__ == "__main__":
    SEED = 22
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t0 = time.time()
    args = parse_args()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else PTBXL_DATA_DIR

    # Load data
    if args.sample_only:
        print("Loading sample data from:", args.sample_dir)
        X, y, meta = load_ptbxl_sample(
            sample_dir=args.sample_dir,
            ptb_path=data_dir,
        )
    else:
        print("Loading full data from:", args.data_dir)
        X, y, meta = load_ptbxl_full(
            data_dir=data_dir,
            subsample_frac=args.subsample_frac,
            sampling_rate=100,
        )

    print("Number of records loaded:", len(meta))

    # Drop "Unknown" labels
    keep = np.array([lbl != "Unknown" for lbl in y], dtype=bool)
    X = X[keep]
    y = [lbl for i, lbl in enumerate(y) if keep[i]]
    meta = meta.loc[keep].reset_index(drop=True)

    # --------------------------------------------------------------------------
    # One epoch using real model and data
    # --------------------------------------------------------------------------
    # Convert data to tensors

    le = LabelEncoder()
    y_tensor = torch.tensor(le.fit_transform(y)).long()  # int class IDs
    X_tensor = torch.tensor(X).float()

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGConvNet(num_classes=len(FIVE_SUPERCLASSES)).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
    print(f"One-epoch training complete. Loss: {loss:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "ecgconvnet_one_epoch.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # --------------------------------------------------------------------------
    # End
    # --------------------------------------------------------------------------
    elapsed = (time.time() - t0) / 60

    summary_path = MODELS_DIR / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"One-epoch training complete.\n")
        f.write(f"Loss: {loss:.4f}\n")
        f.write(f"Runtime: {elapsed:.2f} minutes\n")
    print(f"Training summary saved to: {summary_path}")

    print(f"Total runtime: {elapsed:.2f} minutes")
