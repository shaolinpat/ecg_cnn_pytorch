#!/usr/bin/env python

# --------------------------------------------------------------------------
# Standard Library Imports
# --------------------------------------------------------------------------
import random
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Third-Party Imports (alphabetical)
# --------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------
# Project Imports (ecg_cnn.* in alphabetical order)
# --------------------------------------------------------------------------
from ecg_cnn.config.config_loader import load_training_config
from ecg_cnn.data.data_utils import (
    FIVE_SUPERCLASSES,
    load_ptbxl_full,
    load_ptbxl_sample,
)
from ecg_cnn.models.model_utils import ECGConvNet
from ecg_cnn.paths import (
    DEFAULT_TRAINING_CONFIG,
    MODELS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    PTBXL_DATA_DIR,
)
from ecg_cnn.training.cli_args import parse_args, override_config_with_args
from ecg_cnn.training.trainer import train_one_epoch


# --------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------

SEED = 22
verbose = True
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Main Training Script
# --------------------------------------------------------------------------


def main():
    t0 = time.time()
    args = parse_args()
    if verbose:
        print("CLI parsed subsample_frac =", args.subsample_frac)

    config_path = DEFAULT_TRAINING_CONFIG
    config = load_training_config(config_path)
    print(f"Config file contents: {config}")

    if args.model is not None:
        config.model = args.model

    if args.subsample_frac is not None:
        config.subsample_frac = args.subsample_frac

    # Load data
    data_dir = Path(config.data_dir) if config.data_dir else PTBXL_DATA_DIR
    if config.sample_only:
        print("Loading sample data...")
        X, y, meta = load_ptbxl_sample(
            sample_dir=config.sample_dir,
            ptb_path=data_dir,
        )
    else:
        print("Loading full data...")
        X, y, meta = load_ptbxl_full(
            data_dir=data_dir,
            subsample_frac=config.subsample_frac,
            # subsample_frac=args.subsample_frac,
            sampling_rate=config.sampling_rate,
        )

    print("Loaded", len(meta), "records")

    # Drop unknowns
    keep = np.array([lbl != "Unknown" for lbl in y], dtype=bool)
    X = X[keep]
    y = [lbl for i, lbl in enumerate(y) if keep[i]]
    meta = meta.loc[keep].reset_index(drop=True)

    # Encode targets
    le = LabelEncoder()
    y_tensor = torch.tensor(le.fit_transform(y)).long()
    X_tensor = torch.tensor(X).float()

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGConvNet(num_classes=len(FIVE_SUPERCLASSES)).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Loss: {loss:.4f}")

        if config.save_best and loss < best_loss:
            best_loss = loss
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODELS_DIR / "model_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to: {model_path}")

    # Save summary
    elapsed = (time.time() - t0) / 60
    summary_path = OUTPUT_DIR / "training_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(f"Best Loss: {best_loss:.4f}\n")
        f.write(f"Runtime: {elapsed:.2f} minutes\n")
    print(f"Saved training summary to: {summary_path}")


if __name__ == "__main__":
    main()
