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
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import asdict
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------
# Project Imports (ecg_cnn.* in alphabetical order)
# --------------------------------------------------------------------------
from ecg_cnn.config.config_loader import (
    load_training_config,
    merge_configs,
    normalize_path_fields,
    load_yaml_as_dict,
    TrainConfig,
)
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
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------------------
# Main Training Script
# ------------------------------------------------------------------------------
def main():
    t0 = time.time()
    args = parse_args()

    # 1. Load and merge configs
    base_cfg = load_training_config(DEFAULT_TRAINING_CONFIG)

    print(f"args.config: {args.config}")
    if args.config:
        override_dict = load_yaml_as_dict(Path(args.config))
        config = merge_configs(base_cfg, override_dict)
    else:
        config = base_cfg

    config = normalize_path_fields(config)
    config = override_config_with_args(config, args)

    print(f"config.verbose:  {config.verbose}")
    if config.verbose:
        print("Effective training config:")
        for k, v in vars(config).items():
            print(f"  {k}: {v}")

    # def main():
    #     t0 = time.time()
    #     args = parse_args()

    #     # --------------------------------------------------------------------------
    #     # 1. Load and merge configs
    #     # --------------------------------------------------------------------------
    #     base_config = load_training_config(DEFAULT_TRAINING_CONFIG)

    #     # if args.config:
    #     #     user_config = load_training_config(Path(args.config))
    #     #     config = merge_configs(config, user_config)

    #     # if args.config:
    #     #     override_dict = load_yaml_as_dict(Path(args.config))
    #     #     override_cfg = TrainConfig(**override_dict)
    #     #     config = merge_configs(config, override_cfg)

    #     if args.config:
    #         override_dict = load_yaml_as_dict(Path(args.config))
    #         # Create TrainConfig *only after* merging
    #         override_cfg = TrainConfig(**override_dict)
    #         config = merge_configs(base_config, override_cfg)
    #     else:
    #         config = base_config

    #     # Normalize after merging (e.g., str -> Path)
    #     config = normalize_path_fields(config)

    #     # Apply CLI overrides
    #     config = override_config_with_args(config, args)

    #     print(f"Config file contents: {config}")

    #     # print("Effective training config:")
    #     # for k, v in vars(config).items():
    #     #     print(f"  {k}: {v}")

    #     if config.verbose:
    #         print("Effective training config:")
    #         for k, v in vars(config).items():
    #             print(f"  {k}: {v}")

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

    # Save the final config
    with open(MODELS_DIR / "final_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    time_spent = (time.time() - t0) / 60
    print(f"Elapsed time: {time_spent:.2f} minutes")


if __name__ == "__main__":
    main()
