#!/usr/bin/env python3
"""
train.py

Main training entry point for ECG CNN models.

Run with:
    python -m ecg_cnn.train --config configs/baseline.yaml

"""

# --------------------------------------------------------------------------
# Standard Library Imports
# --------------------------------------------------------------------------
import random
import time
from pathlib import Path
import yaml

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
    RESULTS_DIR,
)
from ecg_cnn.training.cli_args import parse_args, override_config_with_args
from ecg_cnn.training.trainer import train_one_epoch, run_training
from ecg_cnn.utils.grid_utils import is_grid_config, expand_grid


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

    # 1. Load baseline/default config
    base_cfg = load_training_config(DEFAULT_TRAINING_CONFIG)

    # 2. Load and apply YAML config
    if args.config:
        override_dict = load_yaml_as_dict(Path(args.config))

        if is_grid_config(override_dict):
            # Expand grid values into list of config dictionaries
            raw_grid = list(expand_grid(override_dict))  # list of dicts

            # Merge each grid dict with base_cfg to ensure defaults are preserved
            param_grid = [merge_configs(base_cfg, cfg_dict) for cfg_dict in raw_grid]
        else:
            # Single override case: merge directly
            config = merge_configs(base_cfg, override_dict)
            param_grid = [config]
    else:
        param_grid = [base_cfg]

    # 3. Override using CLI arguments (if any)
    param_grid = [override_config_with_args(cfg, args) for cfg in param_grid]

    # 4. Print effective configs
    for i, config in enumerate(param_grid):
        if config.verbose:
            print(f"\n=== Config {i+1}/{len(param_grid)} ===")
            for k, v in vars(config).items():
                print(f"  {k}: {v}")

    # 5. Iterate over all configs and train
    all_summaries = []

    for i, config in enumerate(param_grid):
        print(f"\n===== Starting training run {i+1}/{len(param_grid)} =====")

        tag = f"{config.model}_lr{config.lr}_bs{config.batch_size}_wd{config.weight_decay}".replace(
            ".", ""
        )
        config.tag = tag

        summaries = []

        if config.n_folds and config.n_folds >= 2:
            for fold_idx in range(config.n_folds):
                print(f"\n--- Fold {fold_idx + 1}/{config.n_folds} ---")
                summary = run_training(config, fold_idx=fold_idx, tag=tag)
                summaries.append(summary)
        else:
            summary = run_training(config, tag=tag)
            summaries.append(summary)

        print(f"summaries[0].keys(): {summaries[0].keys()}")
        print(f"summaries:  {summaries}")

        # Save training summary
        summary_path = RESULTS_DIR / f"summary_{config.tag}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)

        # Save config used for this tag
        config_path = RESULTS_DIR / f"config_{config.tag}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(vars(config), f)

        print(f"Saved summary to: {summary_path}")
        print(f"Saved config to: {config_path}")

        # Add all folds to overall results
        all_summaries.extend(summaries)

    # ----------------------------------------------------------
    # Select best summary across all configs/folds (after loop)
    # ----------------------------------------------------------
    if all_summaries:
        best_summary = min(all_summaries, key=lambda d: d["loss"])
        print(
            f"\nBest model: {best_summary['model_path']} (epoch {best_summary['best_epoch']})"
        )

        # best_tag = (
        #     Path(best_summary["model_path"])
        #     .stem.replace("model_best_", "")
        #     .replace(f"_fold{best_summary['fold']}", "")
        # )

        # with open(RESULTS_DIR / "best_tag.txt", "w") as f:
        #     f.write(best_tag)
        # print(f"Saved best tag: {best_tag} to best_tag.txt")

        # # 5. Iterate over all configs and train
        # all_summaries = []

        # for i, config in enumerate(param_grid):
        #     print(f"\n===== Starting training run {i+1}/{len(param_grid)} =====")

        #     tag = f"{config.model}_lr{config.lr}_bs{config.batch_size}_wd{config.weight_decay}".replace(
        #         ".", ""
        #     )
        #     config.tag = tag

        #     summaries = []

        #     if config.n_folds and config.n_folds >= 2:
        #         for fold_idx in range(config.n_folds):
        #             print(f"\n--- Fold {fold_idx + 1}/{config.n_folds} ---")
        #             summary = run_training(config, fold_idx=fold_idx)
        #             summaries.append(summary)
        #     else:
        #         summary = run_training(config)
        #         summaries.append(summary)

        #     print(f"summaries[0].keys(): {summaries[0].keys()}")
        #     print(f"summaries:  {summaries}")

        #     # Save training summary
        #     if summaries:
        #         print("Config keys:", vars(config).keys())
        #         summary_path = OUTPUT_DIR / "results" / f"summary_{config.model}.json"
        #         summary_path.parent.mkdir(parents=True, exist_ok=True)
        #         with open(summary_path, "w") as f:
        #             json.dump(summaries, f, indent=2)

        #     # Save effective config for evaluate.py
        #     config_path = OUTPUT_DIR / "results" / f"config_{config.model}.yaml"
        #     with open(config_path, "w") as f:
        #         yaml.dump(vars(config), f)

        #     print(f"Saved summary to: {summary_path}")
        #     print(f"Saved config to: {config_path}")

        #     best_summary = min(summaries, key=lambda d: d["loss"])
        #     print(
        #         f"Best model: {best_summary['model_path']} (epoch {best_summary['best_epoch']})"
        #     )

        #     best_tag = (
        #         Path(best_summary["model_path"])
        #         .stem.replace("model_best_", "")
        #         .replace(f"_fold{best_summary['fold']}", "")
        #     )

        #     # Save best tag to file for downstream scripts
        #     with open(RESULTS_DIR / "best_tag.txt", "w") as f:
        #         f.write(best_tag)

        time_spent = (time.time() - t0) / 60
        print(f"Elapsed time: {time_spent:.2f} minutes")


if __name__ == "__main__":
    main()
