#!/usr/bin/env python3
"""
train.py

Main training entry point for ECG CNN models. This calls
    ecg_cnn/training/trainer/run_training() to do the actual training.

Run with:
    python -m ecg_cnn.train --config configs/baseline.yaml

"""

import json
import numpy as np
import random
import time
import torch
import yaml

from datetime import datetime
from pathlib import Path

from ecg_cnn.config.config_loader import (
    load_training_config,
    merge_configs,
    load_yaml_as_dict,
)
from ecg_cnn.models.model_utils import ECGConvNet
from ecg_cnn.paths import (
    DEFAULT_TRAINING_CONFIG,
    RESULTS_DIR,
)
from ecg_cnn.training.cli_args import parse_training_args, override_config_with_args
from ecg_cnn.training.trainer import run_training
from ecg_cnn.utils.grid_utils import is_grid_config, expand_grid
from ecg_cnn.utils.plot_utils import format_hparams


# ------------------------------------------------------------------------------
# globals
# ------------------------------------------------------------------------------

SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
_DATA_CACHE = {}

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------


def _acc_value(s):
    """Prefer validation accuracy if present; else fall back to train."""
    va = s.get("val_accs")
    if va is None:
        va = s.get("train_accs")
    # Guard for weird Nones/NaNs/strings
    try:
        return float(va)
    except Exception:
        return float("-inf")


def _loss_value(s):
    """Loss already uses val when available in run_training(); just cast safe."""
    try:
        return float(s.get("loss", float("inf")))
    except Exception:
        return float("inf")


# ------------------------------------------------------------------------------
# Main Training
# ------------------------------------------------------------------------------
def main():
    t0 = time.time()
    args = parse_training_args()

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

        tag = format_hparams(
            model=config.model,
            lr=config.lr,
            bs=config.batch_size,
            wd=config.weight_decay,
            prefix="ecg",  # or "best"/"final" depending on context
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

    if all_summaries:
        # Best by loss (lower is better)
        best_by_loss = min(all_summaries, key=_loss_value)

        # Best by accuracy (higher is better), prefer val_accs but fallback to train_accs
        # Filter out entries that have neither
        candidates_acc = [
            s
            for s in all_summaries
            if s.get("val_accs") is not None or s.get("train_accs") is not None
        ]
        best_by_accuracy = (
            max(candidates_acc, key=_acc_value) if candidates_acc else None
        )

        print(
            f"\nBest model by loss: {best_by_loss['model_path']} (epoch {best_by_loss['best_epoch']})"
        )
        print(f"Best-by-loss summary: {best_by_loss}")

        if best_by_accuracy is not None:
            print(
                f"\nBest model by accuracy: {best_by_accuracy['model_path']} (epoch {best_by_accuracy['best_epoch']})"
            )
            print(f"Best-by-accuracy summary: {best_by_accuracy}")
        else:
            print("\nBest model by accuracy: <none> (no accuracy recorded)")

        # Save a single JSON your other scripts can read directly
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        best_payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tag": tag,  # whatever tag you used for this grid run
            "by_loss": best_by_loss,
            "by_accuracy": best_by_accuracy,
            # Optional: keep the entire list for traceability
            "all_summaries": all_summaries,
        }
        best_path = RESULTS_DIR / f"really_the_best_{tag}.json"
        with open(best_path, "w") as f:
            json.dump(best_payload, f, indent=2)
        print(f"Saved best selections to: {best_path}")

    time_spent = (time.time() - t0) / 60
    print(f"Elapsed time: {time_spent:.2f} minutes")


if __name__ == "__main__":
    main()
