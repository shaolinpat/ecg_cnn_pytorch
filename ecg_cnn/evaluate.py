#!/usr/bin/env python
"""
Unified evaluator for trained ECG CNN runs.

This script:
  1) Locates the most recent `config_*.yaml` in RESULTS_DIR (unchanged behavior).
  2) Loads that config and its embedded `tag` (plus optional `fold`).
  3) Loads the corresponding `summary_<tag>.json` and selects the "best" run:
       - If --fold is passed, use that fold.
       - Else pick the summary entry with the lowest recorded loss.
  4) Loads data using the same sampling/subsample settings from the config.
  5) Restores the trained model weights and runs inference over the dataset.
  6) Prints a classification report and (via evaluate_and_plot) saves artifacts.

Notes
-----
- All original commented-out plotting helpers are preserved exactly as-is.
- Paths come from ecg_cnn.paths; nothing is hardcoded.
- RNG is seeded to 22 for reproducibility.
- Additional validation is included to fail fast with clear messages.

CLI
---
python -m ecg_cnn.evaluate              # evaluate latest config's tag, best
                                        # fold by loss
python -m ecg_cnn.evaluate --fold 3     # force a specific fold
"""

import argparse
import json
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from typing import Any, Dict, List, Optional, Tuple

from ecg_cnn.config.config_loader import load_training_config, TrainConfig
from ecg_cnn.data.data_utils import load_ptbxl_full, FIVE_SUPERCLASSES
from ecg_cnn.models import MODEL_CLASSES
from ecg_cnn.paths import (
    HISTORY_DIR,
    RESULTS_DIR,
    OUTPUT_DIR,
    PTBXL_DATA_DIR,
)
from ecg_cnn.utils.plot_utils import (
    evaluate_and_plot,
)

# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _latest_config_path() -> Path:
    """
    Return the newest `config_*.yaml` in RESULTS_DIR.

    Returns:
        Path to latest config if found, else None. If None a helpful message can
        (and should) be printed at the calling location.
    """
    configs = sorted(RESULTS_DIR.glob("config_*.yaml"), reverse=True)
    return configs[0] if configs else None


def _load_config_and_extras(
    config_path: Path, fold_override: Optional[int]
) -> Tuple[TrainConfig, Dict[str, Any]]:
    """
    Load a potentially-extended training config (with extra keys like `tag`, `fold`)
    and return (TrainConfig, extras_dict). Extras are applied back onto the TrainConfig.

    Args:
        config_path: Path to config YAML.
        fold_override: Optional fold override from CLI.

    Returns:
        (config: TrainConfig, extras: dict)
    """
    raw = load_training_config(config_path, strict=False)

    extras: Dict[str, Any] = {}
    for key in ("fold", "tag", "config"):
        print(f"^^^^^key = {key}")
        if key == "fold" and fold_override is not None:
            extras[key] = fold_override
        else:
            extras[key] = raw.pop(key, None)

    try:
        config = TrainConfig(**raw)
    except TypeError as e:
        raise ValueError(f"Invalid config structure or missing fields: {e}")

    for key, val in extras.items():
        if val is not None:
            setattr(config, key, val)

    return config, extras


def _read_summary(tag: str) -> List[dict]:
    """
    Load the summary JSON for a tag.

    Args:
        tag: Training tag used for filenames.

    Returns:
        List of summary dicts.

    Raises:
        FileNotFoundError: if the summary file does not exist.
        ValueError: if the file is empty or malformed.
    """
    # summary_path = OUTPUT_DIR / "results" / f"summary_{tag}.json"
    summary_path = RESULTS_DIR / f"summary_{tag}.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary for tag '{tag}': {summary_path}. "
            f"Did train.py finish and write summaries?"
        )
    with open(summary_path, "r") as f:
        summaries = json.load(f)
    if not isinstance(summaries, list) or not summaries:
        raise ValueError(f"Summary JSON malformed or empty: {summary_path}")
    return summaries


def _select_best_entry(summaries: List[dict], fold_override: Optional[int]) -> dict:
    """
    Select the best summary entry.

    Priority:
      - If `fold_override` is provided, pick the entry matching that fold.
      - Else pick the entry with the lowest 'loss'.

    Args:
        summaries: List of summary dicts.
        fold_override: Optional fold to force.

    Returns:
        Chosen summary entry.

    Raises:
        ValueError: if a requested fold is not present or entries lack 'loss'.
    """
    if fold_override is not None:
        matching = [s for s in summaries if s.get("fold") == fold_override]
        if not matching:
            raise ValueError(f"No summary entry found for fold {fold_override}")
        return min(matching, key=lambda d: float(d["loss"]))

    # No override: choose lowest loss across all
    if not all("loss" in s for s in summaries):
        raise ValueError("Summary entries missing 'loss' key; cannot select best.")

    print(f"^^^^min loss: {min(summaries, key=lambda d: float(d['loss']))}")
    return min(summaries, key=lambda d: float(d["loss"]))


def _load_history(
    tag: str, fold: Optional[int]
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Load history arrays (train_acc, val_acc, train_loss, val_loss) for the given
        tag/fold.

    Args:
        tag: Training tag.
        fold: 1-based fold index, or None.

    Returns:
        (train_accs, val_accs, train_loss, val_loss). Lists may be empty if
            missing.
    """
    train_accs: List[float] = []
    val_accs: List[float] = []
    train_loss: List[float] = []
    val_loss: List[float] = []

    if fold is None:
        return train_accs, val_accs, train_loss, val_loss

    hist_path = HISTORY_DIR / f"history_{tag}_fold{fold}.json"
    print(f"history_path: {hist_path}")
    if hist_path.exists():
        with open(hist_path, "r") as f:
            hist = json.load(f)
        train_accs = hist.get("train_acc", []) or []
        val_accs = hist.get("val_acc", []) or []
        train_loss = hist.get("train_loss", []) or []
        val_loss = hist.get("val_loss", []) or []
    else:
        print(f"(History not found at {hist_path})")

    return train_accs, val_accs, train_loss, val_loss


def _resolve_ovr_flags(
    config: TrainConfig,
    cli_ovr_enable: Optional[bool] = None,
    cli_ovr_classes: Optional[List[str]] = None,
) -> Tuple[bool, Optional[set]]:
    """
    Resolve one-vs-rest flags from config, environment, and CLI.

    Precedence
    ----------
        CLI > ENV > Config

    Args
    ----
        config: Training config with defaults.
        cli_ovr_enable: Optional boolean from CLI (--enable_ovr or
            --disable_ovr).
        cli_ovr_classes: Optional list of class names from CLI (--ovr_classes).

    Returns:
        (enable_ovr: bool, ovr_classes: Optional[set[str]])
    """

    valid_set = set(FIVE_SUPERCLASSES)

    def _validate_strict(names: Optional[List[str]], source: str) -> List[str]:
        """
        Normalize, de-duplicate, and enforce strict membership.
        If any unknowns are found, exit with a clear message.
        """
        if not names:
            return []
        cleaned = [str(x).strip() for x in names if str(x).strip()]
        # de-duplicate preserving order
        unique, seen = [], set()
        for c in cleaned:
            if c not in seen:
                unique.append(c)
                seen.add(c)

        bad = [c for c in unique if c not in valid_set]
        if bad:
            print(
                f"\nError: unknown OvR class(es) from {source}: {bad}."
                f"\nValid options: {sorted(valid_set)}",
                file=sys.stderr,
            )
            sys.exit(1)

        return unique

    # 1) Start from config
    enable_ovr = bool(getattr(config, "plots_enable_ovr", False))
    ovr_classes = getattr(config, "plots_ovr_classes", []) or []
    if not isinstance(ovr_classes, list):
        ovr_classes = []
    ovr_classes = _validate_strict(ovr_classes, "config")
    if ovr_classes and not enable_ovr:
        enable_ovr = True

    # 2) ENV overrides
    env_classes = os.getenv("ECG_PLOTS_OVR_CLASSES")
    if env_classes is not None:
        if env_classes.strip() == "":
            print(
                "\nError: empty OvR class list provided via envioronment."
                "\nUse ECG_PLOTS_ENABLE_OVR=1 to enable for all classes "
                "or set ECG_PLOTS_OVR_CLASSES=CLASS1,CLASS2 (etc.) for a subset.",
                file=sys.stderr,
            )
            sys.exit(1)
        env_list = [c.strip() for c in env_classes.split(",")]  # if env_classes else []
        env_list = _validate_strict(env_list, "environment")
        # reached only if non-empty AND all valid
        # if env_list:
        ovr_classes = env_list
        enable_ovr = True

    env_enable = os.getenv("ECG_PLOTS_ENABLE_OVR")
    if env_enable is not None:
        enable_ovr = env_enable.strip().lower() in {"1", "true", "yes"}

    # 3) CLI overrides
    if cli_ovr_classes is not None:

        if not cli_ovr_classes:
            print(
                "\nError: empty OvR class list providaed by CLI."
                "\nUse --enable_ovr for all classes or "
                "--ovr_classes CLASS1,CLASS2 (ect.) for a subset.",
                file=sys.stderr,
            )
            sys.exit(1)

        cli_clean = _validate_strict(cli_ovr_classes, "CLI")
        ovr_classes = cli_clean
        if cli_clean and cli_ovr_enable is None:
            enable_ovr = True

    if cli_ovr_enable is not None:
        enable_ovr = cli_ovr_enable
        if cli_ovr_enable is False:
            ovr_classes = []

    return enable_ovr, (set(ovr_classes) if ovr_classes else None)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main(
    enable_ovr: Optional[bool] = None,
    ovr_classes: Optional[set[str]] = None,
    fold_override: Optional[int] = None,
) -> None:
    """
    Entry point for evaluating the latest training run.

    Steps:
      * Load newest config_* file and extract TrainConfig + extras (tag/fold).
      * Load dataset per config settings (subsample/sampling_rate).
      * Choose best summary entry (lowest loss unless fold is forced).
      * Restore the trained model for that entry and run inference.
      * Print a classification report and call evaluate_and_plot to save
        artifacts.

    Args:
        fold_override: Optional 1-based fold index (overrides any fold in
        config/summary).

    Raises:
        FileNotFoundError / ValueError on missing/malformed artifacts.
    """
    t0 = time.time()

    # Find & load the newest config file
    config_path = _latest_config_path()
    if config_path is None:
        print(
            f"No training configs found in {RESULTS_DIR}.\n"
            "Run train.py first or pass --config <path>."
        )
        sys.exit(1)
    print(f"Loading config from: {config_path}")
    config, extras = _load_config_and_extras(config_path, fold_override)

    if not getattr(config, "batch_size", None):
        raise ValueError("Config is missing required field 'batch_size'.")
    if not getattr(config, "model", None):
        raise ValueError("Config is missing required field 'model'.")
    tag = getattr(config, "tag", None) or extras.get("tag")
    if not tag:
        raise ValueError("Config is missing 'tag'; cannot locate summaries/models.")

    print(f"Config file contents: {config}")
    print(f"extra contents: {extras}")

    print(f"    enable_ovr: {enable_ovr}, ovr_classes: {ovr_classes}")

    # Resolve OvR flags
    enable_ovr_cfg, ovr_classes_set = _resolve_ovr_flags(
        config, enable_ovr, ovr_classes
    )

    print(f"    enable_ovr_cfg: {enable_ovr_cfg}, ovr_classes_set: {ovr_classes_set}")

    # Load data for evaluation (same settings as training)
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

    # Encode labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_tensor = torch.tensor(y_encoded).long()
    X_tensor = torch.tensor(X).float()
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Load and pick best summary entry (or forced fold)
    summaries = _read_summary(tag)
    best = _select_best_entry(summaries, fold_override=fold_override)

    # Extract model path / fold / epoch from the chosen entry
    best_model_path = Path(best.get("model_path", ""))
    if not best_model_path:
        raise ValueError("Chosen summary entry lacks 'model_path'.")
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {best_model_path}")

    best_fold = best.get("fold")
    if fold_override is not None:
        best_fold = fold_override

    best_epoch = best.get("best_epoch")

    print(
        f"Evaluating best model from fold {best_fold if best_fold is not None else 'N/A'} "
        f"(epoch {best_epoch})"
    )
    print(f"Loading model: {best_model_path.name}")

    # Build & restore the model
    if config.model not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model '{config.model}'. Add it to ecg_cnn.models.MODEL_CLASSES."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_CLASSES[config.model]
    model = model_cls(num_classes=len(FIVE_SUPERCLASSES)).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Inference
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    all_targets: List[int] = []

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

    avg_loss = total_loss / len(dataset) if len(dataset) > 0 else float("nan")
    print(f"Eval loss: {avg_loss:.4f}")
    print(f"Evaluation completed in {(time.time() - t0) / 60:.2f} minutes.")

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_probs = np.array(all_probs, dtype=np.float32)

    print(f"best_fold:  {best_fold}")
    print(f"tag:  {tag}")
    train_accs, val_accs, train_loss, val_loss = _load_history(tag, best_fold)

    y_true_ep = [int(x) for x in y_true]
    y_pred_ep = [int(x) for x in y_pred]

    print(f"train_accs: {train_accs}")
    print(f"val_accs: {val_accs}")
    print(f"train_loss: {train_loss}")
    print(f"val_loss: {val_loss}")
    print(f"best_fold: {best_fold}")

    # Final composite plots + report
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
        enable_ovr=enable_ovr_cfg,
        ovr_classes=ovr_classes_set,
    )

    print(f"Elapsed time: {(time.time() - t0) / 60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the most recent training config (or a forced fold)."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional 1-based fold index. If omitted, selects the best by lowest loss.",
    )

    # Enalbe flag viea mutually exclusive group
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--enable_ovr",
        dest="enable_ovr",
        action="store_const",
        const=True,
        default=None,  # None = no CLI override; defer to config or env
        help="Enable one-vs-rese plots. Overrides config and env.",
    )
    group.add_argument(
        "--disable_ovr",
        dest="enable_ovr",
        action="store_const",
        const=False,
        help="Disable one-vs-rest plots. Overrieds config and env.",
    )
    parser.add_argument(
        "--ovr_classes",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="Comma-separated class names for OvR analysis (e.g., 'MI, NORM')."
        "Implies --enable_ovr unless --disable_ovr is set.",
    )
    args = parser.parse_args()
    main(
        enable_ovr=args.enable_ovr,
        ovr_classes=args.ovr_classes,
        fold_override=args.fold,
    )
