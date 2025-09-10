#!/usr/bin/env python

# evaluate.py

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
python -m ecg_cnn.evaluate --enable_ovr # produces OvR plots for 5 classes

# SHAP controls:
python -m ecg_cnn.evaluate --shap fast
python -m ecg_cnn.evaluate --shap medium
python -m ecg_cnn.evaluate --shap thorough
python -m ecg_cnn.evaluate --shap off
python -m ecg_cnn.evaluate --shap custom --shap-n 12 --shap-bg 12 --shap-stride 3
"""

import argparse
import json
import numpy as np
import os
import re
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

# from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ecg_cnn.config.config_loader import load_training_config, TrainConfig
from ecg_cnn.data.data_utils import (
    load_ptbxl_full,
    load_ptbxl_sample,
    FIVE_SUPERCLASSES,
)
from ecg_cnn.models import MODEL_CLASSES
from ecg_cnn.paths import (
    ARTIFACTS_DIR,
    HISTORY_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    PTBXL_DATA_DIR,
    RESULTS_DIR,
)
from ecg_cnn.training.cli_args import parse_evaluate_args
from ecg_cnn.utils.plot_utils import (
    evaluate_and_plot,
    shap_sample_background,
    shap_compute_values,
    shap_save_channel_summary,
    save_classification_report_csv,
    save_fold_summary_csv,
)

print(">>> USING LOCAL ecg_cnn.evaluate FROM:", __file__)

# ------------------------------------------------------------------------------
# globals
# ------------------------------------------------------------------------------
SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PTBXL_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# helpers
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


def _as_1d_label_list(y):
    """
    Normalize labels to a flat 1-D list of scalars suitable for scikit-learn.

    Parameters
    ----------
    y : array-like
        Input labels. May be a Python list, NumPy array, Pandas Series, or a
        nested structure such as a list of 1-element arrays.

    Returns
    -------
    list
        A Python list of scalar values with shape (n,).

    Raises
    ------
    ValueError
        If the input cannot be reshaped into a 1-D sequence of scalars.
    """
    arr = np.asarray(y, dtype=object)

    # Handle column vectors (n,1) by flattening
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)

    # Fallback: squeeze any singleton dimensions
    elif arr.ndim > 1:
        arr = np.squeeze(arr)

    # Final check: must be 1-D
    if arr.ndim != 1:
        raise ValueError(
            f"Expected labels to be 1-D after normalization, got shape {arr.shape}"
        )

    # Convert to list of scalars (unwrap 1-element containers)
    out = []
    for item in arr:
        if isinstance(item, (list, np.ndarray)):
            x = np.asarray(item, dtype=object).reshape(-1)
            if x.size == 0:
                raise ValueError("Encountered empty label container")
            out.append(x[0])
        else:
            out.append(item)

    return out


# ------------------------------------------------------------------------------
# SHAP stability/uncertainty report (validated; no imports inside functions)
# ------------------------------------------------------------------------------


def _np_from_tensor(x, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(x)}")


def _validate_3d(name: str, arr: np.ndarray) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray after conversion")
    if arr.ndim != 3:
        raise ValueError(
            f"{name} must have shape (N, C, T); got ndim={arr.ndim} and shape={arr.shape}"
        )
    N, C, T = arr.shape
    if N <= 0 or C <= 0 or T <= 1:
        raise ValueError(f"{name} invalid shape {arr.shape}; need N>0, C>0, T>1")


def _shap_stability_report(sv, *, class_names: Optional[Sequence[str]] = None) -> str:
    """
    Make a numeric stability report for SHAP channel importances.

    Parameters
    ----------
    sv : list of (N, C, T) arrays OR a single (N, C, T) array
         - Multiclass: list length=K, each (N, C, T)
         - Binary:     single (N, C, T)
         Elements may be numpy arrays or torch tensors.

    class_names : optional list of class labels; unused in math, only for context.

    Returns
    -------
    str
        A small ranked table with mean±SEM per channel and a stability flag:
          STABLE (CV < 0.25), OK (0.25-0.50), NOISY (>= 0.50)

    Raises
    ------
    TypeError / ValueError
        If `sv` is the wrong type or has inconsistent/invalid shapes.
    """
    # Normalize to a single ndarray of shape (N, C, T) with |SHAP| averaged across classes
    if isinstance(sv, list):
        if len(sv) == 0:
            raise ValueError("sv is an empty list.")
        sv_np = []
        base_shape = None
        for i, s in enumerate(sv):
            s_np = _np_from_tensor(s, f"sv[{i}]")
            _validate_3d(f"sv[{i}]", s_np)
            if base_shape is None:
                base_shape = s_np.shape
            elif s_np.shape != base_shape:
                raise ValueError(
                    f"sv[{i}] shape {s_np.shape} != sv[0] shape {base_shape}"
                )
            sv_np.append(np.abs(s_np))
        sv_abs = np.mean(np.stack(sv_np, axis=0), axis=0)  # (N, C, T)
    else:
        sv_abs = _np_from_tensor(sv, "sv")
        _validate_3d("sv", sv_abs)
        sv_abs = np.abs(sv_abs)  # (N, C, T)

    # Per-sample, per-channel mean over time: (N, C)
    imp_per_sample = sv_abs.mean(axis=2)
    N_eff, C = imp_per_sample.shape

    # Means and dispersion across samples
    ddof = 1 if N_eff > 1 else 0
    imp_mean = imp_per_sample.mean(axis=0)  # (C,)
    imp_std = imp_per_sample.std(axis=0, ddof=ddof)  # (C,)
    imp_sem = imp_std / np.sqrt(max(N_eff, 1))  # (C,)
    with np.errstate(divide="ignore", invalid="ignore"):
        imp_cv = np.where(imp_mean > 0, imp_std / np.maximum(imp_mean, 1e-12), np.inf)

    # Rank channels by mean importance
    order = np.argsort(-imp_mean)

    # Build table
    lines = []
    lines.append("SHAP channel importance (mean±SEM), stability")
    lines.append("ch\tmean\t\tSEM\t\tCV\tflag")
    for ch in order:
        mean = float(imp_mean[ch])
        sem = float(imp_sem[ch])
        cv = float(imp_cv[ch])
        flag = "STABLE" if cv < 0.25 else ("OK" if cv < 0.5 else "NOISY")
        lines.append(f"{ch:02d}\t{mean:.4g}\t{sem:.4g}\t{cv:.2f}\t{flag}")

    # Guidance focusing on the top-3 channels
    top3 = order[:3]
    high_cv = [int(c) for c in top3 if imp_cv[c] >= 0.5]
    if high_cv:
        lines.append(
            f"\nTop channels {high_cv} flagged NOISY (CV ≥ 0.5). "
            "Increase --shap-n (more samples) first; if still noisy, raise --shap-bg. "
            "Lower --shap-stride for finer temporal detail (slower)."
        )
    else:
        lines.append("\nTop channels look stable (CV < 0.5).")

    return "\n".join(lines)


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------
def main(
    *,
    parsed_args: argparse.Namespace | None = None,
    enable_ovr: Optional[bool] = None,
    ovr_classes: Optional[set[str]] = None,
    fold_override: Optional[int] = None,
    prefer: str | None = None,
    shap_profile: str = "medium",
    shap_n: Optional[int] = None,
    shap_bg: Optional[int] = None,
    shap_stride: Optional[int] = None,
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
      * Optionally compute a SHAP channel-importance summary with bounded cost.

    Args:
        parsed_args: Optional passed in arguments
        enable_ovr: Optional boolen for whether one-versus-rest plots are made
        ovr_classes: Optional value for which list of classes for which to do
            one-versus-rest plots
        fold_override: Optional 1-based fold index (overrides any fold in config/summary)
        shap_profile: off|fast|medium|thorough|custom
        shap_n, shap_bg, shap_stride: used when shap_profile='custom'

    Raises:
        FileNotFoundError / ValueError on missing/malformed artifacts.
    """
    start_time = time.time()  # <— job timer (don’t shadow this)
    print(">>> USING LOCAL ecg_cnn.evaluate FROM main():", __file__)

    # --- normalize args into a single Namespace ---
    if parsed_args is not None:
        args = parsed_args
    else:
        # If any override is provided, build a Namespace from overrides.
        if any(
            v is not None
            for v in (
                enable_ovr,
                ovr_classes,
                fold_override,
                prefer,
                shap_profile,
                shap_n,
                shap_bg,
                shap_stride,
            )
        ):
            args = argparse.Namespace(
                enable_ovr=enable_ovr,
                ovr_classes=ovr_classes,
                fold=fold_override,
                prefer=prefer or "auto",
                shap_profile=shap_profile or "medium",
                shap_n=shap_n,
                shap_bg=shap_bg,
                shap_stride=shap_stride,
            )
        else:
            # Normal CLI path
            args = parse_evaluate_args()

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

    # Prefer sample bundle when requested, else try full PTB-XL and fall back.
    use_sample = bool(getattr(config, "sample_only", False))

    if use_sample:
        X, y, meta = load_ptbxl_sample(
            sample_dir=config.sample_dir,
            ptb_path=None,
            sample_only=True,  # force sample-mode
        )
    else:
        try:
            X, y, meta = load_ptbxl_full(
                data_dir=config.data_dir,
                # sample_dir=config.sample_dir,
                sampling_rate=config.sampling_rate,
                subsample_frac=config.subsample_frac,
            )
        except FileNotFoundError:
            # Graceful fallback so hiring managers can run without the 5GB dataset.
            print("PTB-XL not found; falling back to bundled sample CSVs.")
            X, y, meta = load_ptbxl_sample(
                sample_dir=config.sample_dir,
                ptb_path=None,
                sample_only=True,
            )

    # # Filter unknown labels with y normalized to 1-D
    # y_arr = np.asarray(y).reshape(-1)  # ensure shape (n,)
    # keep = y_arr != "Unknown"

    # Filter unknown labels and normalize y to a flat 1-D list of scalars
    y_list = _as_1d_label_list(y)  # guarantees plain scalars in a 1-D sequence
    y_arr = np.asarray(y_list, dtype=object)
    keep = y_arr != "Unknown"

    X = X[keep]
    y_list = y_arr[keep].tolist()  # still a flat 1-D list of scalars
    meta = meta.loc[keep].reset_index(drop=True)

    # Encode labels to integers (sklearn-free; prevents column-vector warnings)
    classes_, y_encoded = np.unique(
        np.asarray(y_list, dtype=object), return_inverse=True
    )
    y_encoded = y_encoded.astype(int)

    y_tensor = torch.tensor(y_encoded, dtype=torch.long).view(-1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Decide which run to evaluate based on --prefer
    best = None  # dict with keys like model, model_path, fold, best_epoch

    if args.prefer in ("accuracy", "loss", "auto"):
        # Use the most recent "really_the_best_*.json" to decide
        best_json = max(
            RESULTS_DIR.glob("really_the_best_*.json"),
            key=lambda p: p.stat().st_mtime,
            default=None,
        )
        if best_json is not None:
            payload = json.loads(best_json.read_text())
            by_acc = payload.get("by_accuracy")
            by_loss = payload.get("by_loss")

            if args.prefer == "accuracy":
                best = by_acc or by_loss
            elif args.prefer == "loss":
                best = by_loss or by_acc
            else:  # auto: prefer accuracy if present, else loss
                best = by_acc or by_loss

            if best is not None:
                print(f"[prefer={args.prefer}] Using selection from {best_json.name}")

    elif args.prefer == "latest":
        # Ignore best.json; pick newest checkpoint directly
        latest_ckpt = max(
            MODELS_DIR.glob("model_best_*_fold*.pth"),
            key=lambda p: p.stat().st_mtime,
            default=None,
        )
        if latest_ckpt is not None:
            # Derive tag/fold from filename
            m = re.search(r"model_best_(.+?)_fold(\d+)\.pth$", latest_ckpt.name)
            chosen_tag = m.group(1) if m else None
            fold_idx = int(m.group(2)) if m else None

            # Pull minimal info from the matching history file
            be = val_acc = val_loss = train_acc = train_loss = None
            if chosen_tag and fold_idx is not None:
                hist_path = HISTORY_DIR / f"history_{chosen_tag}_fold{fold_idx}.json"
                if hist_path.exists():
                    h = json.loads(hist_path.read_text())
                    be = h.get("best_epoch")
                    val_acc = h.get("val_acc")
                    val_loss = h.get("val_loss")
                    train_acc = h.get("train_acc")
                    train_loss = h.get("train_loss")

            # Try to extract model name from tag; fall back to config.model
            model_name = None
            if chosen_tag:
                parts = chosen_tag.split("_")
                if len(parts) >= 2:
                    model_name = parts[1]

            best = {
                "model": model_name or config.model,
                "model_path": str(latest_ckpt),
                "fold": fold_idx,
                "best_epoch": be,
                "val_accs": val_acc,
                "val_losses": val_loss,
                "train_accs": train_acc,
                "train_losses": train_loss,
            }
            print(f"[prefer=latest] Using newest checkpoint: {latest_ckpt.name}")

    # Fallback: no best JSON, or missing entry, or latest failed
    if best is None:
        summaries = _read_summary(tag)
        best = _select_best_entry(summaries, fold_override=fold_override)
        print(f"[prefer={args.prefer}] Using fallback selection from summary")

    # Extract model path / fold / epoch from the chosen entry
    mp = best.get("model_path")
    if not mp:
        raise ValueError("Chosen summary entry lacks 'model_path'.")
    best_model_path = Path(mp)
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {best_model_path}")

    # Fold / epoch from the chosen entry; allow --fold to override
    best_fold = best.get("fold")
    if fold_override is not None:
        best_fold = fold_override
    chosen_fold = int(best_fold) if best_fold is not None else None

    best_epoch = best.get("best_epoch")

    print(
        f"Evaluating best model from fold {chosen_fold if chosen_fold is not None else 'N/A'} "
        f"(epoch {best_epoch})"
    )
    print(f"Loading model: {best_model_path.name}")

    # IMPORTANT: when 'best' comes from really_the_best_*.json, it includes the
    # model class that produced the checkpoint. Use THAT (not config.model).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = best.get("model") or config.model  # prefer chosen entry's model

    if model_name not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model '{model_name}'. Add it to ecg_cnn.models.MODEL_CLASSES."
        )

    model_cls = MODEL_CLASSES[model_name]
    model = model_cls(num_classes=len(FIVE_SUPERCLASSES)).to(device)

    state = torch.load(best_model_path, map_location=device)
    # strict=True ensures we don’t silently load a mismatched architecture
    model.load_state_dict(state, strict=True)
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
    print(f"Evaluation completed in {(time.time() - start_time) / 60:.2f} minutes.")

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_probs = np.array(all_probs, dtype=np.float32)

    train_accs, val_accs, train_loss, val_loss = _load_history(tag, best_fold)

    y_true_ep = [int(x) for x in y_true]
    y_pred_ep = [int(x) for x in y_pred]

    # Final composite plots + report
    evaluate_and_plot(
        y_true=y_true_ep,
        y_pred=y_pred_ep,
        train_accs=train_accs,
        val_accs=val_accs,
        train_losses=train_loss,
        val_losses=val_loss,
        model=model_name,  # config.model,
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

    # --------------------------------------------------------------------------
    # SHAP summary plot (profiled via --shap; 'off' skips entirely)
    # Profiles map to (N, BG, STRIDE)
    #   fast     -> (4,  4, 10)
    #   medium   -> (8,  8, 5)
    #   thorough -> (16, 16, 2)
    #   custom   -> use --shap-n/--shap-bg/--shap-stride
    # --------------------------------------------------------------------------
    try:
        if shap_profile and shap_profile.lower() != "off":
            profiles = {
                "fast": (4, 4, 10),
                "medium": (8, 8, 5),
                "thorough": (16, 16, 2),
            }
            if shap_profile == "custom":
                n = int(shap_n if shap_n is not None else 8)
                bg = int(shap_bg if shap_bg is not None else 8)
                stride = int(shap_stride if shap_stride is not None else 5)
            else:
                if shap_profile not in profiles:
                    print(f"Unknown --shap profile '{shap_profile}', using 'medium'.")
                    shap_profile = "medium"
                n, bg, stride = profiles[shap_profile]

            # fresh loader so we don't rely on an exhausted dataloader
            small_bs = min(max(8, int(getattr(config, "batch_size", 8))), 64)
            small_loader = DataLoader(dataset, batch_size=small_bs, shuffle=False)

            # gather up to n (no_grad only for data collection)
            X_list: List[torch.Tensor] = []
            with torch.no_grad():
                for xb, _ in small_loader:
                    X_list.append(xb.to(device))
                    if sum(x.shape[0] for x in X_list) >= n:
                        break

            if not X_list:
                print("SHAP: no batches available to explain.")
            else:
                # build explain batch and enforce caps
                X_explain = torch.cat(X_list, dim=0)[:n]  # (N, C, T)

                # optional time-axis downsample to bound cost
                if stride > 1:
                    X_explain = X_explain[:, :, ::stride]

                # background from same distribution, capped
                bg_t = shap_sample_background(X_explain, max_background=bg, seed=SEED)

                # debug + timing
                print(
                    f"SHAP profile='{shap_profile}' -> N={X_explain.shape[0]}, "
                    f"C={X_explain.shape[1]}, T={X_explain.shape[2]}, "
                    f"BG={bg_t.shape[0]}, classes=5, device={device}"
                )

                shap_t0 = time.perf_counter()

                # compute attributions (uses GradientExplainer in shap_compute_values)
                sv = shap_compute_values(model, X_explain, bg_t, device=device)

                print(_shap_stability_report(sv, class_names=FIVE_SUPERCLASSES))

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                shap_t1 = time.perf_counter()
                print(
                    f"SHAP finished in {shap_t1 - shap_t0:.2f}s for the above config."
                )

                # save summary (under outputs/plots/)
                fold_str = f"{best_fold}" if best_fold is not None else "NA"
                fname = f"shap_summary_{config.model}_tag{tag}_fold{fold_str}_epoch{best_epoch}.png"
                out = shap_save_channel_summary(sv, X_explain, PLOTS_DIR, fname)
                print(f"Saved SHAP summary: {out}")
        else:
            if shap_profile and shap_profile.lower() == "off":
                print("SHAP disabled (--shap off).")
    except Exception as e:
        # never let SHAP break evaluation
        print(f"SHAP generation skipped/failed: {e}")
    # --------------------------------------------------------------------------

    # Derive fold_id once for report saving/aggregation
    fold_id = None
    if isinstance(best_fold, int) and best_fold > 0:
        fold_id = best_fold
    else:
        m = re.search(r"fold(\d+)", best_model_path.name)
        if m:
            fold_id = int(m.group(1))
    if fold_id is None:
        fold_id = 1

    # Persist CR CSV for this fold (silent) and aggregate (silent).
    if (
        isinstance(y_true, (list, np.ndarray))
        and isinstance(y_pred, (list, np.ndarray))
        and len(y_true) > 0
        and len(y_true) == len(y_pred)
    ):
        save_classification_report_csv(y_true, y_pred, REPORTS_DIR, tag, fold_id)
        save_fold_summary_csv(REPORTS_DIR, tag)
    else:
        print(
            f"Skipping classification report CSV/summary for fold {fold_id}: "
            "no predictions available or length mismatch."
        )

    # Always report elapsed time
    print(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":

    args = parse_evaluate_args()
    main(
        enable_ovr=args.enable_ovr,
        ovr_classes=args.ovr_classes,
        fold_override=args.fold,
        prefer=args.prefer,
        shap_profile=args.shap_profile,
        shap_n=args.shap_n,
        shap_bg=args.shap_bg,
        shap_stride=args.shap_stride,
    )
