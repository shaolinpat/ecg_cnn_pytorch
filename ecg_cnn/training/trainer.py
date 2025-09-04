# training/trainer.py

import csv
import json
import os
import numpy as np
import re
import time
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from ecg_cnn.config.config_loader import TrainConfig
from ecg_cnn.data.data_utils import (
    FIVE_SUPERCLASSES,
    load_ptbxl_sample,
    load_ptbxl_full,
)
from ecg_cnn.models import model_utils
from ecg_cnn.paths import (
    HISTORY_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    PTBXL_DATA_DIR,
)
from ecg_cnn.training import training_utils
from torch.optim.lr_scheduler import OneCycleLR


# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
SEED = 22

# Simple in-process dataset cache to avoid reloading per fold/run.
# Keyed by data-affecting fields only (NO model/hparams here).
_DATASET_CACHE = {}


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch on the given dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.

    dataloader : DataLoader
        Training data loader.

    optimizer : torch.optim.Optimizer
        Optimizer used for updating model weights.

    criterion : torch.nn.Module
        Loss function.

    device : torch.device
        Device to perform training on ('cpu' or 'cuda').

    Returns
    -------
     tuple of (float, float)
        Training loss and accuracy for the epoch.

    Raises
    ------
    ValueError
        If any argument has an invalid value or type.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise TypeError("dataloader must be a torch.utils.data.DataLoader")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if not callable(criterion):
        raise TypeError("criterion must be callable (e.g., a loss function)")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")

    model.train()
    total_loss = 0.0

    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        # X_batch = X_batch.to(device)
        # y_batch = y_batch.to(device)

        # [perf] non_blocking only matters when pin_memory=True and device is CUDA
        X_batch = X_batch.to(device, non_blocking=True)  # [perf]
        y_batch = y_batch.to(device, non_blocking=True)  # [perf]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # OneCycle is stepped per-batch; others are epoch-level.
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        total_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_on_validation(model, dataloader, criterion, device):
    """
    Evaluate the model on a validation set.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    dataloader : DataLoader
        Validation data loader.

    criterion : torch.nn.Module
        Loss function.

    device : torch.device
        Computation device.


    Returns
    -------
    tuple of (float, float)
        Validation loss and accuracy.

    Raises
    ------
    ValueError
        If any argument has an invalid value or type.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise TypeError("dataloader must be a torch.utils.data.DataLoader")
    if not callable(criterion):
        raise TypeError("criterion must be callable (e.g., a loss function)")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # X_batch = X_batch.to(device)
            # y_batch = y_batch.to(device)

            # [perf] non_blocking only matters when pin_memory=True and device is CUDA
            X_batch = X_batch.to(device, non_blocking=True)  # [perf]
            y_batch = y_batch.to(device, non_blocking=True)  # [perf]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def run_training(
    config: TrainConfig, fold_idx: Optional[int] = None, tag: Optional[str] = None
) -> dict:
    """
    Run training for one configuration. Supports optional fold-based splitting
    and config-specific filename tagging to prevent overwrites during grid
    search.

    Parameters
    ----------
    config : TrainConfig
        Parsed training configuration including model, hyperparams, and fold
        info.

    fold_idx : int, optional
        Fold index for cross-validation. If None, no fold split is used.

    tag : str, optional
        Unique identifier for the current config (e.g., model_lr_bs_wd) to
        disambiguate saved model and history files. Required to prevent filename
        collisions.

    Returns
    -------
    dict
        Summary statistics from the training run (loss, time, etc.).

    Raises
    ------
    ValueError
        If any argument has an invalid value or type.
    """
    # --------------------------------------------------------------------------
    # Input validation (duck-typed to support test doubles)
    # --------------------------------------------------------------------------

    # First: strict type check so tests matching "TrainConfig" pass
    if not isinstance(config, TrainConfig):
        raise TypeError("cfg must be TrainConfig")

    # Second: sanity-check required attributes (catches malformed configs)
    required = [
        "model",
        "lr",
        "batch_size",
        "weight_decay",
        "n_epochs",
        "save_best",
        "sample_only",
        "subsample_frac",
        "sampling_rate",
        "data_dir",
        "sample_dir",
        "n_folds",
        "verbose",
    ]
    missing = [k for k in required if not hasattr(config, k)]
    if missing:
        raise TypeError(f"config is missing required fields: {missing}")

    if fold_idx is not None:
        if not isinstance(fold_idx, int) or fold_idx < 0:
            raise ValueError("fold_idx must be a non-negative integer if provided.")
        if not isinstance(config.n_folds, int) or config.n_folds < 2:
            raise ValueError(
                "config.n_folds must be an integer >= 2 when fold_idx is used."
            )
        if fold_idx >= config.n_folds:
            raise ValueError(
                f"fold_idx {fold_idx} is out of range for {config.n_folds} folds."
            )

    if tag is None:
        raise ValueError("Missing tag — must be provided to disambiguate file outputs.")

    # --------------------------------------------------------------------------
    # Start training
    # --------------------------------------------------------------------------

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # Load and preprocess data
    data_dir = Path(config.data_dir) if config.data_dir else PTBXL_DATA_DIR

    # ----------------------------------------------------------------------
    # Cache dataset in-process so folds/runs reuse the same arrays
    # Key includes only data-affecting fields (not model/hparams).
    # ----------------------------------------------------------------------
    if config.sample_only:
        key = (
            str(Path(config.sample_dir).resolve()) if config.sample_dir else "None",
            "SAMPLE_ONLY",
            int(config.sampling_rate),
        )
        cached = _DATASET_CACHE.get(key)
        if cached is not None:
            X, y, meta = cached
        else:
            # NOTE: tests expect these exact named args; do not change them here.
            X, y, meta = load_ptbxl_sample(
                sample_dir=config.sample_dir,
                ptb_path=data_dir,
            )
            _DATASET_CACHE[key] = (X, y, meta)
    else:
        key = (
            str(data_dir.resolve()),
            float(config.subsample_frac),
            int(config.sampling_rate),
        )
        cached = _DATASET_CACHE.get(key)
        if cached is not None:
            X, y, meta = cached
        else:
            X, y, meta = load_ptbxl_full(
                data_dir=data_dir,
                subsample_frac=config.subsample_frac,
                sampling_rate=config.sampling_rate,
            )
            _DATASET_CACHE[key] = (X, y, meta)

    # Drop unknowns
    keep = np.array([lbl != "Unknown" for lbl in y], dtype=bool)
    X = X[keep]
    y = [lbl for i, lbl in enumerate(y) if keep[i]]
    meta = meta.loc[keep].reset_index(drop=True)

    # Encode targets
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y_encoded).long()

    # --------------------------------------------------------------------------
    # Fold-based split (optional)
    # --------------------------------------------------------------------------
    if fold_idx is not None:

        skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(X_tensor, y_tensor))

        train_idx, val_idx = splits[fold_idx]
        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

        # pin_memory is a no-op on CPU and speeds H2D transfer on CUDA
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=use_cuda,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            pin_memory=use_cuda,
        )
        print(
            f"Fold {fold_idx + 1} of {config.n_folds}: {len(train_idx)} train / {len(val_idx)} val samples"
        )

    else:
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=use_cuda,
        )
        val_dataloader = None
        # print(f"No folds: {len(dataset)} total samples")

    # --------------------------------------------------------------------------
    # Model and optimizer
    # --------------------------------------------------------------------------
    # Use getattr(..., None) so we can raise a clean ValueError for tests
    model_cls = getattr(model_utils, config.model, None)
    if model_cls is None:
        raise ValueError(f"Unknown model name: {config.model}")
    model = model_cls(num_classes=len(FIVE_SUPERCLASSES)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # --------------------------------------------------------------------------
    # Loss: class-weighted CrossEntropy (weights from training fold)
    # --------------------------------------------------------------------------
    y_src = y_tensor[train_idx] if fold_idx is not None else y_tensor
    y_train_np = y_src.numpy()
    num_classes = len(np.unique(y_train_np))
    # training_utils.compute_class_weights returns a torch.Tensor; just move to device
    class_weights = training_utils.compute_class_weights(y_train_np, num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --------------------------------------------------------------------------
    # Scheduler selection (default: ReduceLROnPlateau; optional Cosine/OneCycle via ENV)
    #   ECG_SCHEDULER: "", "cosine", or "onecycle"
    #   ECG_SCHED_TMAX: int (cosine; default = n_epochs)
    #   ECG_SCHED_MAX_LR: float (onecycle; default = config.lr)
    #   ECG_SCHED_PCT_START: float (onecycle; default 0.3)
    #   ECG_SCHED_DIV: float (onecycle; default 25.0)
    #   ECG_SCHED_FINAL_DIV: float (onecycle; default 1e4)
    # --------------------------------------------------------------------------
    _sched_name = os.getenv("ECG_SCHEDULER", "").strip().lower()
    if _sched_name == "cosine":
        _tmax = int(os.getenv("ECG_SCHED_TMAX", str(config.n_epochs)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_tmax)
    elif _sched_name == "onecycle":
        _max_lr = float(os.getenv("ECG_SCHED_MAX_LR", str(config.lr)))
        _pct = float(os.getenv("ECG_SCHED_PCT_START", "0.3"))
        _div = float(os.getenv("ECG_SCHED_DIV", "25.0"))
        _final = float(os.getenv("ECG_SCHED_FINAL_DIV", "10000"))
        steps_per_epoch = len(dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=_max_lr,
            epochs=int(config.n_epochs),
            steps_per_epoch=steps_per_epoch,
            pct_start=_pct,
            div_factor=_div,
            final_div_factor=_final,
        )
    else:
        # Scheduler (ReduceLROnPlateau) — no 'verbose' kw for this torch version
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )

    best_loss = float("inf")
    best_epoch = -1
    model_path = None  # ensure defined even if no improvement

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    # --------------------------------------------------------------------------
    # Training loop with early stopping
    # --------------------------------------------------------------------------
    best_state = None
    bad_epochs = 0
    patience = 5

    for epoch in range(config.n_epochs):
        print(f"Epoch {epoch + 1}/{config.n_epochs}")

        # --- Train epoch ---
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            # Inline training for OneCycle (batch-level stepping); keep train_one_epoch signature intact
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(device, non_blocking=True)  # [perf]
                y_batch = y_batch.to(device, non_blocking=True)  # [perf]

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()  # batch-level

                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            train_loss = total_loss / total
            train_acc = correct / total
        else:
            # Original path (uses your existing helper)
            train_loss, train_acc = train_one_epoch(
                model, dataloader, optimizer, criterion, device
            )

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # --- Validate ---
        if val_dataloader:
            val_loss, val_acc = evaluate_on_validation(
                model, val_dataloader, criterion, device
            )
            print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        else:
            val_loss, val_acc = train_loss, train_acc

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Monitor validation loss when available
        monitored = val_loss if val_dataloader else train_loss

        # Step scheduler:
        # - ReduceLROnPlateau uses monitored metric (val/train loss)
        # - Cosine steps each epoch (no metric)
        # - OneCycle was stepped per-batch above
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(monitored)
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # Save best and reset early-stopping counter
        if config.save_best and monitored < best_loss:

            best_loss = monitored
            best_epoch = epoch + 1
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            fname = (
                f"model_best_{tag}_fold{fold_idx + 1}.pth"
                if fold_idx is not None
                else f"model_best_{tag}.pth"
            )
            model_path = MODELS_DIR / fname
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to: {model_path}")

            bad_epochs = 0
            # cache best weights in memory (so last epoch need not be best)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}; best loss={best_loss:.4f}")
                break

    # Restore best weights if cached
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save history
    if fold_idx is not None:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        history_path = HISTORY_DIR / f"history_{tag}_fold{fold_idx + 1}.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to: {history_path}")

        # ----------------------------------------------------------------------
        # Persist classification report CSV for this fold (parallel to history)
        # ----------------------------------------------------------------------
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        _m = re.search(r"fold(\d+)", history_path.name)
        _fold_num = int(_m.group(1)) if _m else (fold_idx + 1)

        if val_dataloader is not None:
            _y_true, _y_pred = [], []
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_dataloader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    logits = model(X_batch)
                    _, predicted = torch.max(logits, 1)
                    _y_true.extend(y_batch.cpu().numpy().tolist())
                    _y_pred.extend(predicted.cpu().numpy().tolist())

            _cr = classification_report(
                _y_true, _y_pred, output_dict=True, zero_division=0
            )
            _out_csv = (
                ARTIFACTS_DIR / f"classification_report_{tag}_fold{_fold_num}.csv"
            )
            with _out_csv.open("w", newline="") as _fh:
                _w = csv.writer(_fh)
                _w.writerow(["label", "precision", "recall", "f1-score", "support"])
                for _label, _stats in _cr.items():
                    if isinstance(_stats, dict):
                        _w.writerow(
                            [
                                _label,
                                _stats.get("precision"),
                                _stats.get("recall"),
                                _stats.get("f1-score"),
                                _stats.get("support"),
                            ]
                        )
            print(f"Saved classification report to: {_out_csv}")
        # ----------------------------------------------------------------------

    elapsed_min = (time.time() - t0) / 60

    # --- enforce contract in summary: no-fold => val_* is None ---
    val_loss_summary = None if val_dataloader is None else val_loss
    val_acc_summary = None if val_dataloader is None else val_acc

    summary = {
        "loss": best_loss,
        "elapsed_min": elapsed_min,
        "fold": fold_idx + 1 if fold_idx is not None else None,
        "model": model.__class__.__name__,
        "model_path": str(model_path) if config.save_best else None,
        "best_epoch": best_epoch,
        "train_losses": train_loss,
        "val_losses": val_loss_summary,
        "train_accs": train_acc,
        "val_accs": val_acc_summary,
    }

    return summary
