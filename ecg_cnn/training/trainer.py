import json
import numpy as np
import time
import torch
import torch.nn as nn

from pathlib import Path
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
from ecg_cnn.paths import HISTORY_DIR, MODELS_DIR, PTBXL_DATA_DIR
from ecg_cnn.training.training_utils import compute_class_weights


def train_one_epoch(model, dataloader, optimizer, criterion, device):
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
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

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
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

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
    Run training for one configuration. Supports optional fold-based splitting and
    config-specific filename tagging to prevent overwrites during grid search.

    Parameters
    ----------
    config : TrainConfig
        Parsed training configuration including model, hyperparams, and fold info.

    fold_idx : int, optional
        Fold index for cross-validation. If None, no fold split is used.

    tag : str, optional
        Unique identifier for the current config (e.g., model_lr_bs_wd) to disambiguate
        saved model and history files. Required to prevent filename collisions.

    Returns
    -------
    dict
        Summary statistics from the training run (loss, time, etc.).

    Raises
    ------
    ValueError
        If any argument has an invalid value or type.
    """
    # --------------------------------------
    # Input validation
    # --------------------------------------
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be an instance of TrainConfig.")

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
        raise ValueError(
            "Missing `tag` — must be provided to disambiguate file outputs."
        )

    # --------------------------------------
    # Start training
    # --------------------------------------

    t0 = time.time()
    SEED = 22
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    data_dir = Path(config.data_dir) if config.data_dir else PTBXL_DATA_DIR
    if config.sample_only:
        X, y, meta = load_ptbxl_sample(
            sample_dir=config.sample_dir,
            ptb_path=data_dir,
        )
    else:
        X, y, meta = load_ptbxl_full(
            data_dir=data_dir,
            subsample_frac=config.subsample_frac,
            sampling_rate=config.sampling_rate,
        )

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

    # --------------------------------------
    # Fold-based split (optional)
    # --------------------------------------
    if fold_idx is not None:
        # if config.n_folds < 2:
        #     raise ValueError("`n_folds` must be >= 2 if `fold_idx` is provided.")

        skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(X_tensor, y_tensor))

        # if not (0 <= fold_idx < config.n_folds):
        #     raise ValueError(f"Fold index out of range: {fold_idx}")

        train_idx, val_idx = splits[fold_idx]
        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        dataloader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
        print(
            f"Fold {fold_idx + 1} of {config.n_folds}: {len(train_idx)} train / {len(val_idx)} val samples"
        )

    else:
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = None
        print(f"No folds: {len(dataset)} total samples")

    # --------------------------------------
    # Model and optimizer
    # --------------------------------------
    model_cls = getattr(model_utils, config.model, None)
    if model_cls is None:
        raise ValueError(f"Unknown model name: {config.model}")
    model = model_cls(num_classes=len(FIVE_SUPERCLASSES)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    # criterion = nn.CrossEntropyLoss()
    # y_train_np = train_dataset.tensors[1].numpy()
    # num_classes = len(np.unique(y_train_np))
    # class_weights = compute_class_weights(y_train_np, num_classes)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # criterion = nn.CrossEntropyLoss()
    if fold_idx is not None:
        y_src = y_tensor[train_idx]
    else:
        y_src = y_tensor
    y_train_np = y_src.numpy()
    num_classes = len(np.unique(y_train_np))
    class_weights = compute_class_weights(y_train_np, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_loss = float("inf")
    best_epoch = -1

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(config.n_epochs):
        print(f"Epoch {epoch + 1}/{config.n_epochs}")

        # loss = train_one_epoch(model, dataloader, optimizer, criterion, device)

        # print(f"Loss: {loss:.4f}")

        train_loss, train_acc = train_one_epoch(
            model, dataloader, optimizer, criterion, device
        )
        val_loss, val_acc = (None, None)
        if val_dataloader:
            val_loss, val_acc = evaluate_on_validation(
                model, val_dataloader, criterion, device
            )

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        if val_dataloader:
            print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if config.save_best and train_loss < best_loss:
            best_loss = train_loss
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

    # Save history
    if fold_idx is not None:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        history_path = HISTORY_DIR / f"history_{tag}_fold{fold_idx + 1}.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to: {history_path}")

    elapsed = (time.time() - t0) / 60
    summary = {
        "loss": best_loss,
        "elapsed_min": elapsed,
        "fold": fold_idx + 1 if fold_idx is not None else None,
        "model": model.__class__.__name__,
        "model_path": str(model_path) if config.save_best else None,
        "best_epoch": best_epoch,
        "train_losses": train_loss,
        "val_losses": val_loss,
        "train_accs": train_acc,
        "val_accs": val_acc,
    }

    return summary


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse frequency-based class weights for use with nn.CrossEntropyLoss.

    Parameters
    ----------
    y : np.ndarray
        1D array of integer class labels (shape: [n_samples]).
    num_classes : int
        Total number of classes. Must be >= max(y) + 1.

    Returns
    -------
    torch.Tensor
        Tensor of shape [num_classes] with higher weights for minority classes.

    Raises
    ------
    ValueError
        If input is not a valid 1D array of integers, or if num_classes is invalid.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError(f"y must be a numpy ndarray, got {type(y)}")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array of class labels")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must contain integer class labels")
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if y.max() >= num_classes:
        raise ValueError("num_classes must be greater than max(y)")

    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = total / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)


# def run_training(config: TrainConfig, fold_idx: Optional[int] = None) -> dict:
#     """
#     Run training for one configuration. Supports optional fold-based splitting.

#     Parameters
#     ----------
#     config : TrainConfig
#         Parsed training configuration including model, hyperparams, and fold info.

#     fold_idx : int, optional
#         Fold index for cross-validation. If None, no fold split is used.

#     Returns
#     -------
#     dict
#         Summary statistics from the training run (loss, time, etc.).

#     Raises
#     ------
#     ValueError
#         If any argument has an invalid value or type.
#     """
#     # --------------------------------------
#     # Input validation
#     # --------------------------------------
#     if not isinstance(config, TrainConfig):
#         raise TypeError("config must be an instance of TrainConfig.")

#     if fold_idx is not None:
#         if not isinstance(fold_idx, int) or fold_idx < 0:
#             raise ValueError("fold_idx must be a non-negative integer if provided.")
#         if not isinstance(config.n_folds, int) or config.n_folds < 2:
#             raise ValueError(
#                 "config.n_folds must be an integer >= 2 when fold_idx is used."
#             )
#         if fold_idx >= config.n_folds:
#             raise ValueError(
#                 f"fold_idx {fold_idx} is out of range for {config.n_folds} folds."
#             )

#     # --------------------------------------
#     # Start training
#     # --------------------------------------

#     t0 = time.time()
#     SEED = 22
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load and preprocess data
#     data_dir = Path(config.data_dir) if config.data_dir else PTBXL_DATA_DIR
#     if config.sample_only:
#         X, y, meta = load_ptbxl_sample(
#             sample_dir=config.sample_dir,
#             ptb_path=data_dir,
#         )
#     else:
#         X, y, meta = load_ptbxl_full(
#             data_dir=data_dir,
#             subsample_frac=config.subsample_frac,
#             sampling_rate=config.sampling_rate,
#         )

#     # Drop unknowns
#     keep = np.array([lbl != "Unknown" for lbl in y], dtype=bool)
#     X = X[keep]
#     y = [lbl for i, lbl in enumerate(y) if keep[i]]
#     meta = meta.loc[keep].reset_index(drop=True)

#     # Encode targets
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)
#     X_tensor = torch.tensor(X).float()
#     y_tensor = torch.tensor(y_encoded).long()

#     # --------------------------------------
#     # Fold-based split (optional)
#     # --------------------------------------
#     if fold_idx is not None:
#         if config.n_folds < 2:
#             raise ValueError("`n_folds` must be >= 2 if `fold_idx` is provided.")

#         skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
#         splits = list(skf.split(X_tensor, y_tensor))

#         if not (0 <= fold_idx < config.n_folds):
#             raise ValueError(f"Fold index out of range: {fold_idx}")

#         train_idx, val_idx = splits[fold_idx]
#         train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
#         val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
#         dataloader = DataLoader(
#             train_dataset, batch_size=config.batch_size, shuffle=True
#         )
#         val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
#         print(
#             f"Fold {fold_idx + 1} of {config.n_folds}: {len(train_idx)} train / {len(val_idx)} val samples"
#         )

#     else:
#         dataset = TensorDataset(X_tensor, y_tensor)
#         dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
#         val_dataloader = None
#         print(f"No folds: {len(dataset)} total samples")

#     # --------------------------------------
#     # Model and optimizer
#     # --------------------------------------
#     model_cls = getattr(model_utils, config.model, None)
#     if model_cls is None:
#         raise ValueError(f"Unknown model name: {config.model}")
#     model = model_cls(num_classes=len(FIVE_SUPERCLASSES)).to(device)

#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=config.lr, weight_decay=config.weight_decay
#     )
#     criterion = nn.CrossEntropyLoss()

#     best_loss = float("inf")
#     best_epoch = -1

#     history = {
#         "train_loss": [],
#         "val_loss": [],
#         "train_acc": [],
#         "val_acc": [],
#     }

#     for epoch in range(config.n_epochs):
#         print(f"Epoch {epoch + 1}/{config.n_epochs}")

#         # loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
#         # print(f"Loss: {loss:.4f}")

#         train_loss, train_acc = train_one_epoch(
#             model, dataloader, optimizer, criterion, device
#         )
#         val_loss, val_acc = (None, None)
#         if val_dataloader:
#             val_loss, val_acc = evaluate_on_validation(
#                 model, val_dataloader, criterion, device
#             )

#         print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
#         if val_dataloader:
#             print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

#         history["train_loss"].append(train_loss)
#         history["train_acc"].append(train_acc)
#         history["val_loss"].append(val_loss)
#         history["val_acc"].append(val_acc)

#         if config.save_best and train_loss < best_loss:
#             best_loss = train_loss
#             best_epoch = epoch + 1
#             MODELS_DIR.mkdir(parents=True, exist_ok=True)
#             fname = (
#                 f"model_best_fold{fold_idx + 1}.pth"
#                 if fold_idx is not None
#                 else "model_best.pth"
#             )
#             model_path = MODELS_DIR / fname
#             torch.save(model.state_dict(), model_path)
#             print(f"Saved best model to: {model_path}")

#     # Save history
#     if fold_idx is not None:
#         HISTORY_DIR.mkdir(parents=True, exist_ok=True)
#         history_path = HISTORY_DIR / f"history_fold{fold_idx}.json"
#         with open(history_path, "w") as f:
#             json.dump(history, f, indent=2)
#         print(f"Saved training history to: {history_path}")

#     elapsed = (time.time() - t0) / 60
#     summary = {
#         "loss": best_loss,
#         "elapsed_min": elapsed,
#         "fold": fold_idx + 1,
#         "model": model.__class__.__name__,
#         "model_path": str(model_path) if config.save_best else None,
#         "best_epoch": best_epoch,
#         "train_losses": train_loss,
#         "val_losses": val_loss,
#         "train_accs": train_acc,
#         "val_accs": val_acc,
#     }

#     return summary
