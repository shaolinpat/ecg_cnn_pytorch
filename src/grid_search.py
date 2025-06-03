



# src/grid_search.py

import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from itertools import product
from model_utils import ECGConvNet
from plot_utils import format_hparams, save_plot_curves, save_confusion_matrix
#from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------------------------------------------
# INSERT BELOW: GLOBAL SEED & WORKER-INIT (must match main's SEED)
# -------------------------------------------------------------------
SEED = 22

# Seed Python’s built-in random
random.seed(SEED)

# Seed NumPy
np.random.seed(SEED)

# Seed PyTorch CPU
torch.manual_seed(SEED)

# If using CUDA, seed all GPUs
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make cuDNN deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# Define worker_init_fn for DataLoader workers
def seed_worker(worker_id):
    """
    Called once per DataLoader worker subprocess.
    We give each worker a seed of (SEED + worker_id).
    """
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
# -------------------------------------------------------------------
# END INSERT
# -------------------------------------------------------------------





def run_manual_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    unique_labels: list[str],
    num_classes: int,
    out_folder: str,
    model_out_folder: str,
    device: torch.device,
    param_grid: dict
) -> (pd.DataFrame, dict):
    """
    Performs a manual hyperparameter grid search over (epochs, batch_size, lr, weight_decay)
    on the provided X_train/y_train and X_val/y_val. Uses WeightedRandomSampler on the training set
    to mitigate class imbalance, and applies early stopping based on validation loss. For each
    combination:
      - trains until early stopping,
      - saves the best-seen model to `model_out_folder`,
      - calls save_plot_curves(...) and save_confusion_matrix(...) to dump PNGs into `out_folder`.
    Returns:
      - results_df: pandas DataFrame containing one row per hyperparameter combo, with columns
          ["fold", "epochs", "batch_size", "lr", "weight_decay", "val_acc", "model_path"].
      - metrics: a dict mapping (fold, epoch, batch_size, lr, wd) -> 
          {
            "train_losses": [...],
            "val_losses": [...],
            "train_accs": [...],
            "val_accs": [...],
            "y_true": [...],
            "y_pred": [...],
          }
    """

    # combos = list(
    #     np.array(np.meshgrid(
    #         param_grid["epochs"],
    #         param_grid["batch_size"],
    #         param_grid["lr"],
    #         param_grid["weight_decay"]
    #     )).T.reshape(-1, 4)
    # )

    combos = list(product(
        param_grid["epochs"],
        param_grid["batch_size"],
        param_grid["lr"],
        param_grid["weight_decay"]
    ))

    results = []
    metrics = {}

    for combo_idx, (epochs, bs, lr, wd) in enumerate(combos, start=1):
        print(f"=== Combo {combo_idx}/{len(combos)}: epochs={epochs}, bs={bs}, lr={lr}, wd={wd} ===")

        # ------------------------
        # 9C) Build WeightedRandomSampler on y_train
        # ------------------------
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]


        sr_gen = torch.Generator()
        sr_gen.manual_seed(SEED)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=sr_gen
        )

        # ------------------------
        # 9D) Build DataLoaders
        # ------------------------
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            ),
            batch_size=bs,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker 
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            ),
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker 
        )

        # ------------------------
        # 9E) Instantiate model, optimizer, loss
        # ------------------------
        model = ECGConvNet(num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_acc = -1.0
        patience = 5
        patience_counter = 0
        best_state = None
        best_true, best_pred = [], []

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        # ------------------------
        # 9F) Training loop with early stopping
        # ------------------------
        for epoch in range(1, epochs + 1):
            # (a) Training step
            model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct_train += (preds == yb).sum().item()
                total_train += xb.size(0)

            avg_train_loss = running_train_loss / len(train_loader)
            train_acc = correct_train / total_train

            # (b) Validation step
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            epoch_true, epoch_pred = [], []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    vloss = criterion(logits, yb).item()
                    running_val_loss += vloss

                    preds = logits.argmax(dim=1)
                    correct_val += (preds == yb).sum().item()
                    total_val += xb.size(0)

                    epoch_true.extend(yb.cpu().tolist())
                    epoch_pred.extend(preds.cpu().tolist())

            avg_val_loss = running_val_loss / len(val_loader)
            val_acc = correct_val / total_val

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(
                f"  Epoch {epoch}/{epochs}  "
                f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}  "
                f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}"
            )

            # (c) Early stopping logic
            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = model.state_dict()
                best_true, best_pred = epoch_true.copy(), epoch_pred.copy()
                if val_acc > best_acc:
                    best_acc = val_acc
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # ------------------------
        # 9G) Save the best model for this combo
        # ------------------------
        fbase = format_hparams(
            lr=lr,
            bs=bs,
            wd=wd,
            fold=1,
            epochs=epoch,
            prefix="model",
            fname_metric=""
        )
        model_filename = fbase + ".pt"
        model_path = os.path.join(model_out_folder, model_filename)
        torch.save({"model_state_dict": best_state}, model_path)
        print(f"  - Saved model: {model_path}")

        # ------------------------
        # 9H) Plot (loss, acc, confmat) for this combo
        # ------------------------
        # (1) Loss + Accuracy curves
        save_plot_curves(
            x_vals=train_accs,
            y_vals=val_accs,
            x_label="Epoch",
            y_label="Accuracy",
            title_metric="Accuracy",
            fname_metric="accuracy",
            lr=lr,
            bs=bs,
            wd=wd,
            fold=1,
            epochs=epoch,
            out_folder=out_folder,
            prefix="final"
        )
        save_plot_curves(
            x_vals=train_losses,
            y_vals=val_losses,
            x_label="Epoch",
            y_label="Loss",
            title_metric="Loss",
            fname_metric="loss",
            lr=lr,
            bs=bs,
            wd=wd,
            fold=1,
            epochs=epoch,
            out_folder=out_folder,
            prefix="final"
        )

        # (2) Confusion matrix (normalized)
        save_confusion_matrix(
            y_true=best_true,
            y_pred=best_pred,
            class_names=unique_labels,
            lr=lr,
            bs=bs,
            wd=wd,
            fold=1,
            epochs=epoch,
            out_folder=out_folder,
            prefix="final",
            normalize=True
        )

        # ------------------------
        # 9I) Record this combo's results
        # ------------------------
        results.append({
            "fold": 1,
            "epochs": epoch,
            "batch_size": bs,
            "lr": lr,
            "weight_decay": wd,
            "val_acc": best_acc,
            "model_path": model_path
        })
        metrics[(1, epoch, bs, lr, wd)] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "y_true": best_true,
            "y_pred": best_pred,
        }

    # ------------------------
    # 9J) Return results as DataFrame + metrics dict
    # ------------------------
    results_df = pd.DataFrame(results)
    return results_df, metrics

