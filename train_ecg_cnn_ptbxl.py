# ---
# jupyter:
#   jupytext:
#     formats: train_ecg_cnn_ptbxl.py:percent,train_ecg_cnn_ptbxl.ipynb
#     primary: train_ecg_cnn_ptbxl.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
# # ECG CNN PyTorch Demo
#
# This notebook demonstrates:
#
# 0. Imports
# 1. Setup & Config
# 2. Model Definition
# 3. Helper Functions
# 4. Data Loading & Preprocessing
# 5. Grid Search & Training
# 6. Evaluation & Visualization
# 7. External PTB Validation
# 8. Entry Point & Full Runeline: data loading, (subsample), grid search
#    (targeted), early stopping, evaluation, plotting, PTB validation.

# %% [markdown]
# ## 0. Imports

# %%
import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Optionally enable AMP for speed on GPU:
# from torch.cuda.amp import autocast, GradScaler

# %% [markdown]
# ## 1. Setup & Config

# %%
# Reproducibility
SEED = 22
np.random.seed(SEED)
torch.manual_seed(SEED)

# Thread control
torch.set_num_threads(6)
print(f"Using {torch.get_num_threads()} CPU threads")

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# %% [markdown]
# ## 2. Model Definition


# %%
class ECGConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=6)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(3, 2, 1)
        self.conv2 = nn.Conv1d(64, 64, 3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2, 2, 1)
        self.conv3 = nn.Conv1d(64, 64, 3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2, 2, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 22, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 5)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# %% [markdown]
# ## 3. Helper Functions


# %%
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix"):
    if normalize:
        sums = cm.sum(axis=1, keepdims=True)
        cm = cm.astype(float) / np.where(sums != 0, sums, 1)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = f"{cm[i,j]:.2f}" if normalize else f"{int(cm[i,j])}"
        plt.text(
            j,
            i,
            val,
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


# %% [markdown]
# ## 4. Evaluation & Visualization


# %%
def evaluate_and_plot(
    y_true, y_pred, train_accs, val_accs, lr, bs, fold, epochs, out_folder
):
    print(
        f"\n=== Final Evaluation (LR={lr}, BS={bs}, Fold={fold}, Epochs={epochs}) ==="
    )
    print("Classification Report:")
    print(
        classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0)
    )

    # Accuracy curve
    fig, ax = plt.subplots()
    ax.plot(train_accs, label="Training")
    ax.plot(val_accs, label="Validation")
    ax.set_title(
        f"Model Accuracy by Epoch\nLR={lr}, BS={bs}, Fold={fold}, Epochs={epochs}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Model Accuracy")
    ax.legend()
    path = os.path.join(out_folder, f"final_accuracy_{lr}_{bs}_{fold}_{epochs}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved final accuracy plot to {path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    plot_confusion_matrix(
        cm,
        classes=["N", "S", "V", "F", "Q"],
        normalize=True,
        title=f"Normalized Confusion Matrix\nLR={lr}, BS={bs}, Fold={fold}, Epochs={epochs}",
    )
    path = os.path.join(out_folder, f"final_confmat_{lr}_{bs}_{fold}_{epochs}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved final confusion matrix to {path}")


# %% [markdown]
# ## 5. External PTB Validation


# %%
def evaluate_on_ptb(model_path, normal_csv, abnormal_csv, device, out_folder):
    print(f"\n=== PTB Validation: {model_path} ===")
    ptb_n = pd.read_csv(normal_csv, header=None)
    ptb_a = pd.read_csv(abnormal_csv, header=None)
    df = pd.concat([ptb_n, ptb_a], ignore_index=True).dropna()
    df[187] = df[187].astype(int)
    df = df[df[187].isin([0, 1])]
    X = df.iloc[:, :-1].values.reshape(-1, 1, 187)
    y = df.iloc[:, -1].values
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(y, dtype=torch.long),
        ),
        batch_size=32,
    )
    ckpt = torch.load(model_path, map_location=device)
    model = ECGConvNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            preds.extend(torch.argmax(out, 1).cpu().tolist())
            targets.extend(yb.tolist())
    acc = np.mean(np.array(preds) == np.array(targets))
    print(f"PTB Accuracy: {acc:.4f}")
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots()
    plot_confusion_matrix(
        cm[:2, :2],
        classes=["Normal", "Abnormal"],
        normalize=True,
        title="PTB Confusion Matrix",
    )
    path = os.path.join(out_folder, "ptb_confmat.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved PTB confusion matrix to {path}")


# %% [markdown]
# ## 6. Data Loading & Subsample


# %%
def load_data(data_dir="./data", subsample_frac=0.2):
    print(f"Loading data from {data_dir} and subsampling {subsample_frac*100:.0f}%")
    t = pd.read_csv(os.path.join(data_dir, "mitbih_train.csv"), header=None)
    s = pd.read_csv(os.path.join(data_dir, "mitbih_test.csv"), header=None)
    df = pd.concat([t, s], ignore_index=True).dropna()
    df[187] = df[187].astype(int)
    # subsample
    mask = np.random.choice(len(df), size=int(len(df) * subsample_frac), replace=False)
    df = df.iloc[mask]
    X = df.iloc[:, :-1].values.reshape(-1, 1, df.shape[1] - 1)
    y = df.iloc[:, -1].values
    print(f"Final subsample size: {X.shape[0]}")
    return X, y


# %% [markdown]
# ## 7. Grid Search & Training


# %%
def run_grid_search(X, y, out_folder, folds=5):
    """
    Performs grid search with 5-fold CV.
    Tracks train/val loss & acc, saves best model, and generates plots.
    Returns:
      results_df: DataFrame of best val_acc per run
      metrics: dict key=(fold,epochs,batch_size,lr) -> metric lists & preds/true
    """
    # param_grid = {
    #     "epochs": [5, 10],
    #     "batch_size": [10, 32],
    #     "lr": [1e-3, 5e-4, 1e-4],
    # }
    param_grid = {
        "epochs": [5],
        "batch_size": [10],
        "lr": [1e-3, 5e-4],
    }

    combos = list(
        itertools.product(
            param_grid["epochs"], param_grid["batch_size"], param_grid["lr"]
        )
    )
    print(f"Grid search combos: {len(combos)}")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    results = []
    metrics = {}

    for combo_idx, (epochs, bs, lr) in enumerate(combos, start=1):
        print(
            f"=== Combo {combo_idx}/{len(combos)}: epochs={epochs}, bs={bs}, lr={lr} ==="
        )

        for fold, (ti, vi) in enumerate(skf.split(X, y), start=1):
            print(f"  Fold {fold}/{folds}")

            X_tr = torch.tensor(X[ti], dtype=torch.float32)
            y_tr = torch.tensor(y[ti], dtype=torch.long)
            X_vl = torch.tensor(X[vi], dtype=torch.float32)
            y_vl = torch.tensor(y[vi], dtype=torch.long)

            trn = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
            vl = DataLoader(TensorDataset(X_vl, y_vl), batch_size=bs)

            model = ECGConvNet().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()

            best_acc, best_state = -1.0, None
            best_true, best_pred = [], []
            train_losses, val_losses = [], []
            train_accs, val_accs = [], []

            for epoch in range(1, epochs + 1):
                # Training
                model.train()
                running_loss, correct, total = 0.0, 0, 0
                for xb, yb in trn:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    out = model(xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt.step()
                    running_loss += loss.item()
                    preds = torch.argmax(out, 1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
                avg_train_loss = running_loss / len(trn)
                train_acc = correct / total

                # Validation
                model.eval()
                running_vloss, vcorrect, vtotal = 0.0, 0, 0
                true, pred = [], []
                with torch.no_grad():
                    for xb, yb in vl:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)
                        running_vloss += crit(out, yb).item()
                        p = torch.argmax(out, 1)
                        vcorrect += (p == yb).sum().item()
                        vtotal += yb.size(0)
                        true.extend(yb.cpu().tolist())
                        pred.extend(p.cpu().tolist())
                avg_val_loss = running_vloss / len(vl)
                val_acc = vcorrect / vtotal

                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

                print(
                    f"    Epoch {epoch}/{epochs} — loss={avg_train_loss:.4f}/{avg_val_loss:.4f}  "
                    f"acc={train_acc:.4f}/{val_acc:.4f}"
                )

                if val_acc > best_acc:
                    best_acc, best_state = val_acc, model.state_dict()
                    best_true, best_pred = true.copy(), pred.copy()

            # Save best model
            fbase = f"lr{lr:.4f}_bs{bs}_fold{fold}_ep{epochs}"
            model_path = os.path.join(out_folder, f"model_{fbase}.pt")
            torch.save({"model_state_dict": best_state}, model_path)
            print(f"  → Saved model: {model_path}")

            # Plot Loss
            fig, ax = plt.subplots()
            ax.plot(train_losses, label="Training")
            ax.plot(val_losses, label="Validation")
            ax.set_title(
                f"Model Loss by Epoch\nLR={lr}, BS={bs}, Fold={fold}, Epochs={epochs}"
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Model Loss")
            ax.legend()
            lp = os.path.join(out_folder, f"loss_{fbase}.png")
            fig.savefig(lp)
            plt.close(fig)
            print(f"  → Saved loss plot: {lp}")

            # Plot Accuracy
            fig, ax = plt.subplots()
            ax.plot(train_accs, label="Training")
            ax.plot(val_accs, label="Validation")
            ax.set_title(
                f"Model Accuracy by Epoch\nLR={lr}, BS={bs}, Fold={fold}, Epochs={epochs}"
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Model Accuracy")
            ax.legend()
            ap = os.path.join(out_folder, f"accuracy_{fbase}.png")
            fig.savefig(ap)
            plt.close(fig)
            print(f"  → Saved accuracy plot: {ap}")

            # Plot Confusion Matrix
            fig, ax = plt.subplots()
            plot_confusion_matrix(
                confusion_matrix(best_true, best_pred),
                classes=["N", "S", "V", "F", "Q"],
                normalize=True,
                title=(
                    f"Normalized Confusion Matrix\nLR={lr}, BS={bs}, Fold={fold}, Epochs={epochs}"
                ),
            )
            cp = os.path.join(out_folder, f"matrix_{fbase}.png")
            fig.savefig(cp)
            plt.close(fig)
            print(f"  → Saved confusion matrix: {cp}")

            # Summary
            results.append(
                {
                    "fold": fold,
                    "epochs": epochs,
                    "batch_size": bs,
                    "lr": lr,
                    "val_acc": best_acc,
                    "model_path": model_path,
                }
            )
            metrics[(fold, epochs, bs, lr)] = {
                "train_accs": train_accs,
                "val_accs": val_accs,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "y_true": best_true,
                "y_pred": best_pred,
            }

    return pd.DataFrame(results), metrics


# %% [markdown]
# ## 8. Entry Point & Full Run

# %%
if __name__ == "__main__":
    t0 = time.time()
    out_folder = "./Outfiles_pytorch"
    os.makedirs(out_folder, exist_ok=True)
    print("Output folder:", out_folder)

    # Load data (with optional subsample)
    X, y = load_data(data_dir="./data", subsample_frac=0.2)

    # Grid search
    results_df, metrics = run_grid_search(X, y, out_folder)
    results_path = "results_summary_pytorch.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved grid search results to {results_path}/{results_df}")

    # Final evaluation on best
    best_row = results_df.loc[results_df.val_acc.idxmax()]
    model_path = best_row.model_path
    print(f"Evaluating final model: {model_path}")
    key = (best_row.fold, best_row.epochs, best_row.batch_size, best_row.lr)
    m = metrics[key]

    evaluate_and_plot(
        y_true=m["y_true"],
        y_pred=m["y_pred"],
        train_accs=m["train_accs"],
        val_accs=m["val_accs"],
        lr=best_row.lr,
        bs=best_row.batch_size,
        fold=best_row.fold,
        epochs=best_row.epochs,
        out_folder=out_folder,
    )

    # PTB Evaluation
    evaluate_on_ptb(
        model_path,
        "./data/ptbdb_normal.csv",
        "./data/ptbdb_abnormal.csv",
        device,
        out_folder,
    )

    elapsed = (time.time() - t0) / 60
    print(f"Total runtime: {elapsed:.2f} minutes")
