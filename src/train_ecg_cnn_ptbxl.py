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
import argparse
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import os
import random
import time
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset #, WeightedRandomSampler

from data_utils import (
    load_ptbxl_sample,
    load_ptbxl_full
)

from grid_search import run_manual_grid_search
from model_utils import ECGConvNet

from plot_utils import (
    # format_hparams,
    # evaluate_and_plot,
    save_plot_curves,
    save_confusion_matrix,
)

# Optionally enable AMP for speed on GPU:
# from torch.cuda.amp import autocast, GradScaler

# %% [markdown]
# ## 1. Setup & Config

# %%
# Reproducibility
# -----------------------------------------------------------------------
# 1) GLOBAL SEED & DETERMINISM
# -----------------------------------------------------------------------
SEED = 22

# 1a) Seed Python’s built-in RNG
random.seed(SEED)

# 1b) Seed NumPy RNG
np.random.seed(SEED)

# 1c) Seed PyTorch CPU RNG
torch.manual_seed(SEED)

# 1d) Seed PyTorch CUDA RNG (if using GPU)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 1e) Make cuDNN deterministic (slightly slower but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# -----------------------------------------------------------------------
# 2) THREAD CONTROL and DEVICE SELECTION (does NOT affect randomness)
# -----------------------------------------------------------------------
torch.set_num_threads(6)
print(f"Using {torch.get_num_threads()} CPU threads")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")








# -----------------------------------------------------------------------------
# 2) THREAD CONTROL AND DEVICE SELECTION
# -----------------------------------------------------------------------------
torch.set_num_threads(6)
print(f"Using {torch.get_num_threads()} CPU threads")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")



# %% [markdown]
# ### Arg parsing 

# %%
# -----------------------------------------------------------------------------
# 3) ARGPARSE, HELPERS, ETC.
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ECG CNN on PTB-XL")
    p.add_argument(
        "--sample-only",
        action="store_true",
        help="Run on bundled 100-record sample instead of full dataset"
    )
    p.add_argument(
        "--data-dir",
        default="../data/ptbxl/physionet.org/files/ptb-xl/1.0.3",
        help="Path to full PTB-XL data directory"
    )
    p.add_argument(
        "--subsample-frac",
        type=float,
        default=1.0,
        help="Fraction of full data to load (for quick smoke tests)"
    )
    return p.parse_args()

#%%


#%%




# %% [markdown]
# ## 4. Evaluation & Visualization


# %%


# %%




# %% [markdown]
# ## 6. Data Loading & Subsampling


#%%


# %% [markdown]
# ## 8. Entry Point & Full Run

# %%
if __name__ == "__main__":
    t0 = time.time()
    out_folder = "../outputs/plots"
    model_out_folder = "../outputs/models"
    os.makedirs(out_folder, exist_ok=True)
    print("Output folder:", out_folder)

    args = parse_args()

    # 1) Load data (may contain "Unknown" labels)
    if args.sample_only:
        X, y, meta = load_ptbxl_sample(
            sample_dir="../data/larger_sample/",
            ptb_path=args.data_dir
        )
    else:
        X, y, meta = load_ptbxl_full(
            data_dir=args.data_dir,
            subsample_frac=args.subsample_frac,
            sampling_rate=100
        )

    # 2) Flatten each y[i] from ["LABEL"] - "LABEL" or "Unknown"
    y_single = [
        (lbls[0] if isinstance(lbls, list) and len(lbls) > 0
         else (lbls if isinstance(lbls, str) else "Unknown"))
        for lbls in y
    ]

    # 3) Build keep_mask to drop "Unknown"
    keep_mask = np.array([lbl != "Unknown" for lbl in y_single], dtype=bool)

    print("Before filtering out 'Unknown':")
    print("  Old X.shape:", X.shape)
    print("  Old len(y_single):", len(y_single))
    print("  Old labels:", sorted(set(y_single)))
    print("  Old meta.size:", meta.shape[0])

    # 4) Filter out "Unknown" from X, y_single, and meta
    X = X[keep_mask]
    y_single = [lbl for i, lbl in enumerate(y_single) if keep_mask[i]]
    meta = meta.loc[keep_mask].reset_index(drop=True)

    print("After filtering out 'Unknown':")
    print("  New X.shape:", X.shape)
    print("  New len(y_single):", len(y_single))
    print("  New unique labels:", sorted(set(y_single)))
    print("  New meta.size:", meta.shape[0])

    # 5a) Compute num_classes and build integer encoding
    unique_labels = sorted(set(y_single))      # ['CD','HYP','MI','NORM','STTC']
    num_classes = len(unique_labels)           # 5 classes
    label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    # 5b) Convert the list y (e.g. ["CD","STTC","NORM",...]) into a 1D numpy array of ints
    y_enc = np.array([label2idx[lbl] for lbl in y_single], dtype=np.int64)

    # Sanity check (optional):
    print("Loaded sample size:", X.shape[0])
    print("Unique labels:", unique_labels)
    print("Encoded labels shape:", y_enc.shape, "with values in", np.unique(y_enc))

    print("  Unique labels (post-filter):", unique_labels)
    print("  num_classes:", num_classes)
    print("  y_enc unique values:", np.unique(y_enc))  # should be [0 1 2 3 4]

    # 6) Hold out 10% of the 1000 as a TEST set (never used until final evaluation)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.10,    # 10% = 100 samples
        stratify=y_enc,    # preserve class proportions
        random_state=SEED  # repeatabilty guarantee
    )

    print("After splitting off TEST set:")
    print("  X_trainval shape:", X_trainval.shape, "y_trainval shape:", y_trainval.shape)
    print("  X_test shape:", X_test.shape, "y_test shape:", y_test.shape)


    # 7) Split the 900 (X_trainval) into 80% train / 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, 
        y_trainval,
        test_size=0.20,        # 20% of 900 = 180
        stratify=y_trainval,   # preserve class proportions
        random_state=SEED
    )

    from collections import Counter
    ctr = Counter(y_train)   # y_train is your list of mapped labels
    print(f"   !!!   Counter: {ctr} !!!   ")

    # 8A) Train Dataloaded (used for backprop)
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    # The batch_size is set in the grid_search loop:
    train_loader=None

    # 8B) Validation Dataloader (used for computing val_loss / val_acc, no backprop)
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    val_loader=None

    # 8C) Test Dataloader (held out until the very end)
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ),
        batch_size=1,
        shuffle=False
    )


#    # === Step 9: Manual grid search on TRAIN / VAL ===

    # Step 9) Run the entire grid-search function
    param_grid = {
        "epochs":      [20],
        "batch_size":  [16, 32],
        "lr":          [1e-3, 5e-4],
        "weight_decay":[3e-4],
    }
    results_df, metrics = run_manual_grid_search(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        unique_labels=unique_labels,
        num_classes=num_classes,
        out_folder=out_folder,
        model_out_folder=model_out_folder,
        device=device,
        param_grid=param_grid
    )

    # 9J) Save the DataFrame of results
    os.makedirs("../outputs/results", exist_ok=True)
    csv_path = "../outputs/results/results_summary_pytorch.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved grid search results to {csv_path}")



    # Step 10: Final evaluation on the held-out TEST set

    # 10A) Choose the best hyperparameter combo (highest validation accuracy)
    best_row = results_df.loc[results_df.val_acc.idxmax()]
    print("Best hyperparameters by val_acc:", best_row.to_dict())

    # 10B) Load that saved model checkpoint
    checkpoint = torch.load(best_row["model_path"], map_location=device)
    final_model = ECGConvNet(num_classes).to(device)
    final_model.load_state_dict(checkpoint["model_state_dict"])
    final_model.eval()

    # 10C) Evaluate exactly once on test_loader
    all_true_test = []
    all_pred_test = []

    with torch.no_grad():
        for xb, yb in test_loader:      # test_loader has your 100 test samples
            xb, yb = xb.to(device), yb.to(device)
            logits = final_model(xb)
            preds = logits.argmax(dim=1)
            all_true_test.extend(yb.cpu().tolist())
            all_pred_test.extend(preds.cpu().tolist())

    # 10D) Print classification report and confusion matrix on the 100 test examples
    print("=== FINAL TEST SET PERFORMANCE ===")
    print(classification_report(
        all_true_test,
        all_pred_test,
        labels=list(range(len(unique_labels))),
        target_names=unique_labels,
        zero_division=0
    ))

    cm_test = confusion_matrix(
        all_true_test, all_pred_test,
        labels=list(range(len(unique_labels)))
    )

# (Optionally, plot cm_test if you want a figure.)

    # (Optionally plot or save the test-set confusion matrix here if you want)

    # =========================================
    # Step 11: Save best_... plots for the winning combo on VAL
    # =========================================

    # 11A) Build the dictionary key for "best":
    best_key = (
        int(best_row.fold),
        int(best_row.epochs),
        int(best_row.batch_size),
        float(best_row.lr),
        float(best_row.weight_decay)
    )
    best_dict = metrics[best_key]

    # 11B) Unpack the lists for plotting:
    best_train_losses = best_dict["train_losses"]
    best_val_losses   = best_dict["val_losses"]
    best_train_accs   = best_dict["train_accs"]
    best_val_accs     = best_dict["val_accs"]
    best_y_true       = best_dict["y_true"]
    best_y_pred       = best_dict["y_pred"]

    # 11C) Construct a descriptive prefix using those hyperparameters:
    best_lr     = float(best_row.lr)
    best_wd     = float(best_row.weight_decay)
    best_bs     = int(best_row.batch_size)
    best_fold   = int(best_row.fold)
    best_epochs = int(best_row.epochs)

    # 11A) Best accuracy
    save_plot_curves(
        x_vals       = best_train_accs,
        y_vals       = best_val_accs,
        x_label      = "Epoch",
        y_label      = "Accuracy",
        title_metric = "Accuracy",
        fname_metric = "accuracy",
        lr           = best_lr,
        bs           = best_bs,
        wd           = best_wd,
        fold         = best_fold,
        epochs       = best_epochs,
        out_folder   = out_folder,
        prefix       = "best"
        )

     # 11B) Best LOSS
    save_plot_curves(
        x_vals       = best_train_losses,
        y_vals       = best_val_losses,
        x_label      = "Epoch",
        y_label      = "Loss",
        title_metric = "Loss",
        fname_metric = "loss",
        lr           = best_lr,
        bs           = best_bs,
        wd           = best_wd,
        fold         = best_fold,
        epochs       = best_epochs,
        out_folder   = out_folder,
        prefix       = "best"
    )

    # 11C) Best CONFUSION MATRIX
    save_confusion_matrix(
        y_true      = best_y_true,
        y_pred      = best_y_pred,
        class_names = unique_labels,
        lr          = best_lr,
        bs          = best_bs,
        wd          = best_wd,
        fold        = best_fold,
        epochs      = best_epochs,
        out_folder  = out_folder,
        prefix      = "best",
        normalize   = True
    )

    elapsed = (time.time() - t0) / 60
    print(f"Total runtime: {elapsed:.2f} minutes")
