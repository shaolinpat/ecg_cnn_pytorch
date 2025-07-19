# ensemble_test.py

import os
import sys
# Prepend the project root (one level up from this script) to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Import Dataset and model frin src/
from src.data_utils import PTBXLFullDataset
from src.model_utils import ECGConvNet, ECGResNet


def run_5fold_ensemble(
    best_lr, best_bs, best_wd,
    best_dropout_conv, best_dropout_fc, num_epochs,
    meta_csv, scp_csv, ptb_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Instantiate full dataset (loads all records into memory)
    full_dataset = PTBXLFullDataset(meta_csv, scp_csv, ptb_path)
    y_all = full_dataset.y.numpy()  # shape: (N,)
    N = len(full_dataset)

    # 2) Prepare StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_probs = []

    # 3) Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), y_all), start=1):
        print(f"Training fold {fold}")

        # Build Subsets for train & validation
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset   = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=best_bs, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset,   batch_size=best_bs, shuffle=False, num_workers=4)

        # 4) Build a fresh model for this fold
        # model = ECGConvNet(num_classes=5,
        #                    dropout_conv=best_dropout_conv,
        #                    dropout_fc=best_dropout_fc).to(device)



        # Later in the code, when you instantiate:
        use_resnet = True  # or False

        if use_resnet:
            print(">>> Using model: ECGResNet")
            model = ECGResNet(num_classes=5).to(device)
        else:
            print(">>> Using model: ECGConvNet")
            model = ECGConvNet(num_classes=5).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Compute class weights on the TRAIN split only
        train_labels = y_all[train_idx]
        classes_arr = np.unique(train_labels)
        cw = compute_class_weight("balanced", classes=classes_arr, y=train_labels)
        weight_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

        best_val_acc = 0.0
        best_model_path = f"outputs/models/model_fold{fold}.pt"

        for epoch in range(1, num_epochs+1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()



            # After scheduler.step() and before validation:
            model.eval()
            train_correct = 0
            train_total = 0
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(dim=1)
                    train_correct += (preds == yb).sum().item()
                    train_total += yb.size(0)
            train_acc = train_correct / train_total
            print(f" Fold {fold}, Epoch {epoch}, Train Acc {train_acc:.4f}")



            # Evaluate on validation
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_acc = correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

            print(f" Fold {fold}, Epoch {epoch}, Val Acc {val_acc:.4f}")

        # 5) After training, run inference on the entire dataset
        test_loader = DataLoader(full_dataset, batch_size=best_bs, shuffle=False, num_workers=4)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                probs = F.softmax(model(xb), dim=1)
                fold_probs.append(probs.cpu().numpy())
        fold_probs = np.vstack(fold_probs)  # shape: (N, 5)
        test_probs.append(fold_probs)

    # 6) Average probabilities across 5 folds
    all_probs = np.mean(np.stack(test_probs, axis=0), axis=0)  # (N, 5)
    ensemble_preds = np.argmax(all_probs, axis=1)

    # 7) Compute and print final metrics
    print("Ensembled metrics:")
    print(classification_report(y_all, ensemble_preds, target_names=["CD","HYP","MI","NORM","STTC"]))
    cm = confusion_matrix(y_all, ensemble_preds, normalize="true")
    print("Ensembled confusion matrix:\n", cm)


if __name__ == "__main__":
    t0 = time.time()
    # Use hyperparameters found earlier
    best_lr           = 0.001
    best_bs           = 32
    best_wd           = 0.0003
    best_dropout_conv = 0.3
    best_dropout_fc   = 0.5
    num_epochs        = 17

    # Paths to your PTB-XL CSVs and the directory containing all .mat/.hea files
    meta_csv      = "data/ptbxl/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
    scp_csv       = "data/ptbxl/physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"
    ptb_path      = "data/ptbxl/physionet.org/files/ptb-xl/1.0.3" 

    run_5fold_ensemble(
        best_lr, best_bs, best_wd,
        best_dropout_conv, best_dropout_fc,
        num_epochs,
        meta_csv, scp_csv, ptb_path
    )

    elapsed = (time.time() - t0) / 60
    print(f"Total runtime: {elapsed:.2f} minutes")
