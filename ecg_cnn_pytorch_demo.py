# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     primary: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
################################################################################
##
##  Import Block
##
################################################################################

# %% [markdown]
# # ECG Classification Demo
# This notebook shows how we load data, train a 1D CNN, evaluate, and explain with Grad-CAM.

# %%

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Set random seed
SEED = 22
np.random.seed(SEED)
torch.manual_seed(SEED)

# Thread control
torch.set_num_threads(6)
print("Using", torch.get_num_threads(), "threads")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
##
##  ECGConvNet Class Definition
##
################################################################################


class ECGConvNet(nn.Module):
    def __init__(self):
        super(ECGConvNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=6)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

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


################################################################################
##
##  Helper Functions
##
################################################################################


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    Comfusion matrix plotting
    """
    if normalize:
        with np.errstate(all="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm.astype("float"), row_sums, where=row_sums != 0)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def evaluate_and_plot(
    y_true,
    y_pred,
    train_accuracies,
    val_accuracies,
    learning_rate,
    batch_size,
    fold,
    epochs,
    out_folder,
):
    """
    Evaluate and plot a run
    """
    labels = [0, 1, 2, 3, 4]
    class_names = ["N", "S", "V", "F", "Q"]
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.figure()
    if len(train_accuracies) > 1:
        plt.plot(train_accuracies, label="Training")
        plt.plot(val_accuracies, label="Validation")
    else:
        plt.scatter([0], train_accuracies, label="Training")
        plt.scatter([0], val_accuracies, label="Validation")
    plt.title(
        f"Model Accuracy by Epoch\n LR = {learning_rate}, BS = {batch_size}, Folds = {fold}, Epochs = {epochs}"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Model Accuracy")
    plt.legend()
    out_file = f"accuracy_{learning_rate}_{batch_size}_{fold}_{epochs}.png"
    full_path = os.path.join(out_folder, out_file)
    plt.savefig(full_path)
    plt.close()
    plot_confusion_matrix(
        cm,
        classes=class_names,
        normalize=True,
        title=f"Normalized Confusion Matrix\nLR={learning_rate}, BS={batch_size}, Fold={fold}, Epochs={epochs}",
    )
    out_file = f"matrix_{learning_rate}_{batch_size}_{fold}_{epochs}.png"
    full_path = os.path.join(out_folder, out_file)
    plt.savefig(full_path)
    plt.close()


def evaluate_on_ptb(
    model_path, ptb_csv_path_normal, ptb_csv_path_abnormal, device, out_folder
):
    """
    Evaluate a saved model on PTB dataset
    """
    ptb_normal = pd.read_csv(ptb_csv_path_normal, header=None)
    ptb_abnormal = pd.read_csv(ptb_csv_path_abnormal, header=None)
    ptb_df = pd.concat([ptb_normal, ptb_abnormal], ignore_index=True)

    ptb_df.dropna(inplace=True)
    ptb_df[187] = ptb_df[187].astype(int)
    ptb_df = ptb_df[ptb_df[187].isin([0, 1])]

    X_ptb = ptb_df.iloc[:, :-1].values.reshape(-1, 1, 187)
    y_ptb = ptb_df.iloc[:, -1].values

    X_ptb_tensor = torch.tensor(X_ptb, dtype=torch.float32).to(device)
    y_ptb_tensor = torch.tensor(y_ptb, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_ptb_tensor, y_ptb_tensor), batch_size=32)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = ECGConvNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            targets.extend(yb.numpy())

    acc = np.mean(np.array(preds) == np.array(targets))
    print(f"\n--- PTB Evaluation ---")
    print(f"Model loaded from: {model_path}")
    print(f"PTB Accuracy: {acc:.4f}")
    print("Classification Report:")
    print("Classification Report (Classes 0 and 1 Only):")
    print(classification_report(targets, preds, labels=[0, 1], zero_division=0))

    # Save classification report as CSV
    report_dict = classification_report(
        targets, preds, output_dict=True, zero_division=0
    )

    # Keep only class 0 and 1 plus macro/weighted averages
    keep_labels = ["0", "1", "macro avg", "weighted avg"]
    filtered_report_dict = {k: v for k, v in report_dict.items() if k in keep_labels}

    report_df = pd.DataFrame(filtered_report_dict).transpose()
    ptb_report_path = os.path.join(out_folder, "ptb_classification_report.csv")
    report_df.to_csv(ptb_report_path)
    print(f"Saved PTB classification report to {ptb_report_path}")

    # Plot confusion matrix
    cm = confusion_matrix(targets, preds)

    # Restrict to 0 and 1 classes only
    cm = cm[:2, :2]
    class_names = ["Normal", "Abnormal"]

    plot_confusion_matrix(
        cm, classes=class_names, normalize=True, title="PTB Normalized Confusion Matrix"
    )

    ptb_outfile = os.path.join(out_folder, "ptb_confusion_matrix.png")
    plt.savefig(ptb_outfile)
    print(f"Saved PTB confusion matrix to {ptb_outfile}")
    plt.close()


################################################################################
##
##  Main Processing
##
################################################################################

if __name__ == "__main__":
    start_time = time.time()
    out_folder = "./Outfiles_pytorch"
    os.makedirs(out_folder, exist_ok=True)

    mitbih_train_df = pd.read_csv("./data/mitbih_train.csv", header=None)
    mitbih_test_df = pd.read_csv("./data/mitbih_test.csv", header=None)
    combined_df = pd.concat((mitbih_train_df, mitbih_test_df), ignore_index=True)
    combined_df[187] = combined_df[187].astype(int)
    combined_df.dropna(inplace=True)
    combined_df = combined_df.sample(frac=1, random_state=SEED)

    # Extract features and labels
    X = combined_df.iloc[:, :-1].values
    y = combined_df.iloc[:, -1].values
    X = X.reshape(X.shape[0], 1, X.shape[1])

    param_grid = {
        "epochs_list": [5, 10],
        "batch_size_list": [10, 32],
        "learning_rate_list": [1e-3, 5e-4, 1e-4],
    }

    param_combinations = list(
        itertools.product(
            param_grid["epochs_list"],
            param_grid["batch_size_list"],
            param_grid["learning_rate_list"],
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    train_accuracies = []
    val_accuracies = []

    for epochs, batch_size, learning_rate in tqdm(
        param_combinations, desc="Grid Search"
    ):
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            print(
                f"fold = {fold}, epoch = {epochs}, batch = {batch_size}, learning_rate = {learning_rate}"
            )

            # Prepare data
            X_train, X_val = torch.tensor(
                X[train_idx], dtype=torch.float32
            ), torch.tensor(X[val_idx], dtype=torch.float32)
            y_train, y_val = torch.tensor(y[train_idx], dtype=torch.long), torch.tensor(
                y[val_idx], dtype=torch.long
            )
            train_loader = DataLoader(
                TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

            # Prepare model and training tools
            model = ECGConvNet().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            train_losses = []
            val_losses = []
            fold_train_accuracies = []
            fold_val_accuracies = []

            # Track best model state
            best_val_acc = -1.0
            best_model_state = None
            best_model_params = {}

            for epoch in range(epochs):
                model.train()
                running_train_loss = 0.0

                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item()

                # Evaluate
                model.eval()
                train_preds, train_targets = [], []
                val_preds, val_targets = [], []

                with torch.no_grad():
                    for xb_, yb_ in train_loader:
                        xb_ = xb_.to(device)
                        out = model(xb_)
                        preds = torch.argmax(out, dim=1).cpu().numpy()
                        train_preds.extend(preds)
                        train_targets.extend(yb_.numpy())

                    for xb_, yb_ in val_loader:
                        xb_ = xb_.to(device)
                        out = model(xb_)
                        preds = torch.argmax(out, dim=1).cpu().numpy()
                        val_preds.extend(preds)
                        val_targets.extend(yb_.numpy())

                # Capture model accuracy and loss ===
                train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
                val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
                avg_train_loss = running_train_loss / len(train_loader)

                # Save best model state by validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()
                    best_model_params = {
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "fold": fold,
                    }

                fold_train_accuracies.append(train_acc)
                fold_val_accuracies.append(val_acc)
                train_losses.append(avg_train_loss)

                running_val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)
                        loss = criterion(out, yb)
                        running_val_loss += loss.item()

                avg_val_loss = running_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                print(
                    f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_acc:.4f}"
                )

            # Save best model for this fold and grid combo
            if best_model_state:
                model_path = os.path.join(
                    out_folder,
                    f"best_model_lr{learning_rate}_bs{batch_size}_fold{fold}_ep{epochs}.pt",
                )
                torch.save(
                    {"model_state_dict": best_model_state, "params": best_model_params},
                    model_path,
                )
                print(f"Saved best model to {model_path}")

            # Plot losses losses
            plt.figure()
            plt.plot(train_losses, label="Training")
            plt.plot(val_losses, label="Validation")
            plt.title(
                f"Model Loss by Epoch\nLR={learning_rate}, BS={batch_size}, Fold={fold}, Epochs={epochs}"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Model Loss")
            plt.legend()
            plt.savefig(
                f"{out_folder}/loss_{learning_rate}_{batch_size}_{fold}_{epochs}.png"
            )
            plt.close()

            # Plot losses accuracy
            plt.figure()
            plt.plot(fold_train_accuracies, label="Training")
            plt.plot(fold_val_accuracies, label="Validation")
            plt.title(
                f"Model Accuracy by Epoch\nLR={learning_rate}, BS={batch_size}, Fold={fold}, Epochs={epochs}"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Model Accuracy")
            plt.legend()
            plt.savefig(
                f"{out_folder}/accuracy_{learning_rate}_{batch_size}_{fold}_{epochs}.png"
            )
            plt.close()

            # Final eval and print confusion martrix
            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    out = model(xb)
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                    y_pred.extend(preds)
                    y_true.extend(yb.numpy())

            acc = np.mean(np.array(y_pred) == np.array(y_true))
            results.append(
                {
                    "fold": fold,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "val_accuracy": round(acc, 4),
                }
            )

            evaluate_and_plot(
                y_true,
                y_pred,
                fold_train_accuracies,
                fold_val_accuracies,
                learning_rate,
                batch_size,
                fold,
                epochs,
                out_folder,
            )

    # Save file with running results
    pd.DataFrame(results).to_csv("results_summary_pytorch.csv", index=False)
    print("Saved results to results_summary_pytorch.csv")

    ############################################################################
    ##
    ## PTB Validation
    ##
    ############################################################################

    # Select best model for running on PTB data
    results_df = pd.read_csv("results_summary_pytorch.csv")
    best_row = results_df.loc[results_df["val_accuracy"].idxmax()]
    lr = best_row["learning_rate"]
    bs = best_row["batch_size"]
    fold = best_row["fold"]
    ep = best_row["epochs"]

    # Evaluate saved best model on PTB dataset
    best_model_path = f"./Outfiles_pytorch/best_model_lr{lr}_bs{int(bs)}_fold{int(fold)}_ep{int(ep)}.pt"
    ptb_csv_path_normal = "./data/ptbdb_normal.csv"
    ptb_csv_path_abnormal = "./data/ptbdb_abnormal.csv"

    if os.path.exists(best_model_path):
        evaluate_on_ptb(
            best_model_path,
            ptb_csv_path_normal,
            ptb_csv_path_abnormal,
            device,
            out_folder,
        )

        # Generate LaTeX table from PTB report
        report_csv_path = os.path.join(out_folder, "ptb_classification_report.csv")
        if os.path.exists(report_csv_path):
            df = pd.read_csv(report_csv_path, index_col=0)
            df = df.round(3)
            df["support"] = df["support"].astype(int)
            df = df[["precision", "recall", "f1-score", "support"]]

            table_body = df.to_latex(index=True, column_format="lcccc")
            latex_code = (
                r"""
            \begin{table}[H]
                \centering
                \caption{PTB Classification Report (Normal vs. Abnormal)}
                \label{tab:ptb_classification}
                """
                + table_body
                + r"""
                \end{table}
            """
            )
            latex_path = os.path.join(out_folder, "ptb_report_table.tex")
            with open(latex_path, "w") as f:
                f.write(latex_code)
            print(f"Saved LaTeX report to {latex_path}")
        else:
            print("PTB classification report not found. Skipping LaTeX export.")
    else:
        print(f"Model file not found: {best_model_path}")

    # Close out
    time_spent = (time.time() - start_time) / 60
    print(f"Processing time: {time_spent:.2f} minutes")
# %%
