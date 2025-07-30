#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from ecg_cnn.config.config_loader import load_training_config
from ecg_cnn.data.data_utils import load_ptbxl_full, FIVE_SUPERCLASSES
from ecg_cnn.models.model_utils import ECGConvNet
from ecg_cnn.paths import (
    DEFAULT_TRAINING_CONFIG,
    MODELS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    PTBXL_DATA_DIR,
)
from ecg_cnn.training.cli_args import parse_args
from ecg_cnn.utils.plot_utils import (
    save_pr_threshold_curve,
    save_confusion_matrix,
    save_plot_curves,
)

SEED = 22
verbose = True
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    t0 = time.time()

    args = parse_args()
    if verbose:
        print("CLI parsed subsample_frac =", args.subsample_frac)

    config = load_training_config(MODELS_DIR / "final_config.json")
    print(f"Config file contents: {config}")

    if args.subsample_frac is not None:
        config.subsample_frac = args.subsample_frac

    # Load data
    print("Loading data for evaluation...")
    X, y, meta = load_ptbxl_full(
        data_dir=PTBXL_DATA_DIR,
        subsample_frac=config.subsample_frac,
        sampling_rate=100,
    )

    # Filter unknown labels
    keep = np.array([lbl != "Unknown" for lbl in y], dtype=bool)
    X = X[keep]
    y = [lbl for i, lbl in enumerate(y) if keep[i]]
    meta = meta.loc[keep].reset_index(drop=True)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_tensor = torch.tensor(y_encoded).long()
    X_tensor = torch.tensor(X).float()
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Load best model info from summary
    summary_path = OUTPUT_DIR / "results" / f"summary_{config.model}.json"
    with open(summary_path, "r") as f:
        summaries = json.load(f)

    best = min(summaries, key=lambda d: d["loss"])
    best_model_path = Path(best["model_path"])
    best_fold = best["fold"] + 1  # convert to 1-based for display
    best_epoch = best["best_epoch"]

    print(f"Evaluating best model from fold {best_fold} (epoch {best_epoch})")
    print(f"Loading model: {best_model_path.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGConvNet(num_classes=len(FIVE_SUPERCLASSES)).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Run evaluation
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []

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

    avg_loss = total_loss / len(dataset)
    print(f"Eval loss: {avg_loss:.4f}")
    print(f"Evaluation completed in {(time.time() - t0) / 60:.2f} minutes.")

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_probs = np.array(all_probs, dtype=np.float32)

    # Print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=FIVE_SUPERCLASSES, zero_division=0
        )
    )

    # Confusion Matrix
    save_confusion_matrix(
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        class_names=FIVE_SUPERCLASSES,
        out_folder=OUTPUT_DIR / "plots",
        model=config.model,
        lr=config.lr,
        bs=config.batch_size,
        wd=config.weight_decay,
        epoch=best_epoch,
        prefix="eval",
        fname_metric="confusion_matrix",
        normalize=True,
        fold=best_fold,
    )

    # Precision-Recall Curve for NORM class
    norm_class = FIVE_SUPERCLASSES.index("NORM")
    y_true_binary = (y_true == norm_class).astype(int)
    y_probs_binary = y_probs[:, norm_class]

    save_pr_threshold_curve(
        y_true=y_true_binary,
        y_probs=y_probs_binary,
        out_folder=OUTPUT_DIR / "plots",
        model=config.model,
        lr=config.lr,
        bs=config.batch_size,
        wd=config.weight_decay,
        epoch=best_epoch,
        prefix="eval",
        fname_metric="pr_threshold_curve",
        title="Precision & Recall vs Threshold (NORM vs All)",
        fold=best_fold,
    )

    # Fake Loss Curve Demo (optional)
    train_loss = [1.5 - 0.1 * i for i in range(10)]
    val_loss = [1.6 - 0.08 * i for i in range(10)]

    save_plot_curves(
        x_vals=list(range(1, 11)),
        y_vals=val_loss,
        x_label="Epoch",
        y_label="Loss",
        title_metric="Validation Loss Curve",
        out_folder=OUTPUT_DIR / "plots",
        model=config.model,
        lr=config.lr,
        bs=config.batch_size,
        wd=config.weight_decay,
        epoch=best_epoch,
        prefix="eval",
        fname_metric="loss",
        fold=best_fold,
    )

    time_spent = (time.time() - t0) / 60
    print(f"Elapsed time: {time_spent:.2f} minutes")


if __name__ == "__main__":
    main()


# #!/usr/bin/env python

# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import time
# import torch
# import torch.nn as nn

# from pathlib import Path
# from sklearn.metrics import (
#     classification_report,
# )
# from sklearn.preprocessing import LabelEncoder
# from torch.utils.data import TensorDataset, DataLoader


# from ecg_cnn.config.config_loader import load_training_config
# from ecg_cnn.data.data_utils import load_ptbxl_full, FIVE_SUPERCLASSES
# from ecg_cnn.models.model_utils import ECGConvNet
# from ecg_cnn.paths import (
#     DEFAULT_TRAINING_CONFIG,
#     MODELS_DIR,
#     OUTPUT_DIR,
#     PROJECT_ROOT,
#     PTBXL_DATA_DIR,
# )
# from ecg_cnn.training.cli_args import parse_args
# from ecg_cnn.utils.plot_utils import (
#     save_pr_threshold_curve,
#     save_confusion_matrix,
#     save_plot_curves,
#     save_classification_report,
#     evaluate_and_plot,
# )

# SEED = 22
# verbose = True
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# def main():
#     t0 = time.time()

#     args = parse_args()
#     if verbose:
#         print("CLI parsed subsample_frac =", args.subsample_frac)

#     config = load_training_config(MODELS_DIR / "final_config.json")
#     print(f"Config file contents: {config}")

#     if args.subsample_frac is not None:
#         config.subsample_frac = args.subsample_frac

#     # Load data
#     print("Loading data for evaluation...")
#     X, y, meta = load_ptbxl_full(
#         data_dir=PTBXL_DATA_DIR, subsample_frac=config.subsample_frac, sampling_rate=100
#     )

#     # Filter unknown labels
#     keep = np.array([lbl != "Unknown" for lbl in y], dtype=bool)
#     X = X[keep]
#     y = [lbl for i, lbl in enumerate(y) if keep[i]]
#     meta = meta.loc[keep].reset_index(drop=True)

#     # Encode labels
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)
#     y_tensor = torch.tensor(y_encoded).long()
#     X_tensor = torch.tensor(X).float()
#     dataset = TensorDataset(X_tensor, y_tensor)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

#     # Load model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ECGConvNet(num_classes=len(FIVE_SUPERCLASSES)).to(device)
#     model.load_state_dict(torch.load(MODELS_DIR / "model_best.pth"))
#     model.eval()

#     criterion = nn.CrossEntropyLoss()
#     total_loss = 0.0
#     all_preds, all_probs, all_targets = [], [], []

#     with torch.no_grad():
#         for X_batch, y_batch in dataloader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)

#             outputs = model(X_batch)
#             probs = torch.softmax(outputs, dim=1)
#             loss = criterion(outputs, y_batch)
#             total_loss += loss.item() * y_batch.size(0)

#             preds = outputs.argmax(dim=1)
#             all_preds.extend(preds.cpu().numpy().astype(int))
#             all_probs.extend(probs.cpu().numpy())
#             all_targets.extend(y_batch.cpu().numpy().astype(int))

#     avg_loss = total_loss / len(dataset)
#     print(f"Eval loss: {avg_loss:.4f}")
#     print(f"Evaluation completed in {(time.time() - t0) / 60:.2f} minutes.")

#     y_true = np.array(all_targets, dtype=int)
#     y_pred = np.array(all_preds, dtype=int)
#     y_probs = np.array(all_probs, dtype=np.float32)

#     # Classification Report
#     print("\nClassification Report:")
#     print(
#         classification_report(
#             y_true, y_pred, target_names=FIVE_SUPERCLASSES, zero_division=0
#         )
#     )

#     # Confusion Matrix
#     save_confusion_matrix(
#         y_true=y_true.tolist(),
#         y_pred=y_pred.tolist(),
#         class_names=FIVE_SUPERCLASSES,
#         out_folder=OUTPUT_DIR / "plots",
#         model=config.model,
#         lr=config.lr,
#         bs=config.batch_size,
#         wd=config.weight_decay,
#         epoch=config.n_epochs,
#         prefix="eval",
#         fname_metric="confusion_matrix",
#         normalize=True,
#         fold=None,
#     )

#     # Precision-Recall Curve (NORM vs ALL)
#     norm_class = FIVE_SUPERCLASSES.index("NORM")
#     y_true_binary = (y_true == norm_class).astype(int)
#     y_probs_binary = y_probs[:, norm_class]

#     save_pr_threshold_curve(
#         y_true=y_true_binary,
#         y_probs=y_probs_binary,
#         out_folder=OUTPUT_DIR / "plots",
#         model=config.model,
#         lr=config.lr,
#         bs=config.batch_size,
#         wd=config.weight_decay,
#         epoch=config.n_epochs,
#         prefix="eval",
#         fname_metric="pr_threshold_curve",
#         title="Precision & Recall vs Threshold (NORM vs All)",
#         fold=None,
#     )

#     # === Demo Curve Plot (Fake Loss over Epochs)
#     train_loss = [1.5 - 0.1 * i for i in range(10)]
#     val_loss = [1.6 - 0.08 * i for i in range(10)]

#     save_plot_curves(
#         x_vals=list(range(1, 11)),
#         y_vals=val_loss,
#         x_label="Epoch",
#         y_label="Loss",
#         title_metric="Validation Loss Curve",
#         out_folder=OUTPUT_DIR / "plots",
#         model=config.model,
#         lr=config.lr,
#         bs=config.batch_size,
#         wd=config.weight_decay,
#         prefix="eval",
#         fname_metric="loss",
#         epoch=config.n_epochs,
#     )

#     time_spent = (time.time() - t0) / 60

#     print(f"Elapsed time: {time_spent:.2f} minutes")


# if __name__ == "__main__":
#     main()
