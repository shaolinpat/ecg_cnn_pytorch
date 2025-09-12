# demos/run_streamlit_ecg_demo.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit demo for ECG classification.

This version presents model choices that make sense:
  • Best (accuracy) — from really_the_best_*.json
  • Best (loss)     — from really_the_best_*.json
  • Latest checkpoint — newest model_best_*.pth by timestamp
  • Env override (DEMO_MODEL) — if set
  • Bundled demo — demos/checkpoints/model_demo.pth

The default is Best (accuracy) when available, else Best (loss), else Latest, else Env, else Bundled.

Flow:
  1) Pick checkpoint (or accept default)
  2) Upload CSV -> plot waveform
  3) Run inference -> predicted class + confidences

CSV format expected:
  time (optional), lead1..lead12 (numeric). If no "lead*" headers exist, we try to infer the first 12 numeric columns.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Class names (PTB-XL superclasses)
# ------------------------------------------------------------------------------
try:
    from ecg_cnn.data.data_utils import FIVE_SUPERCLASSES

    CLASS_NAMES: List[str] = list(FIVE_SUPERCLASSES)
except Exception:
    CLASS_NAMES = ["NORM", "MI", "CD", "HYP", "STTC"]

# Model registry
from ecg_cnn.models import MODEL_CLASSES  # noqa: E402

RESULTS_DIR = Path("outputs/results")
MODELS_DIR = Path("outputs/models")
BUNDLED_DEMO = Path(__file__).parent / "checkpoints" / "model_demo.pth"
ENV_OVERRIDE = os.getenv("DEMO_MODEL")


@dataclass(frozen=True)
class Candidate:
    label: str
    path: Path
    priority: int  # lower = preferred
    meta: Dict[str, str]


def _fmt_ts(p: Path) -> str:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "unknown time"


def _read_best_json() -> Tuple[Optional[dict], Optional[Path]]:
    """Find newest really_the_best_*.json and load it."""
    if not RESULTS_DIR.exists():
        return None, None
    best_json = max(
        RESULTS_DIR.glob("really_the_best_*.json"),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if not best_json:
        return None, None
    try:
        payload = json.loads(best_json.read_text())
        return payload, best_json
    except Exception:
        return None, best_json


def _candidate_from_best(
    entry: dict, kind: str, json_path: Path
) -> Optional[Candidate]:
    """
    entry should look like:
      {"model": "...", "model_path": "...", "fold": 1, "best_epoch": 3,
       "val_accs": 0.91, "val_losses": 0.42, ...}
    """
    if not entry:
        return None
    mp = entry.get("model_path")
    if not mp:
        return None
    path = Path(mp)
    label = f"Best ({kind}) — {path.name}"
    meta = {
        "tag": entry.get("tag") or "",
        "model": entry.get("model") or "",
        "fold": str(entry.get("fold") or ""),
        "epoch": str(entry.get("best_epoch") or ""),
        "val_acc": (
            f"{entry.get('val_accs'):.4f}" if entry.get("val_accs") is not None else ""
        ),
        "val_loss": (
            f"{entry.get('val_losses'):.4f}"
            if entry.get("val_losses") is not None
            else ""
        ),
        "source": json_path.name,
    }
    # priority: Best (accuracy)=0, Best (loss)=1
    prio = 0 if kind == "accuracy" else 1
    return Candidate(label=label, path=path, priority=prio, meta=meta)


def _latest_checkpoint() -> Optional[Candidate]:
    if not MODELS_DIR.exists():
        return None
    ckpt = max(
        MODELS_DIR.glob("model_best_*_fold*.pth"),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if not ckpt:
        return None
    label = f"Latest — {ckpt.name}"
    meta = {"mtime": _fmt_ts(ckpt)}
    return Candidate(label=label, path=ckpt, priority=2, meta=meta)


def _env_candidate() -> Optional[Candidate]:
    if ENV_OVERRIDE and Path(ENV_OVERRIDE).exists():
        p = Path(ENV_OVERRIDE)
        return Candidate(
            label=f"Env (DEMO_MODEL) — {p.name}", path=p, priority=3, meta={}
        )
    return None


def _bundled_candidate() -> Optional[Candidate]:
    if BUNDLED_DEMO.exists():
        return Candidate(
            label=f"Bundled demo — {BUNDLED_DEMO.name}",
            path=BUNDLED_DEMO,
            priority=4,
            meta={},
        )
    return None


def _collect_candidates() -> List[Candidate]:
    cands: List[Candidate] = []

    payload, best_json = _read_best_json()
    if payload and best_json:
        # include both if present
        if payload.get("by_accuracy"):
            c = _candidate_from_best(payload["by_accuracy"], "accuracy", best_json)
            if c:
                cands.append(c)
        if payload.get("by_loss"):
            c = _candidate_from_best(payload["by_loss"], "loss", best_json)
            if c:
                cands.append(c)

    lat = _latest_checkpoint()
    if lat:
        cands.append(lat)

    envc = _env_candidate()
    if envc:
        cands.append(envc)

    bun = _bundled_candidate()
    if bun:
        cands.append(bun)

    # Deduplicate by path, keep best priority
    dedup: Dict[Path, Candidate] = {}
    for c in cands:
        keep = dedup.get(c.path)
        if (keep is None) or (c.priority < keep.priority):
            dedup[c.path] = c
    out = sorted(
        dedup.values(),
        key=lambda x: (x.priority, -x.path.stat().st_mtime if x.path.exists() else 0),
    )
    return out


def _infer_model_name_from_filename(fname: str) -> str:
    lower = fname.lower()
    for key in MODEL_CLASSES.keys():
        if key.lower() in lower:
            return key
    parts = fname.split("_")
    if len(parts) >= 3 and parts[2] in MODEL_CLASSES:
        return parts[2]
    return next(iter(MODEL_CLASSES.keys()))


def _prepare_input_tensor(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cols = list(df.columns)
    lead_cols = [c for c in cols if c.lower().startswith("lead")]
    if not lead_cols:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        # drop 'time' if present
        numeric_cols = [c for c in numeric_cols if c.lower() != "time"]
        lead_cols = numeric_cols[:12]
    if len(lead_cols) < 12:
        st.warning(
            f"Expected 12 lead columns; found {len(lead_cols)}. Using available leads."
        )
    X = df[lead_cols].to_numpy(dtype=np.float32).T  # (12, T)
    X = X[None, ...]  # (1, 12, T)
    return X, lead_cols


# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------
st.title("ECGConvNet: ECG Classification Demo")

# Model selection
candidates = _collect_candidates()
if candidates:
    # Build informative labels
    def _pretty(c: Candidate) -> str:
        bits = [c.label]
        # add fold/epoch/metrics if available
        f = c.meta.get("fold")
        e = c.meta.get("epoch")
        acc = c.meta.get("val_acc")
        loss = c.meta.get("val_loss")
        src = c.meta.get("source")
        mt = _fmt_ts(c.path)
        extra = []
        if f:
            extra.append(f"fold={f}")
        if e:
            extra.append(f"epoch={e}")
        if acc:
            extra.append(f"val_acc={acc}")
        if loss:
            extra.append(f"val_loss={loss}")
        if src:
            extra.append(f"from {src}")
        extra.append(f"mtime={mt}")
        return f"{c.label}  ({', '.join(extra)})"

    labels = [_pretty(c) for c in candidates]
    default_idx = 0  # list is already priority-sorted
    chosen_label = st.selectbox("Checkpoint selection", labels, index=default_idx)
    chosen = candidates[labels.index(chosen_label)]
    ckpt_path = chosen.path
else:
    ckpt_path = None
    st.info(
        "No checkpoints found.\n\n"
        "Options to enable predictions:\n"
        "• Train a model (creates `outputs/models/model_best_*.pth`)\n"
        "• Set `DEMO_MODEL=/path/to/model.pth`\n"
        "• Include `demos/checkpoints/model_demo.pth`"
    )

# File upload
uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv", "CSV"])
if not uploaded_file:
    st.write("Upload a CSV with columns like: `time, lead1, ..., lead12`.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("Preview of your data:", df.head())

# Plot: prefer time vs lead1 if present; fall back to first two columns
fig, ax = plt.subplots()
cols = list(df.columns)
time_col = next((c for c in cols if c.lower() == "time"), None)
lead1_col = next((c for c in cols if c.lower().startswith("lead")), None)
if time_col and lead1_col:
    ax.plot(df[time_col], df[lead1_col])
    ax.set_xlabel("Time")
    ax.set_ylabel(lead1_col)
else:
    if len(cols) >= 2:
        ax.plot(df.iloc[:, 0], df.iloc[:, 1])
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
    else:
        ax.plot(df.index.values, df.iloc[:, 0])
        ax.set_xlabel("Index")
        ax.set_ylabel(cols[0])
ax.set_title("ECG Signal")
st.pyplot(fig)
st.write("File received:", uploaded_file.name)

# ------------------------------------------------------------------------------
# Prediction (only if a checkpoint is available)
# ------------------------------------------------------------------------------
if ckpt_path is None or not ckpt_path.exists():
    st.info("No checkpoint selected or file missing — showing plot only.")
    st.stop()

st.caption(f"Using checkpoint: `{ckpt_path.name}`")

# Determine model class
model_name = _infer_model_name_from_filename(ckpt_path.name)
if model_name not in MODEL_CLASSES:
    model_name = next(iter(MODEL_CLASSES.keys()))

# Build and load
model = MODEL_CLASSES[model_name](num_classes=len(CLASS_NAMES))
state = torch.load(ckpt_path, map_location="cpu")
state_dict = state.get("state_dict", state)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    st.caption(
        f"(Loaded with missing={len(missing)} unexpected={len(unexpected)} keys)"
    )
model.eval()

# Prepare input and run
X, used_leads = _prepare_input_tensor(df)
with torch.no_grad():
    logits = model(torch.from_numpy(X))
    probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
    pred_idx = int(probs.argmax())
    pred_label = CLASS_NAMES[pred_idx]

st.subheader("Prediction")
st.write(f"**Predicted class:** {pred_label}")
st.dataframe(
    [
        {"Class": CLASS_NAMES[i], "Confidence": float(probs[i])}
        for i in range(len(CLASS_NAMES))
    ],
    hide_index=True,
)
st.caption(f"Leads used: {', '.join(used_leads[:12])}")


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Streamlit demo for ECG classification.

# Behavior:
# - Lets the user upload a CSV (time + lead1..lead12 preferred; time optional).
# - Plots the waveform.
# - Loads a checkpoint and shows predicted class + confidence table.

# Checkpoint selection order (and a UI selector):
# 1) Newest user-trained checkpoint under outputs/models/model_best_*.pth  (preferred)
# 2) DEMO_MODEL env var (if set and exists)
# 3) Bundled demo at demos/checkpoints/model_demo.pth (fallback so hiring managers get predictions)

# If no checkpoint is found, the app still plots the uploaded ECG and explains how to enable predictions.
# """

# from __future__ import annotations

# import os
# from pathlib import Path
# from typing import List

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# import streamlit as st
# import matplotlib.pyplot as plt

# # ------------------------------------------------------------------------------
# # Class names (PTB-XL superclasses)
# # ------------------------------------------------------------------------------
# try:
#     from ecg_cnn.data.data_utils import FIVE_SUPERCLASSES

#     CLASS_NAMES: List[str] = list(FIVE_SUPERCLASSES)
# except Exception:
#     CLASS_NAMES = ["NORM", "MI", "CD", "HYP", "STTC"]

# # Model registry
# from ecg_cnn.models import MODEL_CLASSES  # noqa: E402


# # ------------------------------------------------------------------------------
# # Resolve checkpoint candidates and default selection
# # ------------------------------------------------------------------------------
# def _resolve_checkpoint_selection() -> tuple[Path | None, list[tuple[str, Path]]]:
#     MODELS_DIR = Path("outputs/models")
#     BUNDLED_DEMO = Path(__file__).parent / "checkpoints" / "model_demo.pth"
#     env_override = os.getenv("DEMO_MODEL")

#     candidates: list[tuple[str, Path]] = []

#     # 1) User-trained checkpoints (newest first) — PREFERRED
#     if MODELS_DIR.exists():
#         user_ckpts = sorted(
#             MODELS_DIR.glob("model_best_*_fold*.pth"),
#             key=lambda p: p.stat().st_mtime,
#             reverse=True,
#         )
#         for p in user_ckpts:
#             candidates.append((f"User (newest first): {p.name}", p))

#     # 2) Env override (allow power users/CI to force a path)
#     if env_override:
#         p = Path(env_override)
#         if p.exists():
#             candidates.insert(0, (f"Env DEMO_MODEL: {p.name}", p))

#     # 3) Bundled demo (so hiring managers get predictions without training)
#     if BUNDLED_DEMO.exists():
#         candidates.append((f"Bundled demo: {BUNDLED_DEMO.name}", BUNDLED_DEMO))

#     # Default: newest user-trained if present; else env; else bundled; else None
#     default_idx = 0 if candidates else None
#     if candidates and candidates[0][0].startswith("Env DEMO_MODEL"):
#         # If env is first but we do have user-trained entries, prefer the first user ckpt.
#         for i, (label, _) in enumerate(candidates):
#             if label.startswith("User (newest"):
#                 default_idx = i
#                 break

#     if not candidates:
#         return None, []

#     # Streamlit UI to choose (preselect default)
#     labels = [lbl for (lbl, _) in candidates]
#     sel = st.selectbox("Model checkpoint:", labels, index=default_idx)
#     chosen = dict(candidates)[sel]
#     return chosen, candidates


# # ------------------------------------------------------------------------------
# # Infer model name from filename; fall back if ambiguous
# # ------------------------------------------------------------------------------
# def _infer_model_name_from_filename(fname: str) -> str:
#     """
#     Try to detect a known model class name from the checkpoint filename.
#     Examples:
#       model_best_ecg_ECGConvNet_lr001_bs64_wd0_fold1.pth  -> ECGConvNet
#       model_best_ecg_ECGResNet_lr0003_bs64_wd0_fold1.pth -> ECGResNet
#     If no clear match, return the first key from MODEL_CLASSES.
#     """
#     lower = fname.lower()
#     # direct contains check against known registry keys (case-insensitive)
#     for key in MODEL_CLASSES.keys():
#         if key.lower() in lower:
#             return key

#     # Fallback to a heuristic split (keep for legacy naming)
#     parts = fname.split("_")
#     if len(parts) >= 3 and parts[2] in MODEL_CLASSES:
#         return parts[2]

#     # Last resort: pick the first available model class
#     return next(iter(MODEL_CLASSES.keys()))


# # ------------------------------------------------------------------------------
# # Prepare input tensor from a DataFrame (expects lead columns)
# # ------------------------------------------------------------------------------
# def _prepare_input_tensor(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
#     """
#     Returns (X, lead_cols) where:
#       X has shape (1, C, T) suitable for most 1D CNNs (batch=1).
#       lead_cols is the list of selected lead column names.
#     """
#     cols = list(df.columns)
#     lead_cols = [c for c in cols if c.lower().startswith("lead")]

#     # If no 'lead*' headers, attempt to infer: drop a 'time' col, take first 12 numeric columns
#     if not lead_cols:
#         numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
#         if "time" in [c.lower() for c in cols]:
#             numeric_cols = [c for c in numeric_cols if c.lower() != "time"]
#         lead_cols = numeric_cols[:12]

#     if len(lead_cols) < 12:
#         st.warning(
#             f"Expected 12 lead columns; found {len(lead_cols)}. "
#             "Continuing with available leads; predictions may be less reliable."
#         )

#     # (C, T) -> (1, C, T)
#     X = df[lead_cols].to_numpy(dtype=np.float32).T
#     X = X[None, ...]
#     return X, lead_cols


# # ------------------------------------------------------------------------------
# # Streamlit UI
# # ------------------------------------------------------------------------------
# st.title("ECGConvNet: ECG Classification Demo")

# ckpt_path, all_candidates = _resolve_checkpoint_selection()

# uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv", "CSV"])
# if not uploaded_file:
#     st.write("Upload a CSV with columns like: `time, lead1, ..., lead12`.")
#     st.stop()

# # Read CSV
# df = pd.read_csv(uploaded_file)
# st.write("Preview of your data:", df.head())

# # Plot: prefer time vs lead1 if present, else first two columns
# fig, ax = plt.subplots()
# cols = list(df.columns)
# time_col = next((c for c in cols if c.lower() == "time"), None)
# lead1_col = next((c for c in cols if c.lower().startswith("lead")), None)

# if time_col and lead1_col:
#     ax.plot(df[time_col], df[lead1_col])
#     ax.set_xlabel("Time")
#     ax.set_ylabel(lead1_col)
# else:
#     # Fallback to first two columns as (x, y)
#     if len(cols) >= 2:
#         ax.plot(df.iloc[:, 0], df.iloc[:, 1])
#         ax.set_xlabel(cols[0])
#         ax.set_ylabel(cols[1])
#     else:
#         ax.plot(df.index.values, df.iloc[:, 0])
#         ax.set_xlabel("Index")
#         ax.set_ylabel(cols[0])

# ax.set_title("ECG Signal")
# st.pyplot(fig)
# st.write("File received:", uploaded_file.name)

# # ------------------------------------------------------------------------------
# # Prediction (only if a checkpoint is available)
# # ------------------------------------------------------------------------------
# if ckpt_path is None or not Path(ckpt_path).exists():
#     st.info(
#         "No trained checkpoint found. "
#         "To enable predictions:\n"
#         "• Train a model (checkpoints saved under `outputs/models/`), or\n"
#         "• Set `DEMO_MODEL=/path/to/model_best_...pth`, or\n"
#         "• Include `demos/checkpoints/model_demo.pth` in the repo."
#     )
#     st.stop()

# st.caption(f"Using checkpoint: `{Path(ckpt_path).name}`")

# # Infer model class from filename; fall back to first registry key if ambiguous
# model_name = _infer_model_name_from_filename(Path(ckpt_path).name)
# if model_name not in MODEL_CLASSES:
#     # Should not happen due to _infer_model_name_from_filename, but be defensive
#     model_name = next(iter(MODEL_CLASSES.keys()))

# # Build model on CPU and load weights
# model = MODEL_CLASSES[model_name](num_classes=len(CLASS_NAMES))
# state = torch.load(ckpt_path, map_location="cpu")
# state_dict = state.get("state_dict", state)
# missing, unexpected = model.load_state_dict(state_dict, strict=False)
# if missing or unexpected:
#     st.caption(
#         f"Checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)} keys"
#     )
# model.eval()

# # Prepare (1, C, T) input
# X, used_leads = _prepare_input_tensor(df)

# with torch.no_grad():
#     logits = model(torch.from_numpy(X))
#     probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
#     pred_idx = int(probs.argmax())
#     pred_label = CLASS_NAMES[pred_idx]

# st.subheader("Prediction")
# st.write(f"**Predicted class:** {pred_label}")

# # Confidence table
# conf_rows = [
#     {"Class": CLASS_NAMES[i], "Confidence": float(probs[i])}
#     for i in range(len(CLASS_NAMES))
# ]
# st.dataframe(conf_rows, hide_index=True)

# # Tiny footnote on which leads were used (helps avoid confusion if columns were inferred)
# st.caption(f"Leads used for inference: {', '.join(used_leads[:12])}")


# import matplotlib.pyplot as plt
# import pandas as pd
# import streamlit as st

# st.title("ECGCoveNet: ECG Classification Demo")

# uploaded_file = st.file_uploader("Upload ECG CSV", type=["CSV"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.write("Preview of your data:", df.head())

#     fig, ax = plt.subplots()
#     ax.plot(df.iloc[:, 0], df.iloc[:, 1])
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Amplitude")
#     ax.set_title("ECG Signal")
#     st.pyplot(fig)
#     st.write("File received", uploaded_file.name)
