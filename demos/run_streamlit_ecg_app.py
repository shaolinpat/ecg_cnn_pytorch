# demos/run_streamlit_ecg_app.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app for ECG classification.

This version presents model choices that make sense:
  • Best (accuracy) — from really_the_best_*.json
  • Best (loss)     — from really_the_best_*.json
  • Latest checkpoint — newest model_best_*.pth by timestamp
  • Env override (MODEL_TO_USE) — if set
  • Bundled sample — demos/checkpoints/sample_model_ECGCovNet.pth

The default is Best (accuracy) when available, else Best (loss), else Latest, else Env, else Bundled.

Flow:
  1) Pick checkpoint (or accept default)
  2) Upload CSV -> plot waveform
  3) Run inference -> predicted class + confidences

CSV format expected:
  time (optional), lead1..lead12 (numeric). If no "lead*" headers exist, we try to infer the first 12 numeric columns.
"""

from __future__ import annotations

import altair as alt
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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
BUNDLED_SAMPLE = Path(__file__).parent / "checkpoints" / "sample_model_ECGConvNet.pth"
ENV_OVERRIDE = os.getenv("MODEL_TO_USE")


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
            label=f"Env (MODEL_TO_USE) — {p.name}", path=p, priority=3, meta={}
        )
    return None


def _bundled_candidate() -> Optional[Candidate]:
    if BUNDLED_SAMPLE.exists():
        return Candidate(
            label=f"Bundled sample — {BUNDLED_SAMPLE.name}",
            path=BUNDLED_SAMPLE,
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
st.title("ECG Classification Explorer")

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
    chosen_label = st.selectbox("Checkpoint Selection", labels, index=default_idx)
    chosen = candidates[labels.index(chosen_label)]
    ckpt_path = chosen.path
else:
    ckpt_path = None
    st.info(
        "No checkpoints found.\n\n"
        "Options to enable predictions:\n"
        "• Train a model (creates `outputs/models/model_best_*.pth`)\n"
        "• Set `MODEL_TO_USE=/path/to/model.pth`\n"
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

# --- Optional: SHAP per-lead attribution (fast + safe) ---
try:
    import shap

    st.subheader("Per-lead importance (SHAP)")

    # Background baseline for SHAP (keep tiny to stay fast)
    # Here we use zeros; you can also sample a few windows from the uploaded trace.
    B = 4  # small batch of baselines
    bg = torch.zeros((B, X.shape[1], X.shape[2]), dtype=torch.float32)

    # GradientExplainer expects a callable; we give it the raw model (logits are fine)
    explainer = shap.GradientExplainer(model, bg)

    # SHAP values for all classes: returns a list of length = num_classes,
    # each element shaped like the input (N, C, T). We only need the predicted class.
    # NOTE: Explainer expects torch tensors.
    X_tensor = torch.from_numpy(X).requires_grad_(True)
    shap_vals_all = explainer.shap_values(X_tensor)

    # Handle both (list per class) and array returns (older/newer SHAP behave differently)
    if isinstance(shap_vals_all, list) and len(shap_vals_all) >= (pred_idx + 1):
        sv_pred = shap_vals_all[pred_idx]  # shape (1, C, T)
        sv_pred = sv_pred[0]  # (C, T)
    else:
        # Fallback: try to select along a class axis if present, else treat as (1,C,T)
        sv = shap_vals_all
        # common shapes: (1, C, T) or (1, num_classes, C, T) or (1, C, T, num_classes)
        sv_np = sv.detach().cpu().numpy() if torch.is_tensor(sv) else sv
        if sv_np.ndim == 4 and sv_np.shape[1] == len(CLASS_NAMES):  # (1, classes, C, T)
            sv_pred = sv_np[0, pred_idx]  # (C, T)
        elif sv_np.ndim == 4 and sv_np.shape[-1] == len(
            CLASS_NAMES
        ):  # (1, C, T, classes)
            sv_pred = sv_np[0, :, :, pred_idx]  # (C, T)
        elif sv_np.ndim == 3 and sv_np.shape[0] == 1:  # (1, C, T)
            sv_pred = sv_np[0]  # (C, T)
        else:
            raise RuntimeError(f"Unexpected SHAP value shape: {sv_np.shape}")

    # Reduce time dimension -> per-lead importance
    per_lead_importance = np.mean(np.abs(sv_pred), axis=1)  # (C,)

    # Build lead names if not provided
    lead_names = (
        used_leads[: len(per_lead_importance)]
        if used_leads
        else [f"lead{i+1}" for i in range(len(per_lead_importance))]
    )

    # Create DataFrame
    imp_df = pd.DataFrame(
        {"Lead": lead_names, "Importance": per_lead_importance.astype(float)}
    )

    # Natural sort by numeric suffix if leads look like 'leadN'
    if all(name.startswith("lead") and name[4:].isdigit() for name in imp_df["Lead"]):
        imp_df["Lead_num"] = imp_df["Lead"].str[4:].astype(int)
        imp_df = imp_df.sort_values("Lead_num").drop(columns="Lead_num")
    else:
        imp_df = imp_df.sort_values("Lead")

    # Use the same lead1→lead12 order for both table and chart
    st.dataframe(imp_df, hide_index=True, use_container_width=True)

    # After building imp_df sorted as lead1..lead12
    chart = (
        alt.Chart(imp_df)
        .mark_bar()
        .encode(
            x=alt.X("Lead", sort=list(imp_df["Lead"])),  # force custom order
            y="Importance",
        )
    )

    st.altair_chart(chart, use_container_width=True)

    st.caption("SHAP GradientExplainer: mean |SHAP| over time → per-lead importance")
except Exception as e:
    st.info(
        f"SHAP attributions unavailable ({type(e).__name__}: {e}). "
        "Predictions still work; install/enable SHAP or try a longer sample to see attributions."
    )
