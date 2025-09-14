#!/usr/bin/env python3
# Select representative 5s windows from PTB-XL that your model rates most
# confidently for each superclass, then export CSVs time,lead1..lead12.

import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# project imports
from ecg_cnn.data.data_utils import FIVE_SUPERCLASSES, load_ptbxl_full
from ecg_cnn.models import MODEL_CLASSES
from ecg_cnn.paths import PTBXL_DATA_DIR

CLASS_NAMES = list(FIVE_SUPERCLASSES)


def _save_csv(path: Path, x12_t: np.ndarray, sr: int):
    T = x12_t.shape[1]
    time = np.arange(T, dtype=float) / float(sr)
    cols = {"time": time}
    for i in range(12):
        cols[f"lead{i+1}"] = x12_t[i]
    df = pd.DataFrame(cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _sliding_windows(x12_t: np.ndarray, sr: int, seconds: int, stride: int = 1):
    """Yield (start, end) index pairs for contiguous windows."""
    win = seconds * sr
    step = stride * sr
    T = x12_t.shape[1]
    if T <= win:
        yield 0, T
        return
    for s in range(0, T - win + 1, step):
        yield s, s + win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to model .pth")
    ap.add_argument("--out", default="explorer/samples", help="Output dir")
    ap.add_argument("--seconds", type=int, default=5, help="Window length")
    ap.add_argument("--sr", type=int, default=100, help="Target sampling rate")
    ap.add_argument(
        "--per-class", type=int, default=1, help="How many exemplars per class"
    )
    ap.add_argument("--max-records", type=int, default=500, help="Max records to scan")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load data at the target sampling rate (uses your project’s loader)
    X, y, meta = load_ptbxl_full(
        data_dir=PTBXL_DATA_DIR, sampling_rate=args.sr, subsample_frac=1.0
    )
    X = np.asarray(X)  # (N,12,T)
    y = list(y)

    # Load model
    ckpt = Path(args.checkpoint)
    # Try to infer class from filename; default to first registry key
    model_name = next(iter(MODEL_CLASSES.keys()))
    for k in MODEL_CLASSES.keys():
        if k.lower() in ckpt.name.lower():
            model_name = k
            break

    model = MODEL_CLASSES[model_name](num_classes=len(CLASS_NAMES))
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Prepare map from class -> indices of records containing that class
    class_to_indices = {c: [] for c in CLASS_NAMES}
    for i, lab in enumerate(y):
        lab = str(lab).upper()
        if lab in class_to_indices:
            class_to_indices[lab].append(i)

    # Select per-class exemplars
    saved_counts = {c: 0 for c in CLASS_NAMES}
    for cls in CLASS_NAMES:
        indices = class_to_indices.get(cls, [])
        if not indices:
            print(f"[warn] no records for {cls}")
            continue

        # scan a limited number of records for speed
        indices = indices[: args.max_records]

        best_windows = []  # list of (prob, i, start, end)
        target_idx = CLASS_NAMES.index(cls)

        with torch.no_grad():
            for i in indices:
                x = X[i]  # (12, T)
                for s, e in _sliding_windows(x, args.sr, args.seconds, stride=1):
                    seg = x[:, s:e]  # (12, W)
                    if seg.shape[1] < args.seconds * args.sr:
                        continue
                    inp = torch.from_numpy(
                        seg[None, ...].astype(np.float32)
                    )  # (1,12,W)
                    logits = model(inp)
                    prob = float(F.softmax(logits, dim=1)[0, target_idx])
                    best_windows.append((prob, i, s, e))

        if not best_windows:
            print(f"[warn] no windows found for {cls}")
            continue

        # take top-K by probability
        best_windows.sort(key=lambda t: t[0], reverse=True)
        take = best_windows[: args.per_class]

        for rank, (prob, i, s, e) in enumerate(take, start=1):
            seg = X[i][:, s:e]
            fname = {
                "NORM": "sample_ecg_norm",
                "MI": "sample_ecg_mi",
                "CD": "sample_ecg_cd",
                "HYP": "sample_ecg_hyp",
                "STTC": "sample_ecg_sttc",
            }[cls]
            if args.per_class > 1:
                fname = f"{fname}_{rank}"
            out_path = out / f"{fname}.csv"
            _save_csv(out_path, seg, args.sr)
            print(f"[ok] {cls}  p={prob:.3f}  -> {out_path}")
            saved_counts[cls] += 1

    print("\nSummary:")
    for c in CLASS_NAMES:
        print(f"  {c}: {saved_counts[c]} file(s)")


if __name__ == "__main__":
    main()
