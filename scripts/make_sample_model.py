#!/usr/bin/env python3

# scripts/make_sample_model.py

import torch
from pathlib import Path

from ecg_cnn.models.model_utils import ECGConvNet

# Match ECG setup: 5 classes, 12 leads, 1000 timesteps
NUM_CLASSES = 5
IN_CHANNELS = 12
SEQ_LEN = 1000

# Build the model on CPU with default conv/FC sizes
model = ECGConvNet(
    num_classes=NUM_CLASSES,
    in_channels=IN_CHANNELS,
    seq_len=SEQ_LEN,
    # conv_filters, kernel_sizes, dropouts use your defaults
)

# Make sure the output path exists
ckpt_dir = Path("explorer/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

# sample_model_ECGConvNet.pth — includes the model name so the loader can infer it
for name in ["sample_model_ECGConvNet.pth"]:
    ckpt_path = ckpt_dir / name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": "ECGConvNet",  # helpful metadata if you later read it
            "in_channels": IN_CHANNELS,  # metadata only
            "seq_len": SEQ_LEN,  # metadata only
            "num_classes": NUM_CLASSES,  # metadata only
        },
        ckpt_path,
    )
    print(f"Saved: {ckpt_path}")
