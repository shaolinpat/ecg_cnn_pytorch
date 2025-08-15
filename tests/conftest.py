# tests/conftest.py

"""
Pytest configuration and shared fixtures for the ECG CNN test suite.

This module centralizes reusable test helpers and environment setup:
    - Forces a headless-friendly matplotlib backend for CI.
    - Seeds Python, NumPy, and (optionally) PyTorch RNGs for reproducibility.
    - Provides factories for TrainConfig instances and argparse-like CLI args.
    - Offers a `patch_paths` fixture that redirects all project path constants
      (including results, history, models, output, plots, and PTB-XL data dirs)
      into per-test temporary directories.
    - Supplies small synthetic datasets, default plotting parameters, and a tiny
      torch model for fast forward/eval tests.

Importing this conftest automatically makes these fixtures available to all
tests without explicit import.
"""


from __future__ import annotations

import matplotlib as plt
import numpy as np
import os
import pandas as pd
import pytest
import random

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from pathlib import Path
from types import SimpleNamespace

from ecg_cnn import paths
from ecg_cnn.config.config_loader import TrainConfig
from ecg_cnn.data import data_utils


# -----------------------------------------------------------------------------
# Global, safe defaults for the whole test run
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _matplotlib_agg_backend():
    """Force non-interactive matplotlib backend for headless CI."""
    plt.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _reproducible_rng_state():
    """Make numpy/python/torch RNG deterministic for test stability."""
    SEED = int(os.getenv("TEST_SEED", "22"))
    random.seed(SEED)
    np.random.seed(SEED)
    if torch is not None:
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Factories you can opt into from any test file
# -----------------------------------------------------------------------------


@pytest.fixture
def make_train_config():
    """
    Factory for a valid TrainConfig that passes all validators.
    Override any field via kwargs: cfg = make_train_config(batch_size=8, n_folds=2)
    """

    def _make(**overrides) -> TrainConfig:
        base = dict(
            model="ECGConvNet",
            lr=0.001,
            batch_size=64,
            weight_decay=0.0,
            n_epochs=10,
            n_folds=2,
            save_best=True,
            sample_only=False,
            subsample_frac=1.0,
            sampling_rate=100,
            data_dir=None,
            sample_dir=None,
            verbose=False,
        )
        base.update(overrides)
        return TrainConfig(**base)

    return _make


@pytest.fixture
def make_args():
    """Factory that mimics argparse.Namespace for CLI-related tests."""

    def _make(**kwargs):
        return SimpleNamespace(**kwargs)

    return _make


@pytest.fixture
def patch_paths(monkeypatch, tmp_path: Path):
    """
    Redirect ecg_cnn.paths *and* ecg_cnn.data.data_utils path constants to
    per-test temp folders. Creates the common subdirs that code expects.

    Returns:
        (results_dir, history_dir, models_dir, output_dir, plots_dir, ptbxl_dir)
    """
    # Canonical temp directories
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    models_dir = tmp_path / "models"
    output_dir = tmp_path / "output"
    plots_dir = tmp_path / "plots"
    ptbxl_dir = tmp_path / "ptbxl"

    # Create them so any code that writes will succeed
    for p in (results_dir, history_dir, models_dir, output_dir, plots_dir, ptbxl_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Patch the central paths module
    monkeypatch.setattr(paths, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(paths, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(paths, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(paths, "OUTPUT_DIR", output_dir, raising=False)
    monkeypatch.setattr(paths, "PLOTS_DIR", plots_dir, raising=False)
    monkeypatch.setattr(paths, "PTBXL_DATA_DIR", ptbxl_dir, raising=False)
    # Some code may rely on a project root—point it at tmp_path if present
    if hasattr(paths, "PROJECT_ROOT"):
        monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path, raising=False)

    # Also patch the data_utils module’s copies so “pass None -> default path”
    # branches resolve to the same temp locations.
    try:
        # Create typical subtrees data_utils might expect
        (tmp_path / "data" / "sample").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(data_utils, "PTBXL_DATA_DIR", ptbxl_dir, raising=False)
        if hasattr(data_utils, "PROJECT_ROOT"):
            monkeypatch.setattr(data_utils, "PROJECT_ROOT", tmp_path, raising=False)
        # If data_utils re-exports or caches anything from paths, keep it consistent
        if hasattr(data_utils, "RESULTS_DIR"):
            monkeypatch.setattr(data_utils, "RESULTS_DIR", results_dir, raising=False)
        if hasattr(data_utils, "OUTPUT_DIR"):
            monkeypatch.setattr(data_utils, "OUTPUT_DIR", output_dir, raising=False)
    except Exception:
        # If data_utils isn't imported anywhere in the suite, skip silently.
        pass

    return results_dir, history_dir, models_dir, output_dir, plots_dir, ptbxl_dir


# -----------------------------------------------------------------------------
# Lightweight model/data helpers (opt-in)
# -----------------------------------------------------------------------------


@pytest.fixture
def tiny_model_cls():
    """Return a tiny torch.nn.Module class for forward/eval tests."""
    if torch is None:
        pytest.skip("Tiny model requires torch")

    class TinyModel(torch.nn.Module):
        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes, bias=False)

        def forward(self, x):
            n = x.shape[0]
            return self.fc(x.reshape(n, -1))

    return TinyModel


@pytest.fixture
def make_xy():
    """Build small (X, y, meta) suitable for trainer/evaluate paths."""

    def _make(n: int = 20, classes=("NORM", "MI"), include_unknown: bool = False):
        X = np.random.randn(n, 1, 10).astype(np.float32)
        labels = []
        for i in range(n):
            if include_unknown and i % 7 == 0:
                labels.append("Unknown")
            else:
                labels.append(classes[i % len(classes)])
        meta = pd.DataFrame({"i": range(n)})
        return X, labels, meta

    return _make


@pytest.fixture
def default_hparams():
    """Default plotting params used repeatedly in plot-related tests."""
    return dict(
        model="SomeModel",
        lr=0.001,
        bs=64,
        wd=0.0,
        epoch=10,
        prefix="test",
        fname_metric="some_metric",
        fold=1,
    )
