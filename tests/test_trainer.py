# tests/test_trainer.py
"""
Tests for ecg_cnn.training.trainer

Covers
------
    - train_one_epoch():
        • Type validation errors for all parameters
        • End-to-end single and multi-epoch runs with deterministic loss/accuracy
    - evaluate_on_validation():
        • Happy-path average loss/accuracy computation
        • Model mode switching to eval
        • Type validation errors for all parameters
        • Empty dataloader handling (ZeroDivisionError)
    - run_training():
        • Validation errors for config type, tag presence, fold index rules, and unknown models
        • No-fold path with best model saving and summary generation
        • Folded training path with history file writing and tagged file names
        • Handling of 'Unknown' labels in dataset
        • Best epoch selection when lowest loss is not in epoch 1
        • sample_only branch calling load_ptbxl_sample instead of load_ptbxl_full
    - compute_class_weights():
        • Balanced, imbalanced, single-class, and missing-class scenarios
        • Type, shape, and value checks
        • Validation errors for invalid input arrays and num_classes

Notes
-----
    - Seeding is handled in conftest.py; no per-test/file seeding here
    - Heavy training/data-loading is avoided via DummyModel, DummyConfig, and monkeypatch fixtures
    - tmp_path fixture ensures isolation; no artifacts persist between tests
"""

import json
import numpy as np
import os
import pandas as pd
import pytest
import torch
import torch.nn as nn
import types

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR

from ecg_cnn.config.config_loader import TrainConfig
from ecg_cnn.data import data_utils
from ecg_cnn.models import model_utils
from ecg_cnn.training import training_utils, trainer
from ecg_cnn.training.trainer import (
    _DATASET_CACHE,
    train_one_epoch,
    evaluate_on_validation,
    run_training,
)


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------


class DummyModel(nn.Module):
    """Tiny deterministic model: linear on flattened input -> logits."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(10, num_classes, bias=False)
        with torch.no_grad():
            self.fc.weight.fill_(0.01)

    def forward(self, x):
        # x expected shape: (N, C, L). Flatten to (N, 10)
        n = x.shape[0]
        return self.fc(x.reshape(n, -1))


class DummyConfig:
    """
    Duck-typed training config; we monkeypatch trainer.TrainConfig to this class.
    Only fields actually accessed by run_training are included.
    """

    def __init__(
        self,
        *,
        data_dir=None,
        sample_only=False,
        sample_dir=None,
        subsample_frac=1.0,
        sampling_rate=500,
        n_folds=0,
        batch_size=4,
        model="DummyModel",
        lr=1e-3,
        weight_decay=0.0,
        n_epochs=2,
        save_best=True,
        verbose=True,
    ):
        self.data_dir = data_dir
        self.sample_only = sample_only
        self.sample_dir = sample_dir
        self.subsample_frac = subsample_frac
        self.sampling_rate = sampling_rate
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.save_best = save_best
        self.verbose = verbose


class SimpleModel(nn.Module):
    """
    Minimal deterministic feedforward model for trainer tests.

    Architecture
    ------------
    - Flatten input of shape (12, 1000) to a 1D vector
    - Linear -> ReLU -> Linear to produce logits for 5 classes
    - Xavier-uniform initialization for weights, zero bias

    Notes
    -----
    Uses a local torch.Generator for reproducible weight initialization
    without affecting global RNG state.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 1000, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        # Deterministic init via a local generator
        g = torch.Generator().manual_seed(1337)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(g.initial_seed()))
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleDataset(torch.utils.data.Dataset):
    """
    Tiny synthetic dataset for trainer tests.

    Features
    --------
    - 10 samples of shape (12, 1000) with random values
    - 5-class integer labels in range [0, 4]
    - Data generated with a local torch.Generator for reproducibility
      without changing global RNG state.
    """

    def __init__(self):
        g = torch.Generator().manual_seed(4242)
        self.X = torch.randn(10, 12, 1000, generator=g)
        self.y = torch.randint(0, 5, (10,), generator=g)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class ConstantLogitModel(nn.Module):
    """
    Tiny model that always returns the same logits (favoring class 0).
    Useful for deterministic accuracy checks.
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch, n_classes). Class 0 logit slightly higher.
        batch = x.shape[0]
        out = torch.zeros(batch, self.n_classes, dtype=x.dtype, device=x.device)
        out[:, 0] = 0.5
        out[:, 1] = 0.0
        return out


def _make_balanced_dummy_loader(
    n: int = 20, n_classes: int = 2, batch_size: int = 4
) -> DataLoader:
    """
    Create a balanced two-class dataset with simple inputs.
    """
    X = torch.randn(n, 1, 10)  # (N, C, L)
    y = torch.tensor([i % n_classes for i in range(n)], dtype=torch.long)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ------------------------------------------------------------------------------
# def train_one_epoch(model, dataloader, optimizer, criterion, device):
# ------------------------------------------------------------------------------


def test_train_one_epoch_type_errors():
    model = torch.nn.Linear(10, 2)
    data = torch.randn(4, 10)
    targets = torch.randint(0, 2, (4,))
    dataloader = DataLoader(TensorDataset(data, targets))
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    # Wrong model type
    with pytest.raises(
        TypeError, match=r"^model must be an instance of torch.nn.Module"
    ):
        train_one_epoch("not a model", dataloader, optimizer, criterion, device)

    # Wrong dataloader type
    with pytest.raises(
        TypeError, match=r"^dataloader must be a torch.utils.data.DataLoader"
    ):
        train_one_epoch(model, "not a dataloader", optimizer, criterion, device)

    # Wrong optimizer type
    with pytest.raises(TypeError, match=r"^optimizer must be a torch.optim.Optimizer"):
        train_one_epoch(model, dataloader, "not an optimizer", criterion, device)

    # Wrong criterion type
    with pytest.raises(TypeError, match=r"^criterion must be callable"):
        train_one_epoch(model, dataloader, optimizer, "not callable", device)

    # Wrong device type
    with pytest.raises(TypeError, match=r"^device must be a torch.device"):
        train_one_epoch(model, dataloader, optimizer, criterion, "not a device")


def test_train_one_epoch_end_to_end():
    model = SimpleModel()
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # More stable than SGD on random data
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    initial_loss, initial_accuracy = train_one_epoch(
        model, dataloader, optimizer, criterion, device
    )
    assert isinstance(initial_loss, float)
    assert initial_loss > 0.0

    later_loss, later_accuracy = train_one_epoch(
        model, dataloader, optimizer, criterion, device
    )
    assert isinstance(later_loss, float)
    assert isinstance(later_accuracy, float)
    assert later_loss >= 0.0

    # Keep the loss non-increasing
    assert (
        later_loss <= initial_loss + 1e-6
    ), f"Expected non-increasing loss: {initial_loss:.5f} -> {later_loss:.5f}"


def test_train_one_epoch_onecycle_steps_scheduler(monkeypatch):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny 1D conv model
    class Tiny1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 4, kernel_size=3, padding=1)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            x = torch.relu(self.conv(x))  # (N,4,T)
            x = x.mean(dim=-1)  # (N,4)
            return self.head(x)  # (N,2)

    # Small dataset: 10 samples, 1 channel, 32 timepoints, 2 classes
    X = torch.randn(10, 1, 32)
    y = torch.tensor([0, 1] * 5)
    loader = DataLoader(TensorDataset(X, y), batch_size=5, shuffle=False)

    model = Tiny1D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # OneCycle needs steps_per_epoch
    sched = OneCycleLR(opt, max_lr=1e-3, epochs=1, steps_per_epoch=len(loader))

    # CrossEntropy
    crit = nn.CrossEntropyLoss()

    # Call the helper directly with the scheduler to hit line 105
    train_loss, train_acc = train_one_epoch(
        model, loader, opt, crit, device, scheduler=sched
    )

    # If OneCycle step ran per batch, LR should have changed from initial
    assert isinstance(train_loss, float) and 0.0 <= train_acc <= 1.0
    assert opt.param_groups[0]["lr"] != 1e-3


def test_run_training_cosine_branch_hits_init_and_epoch_step(monkeypatch):
    """Covers cosine init and epoch-level scheduler.step without touching PTB-XL data or real ECGConvNet."""

    tmp = Path(os.getenv("PYTEST_TMPDIR", "/tmp")) / "ecg_sbx_cosine"
    (tmp / "models").mkdir(parents=True, exist_ok=True)
    (tmp / "history").mkdir(parents=True, exist_ok=True)
    (tmp / "artifacts").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ECG_CNN_OUTPUT_DIR", str(tmp))
    monkeypatch.setenv("ECG_CNN_RESULTS_DIR", str(tmp / "results"))
    monkeypatch.setattr("torch.save", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("json.dump", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp / "models", raising=False)
    monkeypatch.setattr(trainer, "HISTORY_DIR", tmp / "history", raising=False)
    monkeypatch.setattr(trainer, "ARTIFACTS_DIR", tmp / "artifacts", raising=False)

    # --- force cosine scheduler branch ---
    monkeypatch.setenv("ECG_SCHEDULER", "cosine")
    monkeypatch.setenv("ECG_SCHED_TMAX", "2")

    # --- force a 2-class universe to match fake labels ---
    monkeypatch.setattr(trainer, "FIVE_SUPERCLASSES", ["MI", "NORM"], raising=False)
    monkeypatch.setattr(model_utils, "FIVE_SUPERCLASSES", ["MI", "NORM"], raising=False)

    # --- tiny, shape-agnostic model to avoid FC size issues ---
    class _TinyNet(nn.Module):
        def __init__(self, in_ch: int = 12, n_classes: int = 2, **_):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(in_ch, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(n_classes))

        def forward(self, x):
            return self.head(self.features(x))

    # patch every plausible construction path to resolve ECGConvNet -> _TinyNet
    monkeypatch.setattr(trainer, "ECGConvNet", _TinyNet, raising=False)
    reg_t = dict(getattr(trainer, "MODEL_CLASSES", {}))
    reg_t["ECGConvNet"] = _TinyNet
    monkeypatch.setattr(trainer, "MODEL_CLASSES", reg_t, raising=False)
    monkeypatch.setattr(model_utils, "ECGConvNet", _TinyNet, raising=False)
    reg_mu = dict(getattr(model_utils, "MODEL_CLASSES", {}))
    reg_mu["ECGConvNet"] = _TinyNet
    monkeypatch.setattr(model_utils, "MODEL_CLASSES", reg_mu, raising=False)
    if hasattr(trainer, "get_model"):
        monkeypatch.setattr(
            trainer, "get_model", lambda *_a, **kw: _TinyNet(**kw), raising=False
        )
    if hasattr(model_utils, "get_model"):
        monkeypatch.setattr(
            model_utils, "get_model", lambda *_a, **kw: _TinyNet(**kw), raising=False
        )

    # --- fake sample loader: (N, 12, 32) + .loc-capable meta ---
    def _fake_sample(sample_dir=None, ptb_path=None):
        N, C, T = 16, 12, 32
        X = np.random.randn(N, C, T).astype(np.float32)
        y = np.array(["MI", "NORM"] * (N // 2))
        meta = pd.DataFrame(
            {"superclass": y, "strat_fold": np.array([0, 1] * (N // 2))},
            index=np.arange(N),
        )
        return X, y, meta

    # patch both the defining module and the import-bound symbol inside trainer
    monkeypatch.setattr(data_utils, "load_ptbxl_sample", _fake_sample)
    monkeypatch.setattr(trainer, "load_ptbxl_sample", _fake_sample, raising=False)

    # --- run ---
    cfg = TrainConfig(
        model="ECGConvNet",
        lr=1e-3,
        batch_size=8,
        weight_decay=5e-4,
        n_epochs=2,  # at least one epoch step
        save_best=True,
        sample_only=True,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        n_folds=2,
        verbose=False,
    )
    summary = trainer.run_training(cfg, fold_idx=0, tag="cov_cosine")
    assert isinstance(summary, dict)
    assert "loss" in summary and "best_epoch" in summary


def test_run_training_onecycle_branch_hits_init_and_inline_loop(monkeypatch):
    """Covers OneCycle init and per-batch inline stepping without touching PTB-XL data or real ECGConvNet."""

    tmp = Path(os.getenv("PYTEST_TMPDIR", "/tmp")) / "ecg_sbx_onecycle"
    (tmp / "models").mkdir(parents=True, exist_ok=True)
    (tmp / "history").mkdir(parents=True, exist_ok=True)
    (tmp / "artifacts").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ECG_CNN_OUTPUT_DIR", str(tmp))
    monkeypatch.setenv("ECG_CNN_RESULTS_DIR", str(tmp / "results"))
    monkeypatch.setattr("torch.save", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("json.dump", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp / "models", raising=False)
    monkeypatch.setattr(trainer, "HISTORY_DIR", tmp / "history", raising=False)
    monkeypatch.setattr(trainer, "ARTIFACTS_DIR", tmp / "artifacts", raising=False)

    # --- force OneCycle scheduler branch ---
    monkeypatch.setenv("ECG_SCHEDULER", "onecycle")
    monkeypatch.setenv("ECG_SCHED_MAX_LR", "0.0015")
    monkeypatch.setenv("ECG_SCHED_PCT_START", "0.3")
    monkeypatch.setenv("ECG_SCHED_DIV", "25")
    monkeypatch.setenv("ECG_SCHED_FINAL_DIV", "10000")

    # --- force a 2-class universe to match fake labels ---
    monkeypatch.setattr(trainer, "FIVE_SUPERCLASSES", ["MI", "NORM"], raising=False)
    monkeypatch.setattr(model_utils, "FIVE_SUPERCLASSES", ["MI", "NORM"], raising=False)

    # --- tiny, shape-agnostic model ---
    class _TinyNet(nn.Module):
        def __init__(self, in_ch: int = 12, n_classes: int = 2, **_):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(in_ch, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(n_classes))

        def forward(self, x):
            return self.head(self.features(x))

    # patch every plausible construction path to resolve ECGConvNet -> _TinyNet
    monkeypatch.setattr(trainer, "ECGConvNet", _TinyNet, raising=False)
    reg_t = dict(getattr(trainer, "MODEL_CLASSES", {}))
    reg_t["ECGConvNet"] = _TinyNet
    monkeypatch.setattr(trainer, "MODEL_CLASSES", reg_t, raising=False)
    monkeypatch.setattr(model_utils, "ECGConvNet", _TinyNet, raising=False)
    reg_mu = dict(getattr(model_utils, "MODEL_CLASSES", {}))
    reg_mu["ECGConvNet"] = _TinyNet
    monkeypatch.setattr(model_utils, "MODEL_CLASSES", reg_mu, raising=False)
    if hasattr(trainer, "get_model"):
        monkeypatch.setattr(
            trainer, "get_model", lambda *_a, **kw: _TinyNet(**kw), raising=False
        )
    if hasattr(model_utils, "get_model"):
        monkeypatch.setattr(
            model_utils, "get_model", lambda *_a, **kw: _TinyNet(**kw), raising=False
        )

    # --- fake sample loader: (N, 12, 32) + .loc-capable meta ---
    def _fake_sample(sample_dir=None, ptb_path=None):
        N, C, T = 12, 12, 32
        X = np.random.randn(N, C, T).astype(np.float32)
        y = np.array(["MI", "NORM"] * (N // 2))
        meta = pd.DataFrame(
            {"superclass": y, "strat_fold": np.array([0, 1] * (N // 2))},
            index=np.arange(N),
        )
        return X, y, meta

    monkeypatch.setattr(data_utils, "load_ptbxl_sample", _fake_sample)
    monkeypatch.setattr(trainer, "load_ptbxl_sample", _fake_sample, raising=False)

    # --- run ---
    cfg = TrainConfig(
        model="ECGConvNet",
        lr=1e-3,
        batch_size=6,
        weight_decay=5e-4,
        n_epochs=2,  # triggers per-batch stepping
        save_best=True,
        sample_only=True,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        n_folds=2,
        verbose=False,
    )
    summary = trainer.run_training(cfg, fold_idx=0, tag="cov_onecycle_inline")
    assert isinstance(summary, dict)
    assert "loss" in summary and "best_epoch" in summary


# ------------------------------------------------------------------------------
# def evaluate_on_validation(model, dataloader, criterion, device):
# ------------------------------------------------------------------------------


def test_evaluate_on_validation_happy_path_cpu():
    device = torch.device("cpu")
    model = ConstantLogitModel(n_classes=2).to(device)
    dataloader = _make_balanced_dummy_loader(n=20, n_classes=2, batch_size=5)
    criterion = nn.CrossEntropyLoss()

    avg_loss, acc = evaluate_on_validation(model, dataloader, criterion, device)

    # Types and ranges
    assert isinstance(avg_loss, float)
    assert isinstance(acc, float)
    assert avg_loss >= 0.0
    assert 0.0 <= acc <= 1.0

    # Since model always favors class 0 and labels are 50/50, expect ~0.5 accuracy.
    assert abs(acc - 0.5) < 1e-6


def test_evaluate_on_validation_sets_eval_mode():
    device = torch.device("cpu")
    model = ConstantLogitModel().to(device)
    model.train()  # Ensure it starts in training mode
    dataloader = _make_balanced_dummy_loader(n=8, n_classes=2, batch_size=4)
    criterion = nn.CrossEntropyLoss()

    _ = evaluate_on_validation(model, dataloader, criterion, device)

    # Function should put the model into eval mode
    assert model.training is False


@pytest.mark.parametrize(
    "bad_model, dataloader, criterion, device, err_msg",
    [
        (
            "not_a_model",
            _make_balanced_dummy_loader(4, 2, 2),
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            r"^model must be an instance of torch.nn.Module",
        ),
        (
            ConstantLogitModel(),
            "not_a_dataloader",
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            r"^dataloader must be a torch.utils.data.DataLoader",
        ),
        (
            ConstantLogitModel(),
            _make_balanced_dummy_loader(4, 2, 2),
            object(),  # not callable
            torch.device("cpu"),
            r"^criterion must be callable",
        ),
        (
            ConstantLogitModel(),
            _make_balanced_dummy_loader(4, 2, 2),
            nn.CrossEntropyLoss(),
            "cpu",  # not a torch.device
            r"^device must be a torch.device",
        ),
    ],
)
def test_evaluate_on_validation_raises_on_bad_types(
    bad_model, dataloader, criterion, device, err_msg
):
    with pytest.raises(TypeError, match=err_msg):
        evaluate_on_validation(bad_model, dataloader, criterion, device)


def test_evaluate_on_validation_empty_dataloader_raises_zero_division():
    device = torch.device("cpu")
    model = ConstantLogitModel().to(device)
    criterion = nn.CrossEntropyLoss()

    # Build an empty dataset/dataloader
    X = torch.empty(0, 1, 10)
    y = torch.empty(0, dtype=torch.long)
    empty_loader = DataLoader(TensorDataset(X, y), batch_size=4)

    with pytest.raises(ZeroDivisionError, match=r"^float division by zero"):
        evaluate_on_validation(model, empty_loader, criterion, device)


# ------------------------------------------------------------------------------
# Fixtures / common patches for run_training
# ------------------------------------------------------------------------------


@pytest.fixture
def patch_trainer_minimal(monkeypatch, tmp_path, make_xy):
    """
    Patch the trainer module so run_training can execute quickly
    without touching real data or big models. Uses shared factories
    from conftest.py where appropriate.
    """
    # Redirect output dirs and paths
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.MODELS_DIR", tmp_path / "models", raising=False
    )
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.HISTORY_DIR", tmp_path / "history", raising=False
    )
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.PTBXL_DATA_DIR", tmp_path / "ptbxl", raising=False
    )

    # Accept our DummyConfig for isinstance checks
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.TrainConfig", DummyConfig, raising=False
    )

    # Make FIVE_SUPERCLASSES 2-long (binary) to match our fake data
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )

    # Provide a model_utils namespace with DummyModel
    model_utils = types.SimpleNamespace(DummyModel=DummyModel)
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.model_utils", model_utils, raising=False
    )

    # Fake loaders: choose full vs sample identically, but use the shared make_xy
    def fake_full(data_dir, subsample_frac, sampling_rate):
        # include_unknown=True ensures filtering code is exercised in some tests
        return make_xy(n=30, classes=("NORM", "MI"), include_unknown=True)

    def fake_sample(sample_dir, ptb_path):
        return make_xy(n=12, classes=("NORM", "MI"), include_unknown=False)

    monkeypatch.setattr(
        "ecg_cnn.training.trainer.load_ptbxl_full", fake_full, raising=False
    )
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.load_ptbxl_sample", fake_sample, raising=False
    )

    # Class weights: simple ones
    def fake_class_weights(y_np, num_classes):
        return torch.ones(num_classes, dtype=torch.float32)

    monkeypatch.setattr(
        training_utils,
        "compute_class_weights",
        fake_class_weights,
        raising=False,
    )

    # train_one_epoch / evaluate_on_validation: return deterministic values per call
    def fake_train_one_epoch(model, dl, opt, crit, device):
        # emulate increasing loss so save_best triggers only on epoch 1
        cnt = getattr(model, "_epoch_cnt", 0) + 1
        setattr(model, "_epoch_cnt", cnt)
        loss = 1.0 + 0.1 * (cnt - 1)  # 1.0, 1.1, 1.2, ...
        acc = 0.6 + 0.05 * (cnt - 1)  # 0.60, 0.65, 0.70, ...
        return float(loss), float(acc)

    def fake_eval(model, dl, crit, device):
        # stable val metrics
        return 0.42, 0.66

    monkeypatch.setattr(
        "ecg_cnn.training.trainer.train_one_epoch", fake_train_one_epoch, raising=False
    )
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.evaluate_on_validation", fake_eval, raising=False
    )

    return tmp_path


# ------------------------------------------------------------------------------
# run_training: validation errors
# ------------------------------------------------------------------------------


def test_run_training_rejects_non_config(monkeypatch):
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.TrainConfig", DummyConfig, raising=False
    )
    with pytest.raises(TypeError, match=r"^cfg must be TrainConfig"):
        run_training(config="not_a_config", fold_idx=None, tag="x")


def test_run_training_requires_tag(patch_trainer_minimal):
    cfg = DummyConfig()
    with pytest.raises(
        ValueError,
        match=r"^Missing tag — must be provided to disambiguate file outputs.",
    ):
        run_training(config=cfg, fold_idx=None, tag=None)


@pytest.mark.parametrize(
    "fold_idx,n_folds,errmsg",
    [
        (-1, 3, r"^fold_idx must be a non-negative integer if provided."),
        (0, 1, r"must be an integer >= 2"),
        (5, 3, r"out of range"),
    ],
)
def test_run_training_fold_idx_validation(
    monkeypatch, patch_trainer_minimal, fold_idx, n_folds, errmsg
):
    cfg = DummyConfig(n_folds=n_folds)
    with pytest.raises(ValueError, match=errmsg):
        run_training(config=cfg, fold_idx=fold_idx, tag="ta")


def test_run_training_unknown_model(monkeypatch, patch_trainer_minimal):
    # Remove DummyModel from model_utils to trigger error
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.model_utils", types.SimpleNamespace(), raising=False
    )
    cfg = DummyConfig()
    with pytest.raises(ValueError, match=r"^Unknown model name"):
        run_training(config=cfg, fold_idx=None, tag="te")


# ------------------------------------------------------------------------------
# run_training: no-fold path
# ------------------------------------------------------------------------------


def test_run_training_no_folds_saves_best_and_summary(patch_trainer_minimal):
    cfg = DummyConfig(n_epochs=2, n_folds=0, save_best=True, batch_size=8)
    summary = run_training(config=cfg, fold_idx=None, tag="model_lr001_bs8_wd0")

    # Summary basics
    assert isinstance(summary, dict)
    for key in [
        "loss",
        "elapsed_min",
        "fold",
        "model",
        "model_path",
        "best_epoch",
        "train_losses",
        "val_losses",
        "train_accs",
        "val_accs",
    ]:
        assert key in summary

    # fold is None on no-fold path
    assert summary["fold"] is None

    # Model path exists and includes tag (no fold suffix on filename)
    model_path = summary["model_path"]
    assert isinstance(model_path, str) and model_path.endswith(".pth")
    assert "model_best_model_lr001_bs8_wd0.pth" in model_path
    assert Path(model_path).exists()

    # val_* are None because no validation loader on no-fold path
    assert summary["val_losses"] is None
    assert summary["val_accs"] is None

    # best_epoch should be 1 given our fake loss pattern
    assert summary["best_epoch"] == 1


# ------------------------------------------------------------------------------
# run_training: fold path
# ------------------------------------------------------------------------------


def test_run_training_with_folds_writes_history_and_tagged_names(
    patch_trainer_minimal, tmp_path, monkeypatch
):
    # Redirect trainer's module-level paths into the tmp sandbox
    hist_dir = tmp_path / "history"
    models_dir = tmp_path / "models"
    artifacts_dir = tmp_path / "artifacts"  # where CR CSVs go

    hist_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(trainer, "HISTORY_DIR", hist_dir, raising=False)
    monkeypatch.setattr(trainer, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(trainer, "ARTIFACTS_DIR", artifacts_dir, raising=False)

    # Minimal config with folds so a val_dataloader exists and CR CSV is written
    cfg = DummyConfig(n_epochs=3, n_folds=3, save_best=True, batch_size=4)

    # Choose fold 0 (1-based in filenames)
    summary = run_training(config=cfg, fold_idx=0, tag="demo_tag")

    # Summary reflects fold numbering
    assert summary["fold"] == 1
    assert summary["model_path"].endswith("model_best_demo_tag_fold1.pth")
    assert Path(summary["model_path"]).exists()

    # History file exists and has correct name/content under tmp_path
    hist_file = hist_dir / "history_demo_tag_fold1.json"
    assert hist_file.exists()

    with hist_file.open("r") as f:
        hist = json.load(f)
    assert len(hist["train_loss"]) == cfg.n_epochs
    assert len(hist["train_acc"]) == cfg.n_epochs
    assert len(hist["val_loss"]) == cfg.n_epochs
    assert len(hist["val_acc"]) == cfg.n_epochs

    # Classification report CSV should also be in the sandbox (since val loader exists)
    cr_csv = artifacts_dir / "classification_report_demo_tag_fold1.csv"
    assert cr_csv.exists()


# ------------------------------------------------------------------------------
# run_training: Unknown labels filtered
# ------------------------------------------------------------------------------


def test_run_training_filters_unknown_labels(
    monkeypatch, patch_trainer_minimal, make_xy
):
    # Force loaders to include 'Unknown' more aggressively
    def loader_with_unknowns(*args, **kwargs):
        return make_xy(n=15, classes=("NORM", "MI"), include_unknown=True)

    monkeypatch.setattr(
        "ecg_cnn.training.trainer.load_ptbxl_full",
        loader_with_unknowns,
        raising=False,
    )

    cfg = DummyConfig(n_folds=0, n_epochs=1)
    summary = run_training(config=cfg, fold_idx=None, tag="t_unknowns")

    # Successful run implies Unknowns did not break encoding/splitting
    assert isinstance(summary, dict)


def test_run_training_no_folds_best_epoch_2(patch_trainer_minimal, monkeypatch):
    """
    Simulate a run where the lowest loss occurs on epoch 2,
    so best_epoch should be 2 in the summary.
    """

    # Override fake_train_one_epoch to make epoch 2 the best
    def fake_train_one_epoch(model, dl, opt, crit, device):
        cnt = getattr(model, "_epoch_cnt", 0) + 1
        setattr(model, "_epoch_cnt", cnt)
        # Loss: higher on epoch 1, lower on epoch 2, then higher again
        if cnt == 1:
            loss = 1.0
        elif cnt == 2:
            loss = 0.5  # best
        else:
            loss = 0.8
        acc = 0.5 + 0.1 * cnt
        return float(loss), float(acc)

    monkeypatch.setattr(
        "ecg_cnn.training.trainer.train_one_epoch", fake_train_one_epoch, raising=False
    )

    cfg = DummyConfig(n_epochs=3, n_folds=0, save_best=True, batch_size=8)
    summary = run_training(config=cfg, fold_idx=None, tag="model_lr001_bs8_wd0_best2")

    assert summary["best_epoch"] == 2
    assert Path(summary["model_path"]).exists()


def test_run_training_sample_only_calls_sample_loader(
    patch_trainer_minimal, monkeypatch
):
    """
    Cover the `if config.sample_only:` branch by ensuring we call load_ptbxl_sample,
    pass through the expected args, and never call load_ptbxl_full.
    """
    called = {"sample": False, "full": False}
    captured = {"ptb_path": None, "sample_dir": None}

    def fake_sample_loader(*, sample_dir, ptb_path):
        called["sample"] = True
        captured["ptb_path"] = ptb_path
        captured["sample_dir"] = sample_dir
        return (  # (X, y, meta)
            np.random.randn(10, 1, 10).astype(np.float32),
            ["NORM", "MI"] * 5,
            __import__("pandas").DataFrame({"i": range(10)}),
        )

    def fake_full_loader(*args, **kwargs):
        called["full"] = True
        raise AssertionError(
            "load_ptbxl_full should NOT be called when sample_only=True"
        )

    monkeypatch.setattr(
        "ecg_cnn.training.trainer.load_ptbxl_sample",
        fake_sample_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.load_ptbxl_full", fake_full_loader, raising=False
    )

    # Provide explicit data_dir and sample_dir to verify they’re forwarded
    data_root = patch_trainer_minimal / "ptbxl_data_root"
    sample_root = patch_trainer_minimal / "sample_dir"

    cfg = DummyConfig(
        sample_only=True,
        sample_dir=str(sample_root),
        data_dir=str(data_root),
        n_epochs=1,
        n_folds=0,
        save_best=False,  # faster, no file write needed here
        verbose=False,
    )

    summary = run_training(config=cfg, fold_idx=None, tag="sample_branch")

    assert isinstance(summary, dict)
    assert called["sample"] is True
    assert called["full"] is False
    # Verify the function received the exact paths we set
    assert captured["ptb_path"] == data_root
    assert captured["sample_dir"] == str(sample_root)


def test_run_training_cache_hit_assigns(monkeypatch, tmp_path):
    """
    Verifies that run_training() uses the in-process dataset cache (no disk load),
    and that all filesystem writes are sandboxed. Also confirms the per-fold
    classification report path is exercised without touching real outputs.
    """

    # --- Sandbox module-level output paths ---
    hist_dir = tmp_path / "history"
    models_dir = tmp_path / "models"
    artifacts_dir = tmp_path / "artifacts"
    hist_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(trainer, "HISTORY_DIR", hist_dir, raising=False)
    monkeypatch.setattr(trainer, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(trainer, "ARTIFACTS_DIR", artifacts_dir, raising=False)

    # --- Make isinstance(config, TrainConfig) accept DummyConfig ---
    monkeypatch.setattr(trainer, "TrainConfig", DummyConfig, raising=False)

    # --- Wire minimal model/utils ---
    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes)

        def forward(self, x):
            return self.fc(x.mean(dim=1))  # [N, 1, 10] -> [N, 10]

    monkeypatch.setattr(
        trainer,
        "model_utils",
        types.SimpleNamespace(DummyModel=DummyModel),
        raising=False,
    )
    monkeypatch.setattr(trainer, "FIVE_SUPERCLASSES", ["A", "B"], raising=False)
    monkeypatch.setattr(
        training_utils,
        "compute_class_weights",
        lambda y_np, n: torch.ones(n),
        raising=False,
    )

    # --- Config that takes the non-sample key path (exercises cache) ---
    cfg = DummyConfig(
        data_dir=str(tmp_path),
        sample_only=False,
        subsample_frac=0.5,
        sampling_rate=100,
        n_folds=2,  # ensures a val_dataloader exists (so CR path triggers)
        batch_size=4,
        model="DummyModel",
        n_epochs=1,  # keep fast
        save_best=False,  # no checkpoint write needed
        verbose=False,
    )

    # --- Seed the exact cache key run_training computes ---
    trainer._DATASET_CACHE.clear()
    data_dir = Path(cfg.data_dir) if cfg.data_dir else trainer.PTBXL_DATA_DIR
    key = (str(data_dir.resolve()), float(cfg.subsample_frac), int(cfg.sampling_rate))
    N = 10
    X = np.random.randn(N, 1, 10).astype("float32")
    y = ["A" if i % 2 == 0 else "B" for i in range(N)]
    meta = pd.DataFrame({"id": np.arange(N)})
    trainer._DATASET_CACHE[key] = (X, y, meta)

    # --- Fail loudly if disk loader is touched on cache hit ---
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("load_ptbxl_full should not be called on cache hit")

    monkeypatch.setattr(
        trainer, "load_ptbxl_full", _should_not_be_called, raising=False
    )

    # --- Execute ---
    out = trainer.run_training(cfg, fold_idx=0, tag="t1")

    # --- Assertions ---
    assert isinstance(out, dict)
    assert len(trainer._DATASET_CACHE) == 1  # cache unchanged
    assert out["fold"] == 1  # 1-based fold in summary
    assert (hist_dir / "history_t1_fold1.json").exists()  # history written to sandbox

    # Trainer writes the per-fold classification report directly into ARTIFACTS_DIR
    cr_path = artifacts_dir / "classification_report_t1_fold1.csv"
    assert cr_path.exists()
    with cr_path.open() as fh:
        header = fh.readline().strip()
    assert header == "label,precision,recall,f1-score,support"


def test_run_training_raises_when_required_fields_missing(monkeypatch):
    # Make isinstance(config, TrainConfig) accept DummyConfig WITHOUT extra imports
    monkeypatch.setitem(run_training.__globals__, "TrainConfig", DummyConfig)

    cfg = DummyConfig()
    # Has to be truly missing (run_training uses hasattr), so delete the attribute
    del cfg.model

    with pytest.raises(
        TypeError, match=r"^config is missing required fields:\s*\['model'\]"
    ):
        run_training(cfg, tag="ok")


def test_run_training_sample_only_cache_hit_uses_cached_data(
    patch_trainer_minimal, monkeypatch
):
    """
    Covers the sample_only cache-hit branch:

        cached = _DATASET_CACHE.get(key)
        if cached is not None:
            X, y, meta = cached
    """

    # Arrange: config that uses sample_only path
    sample_dir = patch_trainer_minimal / "sample_dir"
    sample_dir.mkdir(parents=True, exist_ok=True)

    cfg = DummyConfig(
        model="DummyModel",
        n_epochs=1,
        n_folds=0,
        save_best=False,  # avoid file I/O assertions here
        batch_size=4,
        sample_only=True,
        sample_dir=str(sample_dir),
        sampling_rate=100,  # becomes part of the cache key
        subsample_frac=1.0,
        data_dir=None,  # unused in sample_only
    )

    # Build the exact cache key used by run_training() for sample_only
    key = (str(Path(cfg.sample_dir).resolve()), "SAMPLE_ONLY", int(cfg.sampling_rate))

    # Seed cache with small fake dataset
    X = np.random.randn(8, 10).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    meta = pd.DataFrame({"idx": range(len(y))})
    trainer._DATASET_CACHE[key] = (X, y, meta)

    # Make sure the loader would explode if called (to prove we used the cache)
    def _should_not_be_called(**kwargs):
        raise RuntimeError("load_ptbxl_sample should not be called on cache hit")

    monkeypatch.setattr(trainer, "load_ptbxl_sample", _should_not_be_called)

    # Act
    summary = trainer.run_training(config=cfg, fold_idx=None, tag="cache_hit_sample")

    # Assert: basic summary keys exist
    for k in [
        "loss",
        "elapsed_min",
        "fold",
        "model",
        "model_path",
        "best_epoch",
        "train_losses",
        "val_losses",
        "train_accs",
        "val_accs",
    ]:
        assert k in summary

    # No folds -> fold is None and val_* are None
    assert summary["fold"] is None
    assert summary["val_losses"] is None
    assert summary["val_accs"] is None

    # Sanity: correct model name propagated
    assert summary["model"] == "DummyModel"


def test_run_training_triggers_early_stopping_and_breaks(
    patch_trainer_minimal, monkeypatch, capsys
):
    """
    Covers the early-stopping branch:

        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"...")
                break
    """

    # Arrange: config in NO-FOLD mode (monitor = train loss), small dataset
    data_dir = patch_trainer_minimal / "ptbxl"
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = DummyConfig(
        model="DummyModel",
        n_epochs=10,  # > patience so we can observe the break
        n_folds=0,
        save_best=True,
        batch_size=4,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=str(data_dir),
    )

    # Seed cache for the FULL-DATA path to avoid calling load_ptbxl_full
    key = (
        str(Path(cfg.data_dir).resolve()),
        float(cfg.subsample_frac),
        int(cfg.sampling_rate),
    )
    X = np.random.randn(16, 10).astype(np.float32)
    y = np.array([0, 1] * 8, dtype=np.int64)
    meta = pd.DataFrame({"idx": range(len(y))})
    trainer._DATASET_CACHE[key] = (X, y, meta)

    # Make full loader blow up if touched (it shouldn't be)
    def _should_not_be_called(**kwargs):
        raise RuntimeError(
            "load_ptbxl_full should not be called (cache should be used)"
        )

    monkeypatch.setattr(trainer, "load_ptbxl_full", _should_not_be_called)

    # Force non-improving training so early stopping triggers:
    # epoch 1 -> best (1.0); subsequent epochs -> same loss (1.0) => bad_epochs increments.
    monkeypatch.setattr(trainer, "train_one_epoch", lambda *a, **kw: (1.0, 0.5))
    # No folds => val_dataloader is None, so evaluate_on_validation isn't used, but patch anyway
    monkeypatch.setattr(trainer, "evaluate_on_validation", lambda *a, **kw: (1.0, 0.5))

    # Act
    summary = trainer.run_training(config=cfg, fold_idx=None, tag="early_stop_check")

    out = capsys.readouterr().out

    # Assert: early-stopping message printed and loop broke early
    assert "Early stopping at epoch" in out

    # With patience=5 in run_training, stopping occurs by or before epoch 6
    assert summary["best_epoch"] == 1  # first epoch is the best (monitored=1.0)
    assert summary["best_epoch"] < cfg.n_epochs

    # No folds -> val_* are None in summary
    assert summary["fold"] is None
    assert summary["val_losses"] is None
    assert summary["val_accs"] is None


def test_run_training_skips_val_block(monkeypatch, tmp_path, capsys):
    """
    Cover false branch by making the validation DataLoader None.
    - Real DataLoader for train (type checks pass)
    - 1 epoch so train_loss/acc exist
    - 2 samples/class -> valid StratifiedKFold
    - Class weights length == model num_classes
    - Scheduler patched as a CLASS so isinstance(...) works
    """

    # Sandbox outputs
    monkeypatch.setattr(trainer, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(trainer, "HISTORY_DIR", tmp_path / "history", raising=False)
    monkeypatch.setattr(trainer, "ARTIFACTS_DIR", tmp_path / "artifacts", raising=False)
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path / "models", raising=False)
    monkeypatch.setattr(trainer, "RESULTS_DIR", tmp_path / "results", raising=False)

    # Tiny, valid sample dataset: 2 samples per class (5 classes -> 10 rows)
    FIVE = getattr(trainer, "FIVE_SUPERCLASSES", ["C0", "C1", "C2", "C3", "C4"])

    def _fake_sample(sample_dir=None, ptb_path=None):
        n_classes = len(FIVE)  # 5
        reps = 2  # two per class
        N = n_classes * reps  # 10
        X = np.zeros((N, 1, 8), dtype=np.float32)
        y = [cls for cls in FIVE for _ in range(reps)]
        meta = pd.DataFrame({"id": list(range(N))})
        return X, y, meta

    monkeypatch.setattr(trainer, "load_ptbxl_sample", _fake_sample, raising=True)

    # Force CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=True)

    # Class weights must match model output size (len(FIVE)=5)
    monkeypatch.setattr(
        training_utils,
        "compute_class_weights",
        lambda y, n: torch.ones(len(FIVE)),
        raising=True,
    )

    # Minimal real model so backward() is valid and fast
    class _MiniNet(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.fc = torch.nn.Linear(8, num_classes)  # flatten 1*8 -> logits

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)

    # Make model_utils.<ModelName>() return _MiniNet
    def _make_model(num_classes):
        return _MiniNet(num_classes)

    monkeypatch.setattr(
        trainer,
        "model_utils",
        types.SimpleNamespace(ECGConvNet=_make_model),
        raising=False,
    )

    # Patch LR scheduler with a CLASS (so isinstance(..., ReduceLROnPlateau) works)
    class _DummyReduce:
        def __init__(self, *a, **k):
            pass

        def step(self, *a, **k):
            pass

    monkeypatch.setattr(
        torch.optim.lr_scheduler, "ReduceLROnPlateau", _DummyReduce, raising=True
    )

    # DataLoader wrapper: 1st call -> real DataLoader; 2nd call (val) -> None
    _dl_calls = {"n": 0}

    def _DL_wrapper(*a, **k):
        _dl_calls["n"] += 1
        if _dl_calls["n"] == 1:
            ds = a[0]
            return DataLoader(
                ds,
                batch_size=k.get("batch_size", 2),
                shuffle=k.get("shuffle", False),
                pin_memory=k.get("pin_memory", False),
            )
        return None

    monkeypatch.setattr(trainer, "DataLoader", _DL_wrapper, raising=True)

    # Real TrainConfig; n_folds>=2 since we pass fold_idx; 1 epoch to define train_*.
    cfg = TrainConfig(
        model="ECGConvNet",
        lr=1e-3,
        batch_size=2,
        weight_decay=0.0,
        n_epochs=1,
        save_best=False,
        sample_only=True,  # uses _fake_sample
        subsample_frac=0.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
        n_folds=2,  # required with fold_idx
        plots_enable_ovr=False,
        plots_ovr_classes=[],
    ).finalize()

    # Run: fold path; val loader is None -> skip CSV section (551–586)
    summary = run_training(cfg, fold_idx=0, tag="t")

    out = capsys.readouterr().out
    assert "Saved training history to:" in out
    assert "Saved classification report to:" not in out  # false branch covered

    # Footer: val_* chosen as None; train_* exist (floats)
    assert summary["fold"] == 1
    assert summary["val_accs"] is None
    assert summary["val_losses"] is None
    assert isinstance(summary["train_losses"], float)
    assert isinstance(summary["train_accs"], float)
