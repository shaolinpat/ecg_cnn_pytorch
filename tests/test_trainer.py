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
from pathlib import Path
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_cnn.training.trainer import (
    train_one_epoch,
    evaluate_on_validation,
    run_training,
    compute_class_weights,
)

# ------------------------------------------------------------------------------
# Helpers: lightweight fakes
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
        "ecg_cnn.training.trainer.compute_class_weights",
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
        run_training(config=cfg, fold_idx=fold_idx, tag="t")


def test_run_training_unknown_model(monkeypatch, patch_trainer_minimal):
    # Remove DummyModel from model_utils to trigger error
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.model_utils", types.SimpleNamespace(), raising=False
    )
    cfg = DummyConfig()
    with pytest.raises(ValueError, match=r"^Unknown model name"):
        run_training(config=cfg, fold_idx=None, tag="t")


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
    patch_trainer_minimal, tmp_path
):
    cfg = DummyConfig(n_epochs=3, n_folds=3, save_best=True, batch_size=4)

    # choose fold 0
    summary = run_training(config=cfg, fold_idx=0, tag="demo_tag")

    # Summary reflects fold numbering (1-based in returned dict)
    assert summary["fold"] == 1
    assert summary["model_path"].endswith("model_best_demo_tag_fold1.pth")
    assert Path(summary["model_path"]).exists()

    # History file exists and has correct name
    hist_file = tmp_path / "history" / "history_demo_tag_fold1.json"
    assert hist_file.exists()

    # History contents length should equal n_epochs
    with open(hist_file, "r") as f:
        hist = json.load(f)
    assert len(hist["train_loss"]) == cfg.n_epochs
    assert len(hist["train_acc"]) == cfg.n_epochs
    assert len(hist["val_loss"]) == cfg.n_epochs
    assert len(hist["val_acc"]) == cfg.n_epochs

    # Our fake eval returns fixed val metrics; last entry should reflect that
    assert hist["val_loss"][-1] == pytest.approx(0.42, rel=0, abs=1e-6)
    assert hist["val_acc"][-1] == pytest.approx(0.66, rel=0, abs=1e-6)


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


# ------------------------------------------------------------------------------
# compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor
# ------------------------------------------------------------------------------


def test_compute_class_weights_balanced_two_classes():
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    w = compute_class_weights(y, num_classes=2)
    assert isinstance(w, torch.Tensor)
    assert w.dtype == torch.float32
    assert w.shape == (2,)
    # counts=[2,2], total=4 -> weights = 4/(2*2)=1.0 each
    assert torch.allclose(w, torch.tensor([1.0, 1.0], dtype=torch.float32))


def test_compute_class_weights_imbalanced_two_classes():
    y = np.array([0, 0, 0, 1], dtype=np.int64)  # counts=[3,1], total=4
    w = compute_class_weights(y, num_classes=2)
    # weights = total / (num_classes * counts) = 4/(2*counts) -> [0.666..., 2.0]
    assert w.shape == (2,)
    assert w[0] == pytest.approx(4 / (2 * 3), rel=0, abs=1e-6)
    assert w[1] == pytest.approx(4 / (2 * 1), rel=0, abs=1e-6)
    # minority class gets higher weight
    assert w[1] > w[0]


def test_compute_class_weights_single_class_num_classes_1():
    y = np.array([0, 0, 0, 0], dtype=np.int64)  # counts=[4], total=4
    w = compute_class_weights(y, num_classes=1)
    # weights = 4/(1*4)=1.0
    assert torch.allclose(w, torch.tensor([1.0], dtype=torch.float32))


def test_compute_class_weights_missing_class_gives_inf_weight():
    y = np.array([0, 0, 0], dtype=np.int64)  # counts=[3,0]
    w = compute_class_weights(y, num_classes=2)
    # class 1 has zero count -> division by zero -> inf
    assert torch.isinf(w[1]) and w[1] > 0
    # class 0 finite
    assert torch.isfinite(w[0])


def test_compute_class_weights_type_and_shape():
    y = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
    w = compute_class_weights(y, num_classes=3)
    assert isinstance(w, torch.Tensor)
    assert w.dtype == torch.float32
    assert w.shape == (3,)


# ------------------------------------------------------------------------------
# compute_class_weights: validation/error tests
# ------------------------------------------------------------------------------


def test_compute_class_weights_rejects_non_ndarray():
    with pytest.raises(ValueError, match=r"^y must be a numpy ndarray"):
        compute_class_weights([0, 1, 1], num_classes=2)  # list, not ndarray


def test_compute_class_weights_rejects_non_1d():
    y = np.array([[0, 1], [1, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match=r"^y must be a 1D array of class labels"):
        compute_class_weights(y, num_classes=2)


def test_compute_class_weights_rejects_non_integer_dtype():
    y = np.array([0.0, 1.0], dtype=float)
    with pytest.raises(ValueError, match=r"^y must contain integer class labels"):
        compute_class_weights(y, num_classes=2)


@pytest.mark.parametrize("bad_num_classes", [0, -1, 1.5])
def test_compute_class_weights_rejects_invalid_num_classes(bad_num_classes):
    y = np.array([0, 0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match=r"^num_classes must be a positive integer"):
        compute_class_weights(y, num_classes=bad_num_classes)  # type: ignore[arg-type]


def test_compute_class_weights_rejects_num_classes_le_max_y():
    y = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match=r"^num_classes must be greater than max\(y\)"):
        compute_class_weights(y, num_classes=1)


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
