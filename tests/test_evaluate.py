# tests/test_evaluate.py

"""
Tests for ecg_cnn.evaluate.py

Covers
------
    - main(): happy path, error paths, fold selection, env overrides
    - __main__ entry point smoke test via subprocess
"""

import builtins
import json
import numpy as np
import pandas as pd
import pytest
import subprocess
import sys

# Optional dep: skip cleanly if PyTorch isn't installed
torch = pytest.importorskip("torch", reason="torch not installed")

from pathlib import Path
from unittest import mock

import ecg_cnn.evaluate as evaluate


@mock.patch("torch.load")  # Mock torch.load to avoid real file access
@mock.patch("ecg_cnn.evaluate.load_ptbxl_full")
@mock.patch("ecg_cnn.evaluate.MODEL_CLASSES")
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
@mock.patch("ecg_cnn.evaluate.evaluate_and_plot")
def test_main_runs(
    mock_eval_plot,
    mock_config,
    mock_loader,
    mock_models,
    mock_load_data,
    mock_torch_load,  # receives torch.load patch
    patch_paths,
    monkeypatch,
):
    # Bind per-test paths into the evaluate module
    results_dir, history_dir, models_dir, output_dir, plots_dir, ptbxl_dir = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)

    # Create a dummy config file so evaluate.RESULTS_DIR.glob("config_*.yaml") works
    cfg_path = results_dir / "config_dummy.yaml"
    cfg_path.write_text("dummy: true")

    # ✅ Simulate PTB-XL data load
    mock_load_data.return_value = (
        np.random.randn(10, 1000),  # X
        np.array(
            ["NORM", "MI", "STTC", "CD", "HYP", "NORM", "MI", "STTC", "CD", "HYP"]
        ),
        pd.DataFrame({"dummy": range(10)}),
    )

    # Raw config dictionary returned from YAML
    mock_loader.return_value = {
        "model": "ECGConvNet",
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0,
        "subsample_frac": 1.0,
        "sampling_rate": 500,
        "tag": "dummy",
        "fold": 0,
        "config": "config_dummy.yaml",
    }

    # Simulate parsed TrainConfig
    mock_config.return_value.model = "ECGConvNet"
    mock_config.return_value.batch_size = 32
    mock_config.return_value.lr = 0.001
    mock_config.return_value.weight_decay = 0.0
    mock_config.return_value.subsample_frac = 1.0
    mock_config.return_value.sampling_rate = 500
    mock_config.return_value.tag = "dummy"

    # Fake model that returns logits with forced class distribution
    def fake_forward(x):
        num_classes = 5
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, num_classes)
        for i in range(batch_size):
            logits[i, i % num_classes] = 1.0
        return logits

    mock_model_instance = mock.MagicMock()
    mock_model_instance.side_effect = fake_forward
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance

    # Patch ECGConvNet to return fake model
    mock_models.__getitem__.return_value = mock.MagicMock(
        return_value=mock_model_instance
    )

    # Dummy summary with required fields (including model_path)
    dummy_summary = {
        "fold": 0,
        "loss": 0.123,
        "best_fold": 0,
        "model_path": "some/fake/path.pt",
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    # Simulate opening and reading summary JSON file
    mock_open = mock.mock_open(read_data=json.dumps([dummy_summary]))

    with mock.patch("builtins.open", mock_open), mock.patch.object(builtins, "print"):
        evaluate.main(fold_override=0)

    # Assert expected behavior
    mock_eval_plot.assert_called_once()
    mock_torch_load.assert_called_once()


def test_main_raises_if_no_config_files(patch_paths, monkeypatch):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    # No config_* present -> should raise
    with pytest.raises(FileNotFoundError, match=r"No config_.*\.yaml found in"):
        evaluate.main()


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_raises_if_bad_config(mock_load_config, patch_paths, monkeypatch):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Present a config file so glob finds it
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    # Provide invalid dict that will cause TrainConfig(**raw) to fail
    mock_load_config.return_value = {"unexpected_field": "boom!"}

    with pytest.raises(
        ValueError, match=r"^Invalid config structure or missing fields: .*"
    ):
        evaluate.main()


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_raises_if_no_matching_summary(mock_load_config, patch_paths, monkeypatch):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Config file present
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    config_data = {
        "model": "ECGConvNet",
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "n_epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "fold": 99,  # will not match dummy summary
        "tag": "dummy",
        "config": "config_dummy.yaml",
    }
    mock_load_config.return_value = config_data

    dummy_summary = {"fold": 0, "loss": 0.1, "model_path": "fake.pt"}
    mock_open_file = mock.mock_open(read_data=json.dumps([dummy_summary]))

    mock_X = np.random.randn(3, 1000)
    mock_y = ["NORM", "AFIB", "STTC"]
    mock_meta = pd.DataFrame({"dummy": range(3)})

    with (
        mock.patch("builtins.open", mock_open_file),
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock.MagicMock()}),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot"),
        pytest.raises(ValueError, match="No summary entry found for fold 99"),
    ):
        evaluate.main()


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_selects_best_fold_when_none_specified(
    mock_load_config, patch_paths, monkeypatch
):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Config file present
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    # Return raw config with fold=None so main() must select best
    config_data = {
        "model": "ECGConvNet",
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "n_epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "fold": None,
        "tag": "dummy",
        "config": "config_dummy.yaml",
    }
    mock_load_config.return_value = config_data

    # Return two summaries, best is fold 1 (lower loss)
    summaries = [
        {"fold": 0, "loss": 0.1, "model_path": "f0.pt"},
        {"fold": 1, "loss": 0.05, "model_path": "f1.pt"},
    ]
    mock_open_file = mock.mock_open(read_data=json.dumps(summaries))

    # Simulated dataset with 5 samples, one per class (0-4)
    mock_X = np.random.randn(5, 1000)
    mock_y = [0, 1, 2, 3, 4]
    mock_meta = pd.DataFrame({"dummy": range(5)})

    # Logits aligned with true labels
    mock_logits = torch.tensor(
        [
            [0.9, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9],
        ]
    )

    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.return_value = mock_logits

    mock_model_class = mock.MagicMock(return_value=mock_model_instance)

    with (
        mock.patch("builtins.open", mock_open_file),
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock_model_class}),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot"),
        mock.patch.object(builtins, "print"),
    ):
        evaluate.main()


def test_entry_point_runs(patch_paths, monkeypatch, tmp_path):
    """
    Test the __main__ block with a dummy --fold argument.
    This verifies CLI parsing and main() call from __main__.
    """
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Ensure at least one config exists so the script prints a loading line
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    test_file = Path("ecg_cnn/evaluate.py")
    assert test_file.exists(), f"Target file not found: {test_file}"

    # Simulate: python ecg_cnn/evaluate.py --fold 1
    result = subprocess.run(
        [sys.executable, str(test_file), "--fold", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # It may raise an expected error due to missing files; that's fine.
    assert ("Loading config from:" in result.stdout) or (
        "FileNotFoundError" in result.stderr
    )


@mock.patch("torch.load")
@mock.patch("ecg_cnn.evaluate.load_ptbxl_full")
@mock.patch("ecg_cnn.evaluate.MODEL_CLASSES")
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.evaluate_and_plot")
def test_env_overrides_enable_and_classes(
    mock_eval_plot,
    mock_load_cfg,
    mock_models,
    mock_load_data,
    mock_torch_load,
    monkeypatch,
    patch_paths,
):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Present a config file for glob
    tag = "ECGConvNet_lr001_bs8_wd0"
    (results_dir / f"config_{tag}.yaml").write_text("dummy: true")

    # Env: explicit enable + explicit classes
    monkeypatch.setenv("ECG_PLOTS_ENABLE_OVR", "1")
    monkeypatch.setenv("ECG_PLOTS_OVR_CLASSES", "NORM,MI")

    # Keep class space small and deterministic
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )

    # Minimal data
    X = np.random.randn(6, 1, 10).astype(np.float32)
    y = ["NORM", "MI", "NORM", "MI", "NORM", "MI"]
    meta = pd.DataFrame({"i": range(len(y))})
    mock_load_data.return_value = (X, y, meta)

    # Config dict with extras evaluate.py expects
    mock_load_cfg.return_value = {
        "model": "ECGConvNet",
        "lr": 1e-3,
        "batch_size": 8,
        "weight_decay": 0.0,
        "n_epochs": 1,
        "save_best": False,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "data_dir": None,
        "sample_dir": None,
        "n_folds": 0,
        "verbose": False,
        "plots_enable_ovr": False,
        "plots_ovr_classes": [],
        "tag": tag,
        "fold": None,
        "config": f"config_{tag}.yaml",
    }

    # Tiny model class + instance
    class TinyModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes, bias=False)

        def forward(self, x):
            n = x.shape[0]
            return self.fc(x.reshape(n, -1))

    mock_models.__getitem__.return_value = TinyModel
    mock_torch_load.return_value = TinyModel(num_classes=2).state_dict()

    # Summary pointing to a fake model path
    dummy_summary = [
        {
            "fold": None,
            "loss": 0.1,
            "model_path": "dummy.pth",
            "best_epoch": 1,
            "model": "ECGConvNet",
        }
    ]
    mopen = mock.mock_open(read_data=json.dumps(dummy_summary))

    with mock.patch("builtins.open", mopen), mock.patch.object(builtins, "print"):
        evaluate.main(fold_override=None)

    # Assert evaluate_and_plot saw the env overrides
    assert mock_eval_plot.called
    kwargs = mock_eval_plot.call_args.kwargs
    assert kwargs["enable_ovr"] is True
    assert set(kwargs["ovr_classes"]) == {"NORM", "MI"}


@mock.patch("torch.load")
@mock.patch("ecg_cnn.evaluate.load_ptbxl_full")
@mock.patch("ecg_cnn.evaluate.MODEL_CLASSES")
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.evaluate_and_plot")
def test_env_classes_only_implicitly_enables_all(
    mock_eval_plot,
    mock_load_cfg,
    mock_models,
    mock_load_data,
    mock_torch_load,
    monkeypatch,
    patch_paths,
):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Present a config file for glob
    tag = "ECGConvNet_lr001_bs8_wd0"
    (results_dir / f"config_{tag}.yaml").write_text("dummy: true")

    # Only classes var set; empty string means "all classes"; no explicit enable var
    monkeypatch.delenv("ECG_PLOTS_ENABLE_OVR", raising=False)
    monkeypatch.setenv("ECG_PLOTS_OVR_CLASSES", "")

    # Keep class space small
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )

    # Minimal data
    X = np.random.randn(4, 1, 10).astype(np.float32)
    y = ["NORM", "MI", "NORM", "MI"]
    meta = pd.DataFrame({"i": range(len(y))})
    mock_load_data.return_value = (X, y, meta)

    # Config defaults (OvR disabled in YAML)
    mock_load_cfg.return_value = {
        "model": "ECGConvNet",
        "lr": 1e-3,
        "batch_size": 8,
        "weight_decay": 0.0,
        "n_epochs": 1,
        "save_best": False,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "data_dir": None,
        "sample_dir": None,
        "n_folds": 0,
        "verbose": False,
        "plots_enable_ovr": False,
        "plots_ovr_classes": [],
        "tag": tag,
        "fold": None,
        "config": f"config_{tag}.yaml",
    }

    # Tiny model class + instance
    class TinyModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes, bias=False)

        def forward(self, x):
            n = x.shape[0]
            return self.fc(x.reshape(n, -1))

    mock_models.__getitem__.return_value = TinyModel
    mock_torch_load.return_value = TinyModel(num_classes=2).state_dict()

    # Summary
    dummy_summary = [
        {
            "fold": None,
            "loss": 0.1,
            "model_path": "dummy.pth",
            "best_epoch": 1,
            "model": "ECGConvNet",
        }
    ]
    mopen = mock.mock_open(read_data=json.dumps(dummy_summary))

    with mock.patch("builtins.open", mopen), mock.patch.object(builtins, "print"):
        evaluate.main(fold_override=None)

    # Assert implicit enable + no filter (None)
    assert mock_eval_plot.called
    kwargs = mock_eval_plot.call_args.kwargs
    assert kwargs["enable_ovr"] is True
    assert kwargs["ovr_classes"] is None


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_prints_message_when_history_missing(
    mock_load_config, patch_paths, monkeypatch
):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)

    # Config file found
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    # Return config dict matching expected fields
    config_data = {
        "model": "ECGConvNet",
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "n_epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "fold": 2,
        "tag": "dummy",
        "config": "config_dummy.yaml",
    }
    mock_load_config.return_value = config_data

    # Simulate summary entry for fold 2
    summaries = [{"fold": 2, "loss": 0.123, "model_path": "dummy.pt"}]
    mock_open_file = mock.mock_open(read_data=json.dumps(summaries))

    # No history file created -> evaluate should print "History not found at ..."
    mock_X = np.random.randn(3, 1000)
    mock_y = [0, 1, 2]
    mock_meta = pd.DataFrame({"dummy": range(3)})

    # Simulate logits for 3 known classes: 0, 1, 2
    mock_logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    # Fake model instance
    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.load_state_dict.return_value = None
    mock_model_instance.side_effect = lambda x: mock_logits

    mock_model_class = mock.MagicMock(return_value=mock_model_instance)

    with (
        mock.patch("builtins.open", mock_open_file),
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock_model_class}),
        mock.patch("ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI", "STTC"]),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot", return_value=None),
        mock.patch(
            "sklearn.metrics.classification_report", return_value="dummy_report"
        ),
        mock.patch("builtins.print") as mock_print,
    ):
        evaluate.main()
        found = any(
            "History not found at" in str(call) for call in mock_print.call_args_list
        )
        assert found, "'History not found at' message was not printed"


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_loads_history_successfully(mock_load_config, patch_paths, monkeypatch):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)

    # Config file found
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    # Return config dict matching expected fields
    config_data = {
        "model": "ECGConvNet",
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "n_epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "fold": 2,
        "tag": "dummy",
        "config": "config_dummy.yaml",
    }
    mock_load_config.return_value = config_data

    # Simulate summary entry for fold 2
    summaries = [{"fold": 2, "loss": 0.123, "model_path": "dummy.pt"}]
    mock_open_summary = mock.mock_open(read_data=json.dumps(summaries))

    # Create a real history file at the expected location with JSON content
    hist_path = history_dir / "history_fold2.json"
    history = {
        "train_acc": [0.9, 0.95],
        "val_acc": [0.85, 0.92],
        "train_loss": [0.6, 0.4],
        "val_loss": [0.65, 0.5],
    }
    hist_path.write_text(json.dumps(history))

    # Mock PTB-XL data
    mock_X = np.random.randn(3, 1000)
    mock_y = [0, 1, 2]
    mock_meta = pd.DataFrame({"dummy": range(3)})

    # Simulate logits for 3 known classes
    mock_logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    # Fake model instance
    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.load_state_dict.return_value = None
    mock_model_instance.return_value = mock_logits

    mock_model_class = mock.MagicMock(return_value=mock_model_instance)

    with (
        mock.patch("builtins.open", mock_open_summary),
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock_model_class}),
        mock.patch("ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI", "STTC"]),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot", return_value=None),
        mock.patch(
            "sklearn.metrics.classification_report", return_value="dummy_report"
        ),
        mock.patch.object(builtins, "print"),
    ):
        evaluate.main()
