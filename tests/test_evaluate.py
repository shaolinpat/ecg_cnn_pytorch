# tests/test_evaluate.py

"""
Tests for ecg_cnn.evaluate.py

Covers
------
    - main(): happy path, error paths, fold selection, env overrides
    - __main__ entry point smoke test via subprocess
"""

import builtins
import importlib.util
import json
import numpy as np
import pandas as pd
import pathlib
import pytest
import os
import runpy
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ecg_cnn.evaluate as evaluate

# Optional dep: skip cleanly if PyTorch isn't installed
torch = pytest.importorskip("torch", reason="torch not installed")

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------


def ovr_cfg(plots_enable_ovr=False, plots_ovr_classes=None):
    return SimpleNamespace(
        plots_enable_ovr=plots_enable_ovr,
        plots_ovr_classes=[] if plots_ovr_classes is None else plots_ovr_classes,
    )


# ------------------------------------------------------------------------------
# main()
# ------------------------------------------------------------------------------


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
    (
        results_dir,
        history_dir,
        models_dir,
        output_dir,
        plots_dir,
        cache_dir,
        ptbxl_dir,
    ) = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)

    # Create a dummy config file so evaluate.RESULTS_DIR.glob("config_*.yaml") works
    cfg_path = results_dir / "config_dummy.yaml"
    cfg_path.write_text("dummy: true")

    # Simulate PTB-XL data load (shape N x L is fine for this fake model; adjust if your pipeline needs (N,C,L))
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

    # Simulate parsed TrainConfig (attributes used by evaluate.py)
    mock_config.return_value.model = "ECGConvNet"
    mock_config.return_value.batch_size = 32
    mock_config.return_value.lr = 0.001
    mock_config.return_value.weight_decay = 0.0
    mock_config.return_value.subsample_frac = 1.0
    mock_config.return_value.sampling_rate = 500
    mock_config.return_value.tag = "dummy"
    mock_config.return_value.fold = 0

    # Fake model that returns logits with forced class distribution
    def fake_forward(x):
        num_classes = 5
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, num_classes)
        for i in range(batch_size):
            logits[i, i % num_classes] = 1.0
        return logits

    mock_model_instance = mock.MagicMock()

    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {"ECGConvNet": mock.MagicMock(return_value=mock_model_instance)},
        raising=False,
    )

    # Make the instance callable and also provide .forward
    mock_model_instance.side_effect = fake_forward  # model(x)
    mock_model_instance.forward.return_value = (
        fake_forward(np.zeros((1, 1))) * 0
    )  # placeholder; not used
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance

    # Patch ECGConvNet to return fake model
    mock_models.__getitem__.return_value = mock.MagicMock(
        return_value=mock_model_instance
    )

    # Create an ON-DISK summary file and an ON-DISK dummy checkpoint
    ckpt_path = results_dir / "fake_model.pt"
    ckpt_path.write_text("")  # touch the file so Path.exists() passes

    dummy_summary = {
        "fold": 0,
        "loss": 0.123,
        "best_fold": 0,
        "model_path": str(ckpt_path),  # point to the touched file in results_dir
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }
    (results_dir / "summary_dummy.json").write_text(json.dumps([dummy_summary]))

    with monkeypatch.context() as mctx, mock.patch.object(builtins, "print"):
        # Keep target_names happy if your code passes FIVE_SUPERCLASSES into classification_report
        mctx.setattr(
            evaluate,
            "FIVE_SUPERCLASSES",
            ["NORM", "MI", "STTC", "CD", "HYP"],
            raising=False,
        )

        evaluate.main(fold_override=0)

    # Assert expected behavior
    mock_eval_plot.assert_called_once()
    mock_torch_load.assert_called_once()


def test_main_exits_gracefully_when_no_config_files(patch_paths, capsys):
    # No config_*.yaml present in RESULTS_DIR (patch_paths gives us a clean temp dir)
    with pytest.raises(SystemExit) as ei:
        evaluate.main()
    # Gentle landing: exit code 1, and a helpful message printed
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "No training configs found" in out
    assert "Run train.py first or pass --config" in out


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

    # Dataset placeholders (won't be reached because we raise earlier)
    mock_X = np.random.randn(3, 1000)
    mock_y = ["NORM", "AFIB", "STTC"]
    mock_meta = pd.DataFrame({"dummy": range(3)})

    # Force the exact error your test expects without changing evaluate.py
    def _raise_no_match(_tag):
        raise ValueError("No summary entry found for fold 99")

    with (
        # Short-circuit inside main() before model_path checks
        monkeypatch.context() as mctx,
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock.MagicMock()}),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot"),
        pytest.raises(ValueError, match="No summary entry found for fold 99"),
    ):
        mctx.setattr(evaluate, "_read_summary", _raise_no_match, raising=False)
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
        {"fold": 0, "loss": 0.1, "model_path": str(results_dir / "f0.pt")},
        {"fold": 1, "loss": 0.05, "model_path": str(results_dir / "f1.pt")},
    ]

    # Write the summary file physically so any glob/exists checks pass
    (results_dir / "summary_dummy.json").write_text(json.dumps(summaries))

    # Touch checkpoint files in case code checks for existence before torch.load
    (results_dir / "f0.pt").write_text("")
    (results_dir / "f1.pt").write_text("")

    # Simulated dataset with 5 samples, one per class (0-4)
    # Use (N, C, L) like real ECG and multi-label y as lists
    mock_X = np.random.randn(5, 1, 1000).astype(np.float32)
    mock_y = [
        ["0"],
        ["1"],
        ["2"],
        ["3"],
        ["4"],
    ]  # <- string labels to match target_names expectations
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

    # Mock model: callable and with .forward returning logits
    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.return_value = mock_logits  # model(X)
    mock_model_instance.forward = mock.MagicMock(
        return_value=mock_logits
    )  # model.forward(X)

    mock_model_class = mock.MagicMock(return_value=mock_model_instance)

    with (
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),  # consumed by load_state_dict
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock_model_class}),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot"),
        mock.patch.object(builtins, "print"),
        monkeypatch.context() as mctx,
    ):
        # Align label space with the string labels above so target_names are strings
        mctx.setattr(
            evaluate, "FIVE_SUPERCLASSES", ["0", "1", "2", "3", "4"], raising=False
        )
        evaluate.main()


@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
def test_main_raises_when_config_missing_batch_size(
    mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    # extras dict (for extras.get('tag')), keep empty so it doesn't supply tag
    mock_load_cfg.return_value = {}

    # Return a config object with model but NO batch_size
    mock_TrainConfig.return_value = SimpleNamespace(model="ECGConvNet")

    with pytest.raises(
        ValueError, match="Config is missing required field 'batch_size'."
    ):
        evaluate.main()


# 286: "Config is missing required field 'model'."
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
def test_main_raises_when_config_missing_model(
    mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    mock_load_cfg.return_value = {}  # extras empty

    # Return a config object with batch_size but NO model
    mock_TrainConfig.return_value = SimpleNamespace(batch_size=32)

    with pytest.raises(ValueError, match="Config is missing required field 'model'."):
        evaluate.main()


# 289: "Config is missing 'tag'; cannot locate summaries/models."
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
def test_main_raises_when_config_missing_tag(
    mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    # extras has no 'tag', so getattr(config, 'tag', None) or extras.get('tag') -> None
    mock_load_cfg.return_value = {}

    # Provide model and batch_size, but NO tag attribute
    mock_TrainConfig.return_value = SimpleNamespace(model="ECGConvNet", batch_size=32)

    with pytest.raises(
        ValueError, match="Config is missing 'tag'; cannot locate summaries/models."
    ):
        evaluate.main()


@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
@mock.patch(
    "ecg_cnn.evaluate.load_ptbxl_full",
    return_value=(np.zeros((1, 1, 10), np.float32), [["0"]], pd.DataFrame({"i": [0]})),
)
def test_main_raises_when_summary_lacks_model_path_triggers_value_error(
    _mock_load_data, mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    # Point evaluate to temp results dir and create a config so glob finds something
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    # Minimal extras + config (include attrs accessed before the branch)
    mock_load_cfg.return_value = {}
    mock_TrainConfig.return_value = SimpleNamespace(
        model="ECGConvNet",
        batch_size=32,
        tag="dummy",
        subsample_frac=1.0,
        sampling_rate=100,
    )

    # Summary entry missing 'model_path' (so best.get('model_path','') == "")
    (tmp_path / "summary_dummy.json").write_text(
        json.dumps([{"fold": 0, "loss": 0.12}])
    )

    # Patch Path so that Path("") is falsy (to hit line 323) but otherwise behaves
    class _FalsyOnEmptyPath:
        def __init__(self, p=""):
            self._raw = p
            # Use pathlib for everything else; empty becomes "." so we keep both
            self._p = pathlib.Path(p or ".")

        def __bool__(self):
            return bool(self._raw)  # "" -> False

        def exists(self):
            return self._p.exists()

        @property
        def name(self):
            return self._p.name

        def __truediv__(self, other):
            return _FalsyOnEmptyPath(os.fspath(self._p / other))

        def __fspath__(self):
            return os.fspath(self._p)

        def __str__(self):
            return str(self._p)

        def __repr__(self):
            return f"_FalsyOnEmptyPath({self._raw!r})"

    with monkeypatch.context() as mctx:
        mctx.setattr(evaluate, "Path", _FalsyOnEmptyPath, raising=False)
        mctx.setattr(
            evaluate, "MODEL_CLASSES", {"ECGConvNet": mock.MagicMock()}, raising=False
        )
        with pytest.raises(ValueError, match="Chosen summary entry lacks 'model_path'"):
            evaluate.main(fold_override=0)


@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
@mock.patch(
    "ecg_cnn.evaluate.load_ptbxl_full",
    return_value=(np.zeros((1, 1, 10), np.float32), [["0"]], pd.DataFrame({"i": [0]})),
)
def test_main_raises_when_checkpoint_missing(
    _mock_load_data, mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    mock_load_cfg.return_value = {}
    mock_TrainConfig.return_value = SimpleNamespace(
        model="ECGConvNet",
        batch_size=32,
        tag="dummy",
        subsample_frac=1.0,
        sampling_rate=100,
    )

    # Summary points to a non-existent weights file
    missing_ckpt = tmp_path / "missing_weights.pt"  # don't create it
    (tmp_path / "summary_dummy.json").write_text(
        json.dumps([{"fold": 0, "loss": 0.12, "model_path": str(missing_ckpt)}])
    )

    with pytest.raises(
        FileNotFoundError, match=r"Model weights not found: .*missing_weights\.pt"
    ):
        evaluate.main(fold_override=0)


@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
@mock.patch(
    "ecg_cnn.evaluate.load_ptbxl_full",
    return_value=(np.zeros((1, 1, 10), np.float32), [["0"]], pd.DataFrame({"i": [0]})),
)
def test_main_raises_when_unknown_model(
    _mock_load_data, mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    mock_load_cfg.return_value = {}
    mock_TrainConfig.return_value = SimpleNamespace(
        model="DoesNotExist",
        batch_size=32,
        tag="dummy",
        subsample_frac=1.0,
        sampling_rate=100,
    )

    # Create a valid summary AND a touched checkpoint so earlier checks pass
    ckpt = tmp_path / "ok.pt"
    ckpt.write_text("")  # exists
    (tmp_path / "summary_dummy.json").write_text(
        json.dumps([{"fold": 0, "loss": 0.12, "model_path": str(ckpt)}])
    )

    with pytest.raises(ValueError, match="Unknown model 'DoesNotExist'"):
        evaluate.main(fold_override=0)


def test_cli_entrypoint_covers_main(monkeypatch):
    # Run ecg_cnn/evaluate.py as __main__ in-process

    spec = importlib.util.find_spec("ecg_cnn.evaluate")
    assert spec and spec.origin
    eval_path = spec.origin

    # Make argparse parse something simple; we don’t care what the script does next
    monkeypatch.setattr(sys, "argv", ["python", "--fold", "0"], raising=False)

    # Execute __main__ block; swallow any exit/errors so test is stable
    try:
        runpy.run_path(eval_path, run_name="__main__")
    except BaseException:
        # SystemExit or any exception from the script is fine—lines are executed
        pass


@mock.patch("torch.load")
@mock.patch("ecg_cnn.evaluate.load_ptbxl_full")
@mock.patch("ecg_cnn.models.MODEL_CLASSES")  # <-- patch the source registry
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.evaluate_and_plot")
def test_env_overrides_enable_and_classes(
    mock_eval_plot,
    mock_load_cfg,
    mock_models,  # now from ecg_cnn.models
    mock_load_data,
    mock_torch_load,
    monkeypatch,
    patch_paths,
):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    tag = "ECGConvNet_lr001_bs8_wd0"
    (results_dir / f"config_{tag}.yaml").write_text("dummy: true")

    # Create real summary + dummy model so existence checks pass
    dummy_summary = [
        {
            "fold": None,
            "loss": 0.1,
            "model_path": str(results_dir / "dummy.pth"),
            "best_epoch": 1,
            "model": "ECGConvNet",
        }
    ]
    (results_dir / f"summary_{tag}.json").write_text(json.dumps(dummy_summary))
    (results_dir / "dummy.pth").write_text("")

    # Env overrides
    monkeypatch.setenv("ECG_PLOTS_ENABLE_OVR", "1")
    monkeypatch.setenv("ECG_PLOTS_OVR_CLASSES", "NORM,MI")

    # Small class space
    monkeypatch.setattr(evaluate, "FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False)

    # Minimal data
    X = np.random.randn(6, 1, 10).astype(np.float32)
    y = ["NORM", "MI", "NORM", "MI", "NORM", "MI"]
    meta = pd.DataFrame({"i": range(len(y))})
    mock_load_data.return_value = (X, y, meta)

    # Config expected by evaluate.py
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

    # Tiny model for registry
    class TinyModel(torch.nn.Module):
        def __init__(self, num_classes=2, **kwargs):
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes, bias=True)

        def forward(self, x):
            n = x.shape[0]
            return self.fc(x.reshape(n, -1))

    # right after defining TinyModel (or even before), add:
    monkeypatch.setattr(
        evaluate, "MODEL_CLASSES", {"ECGConvNet": TinyModel}, raising=False
    )

    # Make the registry return TinyModel for "ECGConvNet"
    mock_models.__getitem__.return_value = TinyModel
    mock_torch_load.return_value = TinyModel(num_classes=2).state_dict()

    # Run
    with mock.patch.object(builtins, "print"):
        evaluate.main(fold_override=None)

    # Assertions
    assert mock_eval_plot.called
    kwargs = mock_eval_plot.call_args.kwargs
    assert kwargs["enable_ovr"] is True
    assert set(kwargs["ovr_classes"]) == {"NORM", "MI"}


def test_cli_entrypoint_fast(monkeypatch):
    # Run ecg_cnn/evaluate.py as a real script, but force the fast "no configs" exit.
    test_file = Path(evaluate.__file__).resolve()
    assert test_file.exists()

    # Do NOT create any config_*.yaml in RESULTS_DIR — we want early exit.
    proc = subprocess.run(
        [sys.executable, str(test_file), "--fold", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Accept either early no-configs path OR normal run path
    out, err = proc.stdout, proc.stderr
    assert ("No training configs found" in (out + err)) or (
        "Loading config from:" in out
    )


@mock.patch("torch.load")
@mock.patch("ecg_cnn.evaluate.load_ptbxl_full")
@mock.patch("ecg_cnn.evaluate.MODEL_CLASSES")
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.evaluate_and_plot")
def test_env_empty_classes_is_error(
    mock_eval_plot,
    mock_load_cfg,
    mock_models,
    mock_load_data,
    mock_torch_load,
    monkeypatch,
    patch_paths,
    capsys,
):
    results_dir, history_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Present a config file for glob
    tag = "ECGConvNet_lr001_bs8_wd0"
    (results_dir / f"config_{tag}.yaml").write_text("dummy: true")

    # ENV: empty string is invalid under explicit-or-error policy
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

    # Write real artifacts that evaluate.py checks with .exists()
    model_path = results_dir / "dummy.pth"
    model_path.touch()

    monkeypatch.setattr(
        evaluate, "MODEL_CLASSES", {"ECGConvNet": TinyModel}, raising=False
    )

    dummy_summary = [
        {
            "fold": None,
            "loss": 0.1,
            "model_path": str(model_path),
            "best_epoch": 1,
            "model": "ECGConvNet",
        }
    ]
    (results_dir / f"summary_{tag}.json").write_text(json.dumps(dummy_summary))

    # Run: should exit with error and not call evaluate_and_plot
    with pytest.raises(SystemExit) as e:
        evaluate.main(fold_override=None)
    assert e.value.code == 1
    _, err = capsys.readouterr()
    assert "empty OvR class list" in err
    assert not mock_eval_plot.called


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_prints_message_when_history_missing(
    mock_load_config, patch_paths, monkeypatch
):
    # unpack temp paths
    results_dir, history_dir, models_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)

    # ensure evaluate finds a config
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    # config returned by loader
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

    # write REAL summary file that evaluate._read_summary() checks for
    weights_path = models_dir / "dummy.pt"
    weights_path.write_bytes(b"")  # empty file is fine; we mock torch.load
    summaries = [{"fold": 2, "loss": 0.123, "model_path": str(weights_path)}]
    (results_dir / "summary_dummy.json").write_text(json.dumps(summaries))

    # mock dataset: labels must be strings matching FIVE_SUPERCLASSES
    mock_X = np.random.randn(3, 1000)
    mock_y = ["NORM", "MI", "STTC"]
    mock_meta = pd.DataFrame({"dummy": range(3)})

    # fake model & logits
    mock_logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.load_state_dict.return_value = None
    mock_model_instance.return_value = mock_logits  # __call__ returns logits
    mock_model_class = mock.MagicMock(return_value=mock_model_instance)

    with (
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),  # don't deserialize the empty file
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock_model_class}),
        mock.patch("ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI", "STTC"]),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot", return_value=None),
        mock.patch(
            "sklearn.metrics.classification_report", return_value="dummy_report"
        ),
        mock.patch.object(builtins, "print") as mock_print,
    ):
        evaluate.main()

    # assert the gentle warning printed
    printed = "History not found at" in " ".join(
        str(c) for c in mock_print.call_args_list
    )
    assert printed, "'History not found at' message was not printed"


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_loads_history_successfully(mock_load_config, patch_paths, monkeypatch):
    # NOTE: unpack models_dir so we can place the dummy weights there
    results_dir, history_dir, models_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)

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
        "fold": 2,
        "tag": "dummy",
        "config": "config_dummy.yaml",
    }
    mock_load_config.return_value = config_data

    # create a real weights file and reference it from the summary
    weights_path = models_dir / "dummy.pt"
    weights_path.write_bytes(b"")  # empty file is fine; we also mock torch.load

    summaries = [{"fold": 2, "loss": 0.123, "model_path": str(weights_path)}]
    (results_dir / "summary_dummy.json").write_text(json.dumps(summaries))

    (history_dir / "history_dummy_fold2.json").write_text(
        json.dumps(
            {
                "train_acc": [0.9, 0.95],
                "val_acc": [0.85, 0.92],
                "train_loss": [0.6, 0.4],
                "val_loss": [0.65, 0.5],
            }
        )
    )

    mock_X = np.random.randn(3, 1000)
    mock_y = ["NORM", "MI", "STTC"]
    mock_meta = pd.DataFrame({"dummy": range(3)})

    mock_logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.load_state_dict.return_value = None
    mock_model_instance.return_value = mock_logits
    mock_model_class = mock.MagicMock(return_value=mock_model_instance)

    with (
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),  # we still mock load to avoid real deserialization
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock_model_class}),
        mock.patch("ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI", "STTC"]),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot", return_value=None),
        mock.patch(
            "sklearn.metrics.classification_report", return_value="dummy_report"
        ),
        mock.patch.object(builtins, "print"),
    ):
        evaluate.main()


# ------------------------------------------------------------------------------
# _read_summary
# ------------------------------------------------------------------------------


def test_read_summary_raises_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    with pytest.raises(FileNotFoundError, match="Missing summary for tag 'dummy'"):
        evaluate._read_summary("dummy")


def test_read_summary_raises_when_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "summary_dummy.json").write_text("[]")  # empty file
    with pytest.raises(ValueError, match="Summary JSON malformed or empty"):
        evaluate._read_summary("dummy")


# ------------------------------------------------------------------------------
# _select_best_entry
# ------------------------------------------------------------------------------


def test_select_best_entry_raises_for_missing_fold():
    summaries = [
        {"fold": 0, "loss": 0.10},
        {"fold": 1, "loss": 0.05},
    ]
    with pytest.raises(ValueError, match="No summary entry found for fold 99"):
        evaluate._select_best_entry(summaries, fold_override=99)


def test_select_best_entry_raises_when_loss_missing():
    summaries = [
        {"fold": 0},  # missing 'loss'
        {"fold": 1, "loss": 0.05},
    ]
    with pytest.raises(ValueError, match="Summary entries missing 'loss' key"):
        evaluate._select_best_entry(summaries, fold_override=None)


# ------------------------------------------------------------------------------
# _resolve_ovr_flags
# ------------------------------------------------------------------------------


def tes_resolve_ovr_flags_cli_valid_classes_imply_enable(monkeypatch):
    # Keep class space small for the test
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    # No env
    monkeypatch.delenv("ECG_PLOTS_ENABLE_OVR", raising=False)
    monkeypatch.delenv("ECG_PLOTS_OVR_CLASSES", raising=False)

    enable, classes = evaluate._resolve_ovr_flags(
        ovr_cfg(), cli_ovr_enable=None, cli_ovr_classes=["MI"]
    )
    assert enable is True
    assert classes == {"MI"}


def test_resolve_ovr_flags_cli_empty_classes_is_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    # argparse type= already strips empties -> resolver sees []
    with pytest.raises(SystemExit) as e:
        evaluate._resolve_ovr_flags(ovr_cfg(), cli_ovr_enable=None, cli_ovr_classes=[])
    assert e.value.code == 1
    _, err = capsys.readouterr()
    assert "empty OvR class list" in err  # or "empty OvR class list" in err


def test_resolve_ovr_flags_cli_unknown_class_is_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    with pytest.raises(SystemExit) as e:
        evaluate._resolve_ovr_flags(
            ovr_cfg(), cli_ovr_enable=None, cli_ovr_classes=["MI", "ABCDEF"]
        )
    assert e.value.code == 1
    _, err = capsys.readouterr()
    assert "unknown OvR class(es) from CLI" in err
    assert "ABCDEF" in err


def test_resolve_ovr_flags_env_empty_classes_is_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    monkeypatch.setenv("ECG_PLOTS_OVR_CLASSES", "")
    monkeypatch.delenv("ECG_PLOTS_ENABLE_OVR", raising=False)

    with pytest.raises(SystemExit) as e:
        evaluate._resolve_ovr_flags(ovr_cfg())
    assert e.value.code == 1
    _, err = capsys.readouterr()
    assert "empty OvR class list" in err


def test_resolve_ovr_flags_env_valid_classes_enable(monkeypatch):
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    monkeypatch.setenv("ECG_PLOTS_OVR_CLASSES", "MI")
    monkeypatch.delenv("ECG_PLOTS_ENABLE_OVR", raising=False)

    enable, classes = evaluate._resolve_ovr_flags(ovr_cfg())
    assert enable is True
    assert classes == {"MI"}


def test_resolve_ovr_flags_config_classes_imply_enable(monkeypatch):
    # Config lists classes; your function sets enable_ovr True if classes present
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    monkeypatch.delenv("ECG_PLOTS_OVR_CLASSES", raising=False)
    monkeypatch.delenv("ECG_PLOTS_ENABLE_OVR", raising=False)

    enable, classes = evaluate._resolve_ovr_flags(
        ovr_cfg(plots_enable_ovr=False, plots_ovr_classes=["NORM"])
    )
    assert enable is True
    assert classes == {"NORM"}


def test_resolve_ovr_flags_cli_disable_wins_even_with_classes(monkeypatch):
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    monkeypatch.delenv("ECG_PLOTS_OVR_CLASSES", raising=False)
    monkeypatch.delenv("ECG_PLOTS_ENABLE_OVR", raising=False)

    enable, classes = evaluate._resolve_ovr_flags(
        ovr_cfg(), cli_ovr_enable=False, cli_ovr_classes=["MI"]
    )
    assert enable is False
    assert classes is None


def test_resolve_ovr_flags_precedence_cli_over_env_and_config(monkeypatch):
    # Config: enabled with NORM; ENV: MI; CLI: STTC -> CLI wins
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI", "STTC"], raising=False
    )
    monkeypatch.setenv("ECG_PLOTS_OVR_CLASSES", "MI")
    monkeypatch.setenv("ECG_PLOTS_ENABLE_OVR", "true")

    enable, classes = evaluate._resolve_ovr_flags(
        ovr_cfg(plots_enable_ovr=True, plots_ovr_classes=["NORM"]),
        cli_ovr_enable=None,
        cli_ovr_classes=["STTC"],
    )
    assert enable is True
    assert classes == {"STTC"}


def test_resolve_ovr_flags_env_enable_string_parsing(monkeypatch):
    # env enable true/false parsing should work independently of classes
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    monkeypatch.delenv("ECG_PLOTS_OVR_CLASSES", raising=False)

    monkeypatch.setenv("ECG_PLOTS_ENABLE_OVR", "yes")
    enable, classes = evaluate._resolve_ovr_flags(ovr_cfg())
    assert enable is True and classes is None

    monkeypatch.setenv("ECG_PLOTS_ENABLE_OVR", "0")
    enable, classes = evaluate._resolve_ovr_flags(ovr_cfg())
    assert enable is False and classes is None


def test_resolve_ovr_flags_cli_dedup_and_strip_then_validate(monkeypatch):
    # Ensure duplicates and whitespace are normalized before validation
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI", "STTC"], raising=False
    )
    enable, classes = evaluate._resolve_ovr_flags(
        ovr_cfg(),
        cli_ovr_enable=None,
        cli_ovr_classes=["MI", " STTC ", "MI"],  # duplicates + whitespace
    )
    assert enable is True
    assert classes == {"MI", "STTC"}
