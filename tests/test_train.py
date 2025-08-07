# tests/test_train_main.py

import json

# import pytest
import runpy

# import subprocess
import sys

# import textwrap
from pathlib import Path
from types import SimpleNamespace


# Import the module under test
import ecg_cnn.train as train_mod


def _cfg(model="ECGConvNet", lr=0.001, bs=8, wd=0.0, n_folds=0, verbose=False):
    # matches the attributes train.main expects
    return SimpleNamespace(
        model=model,
        lr=lr,
        batch_size=bs,
        weight_decay=wd,
        n_folds=n_folds,
        verbose=verbose,
        tag=None,  # set during main()
    )


def test_train_main_single_config_no_folds(monkeypatch, tmp_path):
    """
    Covers: args.config present (non-grid), merging, tag calc, single run,
    summary/config saving, best summary selection, and prints.
    """
    # Redirect RESULTS_DIR to tmp
    monkeypatch.setattr(train_mod, "RESULTS_DIR", tmp_path / "results", raising=False)

    # Fake args: --config path
    fake_cfg_path = tmp_path / "cfg.yaml"
    fake_cfg_path.write_text("model: ECGConvNet\n")
    monkeypatch.setattr(
        train_mod,
        "parse_args",
        lambda: SimpleNamespace(config=str(fake_cfg_path)),
        raising=False,
    )

    # Baseline + YAML + merges (non-grid)
    base = _cfg(model="ECGConvNet", lr=0.001, bs=8, wd=0.0, n_folds=0, verbose=False)
    monkeypatch.setattr(
        train_mod, "load_training_config", lambda _: base, raising=False
    )
    monkeypatch.setattr(
        train_mod, "load_yaml_as_dict", lambda p: {"lr": 0.001}, raising=False
    )
    monkeypatch.setattr(train_mod, "is_grid_config", lambda d: False, raising=False)
    monkeypatch.setattr(
        train_mod,
        "merge_configs",
        lambda a, b: _cfg(
            model=a.model,
            lr=b.get("lr", a.lr),
            bs=a.batch_size,
            wd=a.weight_decay,
            n_folds=a.n_folds,
            verbose=a.verbose,
        ),
        raising=False,
    )
    # No CLI overrides in this test
    monkeypatch.setattr(
        train_mod, "override_config_with_args", lambda c, a: c, raising=False
    )

    # Capture run_training calls and return deterministic summary
    calls = []

    def fake_run_training(config, fold_idx=None, tag=None):
        calls.append({"fold_idx": fold_idx, "tag": tag, "lr": config.lr})
        return {
            "loss": 0.5,
            "best_epoch": 1,
            "model_path": "dummy.pth",
            "model": config.model,
        }

    monkeypatch.setattr(train_mod, "run_training", fake_run_training, raising=False)

    # Execute
    train_mod.main()

    # Assertions
    assert len(calls) == 1
    # tag removes '.' characters
    expected_tag = "ECGConvNet_lr0001_bs8_wd0"
    assert calls[0]["tag"] == expected_tag

    # Files saved
    summary_path = tmp_path / "results" / f"summary_{expected_tag}.json"
    config_path = tmp_path / "results" / f"config_{expected_tag}.yaml"
    assert summary_path.exists() and config_path.exists()

    # Summary content shape (list of summaries)
    data = json.loads(summary_path.read_text())
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["loss"] == 0.5


def test_train_main_grid_with_folds(monkeypatch, tmp_path):
    """
    Covers: grid config path, expand_grid, multiple configs,
    fold loop (n_folds >= 2), per-config summary/config saving, and best model print.
    """
    # Redirect RESULTS_DIR
    monkeypatch.setattr(train_mod, "RESULTS_DIR", tmp_path / "results", raising=False)

    # Fake args with a config file
    fake_cfg_path = tmp_path / "grid.yaml"
    fake_cfg_path.write_text("grid: true\n")
    monkeypatch.setattr(
        train_mod,
        "parse_args",
        lambda: SimpleNamespace(config=str(fake_cfg_path)),
        raising=False,
    )

    # Baseline config
    base = _cfg(model="ECGConvNet", lr=0.001, bs=16, wd=0.0, n_folds=0, verbose=True)
    monkeypatch.setattr(
        train_mod, "load_training_config", lambda _: base, raising=False
    )

    # YAML dict is flagged as grid; expand into two dicts
    monkeypatch.setattr(
        train_mod, "load_yaml_as_dict", lambda p: {"grid": True}, raising=False
    )
    monkeypatch.setattr(train_mod, "is_grid_config", lambda d: True, raising=False)
    expand_list = [{"lr": 0.001, "n_folds": 2}, {"lr": 0.0005, "n_folds": 3}]
    monkeypatch.setattr(train_mod, "expand_grid", lambda d: expand_list, raising=False)

    # Merge each dict onto base; keep verbose=True to hit the print loop
    def _merge(a, b):
        return _cfg(
            model=a.model,
            lr=b.get("lr", a.lr),
            bs=a.batch_size,
            wd=a.weight_decay,
            n_folds=b.get("n_folds", a.n_folds),
            verbose=True,
        )

    monkeypatch.setattr(train_mod, "merge_configs", _merge, raising=False)

    # No CLI overrides for simplicity
    monkeypatch.setattr(
        train_mod, "override_config_with_args", lambda c, a: c, raising=False
    )

    # Capture calls and return different losses so "best" selection runs
    calls = []

    def fake_run_training(config, fold_idx=None, tag=None):
        calls.append({"fold_idx": fold_idx, "tag": tag, "lr": config.lr})
        # Make later folds slightly better (lower loss)
        base_loss = 1.0 if config.lr == 0.001 else 0.8
        loss = base_loss - (0.1 * (fold_idx if fold_idx is not None else 0))
        return {
            "loss": loss,
            "best_epoch": 2,
            "model_path": f"dummy_{tag}_fold{(fold_idx+1) if fold_idx is not None else 0}.pth",
            "model": config.model,
        }

    monkeypatch.setattr(train_mod, "run_training", fake_run_training, raising=False)

    # Execute
    train_mod.main()

    # We expect 2 + 3 = 5 training calls
    assert len(calls) == 5

    # Two per-config results files saved
    tags = [
        "ECGConvNet_lr0001_bs16_wd0",
        "ECGConvNet_lr00005_bs16_wd0",
    ]
    for tag in tags:
        summary_path = tmp_path / "results" / f"summary_{tag}.json"
        config_path = tmp_path / "results" / f"config_{tag}.yaml"
        assert summary_path.exists() and config_path.exists()
        # summaries is a list with len = n_folds
        data = json.loads(summary_path.read_text())
        assert isinstance(data, list) and len(data) >= 2


def test_train_main_no_config_uses_base_cfg(monkeypatch, tmp_path):
    # Results dir goes to tmp
    monkeypatch.setattr(train_mod, "RESULTS_DIR", tmp_path / "results", raising=False)

    # Simulate no --config provided
    monkeypatch.setattr(
        train_mod, "parse_args", lambda: SimpleNamespace(config=None), raising=False
    )

    # Base config returned by loader
    base = _cfg(model="ECGConvNet", lr=0.001, bs=8, wd=0.0, n_folds=0, verbose=False)
    monkeypatch.setattr(
        train_mod, "load_training_config", lambda _: base, raising=False
    )

    # CLI overrides: identity (kept for completeness)
    monkeypatch.setattr(
        train_mod, "override_config_with_args", lambda c, a: c, raising=False
    )

    # Make sure YAML/grid helpers are NOT called in this branch
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "This helper should not be called when args.config is None"
        )

    monkeypatch.setattr(
        train_mod, "load_yaml_as_dict", _should_not_be_called, raising=False
    )
    monkeypatch.setattr(
        train_mod, "is_grid_config", _should_not_be_called, raising=False
    )
    monkeypatch.setattr(train_mod, "expand_grid", _should_not_be_called, raising=False)
    monkeypatch.setattr(
        train_mod, "merge_configs", _should_not_be_called, raising=False
    )

    # Stub run_training to avoid real work
    calls = []

    def fake_run_training(config, fold_idx=None, tag=None):
        calls.append({"fold_idx": fold_idx, "tag": tag})
        return {
            "loss": 0.5,
            "best_epoch": 1,
            "model_path": "dummy.pth",
            "model": config.model,
        }

    monkeypatch.setattr(train_mod, "run_training", fake_run_training, raising=False)

    # Execute
    train_mod.main()

    # One training call, no folds
    assert len(calls) == 1
    expected_tag = f"{base.model}_lr{base.lr:.4g}_bs{base.batch_size}_wd{base.weight_decay:.4g}".replace(
        ".", ""
    )
    assert calls[0]["tag"] == expected_tag

    # Files saved
    summary_path = tmp_path / "results" / f"summary_{expected_tag}.json"
    config_path = tmp_path / "results" / f"config_{expected_tag}.yaml"
    assert summary_path.exists() and config_path.exists()

    # Summary contains a list with one item
    data = json.loads(summary_path.read_text())
    assert isinstance(data, list) and len(data) == 1 and data[0]["loss"] == 0.5


def test_train_entrypoint_calls_main(monkeypatch, tmp_path):
    """
    Covers: the __main__ guard in ecg_cnn.train
    We patch source modules BEFORE executing ecg_cnn.train as __main__.
    """

    # 1) Write results under tmp (train.py reads RESULTS_DIR from ecg_cnn.paths)
    import ecg_cnn.paths as paths_mod

    monkeypatch.setattr(paths_mod, "RESULTS_DIR", tmp_path / "results", raising=False)

    # 2) Prevent argparse from seeing pytest args + stub parse_args/overrides
    monkeypatch.setattr(
        sys, "argv", ["python"]
    )  # neutral argv so argparse doesn't choke
    import ecg_cnn.training.cli_args as cli_src

    monkeypatch.setattr(
        cli_src, "parse_args", lambda: SimpleNamespace(config=None), raising=False
    )
    # Important: bypass CLI overrides so your simpler _cfg() is enough
    monkeypatch.setattr(
        cli_src, "override_config_with_args", lambda c, a: c, raising=False
    )

    # 3) Config loader: return a minimal base config; bypass YAML/grid merges
    import ecg_cnn.config.config_loader as cfg_src

    base = _cfg()  # <-- reuse your existing helper
    monkeypatch.setattr(cfg_src, "load_training_config", lambda _: base, raising=False)
    monkeypatch.setattr(cfg_src, "load_yaml_as_dict", lambda *a, **k: {}, raising=False)
    monkeypatch.setattr(cfg_src, "merge_configs", lambda a, b: a, raising=False)

    import ecg_cnn.utils.grid_utils as grid_src

    monkeypatch.setattr(grid_src, "is_grid_config", lambda d: False, raising=False)
    monkeypatch.setattr(grid_src, "expand_grid", lambda d: [], raising=False)

    # 4) Trainer: stub run_training to avoid real work and capture the call
    import ecg_cnn.training.trainer as trainer_src

    calls = []

    def fake_run_training(config, fold_idx=None, tag=None):
        calls.append({"fold_idx": fold_idx, "tag": tag})
        return {
            "loss": 0.5,
            "best_epoch": 1,
            "model_path": "dummy.pth",
            "model": config.model,
        }

    monkeypatch.setattr(trainer_src, "run_training", fake_run_training, raising=False)

    # 5) Execute ecg_cnn.train as __main__ (fresh module namespace; no warning)
    sys.modules.pop("ecg_cnn.train", None)  # ensure fresh exec, avoids runpy warning
    runpy.run_module("ecg_cnn.train", run_name="__main__")

    # 6) Assertions
    assert len(calls) == 1
    expected_tag = (
        f"{base.model}_lr{base.lr:g}_bs{base.batch_size}_wd{base.weight_decay:g}"
    ).replace(".", "")
    summary_path = tmp_path / "results" / f"summary_{expected_tag}.json"
    config_path = tmp_path / "results" / f"config_{expected_tag}.yaml"
    assert summary_path.exists() and config_path.exists()

    data = json.loads(summary_path.read_text())
    assert isinstance(data, list) and len(data) == 1 and data[0]["loss"] == 0.5
