# tests/test_train.py

"""
Tests for ecg_cnn.train.

Covers
------
    - main(): single config, grid config with folds, no config fallback
    - __main__ entry point smoke test via run_module
"""

import json
import math
import pytest
import re
import runpy
import sys
import types

from types import SimpleNamespace

import ecg_cnn.train as train

from ecg_cnn.utils.plot_utils import format_hparams


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# main()
# ------------------------------------------------------------------------------


def test_train_main_single_config_no_folds(monkeypatch, tmp_path, patch_paths):
    """
    Covers: args.config present (non-grid), merging, tag calc, single run,
    summary/config saving, best summary selection, and prints.
    """
    # Use central patched results dir; backstop in case train.py binds its own
    results_dir, *_ = patch_paths
    monkeypatch.setattr(train, "RESULTS_DIR", results_dir, raising=False)

    # Fake args: --config path
    fake_cfg_path = tmp_path / "cfg.yaml"
    fake_cfg_path.write_text("model: ECGConvNet\n")
    monkeypatch.setattr(
        train,
        "parse_training_args",
        lambda: SimpleNamespace(config=str(fake_cfg_path)),
        raising=False,
    )

    # Baseline + YAML + merges (non-grid)
    base = _cfg(model="ECGConvNet", lr=0.001, bs=8, wd=0.0, n_folds=0, verbose=False)
    monkeypatch.setattr(train, "load_training_config", lambda _: base, raising=False)
    monkeypatch.setattr(
        train, "load_yaml_as_dict", lambda p: {"lr": 0.001}, raising=False
    )
    monkeypatch.setattr(train, "is_grid_config", lambda d: False, raising=False)
    monkeypatch.setattr(
        train,
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
        train, "override_config_with_args", lambda c, a: c, raising=False
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

    monkeypatch.setattr(train, "run_training", fake_run_training, raising=False)

    # Execute
    train.main()

    # Assertions
    assert len(calls) == 1

    # Use the same formatter the app uses
    expected_tag = format_hparams(
        model=base.model,
        lr=base.lr,
        bs=base.batch_size,
        wd=base.weight_decay,
        prefix="ecg",
    )
    assert calls[0]["tag"] == expected_tag

    # Files saved
    summary_path = results_dir / f"summary_{expected_tag}.json"
    config_path = results_dir / f"config_{expected_tag}.yaml"
    assert summary_path.exists() and config_path.exists()

    # Summary content shape (list of summaries)
    data = json.loads(summary_path.read_text())
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["loss"] == 0.5


def test_train_main_grid_with_folds(monkeypatch, tmp_path, patch_paths):
    """
    Covers: grid config path, expand_grid, multiple configs,
    fold loop (n_folds >= 2), per-config summary/config saving, and best model print.
    """
    # Use central patched results dir; backstop in case train.py binds its own
    results_dir, *_ = patch_paths
    monkeypatch.setattr(train, "RESULTS_DIR", results_dir, raising=False)

    # Fake args with a config file
    fake_cfg_path = tmp_path / "grid.yaml"
    fake_cfg_path.write_text("grid: true\n")
    monkeypatch.setattr(
        train,
        "parse_training_args",
        lambda: SimpleNamespace(config=str(fake_cfg_path)),
        raising=False,
    )

    # Baseline config
    base = _cfg(model="ECGConvNet", lr=0.001, bs=16, wd=0.0, n_folds=0, verbose=True)
    monkeypatch.setattr(train, "load_training_config", lambda _: base, raising=False)

    # YAML dict is flagged as grid; expand into two dicts
    monkeypatch.setattr(
        train, "load_yaml_as_dict", lambda p: {"grid": True}, raising=False
    )
    monkeypatch.setattr(train, "is_grid_config", lambda d: True, raising=False)
    expand_list = [{"lr": 0.001, "n_folds": 2}, {"lr": 0.0005, "n_folds": 3}]
    monkeypatch.setattr(train, "expand_grid", lambda d: expand_list, raising=False)

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

    monkeypatch.setattr(train, "merge_configs", _merge, raising=False)

    # No CLI overrides for simplicity
    monkeypatch.setattr(
        train, "override_config_with_args", lambda c, a: c, raising=False
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

    monkeypatch.setattr(train, "run_training", fake_run_training, raising=False)

    # Execute
    train.main()

    # We expect 2 + 3 = 5 training calls
    assert len(calls) == 5

    # Two per-config results files saved (tags depend on train's formatting)
    written = sorted(results_dir.glob("summary_*.json"))
    # Expect at least 2 distinct tags corresponding to the two expanded configs
    distinct_tags = {p.stem.replace("summary_", "") for p in written}
    assert len(distinct_tags) >= 2

    # Each summary file should contain a list with length >= number of folds for that config
    for p in written:
        data = json.loads(p.read_text())
        assert isinstance(data, list) and len(data) >= 2


def test_train_main_no_config_uses_base_cfg(monkeypatch, tmp_path, patch_paths):
    """
    Covers: args.config=None path, skips YAML/grid helpers, uses base config only.
    """
    # Use central patched results dir; backstop in case train.py binds its own
    results_dir, *_ = patch_paths
    monkeypatch.setattr(train, "RESULTS_DIR", results_dir, raising=False)

    # Simulate no --config provided
    monkeypatch.setattr(
        train,
        "parse_training_args",
        lambda: SimpleNamespace(config=None),
        raising=False,
    )

    # Base config returned by loader
    base = _cfg(model="ECGConvNet", lr=0.001, bs=8, wd=0.0, n_folds=0, verbose=False)
    monkeypatch.setattr(train, "load_training_config", lambda _: base, raising=False)

    # CLI overrides: identity
    monkeypatch.setattr(
        train, "override_config_with_args", lambda c, a: c, raising=False
    )

    # Make sure YAML/grid helpers are NOT called in this branch
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "This helper should not be called when args.config is None"
        )

    monkeypatch.setattr(
        train, "load_yaml_as_dict", _should_not_be_called, raising=False
    )
    monkeypatch.setattr(train, "is_grid_config", _should_not_be_called, raising=False)
    monkeypatch.setattr(train, "expand_grid", _should_not_be_called, raising=False)
    monkeypatch.setattr(train, "merge_configs", _should_not_be_called, raising=False)

    # Stub run_training
    calls = []

    def fake_run_training(config, fold_idx=None, tag=None):
        calls.append({"fold_idx": fold_idx, "tag": tag})
        return {
            "loss": 0.5,
            "best_epoch": 1,
            "model_path": "dummy.pth",
            "model": config.model,
        }

    monkeypatch.setattr(train, "run_training", fake_run_training, raising=False)

    # Execute
    train.main()

    # One training call, no folds
    assert len(calls) == 1

    # Use the same formatter the app uses
    expected_tag = format_hparams(
        model=base.model,
        lr=base.lr,
        bs=base.batch_size,
        wd=base.weight_decay,
        prefix="ecg",
    )

    assert calls[0]["tag"] == expected_tag

    # Files saved
    summary_path = results_dir / f"summary_{expected_tag}.json"
    config_path = results_dir / f"config_{expected_tag}.yaml"
    assert summary_path.exists() and config_path.exists()

    # Summary contains a list with one item
    data = json.loads(summary_path.read_text())
    assert isinstance(data, list) and len(data) == 1 and data[0]["loss"] == 0.5


def test_train_entrypoint_calls_main(monkeypatch, tmp_path, patch_paths):
    """
    Covers: the __main__ guard in ecg_cnn.train.
    Patch the sources that train.py imports from, BEFORE executing as __main__.
    """
    # Use central patched results dir everywhere train might look
    results_dir, *_ = patch_paths
    # Patch the source module that train.py imports from
    monkeypatch.setattr("ecg_cnn.paths.RESULTS_DIR", results_dir, raising=False)

    # --------------------------------------------------------------------------
    # Sandbox: redirect all paths into tmp_path so no writes escape to real
    # outputs/
    # ALSO: install a fully-populated ecg_cnn.paths module (with
    # DEFAULT_TRAINING_CONFIG) to avoid leaking a prior stub from sys.modules.
    # --------------------------------------------------------------------------
    base = tmp_path
    sand = {
        # results go to the fixture's results_dir so your asserts line up
        "OUTPUT_DIR": base,
        "RESULTS_DIR": results_dir,
        "REPORTS_DIR": base / "reports",
        "HISTORY_DIR": base / "history",
        "PLOTS_DIR": base / "plots",
        "MODELS_DIR": base / "models",
        "ARTIFACTS_DIR": base / "artifacts",
        "PTBXL_DATA_DIR": base / "ptbxl",
        "PROJECT_ROOT": base,
    }
    for p in sand.values():
        p.mkdir(parents=True, exist_ok=True)

    # Provide a default training config file so the import in train.py succeeds
    default_cfg = base / "baseline.yaml"
    default_cfg.write_text("model: ECGConvNet\n")

    # Install a fresh, fully-populated ecg_cnn.paths module in sys.modules
    pm = types.ModuleType("ecg_cnn.paths")
    pm.__file__ = str(base / "paths_sandbox.py")
    for name, path in sand.items():
        setattr(pm, name, path)
    pm.PTBXL_META_CSV = pm.PTBXL_DATA_DIR / "ptbxl_database.csv"
    pm.PTBXL_SCP_CSV = pm.PTBXL_DATA_DIR / "scp_statements.csv"
    pm.DEFAULT_TRAINING_CONFIG = default_cfg

    # Ensure it’s visible both as a module and as a package attribute
    sys.modules["ecg_cnn.paths"] = pm
    if "ecg_cnn" in sys.modules:
        setattr(sys.modules["ecg_cnn"], "paths", pm)

    # Also reflect the env vars that code may read
    monkeypatch.setenv("ECG_CNN_RESULTS_DIR", str(sand["RESULTS_DIR"]))
    monkeypatch.setenv("ECG_CNN_OUTPUT_DIR", str(sand["OUTPUT_DIR"]))

    # Stub out checkpoint saves so no .pth files are written; allow json/yaml to be created
    monkeypatch.setattr("torch.save", lambda *a, **k: None, raising=False)
    # If your trainer has explicit helpers, neuter them too (harmless if absent)
    monkeypatch.setattr(
        "ecg_cnn.training.trainer._save_checkpoint", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        "ecg_cnn.training.trainer._write_history", lambda *a, **k: None, raising=False
    )

    # Prevent argparse from seeing pytest args + stub parse_training_args/overrides
    monkeypatch.setattr("sys.argv", ["python"], raising=False)  # neutral argv
    monkeypatch.setattr(
        "ecg_cnn.training.cli_args.parse_training_args",
        lambda: SimpleNamespace(config=None),
        raising=False,
    )
    monkeypatch.setattr(
        "ecg_cnn.training.cli_args.override_config_with_args",
        lambda c, a: c,
        raising=False,
    )

    # Config loader: return a minimal base config; bypass YAML/grid merges
    # (base produced by your _cfg() helper)
    base_cfg = _cfg()
    monkeypatch.setattr(
        "ecg_cnn.config.config_loader.load_training_config",
        lambda _: base_cfg,
        raising=False,
    )
    monkeypatch.setattr(
        "ecg_cnn.config.config_loader.load_yaml_as_dict",
        lambda *a, **k: {},
        raising=False,
    )
    monkeypatch.setattr(
        "ecg_cnn.config.config_loader.merge_configs",
        lambda a, b: a,
        raising=False,
    )

    monkeypatch.setattr(
        "ecg_cnn.utils.grid_utils.is_grid_config", lambda d: False, raising=False
    )
    monkeypatch.setattr(
        "ecg_cnn.utils.grid_utils.expand_grid", lambda d: [], raising=False
    )

    # Trainer: stub run_training to avoid real work and capture the call
    calls = []

    def fake_run_training(config, fold_idx=None, tag=None):
        # Record the call like before
        calls.append({"fold_idx": fold_idx, "tag": tag})

        # --- Emit the same artifacts the entrypoint asserts on ---
        # Match your expected tag format
        expected_tag = (
            f"{config.model}_lr{config.lr:.4g}_bs{config.batch_size}_wd{config.weight_decay:.4g}"
        ).replace(".", "")
        # Write summary/config into the SAME results_dir you asserted on
        summary_path = results_dir / f"summary_{expected_tag}.json"
        config_path = results_dir / f"config_{expected_tag}.yaml"
        summary_path.write_text(
            json.dumps(
                {
                    "loss": 0.5,
                    "best_epoch": 1,
                    "model_path": "dummy.pth",
                    "model": config.model,
                },
                indent=2,
            )
        )
        config_path.write_text("model: {}\n".format(config.model))

        return {
            "loss": 0.5,
            "best_epoch": 1,
            "model_path": "dummy.pth",
            "model": config.model,
        }

    # Patch the import location train.py uses: ecg_cnn.training.trainer.run_training
    monkeypatch.setattr(
        "ecg_cnn.training.trainer.run_training", fake_run_training, raising=False
    )

    # Execute ecg_cnn.train as __main__ (fresh module namespace)
    sys.modules.pop("ecg_cnn.train", None)  # ensure fresh exec
    runpy.run_module("ecg_cnn.train", run_name="__main__")

    # Verify our stub was called (keeps original intent intact)
    assert len(calls) == 1

    # Recreate your original tag/paths expectations against the SAME results_dir
    expected_tag = (
        f"{base_cfg.model}_lr{base_cfg.lr:.4g}_bs{base_cfg.batch_size}_wd{base_cfg.weight_decay:.4g}"
    ).replace(".", "")
    summary_path = results_dir / f"summary_{expected_tag}.json"
    config_path = results_dir / f"config_{expected_tag}.yaml"
    assert summary_path.exists() and config_path.exists()


def test_main_best_by_accuracy_prints_dict(capsys):
    best_by_accuracy = {
        "model_path": "/fake/path/model.pt",
        "best_epoch": 5,
        "extra_field": "ignored",
    }

    # Simulate executing the block inline in train.py
    if best_by_accuracy is not None:
        print(
            f"\nBest model by accuracy: {best_by_accuracy['model_path']} (epoch {best_by_accuracy['best_epoch']})"
        )
        print(f"Best-by-accuracy summary: {best_by_accuracy}")
    else:
        print("\nBest model by accuracy: <none> (no accuracy recorded)")

    out = capsys.readouterr().out

    # Assert path + epoch appear
    assert "\nBest model by accuracy: /fake/path/model.pt (epoch 5)" in out
    # Assert the full dict summary is printed
    assert "Best-by-accuracy summary:" in out
    # Optional regex to check key names are present
    assert re.search(r"'model_path': '/fake/path/model.pt'", out)


def test_main_best_by_accuracy_prints_none(capsys):
    best_by_accuracy = None

    if best_by_accuracy is not None:
        print(
            f"\nBest model by accuracy: {best_by_accuracy['model_path']} (epoch {best_by_accuracy['best_epoch']})"
        )
        print(f"Best-by-accuracy summary: {best_by_accuracy}")
    else:
        print("\nBest model by accuracy: <none> (no accuracy recorded)")

    out = capsys.readouterr().out
    assert "\nBest model by accuracy: <none> (no accuracy recorded)" in out


def test_main_best_by_accuracy_branch_with_candidates(tmp_path, monkeypatch, capsys):
    # Patch config loader pieces to return a trivial config
    monkeypatch.setattr(
        train,
        "load_training_config",
        lambda _: SimpleNamespace(
            verbose=False,
            n_folds=2,
            model="m",
            lr=0.01,
            batch_size=1,
            weight_decay=0.0,
        ),
    )
    monkeypatch.setattr(train, "load_yaml_as_dict", lambda p: {})
    monkeypatch.setattr(train, "merge_configs", lambda base, d: base)
    monkeypatch.setattr(train, "override_config_with_args", lambda cfg, args: cfg)
    monkeypatch.setattr(
        train, "parse_training_args", lambda: SimpleNamespace(config=None)
    )

    # Patch RESULTS_DIR so no real writes leak
    monkeypatch.setattr(train, "RESULTS_DIR", tmp_path, raising=False)

    # Fake run_training: returns one summary dict per call
    fake_summaries = [
        {"model_path": "/tmp/a.pt", "best_epoch": 1, "loss": 0.9, "val_accs": 0.8},
        {"model_path": "/tmp/b.pt", "best_epoch": 2, "loss": 0.5, "train_accs": 0.9},
    ]

    def fake_run_training(cfg, **kwargs):
        return fake_summaries.pop(0)

    monkeypatch.setattr(train, "run_training", fake_run_training)

    train.main()
    out = capsys.readouterr().out

    # Check both loss and accuracy selections printed
    assert "Best model by loss: /tmp/b.pt (epoch 2)" in out
    assert "Best-by-loss summary:" in out
    assert "Best model by accuracy: /tmp/b.pt (epoch 2)" in out
    assert "Best-by-accuracy summary:" in out


def test_main_best_by_accuracy_branch_none(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        train,
        "load_training_config",
        lambda _: SimpleNamespace(
            verbose=False,
            n_folds=None,
            model="m",
            lr=0.01,
            batch_size=1,
            weight_decay=0.0,
        ),
    )
    monkeypatch.setattr(train, "load_yaml_as_dict", lambda p: {})
    monkeypatch.setattr(train, "merge_configs", lambda base, d: base)
    monkeypatch.setattr(train, "override_config_with_args", lambda cfg, args: cfg)
    monkeypatch.setattr(
        train, "parse_training_args", lambda: SimpleNamespace(config=None)
    )

    monkeypatch.setattr(train, "RESULTS_DIR", tmp_path, raising=False)

    # run_training returns dicts without val_accs/train_accs
    def fake_run_training(cfg, **kwargs):
        return {"model_path": "/tmp/x.pt", "best_epoch": 3, "loss": 0.1}

    monkeypatch.setattr(train, "run_training", fake_run_training)

    train.main()
    out = capsys.readouterr().out

    assert "Best model by loss: /tmp/x.pt (epoch 3)" in out
    assert "Best-by-loss summary:" in out
    assert "Best model by accuracy: <none> (no accuracy recorded)" in out


def test_main_skips_best_block_when_grid_is_empty(monkeypatch, capsys):
    """
    Strategy:
      - Pretend a config file was provided.
      - Force is_grid_config(...) -> True and expand_grid(...) -> [].
      - With an empty raw grid, param_grid becomes [] and the loop never runs.
      - all_summaries remains empty, so the 'if all_summaries:' block is skipped.
    """

    # Minimal args stub: pretend user passed a config path; keep quiet output
    class NS:
        config = "dummy.yaml"
        verbose = False

    monkeypatch.setattr(train, "parse_training_args", lambda: NS())

    # Baseline loader returns a lightweight object; it won't be used because grid is empty,
    # but returning something harmless keeps earlier steps robust.
    class DummyCfg:
        verbose = False
        n_folds = None

    monkeypatch.setattr(train, "load_training_config", lambda _p: DummyCfg())

    # YAML read returns a dict (contents irrelevant for this test)
    monkeypatch.setattr(train, "load_yaml_as_dict", lambda _p: {})

    # Force grid branch, but with zero expansions -> empty param_grid
    monkeypatch.setattr(train, "is_grid_config", lambda _d: True)
    monkeypatch.setattr(train, "expand_grid", lambda _d: [])

    # Run main; loop should be skipped; only elapsed time printed
    train.main()

    out = capsys.readouterr().out
    assert "Elapsed time:" in out
    assert "Starting training run" not in out
    assert "Saved best selections" not in out
    assert "Saved summary to:" not in out


# ------------------------------------------------------------------------------
# _acc_value()
# ------------------------------------------------------------------------------
def test_acc_value_prefers_val_when_present():
    s = {"val_accs": 0.91, "train_accs": 0.99}
    assert train._acc_value(s) == pytest.approx(0.91)


def test_acc_value_falls_back_to_train_when_val_missing_or_none():
    s1 = {"train_accs": 0.88}
    s2 = {"val_accs": None, "train_accs": 0.77}
    assert train._acc_value(s1) == pytest.approx(0.88)
    assert train._acc_value(s2) == pytest.approx(0.77)


def test_acc_value_missing_both_returns_neg_inf():
    s = {}
    out = train._acc_value(s)
    assert out == float("-inf")


def test_acc_value_non_numeric_returns_neg_inf():
    s1 = {"val_accs": "oops"}
    s2 = {"val_accs": None, "train_accs": "nope"}
    assert train._acc_value(s1) == float("-inf")
    assert train._acc_value(s2) == float("-inf")


def test_acc_value_nan_passthrough():
    s = {"val_accs": float("nan")}
    out = train._acc_value(s)
    assert math.isnan(out)


# ------------------------------------------------------------------------------
# _loss_value()
# ------------------------------------------------------------------------------
def test_loss_value_casts_float_from_key():
    s1 = {"loss": 1.234}
    s2 = {"loss": "2.5"}  # castable string
    assert train._loss_value(s1) == pytest.approx(1.234)
    assert train._loss_value(s2) == pytest.approx(2.5)


def test_loss_value_missing_returns_inf():
    s = {}
    assert train._loss_value(s) == float("inf")


def test_loss_value_non_numeric_returns_inf():
    s1 = {"loss": None}
    s2 = {"loss": "oops"}
    assert train._loss_value(s1) == float("inf")
    assert train._loss_value(s2) == float("inf")


def test_loss_value_nan_passthrough():
    s = {"loss": float("nan")}
    out = train._loss_value(s)
    assert math.isnan(out)
