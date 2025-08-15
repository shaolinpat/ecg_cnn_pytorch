# tests/test_config_loader.py

"""
Tests for ecg_cnn.config.config_loader.

Goals
-----
    1) Verify TrainConfig.finalize() normalizes flags and paths.
    2) Validate load_training_config behavior for valid, invalid, and edge
       cases.
    3) Test merge_configs with partial, full, dict-based, and invalid overrides.
    4) Ensure normalize_path_fields correctly converts string paths to Path
       objects.
    5) Verify load_yaml_as_dict handles valid, invalid, and missing files.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from ecg_cnn.config.config_loader import (
    load_training_config,
    merge_configs,
    normalize_path_fields,
    load_yaml_as_dict,
    TrainConfig,
)

# ------------------------------------------------------------------------------
# Local helpers
# ------------------------------------------------------------------------------


def make_temp_yaml(data: dict) -> Path:
    """Write a temporary YAML file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml")
    yaml.safe_dump(data, tmp, sort_keys=True)
    tmp.close()
    return Path(tmp.name)


# ==============================================================================
# A. TrainConfig.finalize
# ==============================================================================


def test_finalize_normalizes_flags_and_paths():
    cfg = TrainConfig(
        model="ECGConvNet",
        lr=0.001,
        batch_size=32,
        weight_decay=0.0,
        n_epochs=3,
        save_best=None,
        sample_only=None,
        subsample_frac=0.2,
        sampling_rate=500,
        data_dir="data/",
        sample_dir="sample/",
        verbose=None,
    ).finalize()

    assert cfg.save_best is False
    assert cfg.sample_only is False
    assert cfg.verbose is False
    assert cfg.n_epochs == 3
    assert cfg.n_folds == 1
    assert isinstance(cfg.data_dir, Path)
    assert isinstance(cfg.sample_dir, Path)


# ==============================================================================
# B. load_training_config
# ==============================================================================


def test_load_training_config_success():
    config_data = {
        "model": "ECGConvNet",
        "lr": 0.001,
        "batch_size": 64,
        "weight_decay": 0.0,
        "n_epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "data_dir": "/data",
        "sample_dir": "/samples",
        "n_folds": 66,
    }
    path = make_temp_yaml(config_data)
    config = load_training_config(path)
    assert isinstance(config, TrainConfig)
    assert config.model == "ECGConvNet"
    assert config.lr == 0.001
    assert config.n_epochs == 10
    assert config.n_folds == 66


def test_load_training_config_file_not_found():
    with pytest.raises(FileNotFoundError, match=r"^Config file not found"):
        load_training_config("nonexistent_config.yaml")


def test_load_training_config_path_is_directory(tmp_path):
    with pytest.raises(ValueError, match=r"^Expected a file"):
        load_training_config(tmp_path)


def test_load_training_config_yaml_parse_error(tmp_path):
    # Malformed YAML
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("{model: ECGConvNet, lr: 0.001,")
    with pytest.raises(ValueError, match=r"^YAML parse error"):
        load_training_config(bad_yaml)


def test_load_training_config_missing_fields():
    path = make_temp_yaml({"model": "ECGConvNet"})
    with pytest.raises(
        ValueError, match=r"^Invalid config structure or missing fields"
    ):
        load_training_config(path)


def test_load_training_config_raises_if_not_dict(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    # Write a YAML list instead of a dict
    bad_yaml.write_text(
        """
    - just: a
    - list: of values
    """
    )
    with pytest.raises(ValueError, match=r"^Config must be a YAML dictionary"):
        load_training_config(bad_yaml)


def test_load_training_config_returns_raw_when_not_strict(tmp_path):
    raw_config = {
        "model": "ECGConvNet",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "n_epochs": 10,
        "batch_size": 32,
        "subsample_frac": 0.2,
        "sampling_rate": 500,
        "data_dir": "/data/ptbxl",
        "sample_dir": None,
        "sample_only": False,
        "save_best": True,
        "verbose": True,
    }
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(f"{k}: {v}" for k, v in raw_config.items())
        .replace("None", "null")
        .replace("True", "true")
        .replace("False", "false")
    )
    result = load_training_config(yaml_path, strict=False)
    assert isinstance(result, dict)
    assert result == raw_config


# ==============================================================================
# C. merge_configs
# ==============================================================================


def test_merge_configs_partial_override_with_dict(make_train_config):
    base = make_train_config(
        model="ECGConvNet",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        n_epochs=10,
        n_folds=1,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        verbose=False,
    )
    override = {
        "model": "Base",
        "lr": 0.01,
        "batch_size": 128,
        "n_epochs": 3,
        "sample_only": True,
        "subsample_frac": 0.2,
        "sampling_rate": 500,
        "data_dir": "data_dir",
    }
    merged = merge_configs(base, override)
    assert merged.lr == 0.01
    assert merged.subsample_frac == 0.2
    assert merged.verbose is False
    assert merged.n_epochs == 3
    assert merged.n_folds == 1
    assert merged.weight_decay == 0.0


def test_merge_configs_partial_override_with_trainconfig(make_train_config):
    base = make_train_config(
        model="ECGConvNet",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        n_epochs=10,
        n_folds=1,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        verbose=False,
    )
    override = make_train_config(
        model="Base",
        lr=0.01,
        batch_size=128,
        n_epochs=3,
        sample_only=True,
        subsample_frac=0.2,
        sampling_rate=500,
        data_dir="data_dir",
    )
    merged = merge_configs(base, override)
    assert merged.lr == 0.01
    assert merged.subsample_frac == 0.2
    assert merged.verbose is False
    assert merged.n_epochs == 3
    assert merged.n_folds == 2
    assert merged.weight_decay == 0.0


def test_merge_configs_all_overrides(make_train_config):
    base = make_train_config()
    override = make_train_config(
        model="AltModel",
        lr=0.005,
        batch_size=128,
        weight_decay=0.01,
        n_epochs=20,
        save_best=False,
        sample_only=True,
        subsample_frac=0.5,
        sampling_rate=500,
        data_dir="/mnt/data",
        sample_dir="samples/alt",
        verbose=True,
        n_folds=3,
    )
    merged = merge_configs(base, override)
    assert merged.model == "AltModel"
    assert merged.verbose is True
    assert merged.save_best is False
    assert merged.n_folds == 3


def test_merge_configs_raises_on_non_trainconfig_inputs(make_train_config):
    base = make_train_config()
    with pytest.raises(TypeError, match=r"^Expected base to be TrainConfig"):
        merge_configs({}, base)
    with pytest.raises(TypeError, match=r"^Expected override to be TrainConfig"):
        merge_configs(base, "invalid")


def test_merge_configs_raises_on_bad_override_type(make_train_config):
    base = make_train_config()
    bad_override = make_train_config()
    bad_override.lr = "not_a_float"  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r"^Invalid type for field"):
        merge_configs(base, bad_override)


def test_merge_configs_optional_field_type_dispatch(make_train_config):
    base = make_train_config()
    override = make_train_config()
    override.sample_dir = "foo/bar"
    merged = merge_configs(base, override)
    assert str(merged.sample_dir) == "foo/bar"


def test_merge_configs_optional_field_with_invalid_type(make_train_config):
    base = make_train_config()
    override = make_train_config()
    override.sample_dir = 12345
    with pytest.raises(ValueError, match=r"^Invalid type for field 'sample_dir'"):
        merge_configs(base, override)


def test_merge_configs_union_field_with_str_triggers_get_args(make_train_config):
    base = make_train_config()
    override = make_train_config()
    override.sample_dir = "some/path"
    merged = merge_configs(base, override)
    merged = normalize_path_fields(merged)
    assert isinstance(merged.sample_dir, Path)
    assert merged.sample_dir == Path("some/path")


def test_merge_configs_union_field_with_invalid_type_triggers_get_args(
    make_train_config,
):
    base = make_train_config()
    override = make_train_config()
    override.sample_dir = 999
    with pytest.raises(ValueError, match=r"^Invalid type for field 'sample_dir'"):
        merge_configs(base, override)


def test_merge_configs_union_field_accepts_path_and_hits_get_args(make_train_config):
    base = make_train_config()
    override = make_train_config()
    override.sample_dir = Path("foo/bar")
    merged = merge_configs(base, override)
    assert merged.sample_dir == Path("foo/bar")
    assert isinstance(merged.sample_dir, Path)


def test_merge_configs_accepts_dict_override(make_train_config):
    base = make_train_config()
    override = {"model": "ECGConvNetV2", "lr": 0.0005, "verbose": True}
    result = merge_configs(base, override)
    assert result.model == "ECGConvNetV2"
    assert result.lr == 0.0005
    assert result.verbose is True


# ==============================================================================
# D. normalize_path_fields
# ==============================================================================


def test_normalize_path_fields_converts_data_dir_str_to_path(make_train_config):
    cfg = make_train_config()
    cfg.data_dir = "some/data/path"
    normalized = normalize_path_fields(cfg)
    assert isinstance(normalized.data_dir, Path)
    assert normalized.data_dir == Path("some/data/path")


def test_normalize_path_fields_converts_sample_dir_str_to_path(make_train_config):
    cfg = make_train_config()
    cfg.sample_dir = "some/sample/path"
    normalized = normalize_path_fields(cfg)
    assert isinstance(normalized.sample_dir, Path)
    assert normalized.sample_dir == Path("some/sample/path")


def test_normalize_path_fields_leaves_none_untouched(make_train_config):
    cfg = make_train_config(data_dir=None, sample_dir=None)
    normalized = normalize_path_fields(cfg)
    assert normalized.data_dir is None
    assert normalized.sample_dir is None


# ==============================================================================
# E. load_yaml_as_dict
# ==============================================================================


def test_load_yaml_as_dict_valid(tmp_path):
    path = tmp_path / "temp_config.yaml"
    path.write_text("lr: 0.001\nbatch_size: 32")
    out = load_yaml_as_dict(path)
    assert isinstance(out, dict)
    assert out["lr"] == 0.001
    assert out["batch_size"] == 32


def test_load_yaml_as_dict_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_yaml_as_dict(Path("nonexistent.yaml"))


def test_load_yaml_as_dict_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("this: is: not: valid: yaml")
    with pytest.raises(ValueError, match=r"^YAML parse error"):
        load_yaml_as_dict(path)


def test_load_yaml_as_dict_not_file(tmp_path):
    with pytest.raises(ValueError, match=r"^Expected a file, got:"):
        load_yaml_as_dict(tmp_path)
