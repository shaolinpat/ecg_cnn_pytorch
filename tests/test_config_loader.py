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
# Helpers
# ------------------------------------------------------------------------------


def make_temp_yaml(data: dict) -> Path:
    """Helper to write a temporary YAML file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml")
    yaml.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


# ------------------------------------------------------------------------------
# def finalize(self) -> "TrainConfig":
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# def load_training_config(path: Path | str) -> TrainConfig:
# ------------------------------------------------------------------------------


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
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_training_config("nonexistent_config.yaml")


def test_load_training_config_path_is_directory(tmp_path):
    with pytest.raises(ValueError, match="Expected a file"):
        load_training_config(tmp_path)  # tmp_path is a dir


def test_load_training_config_yaml_parse_error():
    bad_yaml = "{model: ECGConvNet, lr: 0.001,"  # malformed
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml")
    tmp.write(bad_yaml)
    tmp.close()
    with pytest.raises(ValueError, match="YAML parse error"):
        load_training_config(tmp.name)


def test_load_training_config_missing_fields():
    incomplete_data = {
        "model": "ECGConvNet",  # missing other required fields
    }
    path = make_temp_yaml(incomplete_data)
    with pytest.raises(ValueError, match="Invalid config structure or missing fields"):
        load_training_config(path)


def test_load_training_config_raises_if_not_dict(tmp_path):
    """
    Ensure load_training_config raises ValueError if YAML is not a dict.
    """
    bad_yaml = tmp_path / "bad.yaml"
    # Write a YAML list instead of a dict
    bad_yaml.write_text(
        """
    - just: a
    - list: of values
    """
    )

    with pytest.raises(ValueError, match=r"Config must be a YAML dictionary"):
        load_training_config(bad_yaml)


def test_load_training_config_returns_raw_when_not_strict(tmp_path):
    """
    Ensure load_training_config returns raw dict when strict=False.
    """
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


# ------------------------------------------------------------------------------
# def merge_configs(base: TrainConfig, override: TrainConfig | dict)
#       -> TrainConfig:
# ------------------------------------------------------------------------------


def base_cfg():
    return TrainConfig(
        model="ECGConvNet",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        n_epochs=10,
        save_best=True,
        sample_only=False,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
        n_folds=1,
    )


def test_merge_configs_partial_override():
    override = TrainConfig(
        model="Base",
        lr=0.01,
        batch_size=128,
        weight_decay=0.0,
        n_epochs=3,
        save_best=True,
        sample_only=True,
        subsample_frac=0.2,
        sampling_rate=500,
        data_dir="data_dir",
        sample_dir=None,
    )
    merged = merge_configs(base_cfg(), override)
    assert merged.lr == 0.01
    assert merged.subsample_frac == 0.2
    assert merged.verbose == False


def test_merge_configs_all_overrides():
    override = TrainConfig(
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
    merged = merge_configs(base_cfg(), override)
    assert merged.model == "AltModel"
    assert merged.verbose is True
    assert merged.save_best is False
    assert merged.n_folds == 3


def test_merge_configs_raises_on_non_trainconfig_inputs():
    with pytest.raises(TypeError, match="Expected base to be TrainConfig"):
        merge_configs({}, base_cfg())
    with pytest.raises(TypeError, match="Expected override to be TrainConfig"):
        merge_configs(base_cfg(), "invalid")


def test_merge_configs_raises_on_bad_override_type():
    bad_override = base_cfg()
    bad_override.lr = "not_a_float"
    with pytest.raises(ValueError, match="Invalid type for field"):
        merge_configs(base_cfg(), bad_override)


def test_merge_configs_optional_field_type_dispatch():
    base = base_cfg()
    override = base_cfg()
    override.sample_dir = "foo/bar"  # str to Optional[str|Path]

    merged = merge_configs(base, override)
    assert merged.sample_dir == "foo/bar" or str(merged.sample_dir) == "foo/bar"


def test_merge_configs_optional_field_with_invalid_type():
    base = base_cfg()
    override = base_cfg()
    override.sample_dir = 12345  # not str or Path

    with pytest.raises(ValueError, match="Invalid type for field 'sample_dir'"):
        merge_configs(base, override)


def test_merge_configs_union_field_with_str_triggers_get_args():
    base = base_cfg()
    override = base_cfg()
    override.sample_dir = "some/path"  # str into Optional[Union[str, Path]]

    merged = merge_configs(base, override)
    merged = normalize_path_fields(merged)

    # Should coerce successfully
    assert isinstance(merged.sample_dir, Path)
    assert merged.sample_dir == Path("some/path")


def test_merge_configs_union_field_with_invalid_type_triggers_get_args():
    base = base_cfg()
    override = base_cfg()
    override.sample_dir = 999  # not str, not Path

    with pytest.raises(ValueError, match="Invalid type for field 'sample_dir'"):
        merge_configs(base, override)


def test_merge_configs_union_field_accepts_path_and_hits_get_args():
    base = base_cfg()
    override = base_cfg()
    override.sample_dir = Path("foo/bar")

    # Triggers fallback to get_args(), covering:
    # accepted_types = tuple(t for t in get_args(expected_type) if t is not type(None))
    merged = merge_configs(base, override)

    assert merged.sample_dir == Path("foo/bar")
    assert isinstance(merged.sample_dir, Path)


def test_merge_configs_accepts_dict_override():
    """
    Ensure merge_configs works when the override is a plain dictionary.
    """
    base = load_training_config("configs/baseline.yaml", strict=True)
    override = {
        "model": "ECGConvNetV2",
        "lr": 0.0005,
        "verbose": True,
    }

    result = merge_configs(base, override)
    assert result.model == "ECGConvNetV2"
    assert result.lr == 0.0005
    assert result.verbose is True


# ------------------------------------------------------------------------------
# def normalize_path_fields(cfg: TrainConfig) -> TrainConfig:
# ------------------------------------------------------------------------------


def test_normalize_path_fields_converts_data_dir_str_to_path():
    cfg = base_cfg()
    cfg.data_dir = "some/data/path"

    normalized = normalize_path_fields(cfg)

    assert isinstance(normalized.data_dir, Path)
    assert normalized.data_dir == Path("some/data/path")


def test_normalize_path_fields_converts_sample_dir_str_to_path():
    cfg = base_cfg()
    cfg.sample_dir = "some/sample/path"

    normalized = normalize_path_fields(cfg)

    assert isinstance(normalized.sample_dir, Path)
    assert normalized.sample_dir == Path("some/sample/path")


# ------------------------------------------------------------------------------
# def load_yaml_as_dict(path: Path) -> dict
# ------------------------------------------------------------------------------


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
    with pytest.raises(ValueError, match="YAML parse error"):
        load_yaml_as_dict(path)


def test_load_yaml_as_dict_not_file(tmp_path):
    """
    Ensure load_training_config raises ValueError when given a directory path instead of a file.
    """
    # tmp_path is a Path object pointing to a temporary directory
    with pytest.raises(ValueError, match="Expected a file, got:"):
        load_yaml_as_dict(tmp_path)
