import pytest
import tempfile
import yaml
from pathlib import Path
from ecg_cnn.config.config_loader import load_training_config, TrainConfig

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
# def load_training_config(path: Path | str) -> TrainConfig:
# ------------------------------------------------------------------------------


def test_load_training_config_success():
    config_data = {
        "model": "ECGConvNet",
        "lr": 0.001,
        "batch_size": 64,
        "weight_decay": 0.0,
        "epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "data_dir": "/data",
        "sample_dir": "/samples",
    }
    path = make_temp_yaml(config_data)
    config = load_training_config(path)
    assert isinstance(config, TrainConfig)
    assert config.model == "ECGConvNet"
    assert config.lr == 0.001


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
