"""
test_grid_utils.py

Unit tests for grid_utils.py.
"""

import pytest

from ecg_cnn.utils.grid_utils import is_grid_config, expand_grid


def test_is_grid_config_true():
    config = {
        "lr": [0.001, 0.0005],
        "batch_size": 64,
        "epochs": 10,
    }
    assert is_grid_config(config) is True


def test_is_grid_config_false():
    config = {
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 10,
    }
    assert is_grid_config(config) is False


def test_is_grid_config_invalid_input():
    with pytest.raises(ValueError, match="Expected config_dict to be a dict"):
        is_grid_config("not_a_dict")


def test_expand_grid_single_combination():
    config = {
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 10,
    }
    results = list(expand_grid(config))
    assert len(results) == 1
    assert results[0] == config


def test_expand_grid_multiple_combinations():
    config = {
        "lr": [0.001, 0.0005],
        "batch_size": [32, 64],
        "epochs": 10,
    }
    results = list(expand_grid(config))
    expected = [
        {"lr": 0.001, "batch_size": 32, "epochs": 10},
        {"lr": 0.001, "batch_size": 64, "epochs": 10},
        {"lr": 0.0005, "batch_size": 32, "epochs": 10},
        {"lr": 0.0005, "batch_size": 64, "epochs": 10},
    ]
    assert results == expected


def test_expand_grid_invalid_input():
    with pytest.raises(ValueError, match="Expected config_dict to be a dict"):
        list(expand_grid(["not", "a", "dict"]))
