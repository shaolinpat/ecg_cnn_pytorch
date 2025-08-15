# tests/test_grid_utils.py

"""
Tests for ecg_cnn.utils.grid_utils

Covers
------
    - is_grid_config(): detecting presence of grid-style lists
    - expand_grid(): Cartesian product expansion and validation paths
    - determinism, edge cases, and mutation-safety
"""

import pytest

from ecg_cnn.utils.grid_utils import is_grid_config, expand_grid


# -----------------------
# is_grid_config
# -----------------------


def test_is_grid_config_true():
    config = {"lr": [0.001, 0.0005], "batch_size": 64, "epochs": 10}
    assert is_grid_config(config) is True


def test_is_grid_config_false():
    config = {"lr": 0.001, "batch_size": 64, "epochs": 10}
    assert is_grid_config(config) is False


def test_is_grid_config_invalid_input():
    with pytest.raises(ValueError, match=r"^Expected config_dict to be a dict"):
        is_grid_config("not_a_dict")


def test_is_grid_config_single_length_lists_still_grid():
    # Clarify intent: a single-length list still counts as a "grid"
    config = {"lr": [0.001], "batch_size": 64}
    assert is_grid_config(config) is True


# -----------------------
# expand_grid
# -----------------------


def test_expand_grid_single_combination():
    config = {"lr": 0.001, "batch_size": 64, "epochs": 10}
    results = list(expand_grid(config))
    assert len(results) == 1
    assert results[0] == config


def test_expand_grid_multiple_combinations():
    config = {"lr": [0.001, 0.0005], "batch_size": [32, 64], "epochs": 10}
    results = list(expand_grid(config))
    expected = [
        {"lr": 0.001, "batch_size": 32, "epochs": 10},
        {"lr": 0.001, "batch_size": 64, "epochs": 10},
        {"lr": 0.0005, "batch_size": 32, "epochs": 10},
        {"lr": 0.0005, "batch_size": 64, "epochs": 10},
    ]
    assert results == expected  # order should be stable and predictable


def test_expand_grid_invalid_input():
    with pytest.raises(ValueError, match=r"^Expected config_dict to be a dict"):
        list(expand_grid(["not", "a", "dict"]))


def test_expand_grid_string_values_are_scalar_not_iterable():
    # Strings should be treated as scalar values, not iterables
    config = {"tag": "runA", "lr": [0.1, 0.2]}
    out = list(expand_grid(config))
    assert [d["tag"] for d in out] == ["runA", "runA"]


@pytest.mark.parametrize(
    "bad_iterable",
    [
        set([1, 2]),  # unordered
        tuple([1, 2, 3]),  # allowed? enforce decision: here we treat as invalid
    ],
)
def test_expand_grid_rejects_unsupported_iterables(bad_iterable):
    config = {"lr": bad_iterable, "epochs": 10}
    with pytest.raises(ValueError, match=r"^Unsupported iterable type for grid value"):
        list(expand_grid(config))


def test_expand_grid_rejects_empty_lists():
    config = {"lr": [], "batch_size": [32, 64]}
    with pytest.raises(ValueError, match=r"^Grid list for key 'lr' must be non-empty"):
        list(expand_grid(config))


def test_expand_grid_single_length_list_collapses_to_one_combo():
    config = {"lr": [0.001], "batch_size": [32], "epochs": 10}
    out = list(expand_grid(config))
    assert out == [{"lr": 0.001, "batch_size": 32, "epochs": 10}]


def test_expand_grid_outputs_are_independent_dicts():
    # Ensure expand_grid yields copies, not the same dict mutated each time.
    config = {"lr": [0.1, 0.2], "batch_size": [8, 16]}
    out = list(expand_grid(config))
    out[0]["lr"] = 999  # mutate first result
    # Others should remain unchanged
    assert out[1]["lr"] in (0.1, 0.2) and out[1]["lr"] != 999


def test_expand_grid_preserves_non_grid_keys():
    config = {"lr": [0.1, 0.2], "batch_size": 64, "epochs": 5}
    out = list(expand_grid(config))
    for d in out:
        assert d["batch_size"] == 64 and d["epochs"] == 5
