"""
grid_utils.py

Utility functions for expanding hyperparameter grids and generating configuration
combinations for multi-run training (grid search). Designed to be called by train.py
for flexibility and professionalism.

Author: Your Name
Date: 2025-07-27
"""

import copy
import itertools
from typing import Any, Dict, Generator


def is_grid_config(config_dict: Dict[str, Any]) -> bool:
    """
    Check if the config contains any hyperparameter that is a list.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary parsed from YAML.

    Returns
    -------
    bool
        True if any value in the config is a list (indicating a grid search).
    """
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Expected config_dict to be a dict, got {type(config_dict).__name__}"
        )
    return any(isinstance(value, list) for value in config_dict.values())


def expand_grid(config_dict: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Yield individual configurations from a grid config dictionary.

    Parameters
    ----------
    config_dict : dict
        The original configuration dictionary. Keys with list values are treated as grid axes.

    Yields
    ------
    dict
        A configuration dictionary with one combination of hyperparameters.
    """
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Expected config_dict to be a dict, got {type(config_dict).__name__}"
        )

    grid_params = {k: v for k, v in config_dict.items() if isinstance(v, list)}
    static_params = {k: v for k, v in config_dict.items() if not isinstance(v, list)}

    if not grid_params:
        yield copy.deepcopy(config_dict)
        return

    keys = list(grid_params.keys())
    values_product = itertools.product(*(grid_params[k] for k in keys))

    for values in values_product:
        run_config = static_params.copy()
        run_config.update(dict(zip(keys, values)))
        yield run_config
