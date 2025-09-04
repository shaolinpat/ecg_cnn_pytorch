# utils/grid_utils.py

"""
grid_utils.py

Utility functions for expanding hyperparameter grids and generating configuration
combinations for multi-run training (grid search). Designed to be called by train.py
for flexibility and professionalism.
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

    Contract
    --------
    - Keys with LIST values are treated as grid axes (lists must be non-empty).
    - Tuples or sets are unsupported for grid axes (raise).
    - All other values are static (passed through).
    - This function does NOT validate element types; the consumer should.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary.

    Yields
    ------
    dict
        One configuration per Cartesian combination of grid axes.
    """
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Expected config_dict to be a dict, got {type(config_dict).__name__}"
        )

    grid_params: Dict[str, list] = {}
    static_params: Dict[str, Any] = {}

    for k, v in config_dict.items():
        if isinstance(v, list):
            if len(v) == 0:
                # Tests expect anchored message: r"^Grid list for key 'lr' must be non-empty"
                raise ValueError(f"Grid list for key '{k}' must be non-empty")
            grid_params[k] = v
        elif isinstance(v, (tuple, set)):
            # Tests expect anchored message: r"^Unsupported iterable type for grid value"
            raise ValueError("Unsupported iterable type for grid value")
        else:
            static_params[k] = v

    if not grid_params:
        # No sweep axes -> yield a deep copy of the input
        yield copy.deepcopy(config_dict)
        return

    keys = list(grid_params.keys())
    for values in itertools.product(*(grid_params[k] for k in keys)):
        run = static_params.copy()
        run.update(dict(zip(keys, values)))
        yield run
