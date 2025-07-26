# ecg_cnn/config/config_loader.py

import yaml

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import get_origin, get_args, Union


@dataclass
class TrainConfig:
    """
    Structured configuration for training a model.

    Attributes
    ----------
    model : str
        Name of the model architecture to use.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    weight_decay : float
        L2 regularization strength.
    epochs : int
        Number of training epochs.
    save_best : bool
        Whether to save only the best model based on validation performance.
    sample_only : bool
        Whether to use a small sample dataset for debugging or quick tests.
    subsample_frac : float
        Fraction of the full dataset to load (e.g., 0.1 for 10%).
    sampling_rate : int
        Sampling frequency of the ECG signal (e.g., 100 or 500 Hz).
    data_dir : str | Path | None
        Optional override path to the PTB-XL data directory.
    sample_dir : str | Path | None
        Optional path to the sample dataset directory.
    verbose : bool
        Whether to print detailed status and configuration info during training.
    """

    model: str
    lr: float
    batch_size: int
    weight_decay: float
    epochs: int
    save_best: bool
    sample_only: bool
    subsample_frac: float
    sampling_rate: int
    data_dir: Union[str, Path, None] = None
    sample_dir: Union[str, Path, None] = None
    verbose: bool = False

    def finalize(self) -> "TrainConfig":
        """
        Normalize fields after merging CLI/YAML overrides. Ensures booleans are valid.
        """
        # Optional, but if booleans may still come in as None in some paths, enforce here
        if self.save_best is None:
            self.save_best = False
        if self.sample_only is None:
            self.sample_only = False
        if self.verbose is None:
            self.verbose = False

        # Normalize paths if needed
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.sample_dir, str):
            self.sample_dir = Path(self.sample_dir)

        return self


def load_training_config(path: Path | str, strict: bool = True) -> TrainConfig | dict:
    """
    Load and validate a YAML training config from the given path.

    Parameters
    ----------
    path : Path | str
        Path to the YAML configuration file.
    strict : bool, default=True
        If True, validate the config by instantiating TrainConfig.
        If False, return the raw dictionary without validation.

    Returns
    -------
    TrainConfig or dict
        Parsed and optionally validated configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the file is not a YAML dict or required fields are missing.
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error in config file: {e}")

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML dictionary: {path}")

    if strict:
        try:
            cfg = TrainConfig(**raw)
        except TypeError as e:
            raise ValueError(f"Invalid config structure or missing fields: {e}")
        return normalize_path_fields(cfg)

    return raw


def merge_configs(base: TrainConfig, override: TrainConfig | dict) -> TrainConfig:
    """
    Return a new TrainConfig with override values applied to a base config.

    Parameters
    ----------
    base : TrainConfig
        The base configuration object, typically loaded from baseline.yaml.

    override : TrainConfig or dict
        The override values, either as a TrainConfig instance or a raw dictionary.

    Returns
    -------
    TrainConfig
        A new TrainConfig object with override values merged into the base config.

    Raises
    ------
    TypeError
        If `base` is not a TrainConfig instance, or `override` is neither a TrainConfig nor a dict.

    ValueError
        If an override value is not of the expected type and cannot be used.
    """
    if not (is_dataclass(base) and isinstance(base, TrainConfig)):
        raise TypeError(f"Expected base to be TrainConfig, got {type(base).__name__}")

    if isinstance(override, dict):
        override_dict = override
    elif is_dataclass(override) and isinstance(override, TrainConfig):
        override_dict = asdict(override)
    else:
        raise TypeError(
            f"Expected override to be TrainConfig or dict, got {type(override).__name__}"
        )

    base_dict = asdict(base)

    for f in fields(base):
        name = f.name
        override_val = override_dict.get(name, None)

        if override_val is None:
            continue

        expected_type = f.type

        if get_origin(expected_type) is Union:
            accepted_types = tuple(
                t for t in get_args(expected_type) if t is not type(None)
            )
        else:
            accepted_types = (expected_type,)

        if isinstance(override_val, accepted_types):
            base_dict[name] = override_val
        else:
            raise ValueError(
                f"Invalid type for field '{name}': expected {accepted_types}, got {type(override_val)}"
            )

    return TrainConfig(**base_dict)


def normalize_path_fields(cfg: TrainConfig) -> TrainConfig:
    """
    Ensure all path-like fields in TrainConfig are Path objects if not None.
    """
    if isinstance(cfg.data_dir, str):
        cfg.data_dir = Path(cfg.data_dir)
    if isinstance(cfg.sample_dir, str):
        cfg.sample_dir = Path(cfg.sample_dir)
    return cfg


def load_yaml_as_dict(path: Path) -> dict:
    """
    Load a YAML file as a dictionary without instantiating a TrainConfig.

    Parameters
    ----------
    path : Path
        Path to a YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML content as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.

    ValueError
        If the file cannot be parsed as valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error in config file: {e}")
