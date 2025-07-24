# ecg_cnn/config/config_loader.py

import yaml

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """
    Structured configuration for training a model.

    Attributes:
        model (str): Name of the model architecture to use.
        lr (float): Learning rate.
        batch_size (int): Mini-batch size.
        weight_decay (float): L2 regularization strength.
        epochs (int): Number of training epochs.
        save_best (bool): Whether to save only the best model.
        sample_only (bool): Whether to use a small sample dataset.
        subsample_frac (float): Fraction of the full dataset to load.
        sampling_rate (int): Sampling frequency of the ECG signal.
        data_dir (str | None): Optional override for the PTB-XL data path.
        sample_dir (str | None): Optional path to the sample dataset.
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
    data_dir: str | None = None
    sample_dir: str | None = None


def load_training_config(path: Path | str) -> TrainConfig:
    """
    Load and validate a YAML training config from the given path.

    Parameters
    ----------
        path (Path | str): Path to the YAML configuration file.

    Returns
    -------
        TrainConfig: Parsed and validated configuration object.

    Raises
    ------
        FileNotFoundError
            If the config file does not exist.
        ValueError
            If the config file has incorrect structure or missing fields.
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

    try:
        return TrainConfig(**raw)
    except TypeError as e:
        raise ValueError(f"Invalid config structure or missing fields: {e}")
