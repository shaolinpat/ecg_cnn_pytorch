import argparse

from argparse import Namespace
from pathlib import Path

from ecg_cnn.paths import PTBXL_DATA_DIR
from ecg_cnn.config.config_loader import TrainConfig
from ecg_cnn.utils.validate import validate_hparams


def parse_args():
    """
    Parse command-line arguments for ECG training or evaluation scripts.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with the following attributes:

        - model (str): Model architecture name.

        Training hyperparameters:
        - lr (float): Learning rate for the optimizer.
        - weight_decay (float): L2 regularization strength.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training or evaluation.
        - save_best (bool): Save model only if performance improves.

        Network architecture:
        - kernel_sizes (List[int]): Kernel sizes for Conv1D layers (e.g., 16 3 3).
        - conv_dropout (float): Dropout rate after convolutional layers.
        - fc_dropout (float): Dropout rate in fully connected layers.

        Data loading and preprocessing:
        - sample_only (bool): Whether to run on bundled 100-record sample only.
        - subsample_frac (float): Fraction of full data to load (e.g., 0.1 = 10%).
        - sampling_rate (int): ECG signal sampling rate (100 or 500 Hz).
        - data_dir (str): Path to the full PTB-XL dataset directory.
        - sample_dir (str): Path to directory with sample_ids.csv or per-record CSVs.

    Notes
    -----
    This function is used by CLI entry points like train.py and evaluate.py to configure
    runtime behavior, such as model hyperparameters, input sampling, and override values
    from baseline.yaml without modifying the original configuration file.
    """

    p = argparse.ArgumentParser(description="Train ECG classifier")

    # Model and architecture
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML config to override baseline (partial OK)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture name",
    )

    # Training hyperparameters
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (e.g., 0.001)",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="L2 regularization strength (e.g., 0.0)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training or evaluation",
    )
    p.add_argument(
        "--save-best",
        action="store_true",
        default=None,
        help="Only save model checkpoint if loss improves (default: off)",
    )

    # Convolutional network specifics
    p.add_argument(
        "--kernel-sizes",
        type=_positive_int,
        nargs=3,
        default=None,
        help="Kernel sizes for each Conv1D layer (e.g., 16 3 3)",
    )
    p.add_argument(
        "--conv-dropout",
        type=float,
        default=None,
        help="Dropout rate after conv layers",
    )
    p.add_argument(
        "--fc-dropout",
        type=float,
        default=None,
        help="Dropout rate in fully connected layers",
    )

    # Data loading and sampling
    p.add_argument(
        "--sample-only",
        action="store_true",
        default=None,
        help="Run on bundled 100-record sample instead of full dataset",
    )
    p.add_argument(
        "--subsample-frac",
        type=float,
        default=None,
        help="Fraction of full data to load (e.g., 0.1 loads 10%)",
    )
    p.add_argument(
        "--sampling-rate",
        type=int,
        choices=[100, 500],
        default=None,
        help="Sampling rate of ECG signal (100 or 500 Hz)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to full PTB-XL data directory",
    )
    p.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help="Directory containing sample_ids.csv or per-record CSVs",
    )

    # housekeeping arguments
    p.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Enable verbose output (default: off)",
    )

    return p.parse_args()


def _positive_int(value):
    """
    Validate that the input value is a positive integer.

    Parameters
    ----------
    value : str
        Input value to validate and convert to a positive integer.

    Returns
    -------
    int
        Parsed positive integer.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not an integer or is not greater than zero.


    Examples
    --------
    >>> _positive_int("5")
    5

    >>> _positive_int("-1")
    argparse.ArgumentTypeError: -1 is not a positive integer

    >>> _positive_int("five")
    argparse.ArgumentTypeError: five is not an integer
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


def override_config_with_args(config: TrainConfig, args: Namespace) -> TrainConfig:
    """
    Override fields in the configuration object using validated CLI arguments.

    Parameters
    ----------
    config : TrainConfig
        Training configuration object to update.

    args : argparse.Namespace
        Parsed command-line arguments (e.g., from `argparse.ArgumentParser.parse_args()`).

    Returns
    -------
    TrainConfig
        Updated configuration object with any CLI-specified overrides.

    Raises
    ------
    ValueError
        If any argument has an invalid value or type.
    """
    override_fields = [
        "config",
        "model",
        "lr",
        "batch_size",
        "weight_decay",
        "epochs",
        "save_best",
        "sample_only",
        "subsample_frac",
        "sampling_rate",
        "data_dir",
        "sample_dir",
        "verbose",
    ]

    for field in override_fields:
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                setattr(config, field, value)

    # Validate core numeric and structural parameters using shared logic
    validate_hparams(
        model=config.model,
        lr=config.lr,
        bs=config.batch_size,
        wd=config.weight_decay,
        fold=0,  # Override caller can hardcode fold if unused
        epochs=config.epochs,
        prefix="cli",  # Dummy string to satisfy validation
        fname_metric=None,
    )

    # Additional checks for fields not covered by validate_hparams
    if not (0.0 < config.subsample_frac <= 1.0):
        raise ValueError(
            f"subsample_frac must be in (0.0, 1.0], got {config.subsample_frac}"
        )
    if config.sampling_rate not in (100, 500):
        raise ValueError(
            f"sampling_rate must be 100 or 500, got {config.sampling_rate}"
        )
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError(f"model must be a non-empty string, got: {repr(config.model)}")
    if config.data_dir is not None and not isinstance(config.data_dir, (str, Path)):
        raise ValueError(
            f"data_dir must be a string, Path, or None, got {type(config.data_dir).__name__}"
        )
    if config.sample_dir is not None and not isinstance(config.sample_dir, (str, Path)):
        raise ValueError(
            f"sample_dir must be a string, Path, or None, got {type(config.sample_dir).__name__}"
        )
    if config.verbose is not None and not isinstance(config.verbose, bool):
        raise ValueError(
            f"verbose must be a boolean or None, got {type(config.verbose).__name__}"
        )

    return config
