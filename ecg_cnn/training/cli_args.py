# training/cli_args.py

import argparse

from argparse import Namespace
from pathlib import Path

from ecg_cnn.paths import PTBXL_DATA_DIR
from ecg_cnn.config.config_loader import TrainConfig
from ecg_cnn.utils.validate import validate_hparams_config


def parse_training_args(argv=None):
    """
    Parse command-line arguments for ECG training.py.

    Parameters
    ----------
    argv : list[str] | None
        If None, uses sys.argv[1:]; otherwise parse the provided list.

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
        - subsample_frac (float): Fraction of full data to load (e.g., 0.1 = 10 percent).
        - sampling_rate (int): ECG signal sampling rate (100 or 500 Hz).
        - data_dir (str): Path to the full PTB-XL dataset directory.
        - sample_dir (str): Path to directory with sample_ids.csv or per-record CSVs.

    Notes
    -----
    This function is used by the CLI entry point to train.py to configure
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
        "--n_epochs",
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
        help="Fraction of full data to load (e.g., 0.1 loads 10 percent)",
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
    p.add_argument(
        "--n_folds",
        type=int,
        default=None,
        help="Total number of folds to use for cross-validation (e.g., 5 for 5-fold CV). Default is 1.",
    )

    return p.parse_args(argv)


def parse_evaluate_args(argv=None):
    """
    Parse command-line arguments for ECG evaluate.py.

    Parameters
    ----------
    argv : list[str] | None
        If None, uses sys.argv[1:]; otherwise parse the provided list.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with the following attributes:

        - fold (int or None): 1-based fold index to evaluate. If None, evaluates
          the best run across all folds or follows the evaluator's default.

        OvR (one-vs-rest) controls:
        - enable_ovr (bool or None): Force-enable OvR plots. If None, defer to config.
        - ovr_classes (List[str] or None): Specific classes for OvR analysis.
          Implies --enable_ovr unless --disable_ovr is set.

        SHAP configuration:
        - shap_profile (str): SHAP profile ("off", "fast", "medium", "thorough", "custom").
        - shap_n (int or None): Number of samples to explain (custom mode only).
        - shap_bg (int or None): Background size (custom mode only).
        - shap_stride (int or None): Time downsample stride (::k) (custom mode only).

        Run selection:
        - prefer (str): Which config to evaluate.
          Choices:
            * "auto": Prefer accuracy if available, else loss (default).
            * "accuracy": Always pick best-by-accuracy if recorded.
            * "loss": Always pick best-by-loss if recorded.
            * "latest": Ignore "best" records; use newest config on disk.

    Notes
    -----
    This function is used by the CLI entry point to evaluate.py to control
    evaluation behavior, including fold selection, one-vs-rest diagnostics,
    SHAP explainability trade-offs, and whether to load the best or latest
    available training configuration for analysis.
    """
    p = argparse.ArgumentParser(
        description="Evaluate the most recent training config (or a forced fold).",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional 1-based fold index. If omitted, evaluate the selected run across all folds "
        "or follow the evaluator's default (best-by-metric).",
    )

    # OvR controls (mutually exclusive enable/disable)
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--enable_ovr",
        dest="enable_ovr",
        action="store_const",
        const=True,
        default=None,  # None = no CLI override; defer to config
        help="Enable one-vs-rest plots. Overrides config.",
    )
    group.add_argument(
        "--disable_ovr",
        dest="enable_ovr",
        action="store_const",
        const=False,
        help="Disable one-vs-rest plots. Overrides config.",
    )
    p.add_argument(
        "--ovr_classes",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        metavar="CLASS[,CLASS...]",
        default=None,
        help=(
            "Comma-separated class names for OvR analysis (e.g., 'MI,NORM').\n"
            "Implies --enable_ovr unless --disable_ovr is set."
        ),
    )

    # SHAP controls
    p.add_argument(
        "--shap",
        dest="shap_profile",
        choices=["off", "fast", "medium", "thorough", "custom"],
        default="medium",
        help=(
            "SHAP profile to bound runtime:\n"
            "  off      Disable SHAP\n"
            "  fast     Small sample sizes\n"
            "  medium   Balanced defaults (default)\n"
            "  thorough Larger samples, slower\n"
            "  custom   Use --shap-n / --shap-bg / --shap-stride"
        ),
    )
    p.add_argument(
        "--shap-n",
        type=int,
        default=None,
        help="Custom number of samples to explain. Requires --shap custom.",
    )
    p.add_argument(
        "--shap-bg",
        type=int,
        default=None,
        help="Custom background size. Requires --shap custom.",
    )
    p.add_argument(
        "--shap-stride",
        type=int,
        default=None,
        help="Custom time downsample stride (::k). Requires --shap custom.",
    )

    # Selection of which run/config is preferred
    p.add_argument(
        "--prefer",
        choices=["auto", "accuracy", "loss", "latest"],
        default="auto",
        metavar="MODE",
        help=(
            "Which config to evaluate:\n"
            "  auto      Prefer accuracy if available, else loss (default)\n"
            "  accuracy  Always pick best-by-accuracy if recorded\n"
            "  loss      Always pick best-by-loss if recorded\n"
            "  latest    Ignore 'best' records; use newest config on disk"
        ),
    )

    return p.parse_args(argv)


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
        "n_epochs",
        "save_best",
        "sample_only",
        "subsample_frac",
        "sampling_rate",
        "data_dir",
        "sample_dir",
        "verbose",
        "n_folds",
    ]

    for field in override_fields:
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                setattr(config, field, value)

    # Validate core numeric and structural parameters using shared logic
    validate_hparams_config(
        model=config.model,
        lr=config.lr,
        bs=config.batch_size,
        wd=config.weight_decay,
        n_epochs=config.n_epochs,
        n_folds=config.n_folds,
    )

    # Additional checks for fields not covered by validate_hparams_config
    if not (0.0 < config.subsample_frac <= 1.0):
        raise ValueError(
            f"subsample_frac must be in (0.0, 1.0], got {config.subsample_frac}"
        )
    if config.sampling_rate not in (100, 500):
        raise ValueError(
            f"sampling_rate must be 100 or 500, got {config.sampling_rate}"
        )
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
