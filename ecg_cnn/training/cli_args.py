import argparse
from ecg_cnn.config import PTBXL_DATA_DIR


def parse_args():
    """
    Parse command-line arguments for ECG training script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with the following attributes:
        - sample_dir (str): Path to directory containing sample_ids.csv or per-record CSVs.
        - sample_only (bool): Whether to run on bundled 100-record sample only.
        - data_dir (str): Path to the full PTB-XL dataset directory.
        - subsample_frac (float): Fraction of data to use (e.g., 0.1 loads 10% of records).
        - batch_size (int): Batch size for training (default=32).
        - kernel_sizes (List[int]): Kernel sizes for each Conv1D layer.
        - conv_dropout (float): Dropout rate after conv layers.
        - fc_dropout (float): Dropout rate after FC layers.

    Notes
    -----
    This function is typically used by the CLI entry point (train.py) to configure
    the training environment, including switching between the full PTB-XL dataset
    and smaller sample subsets for debugging or smoke testing.
    """
    p = argparse.ArgumentParser(description="Train ECG classifier")
    p.add_argument(
        "--sample-dir",
        default="data/larger_sample",
        type=str,
        help="Directory containing sample_ids.csv or per-record CSVs",
    )
    p.add_argument(
        "--sample-only",
        action="store_true",
        help="Run on bundled 100-record sample instead of full dataset",
    )
    p.add_argument(
        "--data-dir",
        default=PTBXL_DATA_DIR,
        type=str,
        help="Path to full PTB-XL data directory",
    )
    p.add_argument(
        "--subsample-frac",
        type=float,
        default=1.0,
        help="Fraction of full data to load (for quick smoke tests)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for current run",
    )
    p.add_argument(
        "--kernel-sizes",
        type=positive_int,
        nargs=3,
        default=[16, 3, 3],
        help="Kernel sizes for each Conv1D layer (e.g., 16 3 3)",
    )
    p.add_argument(
        "--conv-dropout",
        type=float,
        default=0.3,
        help="Dropout rate after conv layers (default=0.3)",
    )
    p.add_argument(
        "--fc-dropout",
        type=float,
        default=0.5,
        help="Dropout rate in fully connected layers (default=0.5)",
    )
    return p.parse_args()


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue
