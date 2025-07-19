import argparse


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
        default="../data/ptbxl/physionet.org/files/ptb-xl/1.0.3",
        type=str,
        help="Path to full PTB-XL data directory",
    )
    p.add_argument(
        "--subsample-frac",
        type=float,
        default=1.0,
        help="Fraction of full data to load (for quick smoke tests)",
    )
    return p.parse_args()
