#!/usr/bin/env python
"""
convert_to_streamlit_format.py

Convert a raw ECG CSV (PTB-XL format) into a simplified
12-lead CSV usable with the Streamlit ECG Explorer.

Usage:
    python explorer/convert_to_streamlit_format.py input.csv output.csv
"""

import sys
import pandas as pd


def convert_to_sample(input_path: str, output_path: str) -> None:
    # Load the raw PTB-XL CSV (no headers by default)
    df = pd.read_csv(input_path, header=None)

    # Assume 12 leads; create column names
    col_names = [f"lead{i+1}" for i in range(12)]
    df = df.iloc[:, :12]  # first 12 columns only
    df.columns = col_names

    # Add synthetic time column (0,1,2,…)
    df.insert(0, "time", range(len(df)))

    # Save to new CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python explorer/convert_to_sample.py input.csv output.csv")
        sys.exit(1)
    convert_to_sample(sys.argv[1], sys.argv[2])
