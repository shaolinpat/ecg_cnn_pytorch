#!/usr/bin/env python3
"""
Generate a small sample subset of PTB-XL records for quick demo.
Saves first N records from records100 as CSVs under data/sample.
Usage: python make_ptbxl_sample.py --n_records 100
"""
import os
import argparse
import wfdb
import pandas as pd


def main(n_records, ptb_dir="data/ptbxl", sample_dir="data/sample"):
    # Ensure directories
    os.makedirs(sample_dir, exist_ok=True)
    meta_csv = os.path.join(
        ptb_dir,
        "physionet.org",
        "files",
        "ptb-xl",
        "1.0.3",
        "ptbxl_database.csv",
    )

    df = pd.read_csv(meta_csv)
    # Take the first n_records unique recording names
    recs = df.filename_lr.unique()[:n_records]

    print(f"Generating sample of {len(recs)} records into {sample_dir}...")
    for rec in recs:

        # Read 100 Hz version
        rec_path = os.path.join(
            ptb_dir,
            "physionet.org", "files", "ptb-xl", "1.0.3",
            rec
        )
        record = wfdb.rdrecord(rec_path)
        # record.p_signal: numpy array shape (n_samples, 12)

        rec_name = os.path.basename(rec)
        out_csv = os.path.join(sample_dir, f"{rec_name}_100hz.csv")
        pd.DataFrame(record.p_signal).to_csv(out_csv, index=False, header=False)
    print("Sample generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_records", type=int, default=100,
                        help="Number of records to include in sample")
    args = parser.parse_args()
    main(args.n_records)