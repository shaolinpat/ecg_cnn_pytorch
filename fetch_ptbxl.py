#!/usr/bin/env python3
"""
Download PTB-XL via S3 sync (requires AWS CLI) or fallback to wget.
"""
import os
import subprocess

def main():
    out = "data/ptbxl"
    os.makedirs(out, exist_ok=True)

    try:
        print("Attempting AWS S3 sync...")
        subprocess.run(
            ["aws", "s3", "sync", "--no-sign-request",
             "s3://physionet-open/ptb-xl/1.0.3/", out],
            check=True
        )
        print("S3 sync complete.")
    except Exception:
        print("S3 sync failed; falling back to wget over HTTPS...")
        subprocess.run(
            ["wget", "-r", "-N", "-c", "-np",
             "https://physionet.org/files/ptb-xl/1.0.3/","-P", out],
            check=True
        )
        print("wget download complete.")

if __name__ == "__main__":
    main()