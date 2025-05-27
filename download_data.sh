#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
DATASET="shayanfazeli/heartbeat"
FILES=(
  "mitbih_train.csv"
  "mitbih_test.csv"
  "ptbdb_abnormal.csv"
  "ptbdb_normal.csv"
)

echo "1) Checking for Kaggle CLI..."
if ! command -v kaggle &> /dev/null; then
  echo "  kaggle not found. Install it with:"
  echo "     pip install kaggle"
  exit 1
fi

echo "2) Checking for Kaggle credentials..."
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "  No API token found at ~/.kaggle/kaggle.json."
  echo "     - Go to https://www.kaggle.com → Account → Create New API Token"
  echo "     - Place the downloaded kaggle.json at ~/.kaggle/kaggle.json (chmod 600)"
  exit 1
fi

echo "3) Creating data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

echo "4) Downloading ECG CSVs from Kaggle dataset $DATASET..."
# Download & unzip only the files we want
kaggle datasets download -d "$DATASET" -p "$DATA_DIR" --quiet --unzip

echo "5) Cleaning up extra files..."
# Remove any files not in our FILES list
pushd "$DATA_DIR" > /dev/null
for f in *; do
  if [[ ! " ${FILES[*]} " =~ " $f " ]]; then
    rm -f "$f"
  fi
done
popd > /dev/null

echo "All done! Your data directory now contains:"
ls -lh "$DATA_DIR"