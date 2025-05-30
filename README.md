 
# ECG Signal Classification with CNN (PTB-XL Dataset)

This project implements a 1D Convolutional Neural Network (CNN) to classify ECG signals using the PTB-XL 12-lead ECG dataset. It was developed as a final project for an AI class and originally delivered as a `.py` script, now converted to Jupyter Notebook format.

---

## Tools & Frameworks

- Python 3.x
- PyTorch
- NumPy & Pandas
- Matplotlib & TQDM
- Stratified K-Fold cross-validation
- Confusion matrix and classification report metrics

---

## Model Summary

- A deep 1D CNN for classifying ECG beats into multiple rhythm categories
- Two convolutional layers → pooling → dense → softmax
- Optimized using `Adam` with categorical cross-entropy loss

---

## Key Results

- Model trained and validated using 5-fold stratified cross-validation
- Performance metrics captured in `results_summary_pytorch.csv`
- Confusion matrix and accuracy plotted and saved in `Outfiles_pytorch/`

---

## Visualization Recommendations (To Add)

- Sample waveforms for each class
- Class distribution (before and after resampling, if used)
- Confusion matrix plot
- Training/validation loss curve

---

## Data Setup

Before running the notebook or Streamlit app, fetch the datasets:

```bash
bash download_data.sh
```

---

## How to Run

1. Clone the repo
2. Create environment:

   ```bash
   conda create -n ecg_cnn python=3.11
   conda activate ecg_cnn
   pip install -r requirements.txt
   ```

---

## Quickstart Demo (time < 30 s)

Clone the repo and run on the bundled sample data:

```bash
git clone git@github.com:yourusername/ecg_cnn.git
cd ecg_cnn
conda activate ecg_cnn           # or `pip install -r requirements.txt`
python build_ptbxl_sample.py \
  --n_records 100               # only first 100 records (if you need to regenerate)
python train_ecg_cnn_ptbxl.py   # trains & evaluates on sample subset
```

---

## Full Dataset Reproduction (optional, ~3 GB, may take hours)

If you need the full 12-lead PTB-XL for deeper experiments:

```bash
# 1) Install the downloader tools
pip install wfdb awscli

# 2) Fetch the full PTB-XL archive via S3 (resumable)
python fetch_ptbxl.py   # pulls ~3 GB; time depends on your bandwidth

# 3) (Re)build the 100-record sample if you want to verify
python build_ptbxl_sample.py --n_records 100

# 4) Train/evaluate on the full data directory
python train_ecg_cnn_ptbxl.py \
  --data-dir data/ptbxl/physionet.org/files/ptb-xl/1.0.3
   ```
