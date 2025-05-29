 
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

## Quickstart


```bash
git clone git@github.com:shaolinpat/ecg_cnn_pytorch_demo.git
cd ecg_cnn_pytorch_demo
pip install -r requirements.txt
python ecg_cnn_pytorch_demo.py
```

---

# Full Dataset

## If you want to reproduce the full PTB-XL results:

   ```bash
   pip install wfdb awscli`   # one-time install
   python fetch_ptbxl.py`      # ~3 GB via S3 in <5 min
   python build_ptbxl_sample.py` --n_records 100  # re-generate sample if desired
   python ecg_cnn_pytorch_demo.py`  # now runs on full data
   ```
