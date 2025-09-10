# ECG-CNN (PyTorch) — PTB-XL ECG Classification

[![CI](https://github.com/shaolinpat/ecg_cnn_pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/shaolinpat/ecg_cnn_pytorch/actions/workflows/ci.yml)
[![Coverage (flag)](https://img.shields.io/codecov/c/github/shaolinpat/ecg_cnn_pytorch.svg?flag=flower_classifier&branch=main)](https://codecov.io/gh/shaolinpat/ecg_cnn_pytorch)  
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaolinpat/ecg_cnn_pytorch/blob/main/ecg_cnn/train.py?force_reload=true)

Reproducible, fully-tested deep learning pipeline for 12-lead ECG classification on **PTB-XL**.  
Includes clean training/evaluation CLIs, YAML configs + grids, SHAP explainability, rich plots, and CSV summaries.

> **For hiring managers:** This repo showcases production-grade ML engineering: modular PyTorch, deterministic pipelines, thorough tests, clear experiment tracking, and repeatable results—no hidden notebooks.

---

## Highlights

- **End-to-end**: data → training (single/k-fold) → evaluation → reports
- **Modern PyTorch**: simple model registry, schedulers, clean Trainer
- **Config-first**: YAML configs & grid expansion (`configs/`)
- **Explainability**: SHAP channel-importance summaries
- **Artifacts**: PR/ROC/confusion plots, per-fold reports, fold-level summary CSVs
- **Tested**: extensive pytest suite (unit + behavioral), CI-friendly
- **Fast demo**: ships with tiny sample ECGs to run immediately

---

## Repo Structure (trimmed)
```
ecg_cnn_pytorch/
├── configs/        # Baseline & grid configs (YAML)
├── data/
│   ├── sample/     # Tiny CSV sample for quick runs
│   └── ptbxl/      # (Optional) PTB-XL mirror
├── demos/          # Streamlit demo
├── ecg_cnn/        # Core package
│   ├── config/     # Config loader
│   ├── data/       # Dataset & utilities
│   ├── models/     # Model registry & helpers
│   ├── training/   # Trainer + CLI args + utils
│   ├── utils/      # Plotting, validation, grid utils
│   ├── evaluate.py # Evaluation CLI
│   └── train.py    # Training CLI
├── outputs/        # Default artifacts (ignored in git)
├── tests/          # Pytest suite
├── environment.yml # Conda env (recommended)
└── README.md
```

Additional `outputs_*` folders contain precomputed artifacts to illustrate expected results.

---

## Setup

### 1) Clone the repository

Using HTTPS:
```bash
git clone https://github.com/shaolinpat/ecg_cnn_pytorch.git
cd ecg_cnn_pytorch
```

Using SSH:
```bash
git clone git@github.com:shaolinpat/ecg_cnn_pytorch.git
cd ecg_cnn_pytorch
```


### 2) Environment (Conda recommended)

```bash
conda env create -f environment.yml
conda activate ecg_cnn_env
```

*Pip fallback:*

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart (no downloads)

The repo includes a **tiny sample dataset** under `data/sample/` so you can exercise the full pipeline in seconds.

```bash
# Train a baseline on the sample data
python -m ecg_cnn.train --config configs/baseline.yaml --sample-only
```

Artifacts land under `outputs/`:
- `outputs/results/` — normalized config + run summaries
- `outputs/models/` — `model_best_*_fold*.pth`
- `outputs/history/` — `history_*_fold*.json`
- `outputs/plots/` — accuracy/loss, ROC/PR, confusion matrices, SHAP
- `outputs/reports/` — per-fold classification reports + aggregated fold summary

---

## Full PTB-XL (optional)

Download and stage PTB-XL via the helper script (PhysioNet account & license acceptance required).

```bash
python scripts/fetch_ptbxl.py
```

Then train (omit `--sample-only`), optionally using a grid:

```bash
python -m ecg_cnn.train --config configs/grid.yaml
# or a compact grid:
python -m ecg_cnn.train --config configs/compact_grid.yaml
```

---

## Evaluation CLI

Evaluate and generate reports + plots:

```bash
python -m ecg_cnn.evaluate --fold 1 --prefer latest
```

Outputs appear in `outputs/plots/` and `outputs/reports/`.

---

## Tests

Run the full test suite with coverage and generate a coverage report:

```bash
pytest tests --cov=ecg_cnn --cov-report=term-missing --cov-branch --cov-report=html
```

View the coverage report in a browser (pick by operating system).

Linux:
```bash
xdg-open htmlcov/index.html
```

MacOs:
```bash
open htmlcov/index.html
```

Windows:
```bash
start htmlcov/index.html
```

---

## Demo

An interactive demo is provided under [`demos/run_streamlit_ecg_demo.py`](demos/run_streamlit_ecg_demo.py).  
It uses [Streamlit](https://streamlit.io/) to provide a simple ECG classification interface.

To run the demo locally:

```bash
# From the repo root
python -m streamlit run demos/run_streamlit_ecg_demo.py
```


---

## License

MIT License (see `LICENSE`).
