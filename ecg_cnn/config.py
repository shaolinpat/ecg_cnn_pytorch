# ecg_cnn/config.py

from pathlib import Path

# Get the project root no matter where config.py is imported from
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directory (outside ecg_cnn/)
PTBXL_DATA_DIR = (
    PROJECT_ROOT / "data" / "ptbxl" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
)

# Output folders
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_CSV = OUTPUT_DIR / "results" / "results_summary_pytorch.csv"
