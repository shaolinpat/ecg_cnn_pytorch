# ecg_cnn/paths.py

from pathlib import Path


# ------------------------------------------------------------------------------
# Canonical project paths (single source of truth)
# ------------------------------------------------------------------------------

# Get the project root (top=level repo) no matter where config.py is imported
# from
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------
# Data (PTB-XL)
# ------------------------------------------------------------------------------
# Data directory (outside ecg_cnn/)
PTBXL_DATA_DIR = (
    PROJECT_ROOT / "data" / "ptbxl" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
)

# Common CSVs (these are the usual filenames alongside the records)
PTBXL_META_CSV = PTBXL_DATA_DIR / "ptbxl_database.csv"
PTBXL_SCP_CSV = PTBXL_DATA_DIR / "scp_statements.csv"

# ------------------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
HISTORY_DIR = OUTPUT_DIR / "history"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

# Path to the default training configuration
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
