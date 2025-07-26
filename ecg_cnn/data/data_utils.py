################################################################################
#
# PTB-XL loading and metadata helpers
#
################################################################################


import ast
import json
import numpy as np
import os
import pandas as pd
import wfdb
import torch

from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset

SEED = 22

# The five target superclasses:
FIVE_SUPERCLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]

# Map each frequent raw code into one of the five.
# Note: these would need to be confirmed by subject matter experts

RAW2SUPER = {
    # ---------- ORIGINAL MAPPINGS (unchanged) ----------
    # NORM is already normal
    "NORM": "NORM",  # normal sinus rhythm
    # Myocardial infarction / infarct-related codes -> MI
    "MI": "MI",  # generic myocardial infarction
    "IMI": "MI",  # inferior MI
    "AMI": "MI",  # acute MI
    "ASMI": "MI",  # anteroseptal MI
    "ILMI": "MI",  # old inferior MI
    "ALMI": "MI",  # old lateral MI
    "INJAS": "MI",  # injury, anteroseptal
    # Hypertrophy codes -> HYP
    "HYP": "HYP",  # general hypertrophy label
    "LVH": "HYP",  # left ventricular hypertrophy
    "NDT": "HYP",  # non-diagnostic T-wave, often grouped under voltage criteria
    "LAFB": "HYP",  # left anterior fascicular block (voltage/hypertrophy context)
    "RVH": "HYP",  # right ventricular hypertrophy
    "VCLVH": "HYP",  # variant of left ventricular hypertrophy
    "HVOLT": "HYP",  # high voltage QRS complexes, suggestive of hypertrophy
    "LVOLT": "HYP",  # left ventricular high voltage — subtype of high voltage for LVH
    # Conduction disturbances ("CD") -> CD
    "CD": "CD",  # generic conduction disturbance
    "IRBBB": "CD",  # incomplete right bundle branch block
    "CLBBB": "CD",  # complete left bundle branch block
    "CRBBB": "CD",  # complete right bundle branch block
    "IVCD": "CD",  # interventricular conduction delay
    "PACE": "CD",  # paced beats
    "PVC": "CD",  # premature ventricular contraction
    "1AVB": "CD",  # first-degree AV block
    # ST/T-wave changes -> STTC
    "STTC": "STTC",  # general label for ST-T wave abnormalities
    "NST_": "STTC",  # non-specific T-wave changes
    "ISCAL": "STTC",  # ischemia, lateral
    "ISC_": "STTC",  # generic ischemia code
    # ---------- NEW MAPPINGS (to cover all PTB-XL codes) ----------
    # Conduction Disturbances ("CD")
    "2AVB": "CD",  # second-degree AV block
    "3AVB": "CD",  # third-degree AV block
    "ABQRS": "CD",  # abnormal QRS morphology
    "ILBBB": "CD",  # incomplete left bundle branch block
    "LPFB": "CD",  # left posterior fascicular block
    "LPR": "CD",  # left posterior hemiblock / left posterior division block
    "PRC(S)": "CD",  # PR complex short
    "RVH": "CD",  # right ventricular hypertrophy (also conduction)
    "WPW": "CD",  # Wolf-Parkinson-White syndrome
    "ABQRS": "CD",  # abnormal QRS complex
    # Hypertrophy ("HYP")
    "RAO/LAE": "HYP",  # right atrial or left atrial enlargement
    "SEHYP": "HYP",  # secondary hypertension signs
    "LAFB": "HYP",  # left anterior fascicular block
    "VCLVH": "HYP",  # variant LVH
    "HVOLT": "HYP",  # high voltage
    "LVOLT": "HYP",  # left ventricular voltage
    "LPFB": "HYP",  # left posterior fascicular block (voltage context)
    # Myocardial Infarction ("MI")
    "IPLMI": "MI",  # inferior-posterior left MI
    "IPMI": "MI",  # inferior posterior MI
    "LMI": "MI",  # lateral MI
    "PMI": "MI",  # posterior MI
    "QWAVE": "MI",  # Q waves of infarction
    "ANEUR": "MI",  # aneurysm
    "INJAL": "MI",  # injury, anterolateral
    "INJIL": "MI",  # injury, inferolateral
    "INJIN": "MI",  # injury, inferior
    "INJLA": "MI",  # injury, lateral-anterior
    # Normal Rhythms ("NORM")
    "SR": "NORM",  # sinus rhythm
    "SBRAD": "NORM",  # sinus bradycardia
    "STACH": "NORM",  # sinus tachycardia
    "SARRH": "NORM",  # sinus arrhythmia
    # ST-T Changes & Arrhythmias ("STTC")
    "AFIB": "STTC",  # atrial fibrillation
    "AFLT": "STTC",  # atrial flutter
    "BIGU": "STTC",  # bigeminy
    "DIG": "STTC",  # digitalis effect
    "EL": "STTC",  # electrolyte abnormality
    "INJAL": "STTC",  # injury, anterolateral (also MI, but grouped here if desired)
    "INJAS": "STTC",  # injury, anteroseptal (already in original)
    "INJIL": "STTC",  # injury, inferolateral
    "INJIN": "STTC",  # injury, inferior
    "INJLA": "STTC",  # injury, lateral-anterior
    "INVT": "STTC",  # inverted T waves
    "ISCAL": "STTC",  # ischemia, lateral (already in original)
    "ISCAN": "STTC",  # ischemia, anterior
    "ISCAS": "STTC",  # ischemia, anterior-septal
    "ISCIL": "STTC",  # ischemia, inferolateral
    "ISCIN": "STTC",  # ischemia, inferior
    "ISCLA": "STTC",  # ischemia, lateral-anterior
    "ISC_": "STTC",  # generic ischemia (already in original)
    "LNGQT": "STTC",  # long QT interval
    "LOWT": "STTC",  # low T-wave amplitude
    "PAC": "STTC",  # premature atrial contraction
    "PACE": "STTC",  # paced rhythm (also CD, but can be STTC)
    "PSVT": "STTC",  # paroxysmal supraventricular tachycardia
    "PVC": "STTC",  # premature ventricular contraction (also CD)
    "QWAVE": "STTC",  # Q-wave of MI (also MI, optional)
    "SVARR": "STTC",  # supraventricular arrhythmia
    "SVTAC": "STTC",  # supraventricular tachycardia acute
    "STD_": "STTC",  # ST depression
    "STE_": "STTC",  # ST elevation
    "TAB_": "STTC",  # T-wave abnormality
    "TRIGU": "STTC",  # trigeminy
    "WPW": "STTC",  # Wolf-Parkinson-White (also conduction)
    # Any raw label not explicitly listed here continues to be dropped (Unknown)
}

# Map each super-label to an integer index 0..4
LABEL2IDX = {"CD": 0, "HYP": 1, "MI": 2, "NORM": 3, "STTC": 4}


def build_full_X_y(meta_csv, scp_csv, ptb_path):
    """
    Loads PTB-XL records and converts them into:
      - X_all: NumPy array of shape (N, 12, 1000) (first 1000 samples per record)
      - y_all: Integer label array of shape (N,) with values 0..4
      - meta_kept: DataFrame containing metadata for the retained records

    Skips records that:
      - Cannot be mapped to a super-label
      - Are unreadable from disk
      - Are missing required metadata fields

    Parameters
    ----------
    meta_csv : str
        Path to PTB-XL metadata CSV (must contain 'ecg_id', 'filename_lr', and 'scp_codes').
    scp_csv : str
        Path to SCP-statements CSV. [Currently unused]
    ptb_path : str
        Root directory of PhysioNet PTB-XL data (e.g. 'data/ptbxl/.../1.0.3').

    Returns
    -------
    X_all : np.ndarray
        ECG signal array of shape (N, 12, 1000)
    y_all : np.ndarray
        Integer label array of shape (N,)
    meta_kept : pd.DataFrame
        Filtered metadata for kept records

    Raises
    ------
    FileNotFoundError
        If `meta_csv` or `scp_csv` is not found
    NotADirectoryError
        If `ptb_path` isn't a directory

    """

    # Normalize path input
    ptb_path = Path(ptb_path)

    # -----------------------------
    # Input validation
    # -----------------------------
    if not os.path.isfile(meta_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {meta_csv}")

    if not os.path.isfile(scp_csv):
        raise FileNotFoundError(f"SCP statements CSV not found: {scp_csv}")

    if not ptb_path.is_dir():
        raise NotADirectoryError(f"PTB-XL root directory not found: {ptb_path}")

    # -----------------------------
    # Load metadata
    # -----------------------------
    sample_meta = pd.read_csv(
        meta_csv, index_col="ecg_id", converters={"scp_codes": ast.literal_eval}
    )
    scp_df = pd.read_csv(scp_csv, index_col=0)  # [NOTE: currently unused]

    X_list, y_list, keep_meta = [], [], []

    for ecg_id, row in sample_meta.iterrows():
        rec_path = row.get("filename_lr")
        if not isinstance(rec_path, str) or not rec_path.strip():
            print(f"Skipping {ecg_id}: filename_lr is missing or invalid")
            continue  # skip if path is missing or empty

        raw_list = row.get("scp_codes", [])
        super_labels = [RAW2SUPER[r] for r in raw_list if r in RAW2SUPER]

        if not super_labels:
            print(f"Skipping {ecg_id}: no valid super labels in {raw_list}")
            continue  # skip if no valid labels

        y_str = super_labels[0]
        y_idx = LABEL2IDX[y_str]

        full_path = Path(ptb_path) / str(rec_path)

        print(f"Trying record {ecg_id} at {full_path}")

        try:
            signal, _ = wfdb.rdsamp(str(full_path))
            sig = signal[:1000, :].T  # shape (12, 1000)
        except Exception as e:
            print("wfdb.rdsamp() failed on", full_path)
            print("Exception:", repr(e))
            continue  # skip unreadable record

        print(f"Read success: {full_path}")

        X_list.append(sig)
        y_list.append(y_idx)
        keep_meta.append(row)

    if not X_list:
        raise ValueError("No valid records were loaded.")

    X_all = np.stack(X_list, axis=0)
    y_all = np.array(y_list, dtype=np.int64)
    meta_kept = pd.DataFrame(keep_meta)

    return X_all, y_all, meta_kept


def select_primary_label(label_list):
    """
    Select a single diagnostic superclass from a list of possible labels.

    Given a list of diagnostic superclass labels (e.g., ["MI", "NORM"]),
    this function returns the first known class found according to a fixed
    priority defined in FIVE_SUPERCLASSES. If no recognized label is found,
    it returns "Unknown".

    Parameters
    ----------
    label_list : list or set of str
        A collection of diagnostic superclass labels. Typical inputs are the
        outputs from `aggregate_diagnostic()`.

    Returns
    -------
    str
        A single superclass label (e.g., "MI", "NORM") if a known class is found,
        or "Unknown" if the input is empty or contains no recognized classes.

    Raises
    ------
    TypeError
        If `label_list` is not a list, set, or tuple of strings.
    """

    if not isinstance(label_list, (list, set, tuple)):
        raise TypeError(
            f"label_list must be a list, set, or tuple, got {type(label_list)}"
        )

    if not all(isinstance(label, str) for label in label_list):
        raise TypeError("All elements in label_list must be strings.")

    for cls in FIVE_SUPERCLASSES:
        if cls in label_list:
            return cls
    return "Unknown"


def aggregate_diagnostic(codes, agg_df, restrict=True):
    """
    Map raw SCP codes to a list of diagnostic superclass labels.

    Parameters
    ----------
    codes : dict
        A dictionary of SCP codes and their corresponding scores or weights.

    agg_df : pd.DataFrame
        DataFrame containing SCP code metadata with columns:
        - 'diagnostic' (1 for diagnostic codes)
        - 'diagnostic_class' (e.g., 'NORM', 'MI', etc.)

    restrict : bool, optional (default=True)
        If True, only return diagnostic classes that are in FIVE_SUPERCLASSES.
        If False, return all diagnostic_class values found in agg_df.

    Returns
    -------
    list of str
        A list of diagnostic superclass labels. Returns ['Unknown'] if no valid
        diagnostic class is found.

    Raises
    ------
    ValueError
        If `codes` is not a dictionary
        If `agg_df` is not a pandas DataFrame
    """

    if not isinstance(codes, dict):
        raise ValueError(f"codes must be a dictionary, got {type(codes)}")

    if not isinstance(agg_df, pd.DataFrame):
        raise ValueError(f"agg_df must be a pandas DataFrame, got {type(agg_df)}")

    out = []
    for c in codes:
        if c in agg_df.index:
            row = agg_df.loc[c]
            if isinstance(row, pd.Series) and row.get("diagnostic", 0) == 1:
                cls = row.get("diagnostic_class")
                if not restrict or cls in FIVE_SUPERCLASSES:
                    out.append(cls)

    return list(set(out)) if out else ["Unknown"]


def load_ptbxl_meta(ptb_path):
    """
    Load PTB-XL metadata and add a diagnostic_superclass column.

    Parameters
    ----------
    ptb_path : str
        Root directory of PhysioNet PTB-XL data (e.g. 'data/ptbxl/.../1.0.3').

    Returns
    -------
    df
        A DataFrame created from 'ptbxl_database.csv' together with the column
        'diagnostic_superclass' appended.

    Raises
    ------
    NotADirectoryError
        If `ptb_path` is not a directory
    """

    # Normalize path input
    ptb_path = Path(ptb_path)

    # -----------------------------
    # Input validation
    # -----------------------------
    if not os.path.isdir(ptb_path):
        raise NotADirectoryError(f"Input not a directory: {ptb_path}")

    # -----------------------------
    # Build the DataFrame
    # -----------------------------
    meta_csv = os.path.join(ptb_path, "ptbxl_database.csv")
    df = pd.read_csv(
        meta_csv, index_col="ecg_id", converters={"scp_codes": ast.literal_eval}
    )

    scp_csv = os.path.join(ptb_path, "scp_statements.csv")
    agg_df = pd.read_csv(scp_csv, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    df["diagnostic_superclass"] = df.scp_codes.apply(
        lambda codes: aggregate_diagnostic(codes, agg_df)
    )

    return df


def load_ptbxl_sample(sample_dir, ptb_path):
    """
    Load a sample of PTB-XL ECG signals based on either a CSV file of IDs or individual ECG files.

    This function supports two use cases:
    1. If `sample_dir` contains a file named "sample_ids.csv", it is expected to have a single column
       named "ecg_id", and the function loads only those ECGs.
    2. Otherwise, `sample_dir` is assumed to contain multiple CSV files, each named like "<ecg_id>_*.csv".
       In this case, all ECGs with matching filenames are loaded.

    Parameters
    ----------
    sample_dir : str or Path
        Path to the directory containing either "sample_ids.csv" or individual ECG CSV files.

    ptb_path : str or Path
        Root directory of the PTB-XL dataset, used to locate the full metadata.

    Returns
    -------
    X : np.ndarray
        ECG signal data of shape (N, 12, T), where N is the number of samples and T is the number of time steps.

    y : list of str
        Diagnostic superclass labels (e.g., "NORM", "MI", etc.), one per sample.

    sample_meta : pd.DataFrame
        Metadata for the loaded samples, including at minimum the "ecg_id" and "diagnostic_superclass" columns.

    Raises
    ------
    NotADirectoryError
        If `sample_dir` is not a directory
        If `ptb_path` is not a directory

    Notes
    -----
    - Assumes that `ptb_path` contains the files "ptbxl_database.csv" and "scp_statements.csv".
    - Assumes that ECG signal files are in CSV format with 12 columns (one per lead).
    """

    # Normalize path inputs
    sample_dir = Path(sample_dir)
    ptb_path = Path(ptb_path)

    # -----------------------------
    # Input validation
    # -----------------------------
    if not sample_dir.is_dir():
        raise NotADirectoryError(f"Input not a directory: {sample_dir}")

    if not ptb_path.is_dir():
        raise NotADirectoryError(f"Input not a directory: {ptb_path}")

    # 1) Check if there's a "sample_ids.csv" in sample_dir:
    list_of_files = os.listdir(sample_dir)
    if "sample_ids.csv" in list_of_files:
        # load IDs from single sample_ids.csv
        ids_df = pd.read_csv(os.path.join(sample_dir, "sample_ids.csv"))
        ecg_ids = ids_df["ecg_id"].tolist()
        print("Sample IDs loaded:", ecg_ids[:5], "... total:", len(ecg_ids))
    else:
        # assume each file is "<ecg_id>_<whatever>.csv"
        ecg_ids = []
        for fname in list_of_files:
            if not fname.lower().endswith(".csv"):
                continue
            # take everything before the first '_' as the ecg_id
            # (e.g. "00017_lr.csv", or "17.csv")
            try:
                ecg_id = int(fname.split("_", 1)[0])
            except ValueError:
                # if filename doesn’t start with an integer, skip
                continue
            ecg_ids.append(ecg_id)

    # 2) Grab the full metadata so we can look up each ECGs scp_codes/filename
    full_meta_df = load_ptbxl_meta(ptb_path)  # this df is indexed by ecg_id

    # 3) Subset to only those ecg_ids
    sample_meta = full_meta_df.loc[ecg_ids].copy()

    X_list = []
    y_list = []
    for ecg_id, row in sample_meta.iterrows():
        # Use the 100 Hz signal filename
        rec_path = row["filename_lr"]
        full_path = os.path.join(ptb_path, rec_path)

        # Read the WFDB record
        signal, _ = wfdb.rdsamp(full_path)
        X_list.append(signal.T)

        # Derive the five-class label from scp_codes (must match your RAW2SUPER mapping)
        y_lbl = raw_to_five_class(row["scp_codes"])
        y_list.append(y_lbl)

    X = np.stack(X_list, axis=0)  # shape = (N, 12, T)
    y = y_list  # list of length N (strings, possibly "Unknown")

    return X, y, sample_meta


def load_ptbxl_full(data_dir, subsample_frac, sampling_rate=100):
    """
    Load all PTB-XL ECG records from the given directory, with optional subsampling.

    This function loads signal data for all ECG records listed in the PTB-XL metadata
    file located in `data_dir`. You may subsample the data for quick experimentation.

    Parameters
    ----------
    data_dir : str or Path
        Root directory of the PTB-XL dataset. Must contain `ptbxl_database.csv` and
        the waveform data under "records100" or "records500".

    subsample_frac : float
        Fraction (0.0 to 1.0] of records to load. If less than 1.0, a random subset
        of the records is selected using a fixed global seed.

    sampling_rate : int, optional (default=100)
        Sampling rate to use. Must be either 100 or 500. Determines which file column
        to read in the metadata: `filename_lr` (100 Hz) or `filename_hr` (500 Hz).

    Returns
    -------
    X : np.ndarray
        ECG signal data of shape (M, 12, T), where M is the number of loaded records
        and T is the number of time steps (varies by sampling rate).

    y : list of str
        Diagnostic superclass labels for each record, mapped via `raw_to_five_class()`.

    full_meta : pd.DataFrame
        Subset of the PTB-XL metadata corresponding to the loaded ECGs (length M).

    Raises
    ------
    ValueError
        If `subsample_frac` is not in (0.0, 1.0], or `sampling_rate` is not 100 or 500.

    FileNotFoundError
        If required waveform files or metadata are missing.

    Notes
    -----
    - The metadata is loaded via `load_ptbxl_meta(data_dir)`, and must contain the proper
      columns (`filename_lr` or `filename_hr`, and `scp_codes`).
    - Metadata is loaded via `load_ptbxl_meta()` and filtered to match valid ECGs.
    - Reproducible subsampling is done using SEED = 22 before loading files.
    - Label mapping is handled by an external helper: `raw_to_five_class()`.
    - Reproducible subsampling is achieved using global SEED = 22.
    """
    SEED = 22

    # Normalize and validate inputs
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Invalid data directory: {data_dir}")

    if not (0.0 < subsample_frac <= 1.0):
        raise ValueError(f"subsample_frac must be in (0.0, 1.0], got {subsample_frac}")

    if sampling_rate not in (100, 500):
        raise ValueError(f"sampling_rate must be 100 or 500, got {sampling_rate}")

    meta_df = load_ptbxl_meta(data_dir)
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    filenames = meta_df[col].tolist()
    ecg_ids = meta_df.index.tolist()

    print(f"subsample_frac: {subsample_frac}")
    if subsample_frac < 1.0:
        np.random.seed(SEED)
        idx = np.random.choice(
            len(ecg_ids), int(len(ecg_ids) * subsample_frac), replace=False
        )
        ecg_ids = [ecg_ids[i] for i in idx]
        filenames = [filenames[i] for i in idx]

    ids = []
    X_list = []

    for ecg_id, rec in zip(ecg_ids, filenames):
        full_path = data_dir / rec

        try:
            signal, _ = wfdb.rdsamp(str(full_path))
        except Exception as e:
            print(f"!!! FAILED on {full_path}: {e}")
            continue

        X_list.append(signal.T)
        ids.append(ecg_id)

    if not X_list:
        raise ValueError("No valid records were loaded.")

    X = np.stack(X_list, axis=0)
    full_meta = meta_df.loc[ids].copy()

    # Derive y from raw scp_codes via your mapping helper
    y = [raw_to_five_class(s) for s in full_meta["scp_codes"]]

    print(f"Loaded {len(ids)} records after subsampling.")
    return X, y, full_meta


def raw_to_five_class(scp_entry) -> str:
    """
    Map the dominant raw SCP label to one of five high-level classes.

    Parameters
    ----------
    scp_entry : str or dict
        A dictionary of SCP codes and their associated weights (e.g., {'MI': 1.0}),
        or a string representation of such a dict
        (e.g., "{'MI': 1.0, 'SR': 0.0}").

    Returns
    -------
    str
        One of the five superclasses: ['CD', 'HYP', 'MI', 'NORM', 'STTC'],
        or "Unknown" if the input cannot be parsed or mapped.

    Raises
    ------
    No raised exceptions

    Notes
    -----
    - If `scp_entry` is a string, it is parsed using `ast.literal_eval`.
    - If parsing fails or if the input is neither a string nor a dict,
      returns "Unknown".
    - If the resulting dict is empty or its top code is unmapped in RAW2SUPER,
      returns "Unknown".
    """
    if isinstance(scp_entry, str):
        try:
            d = ast.literal_eval(scp_entry)  # safely parse Python-style string
        except (ValueError, SyntaxError):
            return "Unknown"
    elif isinstance(scp_entry, dict):
        d = scp_entry
    else:
        return "Unknown"

    if not d:
        return "Unknown"

    top_key = max(d, key=d.get)
    return RAW2SUPER.get(top_key, "Unknown")
