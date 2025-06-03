################################################################################
#
# PTB-XL loading and metadata helpers
#
################################################################################

import ast
import numpy as np
import os
import pandas as pd
import wfdb 

SEED = 22

# The five target superclasses:
FIVE_SUPERCLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# Map each frequent raw code into one of the five.
# Note: these would need to be confirmed by subject matter experts

RAW2SUPER = {
    # ---------- ORIGINAL MAPPINGS (unchanged) ----------
    # NORM is already normal
    "NORM": "NORM",

    # Myocardial infarction / infarct-related codes -> MI
    "IMI":   "MI",   # inferior MI
    "AMI":   "MI",   # acute MI
    "ASMI":  "MI",   # anteroseptal MI
    "ILMI":  "MI",   # old inferior MI
    "ALMI":  "MI",   # old lateral MI
    "INJAS": "MI",   # injury, anteroseptal

    # Hypertrophy codes -> HYP
    "LVH":   "HYP",  # left ventricular hypertrophy
    "NDT":   "HYP",  # non-diagnostic T-wave, often grouped under voltage criteria
    "LAFB":  "HYP",  # left anterior fascicular block (voltage/hypertrophy context)
    "RVH":   "HYP",
    "VCLVH": "HYP",
    "HVOLT": "HYP",
    "LVOLT": "HYP",

    # Conduction disturbances ("CD") -> CD
    "IRBBB": "CD",   # incomplete right bundle branch block
    "CLBBB": "CD",   # complete left bundle branch block
    "CRBBB": "CD",   # complete right bundle branch block
    "IVCD":  "CD",   # interventricular conduction delay
    "PACE":  "CD",   # paced beats
    "PVC":   "CD",   # premature ventricular contraction
    "1AVB":  "CD",   # first-degree AV block

    # ST/T-wave changes -> STTC
    "NST_":  "STTC", # non-specific T-wave changes
    "ISCAL": "STTC", # ischemia, lateral
    "ISC_":  "STTC", # generic ischemia code

    # ---------- NEW MAPPINGS (to cover all PTB-XL codes) ----------
    # Conduction Disturbances ("CD")
    "2AVB":       "CD",   # second-degree AV block
    "3AVB":       "CD",   # third-degree AV block
    "ABQRS":      "CD",   # abnormal QRS morphology
    "ILBBB":      "CD",   # incomplete left bundle branch block
    "LPFB":       "CD",   # left posterior fascicular block
    "LPR":        "CD",   # left posterior hemiblock / left posterior division block
    "PRC(S)":     "CD",   # PR complex short
    "RVH":        "CD",   # right ventricular hypertrophy (also conduction)
    "WPW":        "CD",   # Wolf-Parkinson-White syndrome
    "ABQRS":      "CD",   # abnormal QRS complex

    # Hypertrophy ("HYP")
    "RAO/LAE":    "HYP",  # right atrial or left atrial enlargement
    "SEHYP":      "HYP",  # secondary hypertension signs
    "LAFB":       "HYP",  # left anterior fascicular block
    "VCLVH":      "HYP",  # variant LVH
    "HVOLT":      "HYP",  # high voltage
    "LVOLT":      "HYP",  # left ventricular voltage
    "LPFB":       "HYP",  # left posterior fascicular block (voltage context)

    # Myocardial Infarction ("MI")
    "IPLMI":      "MI",   # inferior-posterior left MI
    "IPMI":       "MI",   # inferior posterior MI
    "LMI":        "MI",   # lateral MI
    "PMI":        "MI",   # posterior MI
    "QWAVE":      "MI",   # Q waves of infarction
    "ANEUR":      "MI",   # aneurysm
    "INJAL":      "MI",   # injury, anterolateral
    "INJIL":      "MI",   # injury, inferolateral
    "INJIN":      "MI",   # injury, inferior
    "INJLA":      "MI",   # injury, lateral-anterior

    # Normal Rhythms ("NORM")
    "SR":         "NORM", # sinus rhythm
    "SBRAD":      "NORM", # sinus bradycardia
    "STACH":      "NORM", # sinus tachycardia
    "SARRH":      "NORM", # sinus arrhythmia

    # ST-T Changes & Arrhythmias ("STTC")
    "AFIB":       "STTC", # atrial fibrillation
    "AFLT":       "STTC", # atrial flutter
    "BIGU":       "STTC", # bigeminy
    "DIG":        "STTC", # digitalis effect
    "EL":         "STTC", # electrolyte abnormality
    "INJAL":      "STTC", # injury, anterolateral (also MI, but grouped here if desired)
    "INJAS":      "STTC", # injury, anteroseptal (already in original)
    "INJIL":      "STTC", # injury, inferolateral
    "INJIN":      "STTC", # injury, inferior
    "INJLA":      "STTC", # injury, lateral-anterior
    "INVT":       "STTC", # inverted T waves
    "ISCAL":      "STTC", # ischemia, lateral (already in original)
    "ISCAN":      "STTC", # ischemia, anterior
    "ISCAS":      "STTC", # ischemia, anterior-septal
    "ISCIL":      "STTC", # ischemia, inferolateral
    "ISCIN":      "STTC", # ischemia, inferior
    "ISCLA":      "STTC", # ischemia, lateral-anterior
    "ISC_":       "STTC", # generic ischemia (already in original)
    "LNGQT":      "STTC", # long QT interval
    "LOWT":       "STTC", # low T-wave amplitude
    "PAC":        "STTC", # premature atrial contraction
    "PACE":       "STTC", # paced rhythm (also CD, but can be STTC)
    "PSVT":       "STTC", # paroxysmal supraventricular tachycardia
    "PVC":        "STTC", # premature ventricular contraction (also CD)
    "QWAVE":      "STTC", # Q-wave of MI (also MI, optional)
    "SVARR":      "STTC", # supraventricular arrhythmia
    "SVTAC":      "STTC", # supraventricular tachycardia acute
    "STD_":       "STTC", # ST depression
    "STE_":       "STTC", # ST elevation
    "TAB_":       "STTC", # T-wave abnormality
    "TRIGU":      "STTC", # trigeminy
    "WPW":        "STTC", # Wolf-Parkinson-White (also conduction)

    # Any raw label not explicitly listed here continues to be dropped (Unknown)
}



# RAW2SUPER = {
#     # NORM is already normal
#     'NORM': 'NORM',

#     # Myocardial infarction / infarct-related codes -> MI   
#     'IMI':   'MI',   # inferior MI
#     'AMI':   'MI',   # acute MI
#     'ASMI':  'MI',   # anteroseptal MI
#     'ILMI':  'MI',   # old inferior MI
#     'ALMI':  'MI',   # old lateral MI
#     'INJAS': 'MI',   # injury, anteroseptal

#     # Hypertrophy codes -> HYP
#     'LVH':   'HYP',  # left ventricular hypertrophy
#     'NDT':   'HYP',  # non-diagnostic T-wave, often grouped under voltage criteria
#     'LAFB':  'HYP',  # left anterior fascicular block (voltage/hypertrophy context)
#     'RVH':   'HYP',
#     'VCLVH': 'HYP',
#     'HVOLT': 'HYP',
#     'LVOLT': 'HYP',

#     # Conduction disturbances ("CD") -> CD
#     'IRBBB': 'CD',   # incomplete right bundle branch block
#     'CLBBB': 'CD',   # complete left bundle branch block
#     'CRBBB': 'CD',   # complete right bundle branch block
#     'IVCD':  'CD',   # interventricular conduction delay
#     'PACE':  'CD',   # paced beats
#     'PVC':   'CD',   # premature ventricular contraction
#     '1AVB':  'CD',   # first-degree AV block

#     #ST/T-wave changes -> STTC
#     'NST_':  'STTC', # non-specific T-wave changes
#     'ISCAL': 'STTC', # ischemia, lateral
#     'ISC_':  'STTC', # generic ischemia code

#     # Any code not explicitly listed here is "Unknown" (and will be dropped) 
# }


def aggregate_diagnostic(codes, agg_df):
    """Map a list of SCP codes to the diagnostic_superclass list."""
    """
    Look up each raw scp_code in agg_df (where agg_df.diagnostic == 1),
    and return one of the five superclasses: 'NORM','MI','HYP','STTC','CD'.
    If none of those five apply, return 'Unknown'.
    """
    out = []
    for c in codes:
        if c in agg_df.index:
            out.append(agg_df.loc[c].diagnostic_class)
    return list(set(out))


def load_ptbxl_meta(ptb_path):
    """Load PTB-XL metadata and add a diagnostic_superclass column."""
    meta_csv = os.path.join(ptb_path, "ptbxl_database.csv")
    df = pd.read_csv(meta_csv, index_col="ecg_id", converters={"scp_codes": ast.literal_eval})

    scp_csv = os.path.join(ptb_path, "scp_statements.csv")
    agg_df  = pd.read_csv(scp_csv, index_col=0)
    agg_df  = agg_df[agg_df.diagnostic == 1]

    df["diagnostic_superclass"] = df.scp_codes.apply(lambda codes: aggregate_diagnostic(codes, agg_df))
    return df


def load_ptbxl_sample(sample_dir, ptb_path):
    """
    If sample_dir contains a file named "sample_ids.csv", we read that single 
    file (one column "ecg_id") and load exactly those IDs. Otherwise, we assume 
    sample_dir contains N individual ECG CSVs named "<ecg_id>_something.csv". 
    This supports a variety of testing scenarios.

    Returns X (Nx12xT), y (list length N), and sample_meta (DataFrame length N).
    """

    # 1) Check if there's a "sample_ids.csv" in sample_dir:
    list_of_files = os.listdir(sample_dir)
    if "sample_ids.csv" in list_of_files:
        # load IDs from single sample_ids.csv
        ids_df = pd.read_csv(os.path.join(sample_dir, "sample_ids.csv"))
        ecg_ids = ids_df["ecg_id"].tolist()

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
    y = y_list                    # list of length N (strings, possibly "Unknown")

    return X, y, sample_meta


def load_ptbxl_full(data_dir, subsample_frac, sampling_rate=100):
    """
    Loads all PTB-XL records from data_dir, optionally subsampling.
    Returns X (Mx12xT), y (list length M), and full_meta (DataFrame length M).
    """
    meta_df = load_ptbxl_meta(data_dir)

    col  = "filename_lr" if sampling_rate == 100 else "filename_hr"
    recs = meta_df[col].unique().tolist()

    if subsample_frac < 1.0:
        np.random.seed(SEED)
        recs = list(np.random.choice(recs, int(len(recs)*subsample_frac), replace=False))

    ids   = []
    X_list = []

    for rec in recs:
        full_path = os.path.join(data_dir, rec)
        signal, _ = wfdb.rdsamp(full_path)
        X_list.append(signal.T)

        parts = rec.split("/")                 # ["records100", "00000", "00017_lr"]
        filename = parts[2]                    # "00017_lr"
        ecg_id = int(filename.split("_")[0])   # -> 17
        ids.append(ecg_id)


    X         = np.stack(X_list, axis=0)
    full_meta = meta_df.loc[ids].copy()

    # Derive y from raw scp_codes via your mapping helper
    y = [raw_to_five_class(s) for s in full_meta["scp_codes"]]

    return X, y, full_meta


def raw_to_five_class(scp_entry) -> str:
    """
    Accept either a JSON‐string (e.g. "{'AFLT':100.0,'SR':0.0}") or a dict ({"AFLT":100.0,"SR":0.0}).
    Return one of ['CD','HYP','MI','NORM','STTC'], or "Unknown" if unmapped.
    """
    # 1) If it's already a dict, use it directly; otherwise parse the string
    if isinstance(scp_entry, dict):
        d = scp_entry
    else:
        # Convert single quotes - double quotes so json.loads works
        safe = scp_entry.replace("'", '"')
        d = json.loads(safe)

    # 2) Pick the key with the highest percentage
    top_key = max(d, key=lambda k: d[k])

    # 3) Map via your RAW2SUPER dictionary (imported or defined above)
    return RAW2SUPER.get(top_key, "Unknown")