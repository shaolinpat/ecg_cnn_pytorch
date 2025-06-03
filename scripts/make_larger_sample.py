 
import pandas as pd
import numpy as np
import json
import os

SEED = 22

# 1) Load the full PTB-XL metadata (adjust the path if needed):
meta = pd.read_csv(
    "./data/ptbxl/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv",
    index_col="ecg_id"
)

# 2) Define raw-five mapping (must match your utils.raw_to_five_class):
RAW2SUPER = {
    'NORM': 'NORM',
    'IMI':   'MI',
    'AMI':   'MI',
    'ASMI':  'MI',
    'ILMI':  'MI',
    'ALMI':  'MI',
    'INJAS': 'MI',
    'LVH':   'HYP',
    'NDT':   'HYP',
    'LAFB':  'HYP',
    'RVH':   'HYP',
    'VCLVH': 'HYP',
    'HVOLT': 'HYP',
    'LVOLT': 'HYP',
    'CD':    'CD',
    'AFLT':  'CD',
    'AFIB':  'CD',
    'ABQRS': 'CD',
    'LAD':   'CD',
    'LAFB':  'CD',
    'RAD':   'CD',
    'LBBB':  'CD',
    'RBBB':  'CD',
    'CLBBB': 'CD',
    'CRBBB': 'CD',
    'IRBBB': 'CD',
    'ILBBB': 'CD',
    'IVCD':  'CD',
    'PACE':  'CD',
    'PVC':   'CD',
    '1AVB':  'CD',
    'PSVT':  'CD',
    'SVTAC': 'CD',
    'SVARR': 'CD',
    'TRIGU': 'CD',
    'WPW':   'CD',
    'STTC':  'STTC',
    'NST_':  'STTC',
    'ISCAL': 'STTC',
    'ISC_':  'STTC',
    'STD_':  'STTC',
    'STE_':  'STTC',
    'STACH': 'STTC',
    # anything else → Unknown
}

def raw_to_five(scp_str):
    """Return one of ['CD','HYP','MI','NORM','STTC'] or 'Unknown'."""
    d = json.loads(scp_str.replace("'", '"'))
    top = max(d, key=lambda k: d[k])
    return RAW2SUPER.get(top, "Unknown")

# 3) Add a 'five_class' column to meta
meta["five_class"] = meta["scp_codes"].apply(raw_to_five)

# 4) Remove any "Unknown" rows
meta = meta[meta["five_class"] != "Unknown"]

# 5) For reproducibility
np.random.seed(SEED)

# 6) For each of the five valid classes, sample up to N examples
samples_per_class = 1000
sampled_ids = []

for cls in ["CD", "HYP", "MI", "NORM", "STTC"]:
    cls_indices = meta[meta["five_class"] == cls].index.to_list()
    if len(cls_indices) < samples_per_class:
        raise ValueError(f"Not enough examples of class {cls} to sample {samples_per_class}")
    chosen = np.random.choice(cls_indices, samples_per_class, replace=False)
    sampled_ids.extend(chosen.tolist())

# 7) Shuffle the sampled IDs
sampled_ids = np.array(sampled_ids)
np.random.shuffle(sampled_ids)

# 8) Save the list of ECG IDs to CSV so your 'load_ptbxl_sample' can see them
out_dir = "./data/larger_sample"
os.makedirs(out_dir, exist_ok=True)

# Save a single‐column CSV named "sample_ids.csv" with column header "ecg_id"
pd.DataFrame({"ecg_id": sampled_ids}).to_csv(
    os.path.join(out_dir, "sample_ids.csv"),
    index=False
)

print(f"Saved {len(sampled_ids)} IDs to {out_dir}/sample_ids.csv")
