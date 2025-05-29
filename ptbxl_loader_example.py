import ast
import os
import pandas as pd
import numpy as np
import wfdb

ptb_path = "./data/ptbxl/physionet.org/files/ptb-xl/1.0.3"
sample_path = './data/sample/'

# Load metadata with scp_codes as the ground truth
meta_csv = os.path.join(ptb_path, "ptbxl_database.csv")
meta_df = pd.read_csv(meta_csv, index_col='ecg_id')
meta_df.scp_codes = meta_df.scp_codes.apply(lambda x: ast.literal_eval(x))

# Select codes as diagnostic superclass
scp_csv = os.path.join(ptb_path, "scp_statements.csv")

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(scp_csv, index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
meta_df['diagnostic_superclass'] = meta_df.scp_codes.apply(aggregate_diagnostic)

files = [f for f in os.listdir(sample_path) if f.endswith(".csv")]

X_samples = []
sample_ids = []

for file in files:
    ecg_id = int(file.split("_")[0])
    sample_ids.append(ecg_id)

    arr = pd.read_csv(os.path.join(sample_path, file), header=None).values
    X_samples.append(arr.T)

sample_meta = meta_df.loc[sample_ids].copy()
X = np.stack(X_samples, axis=0)
y = np.array(sample_meta.diagnostic_superclass.tolist(), dtype=object)

# Split data into train and test
test_fold = 10
mask = sample_meta.strat_fold != test_fold

# Train
X_train = X[mask]
y_train = y[mask].tolist()

# Test
X_test  = X[~mask]
y_test  = y[~mask].tolist()

print(X_train.shape)
print(y_train[:5])
print(X_test.shape)
print(y_test[:5])
