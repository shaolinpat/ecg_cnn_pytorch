# utils/find_labels.py

import pandas as pd
import torch
import torch.nn as nn

meta_csv = "data/ptbxl/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
df_meta = pd.read_csv(
    meta_csv, index_col="ecg_id", converters={"scp_codes": lambda x: eval(x)}
)
all_raw = set()
for lst in df_meta["scp_codes"]:
    all_raw.update(lst)
print("ALL raw labels in PTB-XL:", sorted(all_raw))


x = torch.randn(1, 12, 1000)
conv1 = nn.Conv1d(12, 64, kernel_size=16, padding=7)
y = conv1(x)
print(y.shape)  # should be (1, 64, 999)
pool1 = nn.MaxPool1d(3, stride=2, padding=1)
z = pool1(y)
print(z.shape)  # should be (1, 64, 500)
