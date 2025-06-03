 
import pandas as pd
meta_csv = "data/ptbxl/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
df_meta = pd.read_csv(meta_csv, index_col="ecg_id", converters={"scp_codes": lambda x: eval(x)})
all_raw = set()
for lst in df_meta["scp_codes"]:
    all_raw.update(lst)
print("ALL raw labels in PTB-XL:", sorted(all_raw))


import torch
import torch.nn as nn

# L_out = (L_in + 2*pad - kernel_size) / stride + 1

x = torch.randn(1,12,1000)
conv1 = nn.Conv1d(12,64,kernel_size=16,padding=7)
y = conv1(x)
print(y.shape)   # should be (1, 64, 999)
pool1 = nn.MaxPool1d(3, stride=2, padding=1)
z = pool1(y)
print(z.shape)   # should be (1, 64, 500)


# For a 1-D convolution (Conv1d), the output length L_out (assuming stride=1) is:
# L_out = (L_in + 2*padding - kernel_size) / stride + 1
# L_out_conv1 = (1000 + 2*7 - 16) / 1 + 1
#             = (1000 + 14 - 16) / 1 + 1
#             = 998 / 1 + 1
#             = 999

# For a 1-D max-pool (MaxPool1d), if you have kernel_size=k, stride=s, and padding=p, the output length is:
# L_out = floor((L_in + 2*p - k) / s) + 1

# L_out_pool1 = floor((999 + 2*1 - 3) / 2) + 1
#             = floor((999 + 2 - 3) / 2) + 1
#             = floor(998 / 2) + 1
#             = 499 + 1
#             = 500