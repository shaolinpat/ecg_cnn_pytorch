 
import pandas as pd
import json


# 1) Load the PTB-XL metadata CSV (adjust path as needed if your working directory differs)
meta = pd.read_csv("./data/ptbxl/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv", index_col="ecg_id")


# 2) Helper to parse the scp_codes JSON string and pick the top code
def get_top_code(scp_json_str):
    d = json.loads(scp_json_str.replace("'", '"'))
    return max(d, key=lambda k: d[k])

# 3) Create a new column “top_code” containing that highest‐percentage key
meta["top_code"] = meta["scp_codes"].apply(get_top_code)

# 4) Count how many times each code appears, sort descending
freq = meta["top_code"].value_counts().reset_index()
freq.columns = ["scp_code", "count"]

# 5) Display the top 20 most frequent raw SCP codes
print("Top 20 raw SCP codes by frequency:")
print(freq.head(20).to_string(index=False))
