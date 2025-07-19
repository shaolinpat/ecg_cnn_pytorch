import numpy as np
from ecg_cnn.data.data_utils import load_ptbxl_full, load_ptbxl_sample

# Adjust these paths as needed for your environment:
FULL_DATA_DIR = "./data/ptbxl/physionet.org/files/ptb-xl/1.0.3"
SAMPLE_DIR = "./data/sample"


def test_loaders():
    # 1) Test load_ptbxl_full on a small subsample (1% of records)
    Xf, yf, meta_f = load_ptbxl_full(
        data_dir=FULL_DATA_DIR,
        subsample_frac=0.01,  # load ~1% to keep this quick
        sampling_rate=100,
    )
    print("=== Full loader (1% subsample) ===")
    print("X.shape:", Xf.shape)
    print("Raw unique labels:", sorted(set(yf)))

    # Filter out "Unknown" before counting final
    keep_f = np.array([lbl != "Unknown" for lbl in yf], dtype=bool)
    Xf2 = Xf[keep_f]
    yf2 = [lbl for i, lbl in enumerate(yf) if keep_f[i]]
    meta_f2 = meta_f.loc[keep_f].reset_index(drop=True)
    print("After dropping 'Unknown':")
    print("  X.shape:", Xf2.shape)
    print("  Remaining classes:", sorted(set(yf2)))
    print()

    # 2) Test load_ptbxl_sample on the 100-record subset
    Xs, ys, meta_s = load_ptbxl_sample(sample_dir=SAMPLE_DIR, ptb_path=FULL_DATA_DIR)
    print("=== Sample loader (100 records) ===")
    print("X.shape:", Xs.shape)
    print("Raw unique labels:", sorted(set(ys)))

    keep_s = np.array([lbl != "Unknown" for lbl in ys], dtype=bool)
    Xs2 = Xs[keep_s]
    ys2 = [lbl for i, lbl in enumerate(ys) if keep_s[i]]
    meta_s2 = meta_s.loc[keep_s].reset_index(drop=True)
    print("After dropping 'Unknown':")
    print("  X.shape:", Xs2.shape)
    print("  Remaining classes:", sorted(set(ys2)))


if __name__ == "__main__":
    test_loaders()
