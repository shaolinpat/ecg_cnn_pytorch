import torch
from torch.utils.data import Dataset
from ecg_cnn.data.data_utils import build_full_X_y


class PTBXLFullDataset(Dataset):
    """
    A PyTorch Dataset for the full PTB-XL ECG dataset.

    This dataset loads all ECG signals and their corresponding labels into memory
    at initialization time, allowing fast access during training or evaluation.

    Each item returned by __getitem__ is a tuple:
        (ecg_tensor, label_int)

    - ecg_tensor: A torch.FloatTensor of shape (12, T), where T is the signal length.
    - label_int: An integer corresponding to the five-class diagnosis label.

    This dataset assumes that the input has already been preprocessed into
    NumPy arrays or tensors, and that the label mapping is consistent with
    the five-class classification scheme.
    """

    def __init__(self, meta_csv, scp_csv, ptb_path):
        # Call the function above to build X_all, y_all (and keep meta if needed)
        X_all, y_all, _ = build_full_X_y(meta_csv, scp_csv, ptb_path)
        self.X = torch.from_numpy(X_all).float()  # Tensor shape: (N, 12, T)
        self.y = torch.from_numpy(y_all).long()  # Tensor shape: (N,)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # No squeeze necessary
        assert x.shape == (12, 1000), f"Expected shape (12, 1000), got {x.shape}"
        return x, int(self.y[idx])
