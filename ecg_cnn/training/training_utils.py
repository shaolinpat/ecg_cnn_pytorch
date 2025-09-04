# training/training_utils.py

import numpy as np
import torch


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse frequency-based class weights for use with nn.CrossEntropyLoss.

    Parameters
    ----------
    y : np.ndarray
        1D array of integer class labels (shape: [n_samples]).
    num_classes : int
        Total number of classes. Must be >= max(y) + 1.

    Returns
    -------
    torch.Tensor
        Tensor of shape [num_classes] with higher weights for minority classes.

    Raises
    ------
    ValueError
        If input is not a valid 1D array of integers, or if num_classes is invalid.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError(f"y must be a numpy ndarray, got {type(y)}")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array of class labels")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must contain integer class labels")
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if y.max() >= num_classes:
        raise ValueError("num_classes must be greater than max(y)")

    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = total / (num_classes * counts.astype(np.float64))

    return torch.tensor(weights, dtype=torch.float32)
