import numpy as np
import pytest
from ecg_cnn.training.training_utils import compute_class_weights
import torch


def test_compute_class_weights_basic():
    y = np.array([0, 1, 0, 1, 1, 2])  # class counts: [2, 3, 1]
    weights = compute_class_weights(y, num_classes=3)
    expected = torch.tensor([1.0, 0.6666667, 2.0])  # total = 6, class freq = [2, 3, 1]
    assert torch.allclose(weights, expected, rtol=1e-4)


def test_invalid_y_type():
    with pytest.raises(ValueError, match="y must be a numpy ndarray"):
        compute_class_weights([0, 1, 2], num_classes=3)


def test_invalid_y_dim():
    y = np.array([[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="y must be a 1D array"):
        compute_class_weights(y, num_classes=4)


def test_invalid_y_dtype():
    y = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match="y must contain integer"):
        compute_class_weights(y, num_classes=3)


def test_invalid_num_classes():
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        compute_class_weights(y, num_classes=0)


def test_num_classes_too_small():
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="greater than max"):
        compute_class_weights(y, num_classes=2)
