# tests/test_training_utils.py
"""
Tests for ecg_cnn.training.training_utils.

Covers
------
    - compute_class_weights():
        * Correct output for balanced and unbalanced class distributions
        * Handling of missing classes (infinite weight)
        * Type, shape, and dtype checks
        * Input validation: type, dimensionality, dtype, num_classes
"""

import numpy as np
import pytest
import torch

from ecg_cnn.training.training_utils import compute_class_weights


def test_compute_class_weights_basic():
    y = np.array([0, 1, 0, 1, 1, 2], dtype=np.int64)  # class counts: [2, 3, 1]
    weights = compute_class_weights(y, num_classes=3)
    expected = torch.tensor([1.0, 0.6666667, 2.0])  # total = 6, class freq = [2, 3, 1]
    assert torch.allclose(weights, expected, rtol=1e-4)


def test_invalid_y_type():
    with pytest.raises(ValueError, match=r"^y must be a numpy ndarray"):
        compute_class_weights([0, 1, 2], num_classes=3)


def test_invalid_y_dim():
    y = np.array([[0, 1], [2, 3]])
    with pytest.raises(ValueError, match=r"^y must be a 1D array"):
        compute_class_weights(y, num_classes=4)


def test_invalid_y_dtype():
    y = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match=r"^y must contain integer class labels"):
        compute_class_weights(y, num_classes=3)


def test_invalid_num_classes():
    y = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match=r"^num_classes must be a positive integer"):
        compute_class_weights(y, num_classes=0)


def test_num_classes_too_small():
    y = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match=r"^num_classes must be greater than max\(y\)"):
        compute_class_weights(y, num_classes=2)


def test_compute_class_weights_type_and_shape():
    y = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
    w = compute_class_weights(y, num_classes=3)
    assert isinstance(w, torch.Tensor)
    assert w.dtype == torch.float32
    assert w.shape == (3,)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide")
def test_compute_class_weights_missing_class_inf():
    y = np.array([0, 0, 0], dtype=np.int64)  # counts [3,0]
    w = compute_class_weights(y, num_classes=2)
    assert torch.isinf(w[1]) and w[1] > 0
    assert torch.isfinite(w[0])


def test_compute_class_weights_balanced_two_classes():
    y = np.array([0, 1, 0, 1], dtype=np.int64)  # counts [2,2]
    w = compute_class_weights(y, num_classes=2)
    assert torch.allclose(w, torch.tensor([1.0, 1.0], dtype=torch.float32))
