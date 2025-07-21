import numpy as np
import pytest
import random
import torch
from ecg_cnn.models.model_utils import ECGConvNet


SEED = 22
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def test_ecgconvnet_num_parameters():
    model = ECGConvNet(num_classes=4)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0, "Model should have trainable parameters"


def test_ecgconvnet_default_output_shape():
    model = ECGConvNet(num_classes=5)
    dummy_input = torch.randn(4, 12, 1000)  # batch=4, 12 leads, 1000 time steps
    output = model(dummy_input)
    assert output.shape == (4, 5), "Output shape should be (batch_size, num_classes)"


def test_ecgconvnet_custom_kernel_sizes():
    model = ECGConvNet(num_classes=3, kernel_sizes=[7, 5, 3])
    dummy_input = torch.randn(2, 12, 1000)
    output = model(dummy_input)
    assert output.shape == (2, 3), "Custom kernel sizes should not affect output shape"


def test_ecgconvnet_dropout_forward():
    model = ECGConvNet(num_classes=2, conv_dropout=0.1, fc_dropout=0.2)
    model.train()  # enable dropout
    dummy_input = torch.randn(1, 12, 1000)
    output1 = model(dummy_input)
    output2 = model(dummy_input)
    # Dropout means output1 != output2
    assert not torch.equal(
        output1, output2
    ), "Dropout should produce stochastic output in training mode"


def test_get_flattened_size_valid_input():
    model = ECGConvNet(num_classes=3)
    size = model._get_flattened_size(in_channels=12, seq_len=1000)
    assert isinstance(size, int)
    assert size > 0


def test_get_flattened_size_custom_input():
    model = ECGConvNet(num_classes=5, kernel_sizes=[5, 3, 3], fc_dropout=0.2)
    size = model._get_flattened_size(in_channels=12, seq_len=800)
    assert isinstance(size, int)
    assert size > 0


def test_get_flattened_size_invalid_in_channels():
    model = ECGConvNet(num_classes=5)
    with pytest.raises(ValueError, match=f"in_channels must be an int"):
        model._get_flattened_size(in_channels="twelve", seq_len=800)


def test_get_flattened_size_inchannels_loo_low():
    model = ECGConvNet(num_classes=5)
    with pytest.raises(ValueError, match=r"in_channels must be an int in \[1, 32\]"):
        model._get_flattened_size(in_channels=0, seq_len=800)


def test_get_flattened_size_inchannels_loo_high():
    model = ECGConvNet(num_classes=5)
    with pytest.raises(ValueError, match=r"in_channels must be an int in \[1, 32\]"):
        model._get_flattened_size(in_channels=33, seq_len=800)


def test_get_flattened_size_invalid_seq_len():
    model = ECGConvNet(num_classes=5)
    with pytest.raises(ValueError, match=f"seq_len must be an int"):
        model._get_flattened_size(in_channels=12, seq_len="800")


def test_get_flattened_size_seq_len_too_low():
    model = ECGConvNet(num_classes=5)
    with pytest.raises(ValueError, match=f"seq_len must be an int >= 8"):
        model._get_flattened_size(in_channels=12, seq_len=7)
