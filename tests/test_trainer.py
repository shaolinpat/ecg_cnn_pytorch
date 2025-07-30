import numpy as np
import pytest
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ecg_cnn.training.trainer import (
    train_one_epoch,
)

SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 1000, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # 5 classes
        )

    def forward(self, x):
        return self.net(x)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self):
        torch.manual_seed(SEED)
        self.X = torch.randn(10, 12, 1000)
        self.y = torch.randint(0, 5, (10,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def test_train_one_epoch_type_errors():
    model = torch.nn.Linear(10, 2)
    data = torch.randn(4, 10)
    targets = torch.randint(0, 2, (4,))
    dataloader = DataLoader(TensorDataset(data, targets))
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    # Wrong model type
    with pytest.raises(TypeError, match="model must be an instance of torch.nn.Module"):
        train_one_epoch("not a model", dataloader, optimizer, criterion, device)

    # Wrong dataloader type
    with pytest.raises(
        TypeError, match="dataloader must be a torch.utils.data.DataLoader"
    ):
        train_one_epoch(model, "not a dataloader", optimizer, criterion, device)

    # Wrong optimizer type
    with pytest.raises(TypeError, match="optimizer must be a torch.optim.Optimizer"):
        train_one_epoch(model, dataloader, "not an optimizer", criterion, device)

    # Wrong criterion type
    with pytest.raises(TypeError, match="criterion must be callable"):
        train_one_epoch(model, dataloader, optimizer, "not callable", device)

    # Wrong device type
    with pytest.raises(TypeError, match="device must be a torch.device"):
        train_one_epoch(model, dataloader, optimizer, criterion, "not a device")


def test_train_one_epoch_end_to_end():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator().manual_seed(SEED)
    model = SimpleModel()
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, generator=g)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    initial_loss, initial_accuracy = train_one_epoch(
        model, dataloader, optimizer, criterion, device
    )

    assert isinstance(initial_loss, float), "train_one_epoch should return a float"
    assert initial_loss > 0.0, "Initial loss should be positive"

    # Optional: second epoch to check for progress
    later_loss, later_accuracy = train_one_epoch(
        model, dataloader, optimizer, criterion, device
    )

    assert isinstance(later_loss, float), "Loss should be a float"
    assert isinstance(later_accuracy, float), "Accuracy should be a float"
    assert later_loss >= 0.0
    # Allow +/-1.0 wiggle room to avoid flakeouts on tiny datasets
    assert (
        abs(later_loss - initial_loss) <= 1.0
    ), f"Loss drifted: {initial_loss:.5f} -> {later_loss:.5f}"
