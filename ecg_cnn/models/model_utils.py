import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGConvNet(nn.Module):
    """

    A 1D Convolutional Neural Network for multi-class ECG classification.


    Architecture Overview:

        - 3 convolutional blocks, each with Conv1d → BatchNorm → ReLU → MaxPool

        - Dropout after final conv block

        - Flatten

        - 2 fully connected layers with dropout and ReLU

        - Final output layer with logits for classification


    Parameters

    ----------

    num_classes : int

        Number of output classes for classification.

    in_channels : int, default=12

        Number of input channels (e.g., ECG leads).

    conv_filters : int, default=64

        Number of filters in each convolutional layer.

    kernel_sizes : list of int, default=[16, 3, 3]

        Kernel sizes for each convolutional layer.

    conv_dropout : float, default=0.3

        Dropout probability after final conv block.

    fc_dropout : float, default=0.5

        Dropout probability applied after each FC layer.



    Input shape

    -----------

    (batch_size, in_channels, time_steps)



    Output shape

    ------------

    (batch_size, num_classes)

    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 12,
        conv_filters: int = 64,
        kernel_sizes: list = [16, 3, 3],
        conv_dropout: float = 0.3,
        fc_dropout: float = 0.5,
    ):
        super().__init__()

        assert len(kernel_sizes) == 3, "Expected 3 kernel sizes for 3 conv blocks"

        # --- Convolutional Blocks ---
        self.conv1 = nn.Conv1d(
            in_channels,
            conv_filters,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
        )
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_sizes[1])
        self.bn2 = nn.BatchNorm1d(conv_filters)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_sizes[2])
        self.bn3 = nn.BatchNorm1d(conv_filters)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        # --- Dropout after conv stack ---
        self.dropout_conv = nn.Dropout(p=conv_dropout)

        # --- FC Layers ---
        self.flatten = nn.Flatten()
        flat_size = self._get_flattened_size(in_channels, seq_len=1000)
        self.fc1 = nn.Linear(flat_size, 64)
        # self.fc1 = nn.Linear(
        #     conv_filters * 125, 64
        # )  # Input length should match post-pooling shape
        self.dropout_fc1 = nn.Dropout(p=fc_dropout)
        self.fc2 = nn.Linear(64, 32)
        self.dropout_fc = nn.Dropout(p=fc_dropout)

        # --- Output Layer ---
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, time_steps)

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # → (batch, filters, 500)

        # Conv Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # → (batch, filters, 250)

        # Conv Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # → (batch, filters, 125)

        x = self.dropout_conv(x)  # → (batch, filters, 125)
        x = self.flatten(x)  # → (batch, filters * 125)

        x = F.relu(self.dropout_fc1(self.fc1(x)))  # → (batch, 64)
        x = F.relu(self.fc2(x))  # → (batch, 32)
        x = self.dropout_fc(x)  # → (batch, 32)

        return self.out(x)  # → (batch, num_classes)

    def _get_flattened_size(self, in_channels, seq_len):
        """
        Computes the flattened feature size after all convolution, batch normalization,
        pooling, and dropout layers, given a dummy input with specified shape.

        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., 12 for 12-lead ECG).

        seq_len : int
            Length of the input time series (e.g., 1000 timesteps).

        Returns
        -------
        int
            Size of the flattened output tensor after the final convolutional layer,
            used to define the first fully connected layer.
        """
        if not isinstance(in_channels, int) or not (1 <= in_channels <= 32):
            raise ValueError(
                f"in_channels must be an int in [1, 32], got {in_channels}"
            )

        if not isinstance(seq_len, int) or seq_len < 8:
            raise ValueError(f"seq_len must be an int >= 8, got {seq_len}")

        with torch.no_grad():
            x = torch.zeros(1, in_channels, seq_len)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.dropout_conv(x)
            return x.view(1, -1).shape[1]


# -----------------------------------------------------------------
# New: a small 1D-ResNet (add this below ECGConvNet)
# -----------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out)
        return out


class ECGResNet(nn.Module):
    def __init__(self, num_classes):
        super(ECGResNet, self).__init__()
        # Initial convolution: 12→64, k=7, s=2, p=3
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # Adaptive pooling → flatten → FC
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # collapses time dimension to 1
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 12, 1000)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # roughly (batch, 64, 250)

        x = self.layer1(x)  # (batch, 64, 250)
        x = self.layer2(x)  # (batch,128,125)
        x = self.layer3(x)  # (batch,256, 63)

        x = self.avgpool(x)  # (batch,256, 1)
        x = x.view(x.size(0), -1)  # (batch,256)
        return self.fc(x)  # (batch,num_classes)
