import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ECGConvNet",
    "ECGResNet",
    "ECGInceptionNet",
    "ResidualBlock1D",
    "InceptionBlock1D",
]


# ------------------------------------------------------------------------------
# ECGConvNet: Baseline CNN for ECG classification
# ------------------------------------------------------------------------------
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
    seq_len : int, default=1000
        Length of the input time series.

    Input shape
    -----------
    (batch_size, in_channels, time_steps)

    Output shape
    ------------
    (batch_size, num_classes)

    Note
    ----
    This class must be imported and registered in ecg_cnn/models/__init__.py:
        from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet
        MODEL_CLASSES = { ... }
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 12,
        conv_filters: int = 64,
        kernel_sizes: list = [16, 3, 3],
        conv_dropout: float = 0.3,
        fc_dropout: float = 0.5,
        seq_len: int = 1000,
    ):
        """
        Initialize ECGConvNet model.

        Parameters
        ----------
        num_classes : int
            Number of output classes.
        in_channels : int
            Number of input channels (e.g., 12-lead ECG).
        conv_filters : int
            Number of filters in each convolutional layer.
        kernel_sizes : list of int
            Kernel sizes for the convolutional layers.
        conv_dropout : float
            Dropout probability after convolutional layers.
        fc_dropout : float
            Dropout probability after fully connected layers.
        seq_len : int
            Length of the input time series.
        """
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
        flat_size = self._get_flattened_size(in_channels, seq_len)
        self.fc1 = nn.Linear(flat_size, 64)
        self.dropout_fc1 = nn.Dropout(p=fc_dropout)
        self.fc2 = nn.Linear(64, 32)
        self.dropout_fc = nn.Dropout(p=fc_dropout)

        # --- Output Layer ---
        self.out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, time_steps)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # → (batch, filters, ~500)

        # Conv Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # → (batch, filters, ~250)

        # Conv Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # → (batch, filters, ~125)

        x = self.dropout_conv(x)  # → (batch, filters, ~125)
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

    def __repr__(self):
        """
        Returns a concise string summarizing key model hyperparameters.

        Returns
        -------
        str
            String representation including in_channels, conv_filters, kernel_sizes,
            conv_dropout, fc_dropout, and number of output classes.
        """
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.conv1.in_channels}, "
            f"conv_filters={self.conv1.out_channels}, "
            f"kernel_sizes={[self.conv1.kernel_size[0], self.conv2.kernel_size[0], self.conv3.kernel_size[0]]}, "
            f"conv_dropout={self.dropout_conv.p}, "
            f"fc_dropout={self.dropout_fc.p}, "
            f"num_classes={self.out.out_features})"
        )


class ResidualBlock1D(nn.Module):
    """
    A 1D residual block for convolutional neural networks, adapted from ResNet.

    This block consists of two Conv1D layers with BatchNorm and ReLU activations.
    If the input and output dimensions differ, a downsampling path is applied to
    the identity connection to match shapes.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, default=1
        Stride for the first convolution. Also used for downsampling if needed.
    kernel_size : int, default=3
        Size of the 1D convolution kernel for both conv layers.
    padding : int, default=1
        Padding used in convolution layers.
    bias : bool, default=False
        Whether to use bias in convolution layers.

    Notes
    -----
    - Used as a building block for ECGResNet.
    - Must be imported and registered in `ecg_cnn/models/__init__.py` to be
      available for training and evaluation via string name lookup:

        from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet
        MODEL_CLASSES = {
            "ECGConvNet": ECGConvNet,
            "ECGResNet": ECGResNet,
            "ECGInceptionNet": ECGInceptionNet,
        }
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ):
        """
        Initialize the ResidualBlock1D module.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        stride : int, default=1
            Stride for the first convolution.
        kernel_size : int, default=3
            Size of the 1D convolution kernel.
        padding : int, default=1
            Padding used in the convolution layers.
        bias : bool, default=False
            Whether to use bias in the convolution layers.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length_out)
        """
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

    def __repr__(self) -> str:
        """
        Return a string representation of the ResidualBlock1D.

        Returns
        -------
        str
            Summary string with key initialization parameters.
        """
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"stride={self.stride}, "
            f"kernel_size={self.kernel_size}, "
            f"padding={self.padding}, "
            f"bias={self.bias})"
        )


class ECGResNet(nn.Module):
    """
    A compact 1D ResNet-style convolutional network for ECG classification.

    This model uses an initial convolution followed by three residual stages
    with increasing channel widths and downsampling. It concludes with global
    average pooling and a fully connected classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes for classification.
    in_channels : int, default=12
        Number of input channels (e.g., 12 for 12-lead ECG).
    base_widths : tuple[int, int, int], default=(64, 128, 256)
        Output channels for each residual stage.
    num_blocks : tuple[int, int, int], default=(2, 2, 2)
        Number of residual blocks in each stage.
    kernel_size : int, default=7
        Kernel size for the initial convolution.
    initial_stride : int, default=2
        Stride for the initial convolution.
    initial_padding : int, default=3
        Padding for the initial convolution.
    pool_kernel : int, default=3
        Kernel size for the initial max-pooling.
    pool_stride : int, default=2
        Stride for the initial max-pooling.
    pool_padding : int, default=1
        Padding for the initial max-pooling.

    Notes
    -----
    This model must be registered in ecg_cnn/models/__init__.py:

        from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet
        MODEL_CLASSES = {
            "ECGConvNet": ECGConvNet,
            "ECGResNet": ECGResNet,
            "ECGInceptionNet": ECGInceptionNet,
        }
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 12,
        base_widths: tuple[int, int, int] = (64, 128, 256),
        num_blocks: tuple[int, int, int] = (2, 2, 2),
        kernel_size: int = 7,
        initial_stride: int = 2,
        initial_padding: int = 3,
        pool_kernel: int = 3,
        pool_stride: int = 2,
        pool_padding: int = 1,
    ):
        """
        Initialize the ECGResNet model with parameterized architecture.
        """
        super(ECGResNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_widths = base_widths
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.initial_stride = initial_stride
        self.initial_padding = initial_padding
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        width1, width2, width3 = base_widths
        blocks1, blocks2, blocks3 = num_blocks

        # Initial convolution
        self.conv1 = nn.Conv1d(
            in_channels,
            width1,
            kernel_size=kernel_size,
            stride=initial_stride,
            padding=initial_padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(width1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(
            kernel_size=pool_kernel,
            stride=pool_stride,
            padding=pool_padding,
        )

        # Residual stages
        self.layer1 = self._make_layer(width1, width1, blocks1, stride=1)
        self.layer2 = self._make_layer(width1, width2, blocks2, stride=2)
        self.layer3 = self._make_layer(width2, width3, blocks3, stride=2)

        # Final pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(width3, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """
        Construct a residual stage from multiple ResidualBlock1D modules.

        Parameters
        ----------
        in_channels : int
            Input channels for the first block.
        out_channels : int
            Output channels for all blocks.
        num_blocks : int
            Number of blocks in this stage.
        stride : int
            Stride for the first block. Remaining blocks use stride=1.

        Returns
        -------
        nn.Sequential
            A sequence of ResidualBlock1D modules.
        """
        layers = [ResidualBlock1D(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ECGResNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes)
        """
        # x: (batch, in_channels, sequence_length)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # → (batch, width1, L/4)

        x = self.layer1(x)  # → (batch, width1, L/4)
        x = self.layer2(x)  # → (batch, width2, L/8)
        x = self.layer3(x)  # → (batch, width3, L/16)

        x = self.avgpool(x)  # → (batch, width3, 1)
        x = x.view(x.size(0), -1)  # → (batch, width3)
        return self.fc(x)  # → (batch, num_classes)

    def __repr__(self) -> str:
        """
        Return a string representation of the ECGResNet model.

        Returns
        -------
        str
            Summary string with initialization parameters.
        """
        return (
            f"ECGResNet(num_classes={self.num_classes}, "
            f"in_channels={self.in_channels}, "
            f"base_widths={self.base_widths}, "
            f"num_blocks={self.num_blocks}, "
            f"kernel_size={self.kernel_size}, "
            f"initial_stride={self.initial_stride}, "
            f"initial_padding={self.initial_padding}, "
            f"pool_kernel={self.pool_kernel}, "
            f"pool_stride={self.pool_stride}, "
            f"pool_padding={self.pool_padding})"
        )


class InceptionBlock1D(nn.Module):
    """
    A 1D Inception-style block for multi-scale feature extraction in time series.

    This block uses parallel convolutional branches with different kernel sizes
    to capture local patterns at multiple receptive fields. Each branch starts
    with a bottleneck 1x1 convolution followed by a wide-kernel convolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels per branch.
    kernel_sizes : tuple of int, optional (default=(3, 5, 7))
        The kernel sizes used for each parallel branch.
    bottleneck_channels : int, optional (default=32)
        The number of channels in the bottleneck 1x1 convolution.

    Output Shape
    ------------
    Tensor of shape (batch_size, out_channels * len(kernel_sizes), sequence_length)

    Notes
    -----
    - Output channels are concatenated across all branches.
    - BatchNorm is applied after concatenation.
    - Used in ECGInceptionNet or similar architectures.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (3, 5, 7),
        bottleneck_channels: int = 32,
    ):
        """
        Initialize the InceptionBlock1D module.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels per branch.

        kernel_sizes : tuple of int, default=(3, 5, 7)
            The kernel sizes used for each parallel branch.

        bottleneck_channels : int, default=32
            Number of channels for bottleneck 1x1 convolutions before wide kernels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.bottleneck_channels = bottleneck_channels

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(
                    bottleneck_channels, out_channels, kernel_size=k, padding=k // 2
                ),
                nn.ReLU(),
            )
            self.branches.append(branch)

        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Inception block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels * branches, sequence_length)
        """
        out = [branch(x) for branch in self.branches]
        out = torch.cat(out, dim=1)
        out = self.bn(out)
        return out

    def __repr__(self) -> str:
        """
        Return a string representation of the InceptionBlock1D.

        Returns
        -------
        str
            Summary string with initialization parameters.
        """
        return (
            f"InceptionBlock1D(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_sizes={self.kernel_sizes}, "
            f"bottleneck_channels={self.bottleneck_channels})"
        )


class ECGInceptionNet(nn.Module):
    """
    Inception-style 1D CNN for ECG classification.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 12 for 12-lead ECG).
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout rate before final classification layer.

    Notes
    -----
    - It must be imported and registered in `ecg_cnn/models/__init__.py` to be
      available for training and evaluation via string name lookup.

        from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet
        MODEL_CLASSES = {
            "ECGConvNet": ECGConvNet,
            "ECGResNet": ECGResNet,
            "ECGInceptionNet": ECGInceptionNet,
        }
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        dropout: float = 0.5,
        block_channels: tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: tuple[int, int, int] = (3, 5, 7),
        bottleneck_channels: int = 32,
    ):
        """
        Initialize the Inception-style 1D CNN for ECG classification.

        Parameters
        ----------
        in_channels : int, default=12
            Number of input channels (e.g., 12-lead ECG).
        num_classes : int, default=5
            Number of output classes for classification.
        dropout : float, default=0.5
            Dropout rate applied before the final classification layer.
        block_channels : tuple of 3 ints, default=(32, 64, 128)
            Output channels for each of the three Inception blocks.
        kernel_sizes : tuple of 3 ints, default=(3, 5, 7)
            Kernel sizes for each branch within the Inception blocks.
        bottleneck_channels : int, default=32
            Number of channels for the initial 1x1 bottleneck convolutions.

        Notes
        -----
        - This model consists of three Inception-style blocks with configurable
          multi-scale branches and channel widths.
        - Each Inception block reduces input via 1x1 bottleneck, applies multiple
          kernel convolutions, and concatenates results.
        - Adaptive average pooling collapses the temporal axis before dropout and
          final classification.
        - Output shape: (batch_size, num_classes)

        To make this model available by name, register it in:
            ecg_cnn/models/__init__.py

            Example:
                from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet
                MODEL_CLASSES = {
                    "ECGConvNet": ECGConvNet,
                    "ECGResNet": ECGResNet,
                    "ECGInceptionNet": ECGInceptionNet,
                }
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.block_channels = block_channels
        self.kernel_sizes = kernel_sizes
        self.bottleneck_channels = bottleneck_channels

        self.block1 = InceptionBlock1D(
            in_channels,
            out_channels=block_channels[0],
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
        )
        self.block2 = InceptionBlock1D(
            block_channels[0] * len(kernel_sizes),
            out_channels=block_channels[1],
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
        )
        self.block3 = InceptionBlock1D(
            block_channels[1] * len(kernel_sizes),
            out_channels=block_channels[2],
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(block_channels[2] * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward pass through the ECGInceptionNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, signal_length)

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

    def __repr__(self) -> str:
        """
        Return a string representation of the ECGInceptionNet model.

        Returns
        -------
        str
            Summary string with initialization parameters.
        """
        return (
            f"ECGInceptionNet(in_channels={self.in_channels}, "
            f"num_classes={self.num_classes}, "
            f"dropout={self.dropout_rate}, "
            f"block_channels={self.block_channels}, "
            f"kernel_sizes={self.kernel_sizes}, "
            f"bottleneck_channels={self.bottleneck_channels})"
        )


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        A PyTorch model.

    Returns
    -------
    int
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# __all__ = [
#     "ECGConvNet",
#     "ECGResNet",
#     "ECGInceptionNet",
#     "ResidualBlock1D",
#     "InceptionBlock1D",
# ]


# class ECGConvNet(nn.Module):
#     """
#     A 1D Convolutional Neural Network for multi-class ECG classification.

#     Architecture Overview:
#         - 3 convolutional blocks, each with Conv1d → BatchNorm → ReLU → MaxPool
#         - Dropout after final conv block
#         - Flatten
#         - 2 fully connected layers with dropout and ReLU
#         - Final output layer with logits for classification

#     Parameters
#     ----------
#     num_classes : int
#         Number of output classes for classification.
#     in_channels : int, default=12
#         Number of input channels (e.g., ECG leads).
#     conv_filters : int, default=64
#         Number of filters in each convolutional layer.
#     kernel_sizes : list of int, default=[16, 3, 3]
#         Kernel sizes for each convolutional layer.
#     conv_dropout : float, default=0.3
#         Dropout probability after final conv block.
#     fc_dropout : float, default=0.5
#         Dropout probability applied after each FC layer.

#     Input shape
#     -----------
#     (batch_size, in_channels, time_steps)

#     Output shape
#     ------------
#     (batch_size, num_classes)

#     Note
#     ----
#     This class must be imported and registered in ecg_cnn/models/__init__.py:
#         from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet
#         MODEL_CLASSES = { ... }
#     """

#     def __init__(
#         self,
#         num_classes: int,
#         in_channels: int = 12,
#         conv_filters: int = 64,
#         kernel_sizes: list = [16, 3, 3],
#         conv_dropout: float = 0.3,
#         fc_dropout: float = 0.5,
#     ):
#         super().__init__()

#         assert len(kernel_sizes) == 3, "Expected 3 kernel sizes for 3 conv blocks"

#         # --- Convolutional Blocks ---
#         self.conv1 = nn.Conv1d(
#             in_channels,
#             conv_filters,
#             kernel_size=kernel_sizes[0],
#             padding=kernel_sizes[0] // 2,
#         )
#         self.bn1 = nn.BatchNorm1d(conv_filters)
#         self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_sizes[1])
#         self.bn2 = nn.BatchNorm1d(conv_filters)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

#         self.conv3 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_sizes[2])
#         self.bn3 = nn.BatchNorm1d(conv_filters)
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

#         # --- Dropout after conv stack ---
#         self.dropout_conv = nn.Dropout(p=conv_dropout)

#         # --- FC Layers ---
#         self.flatten = nn.Flatten()
#         flat_size = self._get_flattened_size(in_channels, seq_len=1000)
#         self.fc1 = nn.Linear(flat_size, 64)
#         # self.fc1 = nn.Linear(conv_filters * 125, 64)
#         # --> Used if input is fixed-length (1000 timesteps) and shape is known
#         self.dropout_fc1 = nn.Dropout(p=fc_dropout)
#         self.fc2 = nn.Linear(64, 32)
#         self.dropout_fc = nn.Dropout(p=fc_dropout)

#         # --- Output Layer ---
#         self.out = nn.Linear(32, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the model.

#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, in_channels, time_steps)

#         Returns
#         -------
#         torch.Tensor
#             Output logits of shape (batch_size, num_classes)
#         """
#         # Conv Block 1
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # → (batch, filters, 500)

#         # Conv Block 2
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # → (batch, filters, 250)

#         # Conv Block 3
#         x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # → (batch, filters, 125)

#         x = self.dropout_conv(x)  # → (batch, filters, 125)
#         x = self.flatten(x)  # → (batch, filters * 125)

#         x = F.relu(self.dropout_fc1(self.fc1(x)))  # → (batch, 64)
#         x = F.relu(self.fc2(x))  # → (batch, 32)
#         x = self.dropout_fc(x)  # → (batch, 32)

#         return self.out(x)  # → (batch, num_classes)

#     def _get_flattened_size(self, in_channels, seq_len):
#         """
#         Computes the flattened feature size after all convolution, batch normalization,
#         pooling, and dropout layers, given a dummy input with specified shape.

#         Parameters
#         ----------
#         in_channels : int
#             Number of input channels (e.g., 12 for 12-lead ECG).

#         seq_len : int
#             Length of the input time series (e.g., 1000 timesteps).

#         Returns
#         -------
#         int
#             Size of the flattened output tensor after the final convolutional layer,
#             used to define the first fully connected layer.
#         """
#         if not isinstance(in_channels, int) or not (1 <= in_channels <= 32):
#             raise ValueError(
#                 f"in_channels must be an int in [1, 32], got {in_channels}"
#             )

#         if not isinstance(seq_len, int) or seq_len < 8:
#             raise ValueError(f"seq_len must be an int >= 8, got {seq_len}")

#         with torch.no_grad():
#             x = torch.zeros(1, in_channels, seq_len)
#             x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#             x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#             x = self.pool3(F.relu(self.bn3(self.conv3(x))))
#             x = self.dropout_conv(x)
#             return x.view(1, -1).shape[1]


# # -----------------------------------------------------------------
# # New: a small 1D-ResNet (add this below ECGConvNet)
# # -----------------------------------------------------------------
# class ResidualBlock1D(nn.Module):
#     """
#     A 1D residual block for convolutional neural networks, adapted from ResNet.

#     This block consists of two Conv1D layers with BatchNorm and ReLU activations.
#     If the input and output dimensions differ, a downsampling path is applied to
#     the identity connection to match shapes.

#     Parameters
#     ----------
#     in_channels : int
#         Number of input channels.
#     out_channels : int
#         Number of output channels.
#     stride : int, optional (default=1)
#         Stride of the first convolution. If not 1 or if the input/output channels
#         differ, the identity path is downsampled.

#     Notes
#     -----
#     - This class is used as a building block for ECGResNet.
#     - It must be imported and registered in `ecg_cnn/models/__init__.py` to be
#       available for training and evaluation via string name lookup.

#         from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet
#         MODEL_CLASSES = {
#             "ECGConvNet": ECGConvNet,
#             "ECGResNet": ECGResNet,
#             "ECGInceptionNet": ECGInceptionNet,
#         }
#     """

#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=False,
#         )
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.conv2 = nn.Conv1d(
#             out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm1d(out_channels)

#         self.downsample = None
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv1d(
#                     in_channels, out_channels, kernel_size=1, stride=stride, bias=False
#                 ),
#                 nn.BatchNorm1d(out_channels),
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(identity)

#         out += identity
#         out = F.relu(out)
#         return out


# class ECGResNet(nn.Module):
#     """
#     A compact 1D ResNet-style convolutional network for ECG classification.

#     This model uses an initial convolution followed by three residual stages
#     with increasing channel widths and downsampling. It concludes with global
#     average pooling and a fully connected classification head.

#     Parameters
#     ----------
#     num_classes : int
#         Number of output classes for classification.

#     Architecture
#     ------------
#     - Input shape: (batch_size, 12, 1000) for 12-lead ECGs
#     - Conv1: 12 → 64 channels, kernel=7, stride=2, padding=3
#     - MaxPool1d: kernel=3, stride=2, padding=1
#     - Residual layers:
#         - layer1: 2 blocks @ 64 channels, no downsampling
#         - layer2: 2 blocks @ 128 channels, downsample (stride=2)
#         - layer3: 2 blocks @ 256 channels, downsample (stride=2)
#     - AdaptiveAvgPool1d: collapses time to length 1
#     - Fully connected layer: 256 → num_classes

#     Notes
#     -----
#     Each residual block uses two 1D convolutions with BatchNorm and ReLU.
#     If dimensions do not match, the identity connection is projected with a
#     1x1 convolution.
#     """

#     def __init__(self, num_classes):
#         super(ECGResNet, self).__init__()
#         # Initial convolution: 12→64, k=7, s=2, p=3
#         self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # Residual stages
#         self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
#         self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
#         self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

#         # Adaptive pooling → flatten → FC
#         self.avgpool = nn.AdaptiveAvgPool1d(1)  # collapses time dimension to 1
#         self.fc = nn.Linear(256, num_classes)

#     def _make_layer(self, in_channels, out_channels, num_blocks, stride):
#         layers = []
#         layers.append(ResidualBlock1D(in_channels, out_channels, stride))
#         for _ in range(1, num_blocks):
#             layers.append(ResidualBlock1D(out_channels, out_channels, stride=1))
#         return nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (batch, 12, 1000)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)  # roughly (batch, 64, 250)

#         x = self.layer1(x)  # (batch, 64, 250)
#         x = self.layer2(x)  # (batch,128,125)
#         x = self.layer3(x)  # (batch,256, 63)

#         x = self.avgpool(x)  # (batch,256, 1)
#         x = x.view(x.size(0), -1)  # (batch,256)
#         return self.fc(x)  # (batch,num_classes)


# class InceptionBlock1D(nn.Module):
#     """
#     A 1D Inception-style block for multi-scale feature extraction in time series.

#     This block uses parallel convolutional branches with different kernel sizes
#     to capture local patterns at multiple receptive fields. Each branch starts
#     with a bottleneck 1x1 convolution followed by a wide-kernel convolution.

#     Parameters
#     ----------
#     in_channels : int
#         Number of input channels.
#     out_channels : int
#         Number of output channels per branch.
#     kernel_sizes : tuple of int, optional (default=(3, 5, 7))
#         The kernel sizes used for each parallel branch.
#     bottleneck_channels : int, optional (default=32)
#         The number of channels in the bottleneck 1x1 convolution.

#     Output Shape
#     ------------
#     Tensor of shape (batch_size, out_channels * len(kernel_sizes), sequence_length)

#     Notes
#     -----
#     - Output channels are concatenated across all branches.
#     - BatchNorm is applied after concatenation.
#     """

#     def __init__(
#         self, in_channels, out_channels, kernel_sizes=(3, 5, 7), bottleneck_channels=32
#     ):
#         super().__init__()

#         self.branches = nn.ModuleList()
#         for k in kernel_sizes:
#             branch = nn.Sequential(
#                 nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv1d(
#                     bottleneck_channels, out_channels, kernel_size=k, padding=k // 2
#                 ),
#                 nn.ReLU(),
#             )
#             self.branches.append(branch)

#         self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = [branch(x) for branch in self.branches]
#         out = torch.cat(out, dim=1)
#         out = self.bn(out)
#         return out


# class ECGInceptionNet(nn.Module):
#     """
#     Inception-style 1D CNN for ECG classification.

#     Parameters
#     ----------
#     in_channels : int
#         Number of input channels (e.g., 12 for 12-lead ECG).
#     num_classes : int
#         Number of output classes.
#     dropout : float
#         Dropout rate before final classification layer.

#     Notes
#     -----
#     - It must be imported and registered in `ecg_cnn/models/__init__.py` to be
#       available for training and evaluation via string name lookup.

#         from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet
#         MODEL_CLASSES = {
#             "ECGConvNet": ECGConvNet,
#             "ECGResNet": ECGResNet,
#             "ECGInceptionNet": ECGInceptionNet,
#         }
#     """

#     def __init__(
#         self,
#         in_channels: int = 12,
#         num_classes: int = 5,
#         dropout: float = 0.5,
#         block_channels: tuple[int, int, int] = (32, 64, 128),
#         kernel_sizes: tuple[int, int, int] = (3, 5, 7),
#         bottleneck_channels: int = 32,
#     ):
#         """
#         Initialize the Inception-style 1D CNN for ECG classification.

#         Parameters
#         ----------
#         in_channels : int, default=12
#             Number of input channels (e.g., 12-lead ECG).
#         num_classes : int, default=5
#             Number of output classes for classification.
#         dropout : float, default=0.5
#             Dropout rate applied before the final classification layer.
#         block_channels : tuple of 3 ints, default=(32, 64, 128)
#             Output channels for each of the three Inception blocks.
#         kernel_sizes : tuple of 3 ints, default=(3, 5, 7)
#             Kernel sizes for each branch within the Inception blocks.
#         bottleneck_channels : int, default=32
#             Number of channels for the initial 1x1 bottleneck convolutions.

#         Notes
#         -----
#         - This model consists of three Inception-style blocks with configurable
#         multi-scale branches and channel widths.
#         - Each Inception block reduces input via 1x1 bottleneck, applies multiple
#         kernel convolutions, and concatenates results.
#         - Adaptive average pooling collapses the temporal axis before dropout and
#         final classification.
#         - Output shape: (batch_size, num_classes)

#         To make this model available by name, register it in:
#             ecg_cnn/models/__init__.py

#             Example:
#                 from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet
#                 MODEL_CLASSES = {
#                     "ECGConvNet": ECGConvNet,
#                     "ECGResNet": ECGResNet,
#                     "ECGInceptionNet": ECGInceptionNet,
#                 }
#         """
#         super().__init__()

#         self.block1 = InceptionBlock1D(
#             in_channels,
#             out_channels=block_channels[0],
#             kernel_sizes=kernel_sizes,
#             bottleneck_channels=bottleneck_channels,
#         )
#         self.block2 = InceptionBlock1D(
#             block_channels[0] * len(kernel_sizes),
#             out_channels=block_channels[1],
#             kernel_sizes=kernel_sizes,
#             bottleneck_channels=bottleneck_channels,
#         )
#         self.block3 = InceptionBlock1D(
#             block_channels[1] * len(kernel_sizes),
#             out_channels=block_channels[2],
#             kernel_sizes=kernel_sizes,
#             bottleneck_channels=bottleneck_channels,
#         )

#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(block_channels[2] * len(kernel_sizes), num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Apply forward pass through the ECGInceptionNet.

#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, in_channels, signal_length)

#         Returns
#         -------
#         torch.Tensor
#             Output logits of shape (batch_size, num_classes)
#         """
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.pool(x).squeeze(-1)
#         x = self.dropout(x)
#         return self.fc(x)
