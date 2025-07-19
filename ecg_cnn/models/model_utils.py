import torch.nn as nn
import torch.nn.functional as F

class ECGConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ECGConvNet, self).__init__()
        # Now we have 1-lead inputs, so in_channels=12
        #self.conv1 = nn.Conv1d(12, 64, kernel_size=6)
        self.conv1 = nn.Conv1d(12, 64, kernel_size=16, padding=7)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.flatten = nn.Flatten()

        # Compute the final flattened size: 64 channels x 124 timesteps = 7936
        self.fc1 = nn.Linear(64 * 125, 64)
        self.dropout_fc1 = nn.Dropout(p=0.5)    # <-- new dropout after fc1
        self.fc2 = nn.Linear(64, 32)

        self.out = nn.Linear(32, num_classes)

        # after the last pooling layer (conv3-pool3) to regularize convolutional features
        self.dropout_conv = nn.Dropout(p=0.3)

        # before the final classification, after fc2
        self.dropout_fc = nn.Dropout(p=0.5)

    def forward(self, x):
        # x shape: (batch_size, 12, 1000)
        # Conv block 1
        x = self.conv1(x)                   # -> (batch, 64, 995)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)                   # -> (batch, 64, 498)

        # Conv block 2
        x = self.conv2(x)                   # -> (batch, 64, 496)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)                   # -> (batch, 64, 249)

        # Conv block 3
        x = self.conv3(x)                   # -> (batch, 64, 247)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)                   # -> (batch, 64, 124)

        # --- APPLY DROPOUT after the last pooling (conv) block ---
        x = self.dropout_conv(x)            # (batch, 64, 124), randomly zero 30% of channels each forward

        # Flatten
        x = self.flatten(x)                 # -> (batch, 64 * 124 = 7936)

         # Fully connected layers
        x = self.fc1(x)                     # -> (batch, 64)               # changed: removed F.relu here
        x = self.dropout_fc1(x)             # -> (batch, 64), new dropout  # new: apply dropout right after fc1
        x = F.relu(x)                       # -> (batch, 64)               # changed: moved ReLU after dropout

        x = F.relu(self.fc2(x))             # -> (batch, 32)               # unchanged: fc2 + ReLU


        # --- APPLY DROPOUT before the final classifier ---
        x = self.dropout_fc(x)              # (batch, 32), randomly zero 50% of features each forward

        return self.out(x)                  # -> (batch, num_classes)
    

# -----------------------------------------------------------------
# New: a small 1D-ResNet (add this below ECGConvNet)
# -----------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
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
        self.bn1   = nn.BatchNorm1d(64)
        self.relu  = nn.ReLU(inplace=True)
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
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.maxpool(x)    # roughly (batch, 64, 250)

        x = self.layer1(x)     # (batch, 64, 250)
        x = self.layer2(x)     # (batch,128,125)
        x = self.layer3(x)     # (batch,256, 63)

        x = self.avgpool(x)    # (batch,256, 1)
        x = x.view(x.size(0), -1)  # (batch,256)
        return self.fc(x)         # (batch,num_classes)    