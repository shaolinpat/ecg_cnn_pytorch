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