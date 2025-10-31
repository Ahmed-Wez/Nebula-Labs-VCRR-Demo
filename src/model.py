# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, num_classes=20, pool_output_size=(4,4)):
        super().__init__()
        # Conv backbone
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Adaptive pooling to make fully-connected input size stable irrespective of intermediate sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_output_size)  # e.g., (4,4)
        # will compute flatten size dynamically in forward, but define a reasonably sized fc for feature dim
        flattened_dim = 128 * pool_output_size[0] * pool_output_size[1]
        self.fc1 = nn.Linear(flattened_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (B, 3, H, W)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))   # now downsampled
        x = self.adaptive_pool(x)              # make spatial dims fixed (e.g., 4x4)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x