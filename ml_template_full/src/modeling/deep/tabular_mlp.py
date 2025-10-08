import torch
from torch import nn

class TabularMLP(nn.Module):
    def __init__(self, in_features, out_features=1, task="regression"):
        super().__init__()
        self.task = task
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        return self.net(x)
