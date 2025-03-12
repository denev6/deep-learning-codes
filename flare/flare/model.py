import torch
import torch.nn as nn
import torch.nn.functional as F

class FlareModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

    def optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_func(self):
        """Define loss function that gets 'output' and 'target' as input."""
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def scheduler(self):
        return None
