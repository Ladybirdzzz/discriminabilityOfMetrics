import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP


class MyVGNAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, model='VGNAE', scaling_factor=1.8):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)
        self.scaling_factor=scaling_factor
        self.model=model

    def forward(self, x, edge_index):
        if self.model == 'GNAE':
            x = self.linear1(x)
            x = F.normalize(x, p=2, dim=1) * self.scaling_factor
            x = self.propagate(x, edge_index)
            return x

        if self.model == 'VGNAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)

            x = self.linear2(x)
            x = F.normalize(x, p=2, dim=1) * self.scaling_factor
            x = self.propagate(x, edge_index)
            return x, x_

        return x