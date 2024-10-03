import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class MyGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels*heads, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1))