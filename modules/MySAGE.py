import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class MySAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1))
