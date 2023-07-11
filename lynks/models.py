import torch
import torch.nn.functional as F

from lynks.layers import GCNConv


class GCNModel(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()
        self.conv1 = GCNConv(in_channels=c_in, out_channels=c_out)
        self.conv2 = GCNConv(in_channels=c_out, out_channels=c_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
