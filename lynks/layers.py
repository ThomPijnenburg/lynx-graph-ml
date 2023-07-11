import torch
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    """Implementation follows pytorch geometric docs
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        self._in_channels = in_channels
        self._out_channels = out_channels

        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def __repr__(self):
        return f"GCNConv({self._in_channels}, {self._out_channels})"

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge index has shape [2, E]

        # 1. add self loop to adjacency
        edge_index, _ = add_self_loops(edge_index)

        # 2. linear transform node features
        x = self.lin(x)

        # 3. compute normalisation
        # get degrees
        row, col = edge_index
        out_degree = degree(col, num_nodes=x.size(0), dtype=x.dtype)
        out_degree_inv_sqrt = out_degree.pow(-0.5)

        # fill divergent values
        out_degree_inv_sqrt[out_degree_inv_sqrt == float('inf')] = 0

        # norm vector
        norm = out_degree_inv_sqrt[row] * out_degree_inv_sqrt[col]

        # 4. aggregate features
        out = self.propagate(edge_index, x=x, norm=norm)

        # apply final bias
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j of shape [E, out_channels]

        # this is called by propagate method. Here we normalise the node features
        return norm.view(-1, 1) * x_j
