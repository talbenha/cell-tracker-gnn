from typing import Optional, Callable, List
import copy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, ReLU

from src.models.modules.pdn_conv import PDNConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from torch_geometric.typing import Adj

from src.models.modules.mlp import MLP

"""
Our implementation is based on the BasicGNN - An abstract class for implementing basic GNN models.
it is provided by PyTorch Geometric in:
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html
"""

class EedgePath_MPNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models dictated by the edges.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden and output sample.
        num_layers (int): Number of message passing layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 in_edge_channels: int, hidden_edge_channels: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last'):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.in_edge_channels = in_edge_channels
        self.hidden_edge_channels = hidden_edge_channels

        self.out_channels = hidden_channels
        if jk == 'cat':
            self.out_channels = num_layers * hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        from torch.nn.modules.distance import CosineSimilarity
        self.distance = CosineSimilarity()
        self.convs = ModuleList()
        self.fcs = ModuleList()

        self.jk = None
        if jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        self.norms = None
        if norm is not None:
            self.norms = ModuleList(
                [copy.deepcopy(norm) for _ in range(num_layers)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if self.jk is not None:
            self.jk.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_feat: Tensor, *args, **kwargs) -> Tensor:

        src, trg = edge_index
        xs: List[Tensor] = []
        edge_features: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_feat, *args, **kwargs)

            x_src, x_trg = x[src], x[trg]
            similar = self.distance(x_src, x_trg)
            edge_feat = torch.cat((edge_feat, x_src, x_trg, torch.abs(x_src - x_trg), similar[:, None]), dim=-1)
            edge_feat = self.fcs[i](edge_feat)

            if self.norms is not None:
                x = self.norms[i](x)

            if self.act is not None:
                x = self.act(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_feat = F.dropout(edge_feat, p=self.dropout, training=self.training)

            if self.jk is not None:
                xs.append(x)

            if self.jk is not None:
                edge_features.append(edge_feat)

        return edge_feat

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class CellTrack_GNN(EedgePath_MPNN):
    def __init__(self,
                 in_channels: int, hidden_channels: int,
                 in_edge_channels: int, hidden_edge_channels_linear: int,
                 hidden_edge_channels_conv: int,
                 num_layers: int,
                 num_nodes_features: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels,
                         in_edge_channels, hidden_edge_channels_linear,
                         num_layers, dropout,
                         act, norm, jk)
        assert in_edge_channels == hidden_edge_channels_linear[-1]
        in_edge_dims = in_edge_channels + num_nodes_features * in_channels + 1
        self.convs.append(PDNConv(in_channels, hidden_channels, in_edge_channels,
                                  hidden_edge_channels_conv, **kwargs))
        self.fcs.append(MLP(in_edge_dims, hidden_edge_channels_linear, dropout_p=dropout))
        for _ in range(1, num_layers):
            self.convs.append(
                PDNConv(hidden_channels, hidden_channels, in_edge_channels,
                        hidden_edge_channels_conv, **kwargs))
            self.fcs.append(MLP(in_edge_dims, hidden_edge_channels_linear, dropout_p=dropout))
