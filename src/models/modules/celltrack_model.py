import torch
import torch.nn as nn
from torch.nn.modules.distance import CosineSimilarity

from src.models.modules.mlp import MLP
import src.models.modules.edge_mpnn as edge_mpnn


class CellTrack_Model(nn.Module):
    def __init__(self,
                 hand_NodeEncoder_dic={},
                 learned_NodeEncoder_dic={},
                 intialize_EdgeEncoder_dic={},
                 message_passing={},
                 edge_classifier_dic={}
                 ):
        super(CellTrack_Model, self).__init__()
        self.distance = CosineSimilarity()
        self.handcrafted_node_embedding = MLP(**hand_NodeEncoder_dic)
        self.learned_node_embedding = MLP(**learned_NodeEncoder_dic)
        self.learned_edge_embedding = MLP(**intialize_EdgeEncoder_dic)

        edge_mpnn_class = getattr(edge_mpnn, message_passing.target)
        self.message_passing = edge_mpnn_class(**message_passing.kwargs)

        self.edge_classifier = MLP(**edge_classifier_dic)

    def forward(self, x, edge_index, edge_feat):
        x1, x2 = x
        x_init = torch.cat((x1, x2), dim=-1)
        src, trg = edge_index
        similarity1 = self.distance(x_init[src], x_init[trg])
        abs_init = torch.abs(x_init[src] - x_init[trg])
        x1 = self.handcrafted_node_embedding(x1)
        x2 = self.learned_node_embedding(x2)
        x = torch.cat((x1, x2), dim=-1)
        src, trg = edge_index
        similarity2 = self.distance(x[src], x[trg])
        edge_feat_in = torch.cat((abs_init, similarity1[:, None], x[src], x[trg], torch.abs(x[src] - x[trg]), similarity2[:, None]), dim=-1)
        edge_init_features = self.learned_edge_embedding(edge_feat_in)
        edge_feat_mp = self.message_passing(x, edge_index, edge_init_features)
        pred = self.edge_classifier(edge_feat_mp).squeeze()
        return pred