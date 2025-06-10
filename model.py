"""
model.py

Defines the NGCF + R-GCN hybrid model for MovieRecGNN using PyTorch Geometric.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, RGCNConv
from torch_geometric.data import HeteroData

class NGCFMessagePassing(MessagePassing):
    """
    NGCF-style message passing for user-item bipartite graph.
    """
    def __init__(self, in_channels, out_channels):
        super(NGCFMessagePassing, self).__init__(aggr='add')
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # Bi-interaction between embeddings
        bi = x_i * x_j
        return self.lin1(x_j) + self.lin2(bi)

class MovieRecGNN(torch.nn.Module):
    """
    Hybrid NGCF + R-GCN model for movie recommendation.
    """
    def __init__(self, hidden_channels, num_relations, num_entities):
        super(MovieRecGNN, self).__init__()
        self.hidden_channels = hidden_channels
        # NGCF for user->movie interactions
        self.ngcf = NGCFMessagePassing(hidden_channels, hidden_channels)
        # R-GCN for movie->entity relations
        self.rgcn = RGCNConv(hidden_channels, hidden_channels, num_relations)
        # Final embedding size (concatenation of original + message)
        self.emb_dim = hidden_channels * 2

    def forward(self, data: HeteroData):
        # Node features
        user_x = data['user'].x
        movie_x = data['movie'].x
        entity_x = data['entity'].x

        # NGCF message passing on user->movie edges
        um_edge_index = data['user', 'rates', 'movie'].edge_index
        movie_msg = self.ngcf(movie_x, um_edge_index)

        # R-GCN on movie->entity edges (treat all as single relation)
        me_edge_index = data['movie', 'to_entity', 'entity'].edge_index
        edge_type = torch.zeros(me_edge_index.size(1), dtype=torch.long, device=movie_x.device)
        entity_msg = self.rgcn(entity_x, me_edge_index, edge_type)

        # Combine embeddings
        # For movies: original + aggregated messages
        movie_emb = torch.cat([movie_x, movie_msg + entity_msg[:movie_msg.size(0)]], dim=1)
        # For users: original + NGCF from reverse edges
        um_rev = torch.stack([um_edge_index[1], um_edge_index[0]], dim=0)
        user_msg = self.ngcf(user_x, um_rev)
        user_emb = torch.cat([user_x, user_msg], dim=1)

        return user_emb, movie_emb

    def loss(self, user_emb, movie_emb, pos_edge_index, neg_edge_index):
        """
        BPR loss for implicit feedback.
        pos_edge_index: [2, num_pos] user->positive movie
        neg_edge_index: [2, num_neg] user->negative movie
        """
        u_pos = user_emb[pos_edge_index[0]]
        m_pos = movie_emb[pos_edge_index[1]]
        u_neg = user_emb[neg_edge_index[0]]
        m_neg = movie_emb[neg_edge_index[1]]

        pos_score = (u_pos * m_pos).sum(dim=1)
        neg_score = (u_neg * m_neg).sum(dim=1)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss
