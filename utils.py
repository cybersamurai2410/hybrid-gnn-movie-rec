"""
utils.py

Utility functions for negative sampling and train/test split.
"""

import torch
import random

def negative_sampling(pos_edge_index, num_movies, num_negatives=4):
    """
    Generate negative samples for each positive edge.
    Returns edge_index of shape [2, num_pos * num_negatives].
    """
    users = pos_edge_index[0].tolist()
    neg_u, neg_m = [], []
    for u in users:
        for _ in range(num_negatives):
            neg_u.append(u)
            neg_m.append(random.randrange(num_movies))
    return torch.tensor([neg_u, neg_m], dtype=torch.long)

def train_test_split(pos_edge_index, test_ratio=0.2):
    """
    Split positive edges into train and test sets.
    """
    num_edges = pos_edge_index.size(1)
    perm = torch.randperm(num_edges)
    test_size = int(num_edges * test_ratio)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    train_edges = pos_edge_index[:, train_idx]
    test_edges = pos_edge_index[:, test_idx]
    return train_edges, test_edges
