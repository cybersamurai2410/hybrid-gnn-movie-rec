"""
eval.py

Evaluation script for MovieRecGNN with HR@10 and NDCG@10 visualization.
"""

import torch
import argparse
import matplotlib.pyplot as plt
from build_graph import build_graph
from model import MovieRecGNN
from utils import train_test_split

def hit_rate(preds, true_edges, k):
    """Compute Hit Rate@k."""
    hits = 0
    users, items = true_edges
    for u, i in zip(users.tolist(), items.tolist()):
        topk = preds[u][:k]
        if i in topk:
            hits += 1
    return hits / len(users)

def ndcg(preds, true_edges, k):
    """Compute NDCG@k."""
    import math
    dcg = 0.0
    users, items = true_edges
    for u, i in zip(users.tolist(), items.tolist()):
        topk = preds[u][:k]
        if i in topk:
            rank = topk.index(i) + 1
            dcg += 1 / math.log2(rank + 1)
    idcg = len(users) * (1 / math.log2(2))
    return dcg / idcg

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = build_graph(args.imdb_folder, args.interactions).to(device)

    # Load model
    model = MovieRecGNN(hidden_channels=args.hidden_dim,
                        num_relations=1,
                        num_entities=data['entity'].num_nodes).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Split edges
    pos_edge = data['user','rates','movie'].edge_index
    _, test_pos = train_test_split(pos_edge, test_ratio=0.2)
    user_emb, movie_emb = model(data)

    # Compute full score matrix and top-k
    scores = torch.matmul(user_emb, movie_emb.t())
    _, idx = torch.topk(scores, k=args.k, dim=1)
    preds = {u: idx[u].tolist() for u in range(idx.size(0))}

    hr = hit_rate(preds, test_pos, args.k)
    n10 = ndcg(preds, test_pos, args.k)
    print(f'HR@{args.k}: {hr:.4f}')
    print(f'NDCG@{args.k}: {n10:.4f}')

    # === PLOT METRICS ===
    metrics = {
        f'HR@{args.k}': hr,
        f'NDCG@{args.k}': n10
    }
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    print("Saved evaluation_metrics.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdb_folder', type=str, default='data/imdb')
    parser.add_argument('--interactions', type=str, default=None)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    main(args)
