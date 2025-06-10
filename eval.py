"""
Evaluation script for MovieRecGNN with HR@k, NDCG@k, Precision@k, and Recall@k + visualization.
"""

import torch
import argparse
import matplotlib.pyplot as plt
from build_graph import build_graph
from model import MovieRecGNN
from utils import train_test_split

def hit_rate(preds, true_edges, k):
    hits = 0
    users, items = true_edges
    for u, i in zip(users.tolist(), items.tolist()):
        if i in preds[u][:k]:
            hits += 1
    return hits / len(users)

def ndcg(preds, true_edges, k):
    import math
    dcg = 0.0
    users, items = true_edges
    for u, i in zip(users.tolist(), items.tolist()):
        if i in preds[u][:k]:
            rank = preds[u][:k].index(i) + 1
            dcg += 1 / math.log2(rank + 1)
    idcg = len(users) * (1 / math.log2(2))
    return dcg / idcg

def precision(preds, true_edges, k):
    users, items = true_edges
    correct = 0
    for u, i in zip(users.tolist(), items.tolist()):
        if i in preds[u][:k]:
            correct += 1
    return correct / (len(users) * k)

def recall(preds, true_edges, k):
    users, items = true_edges
    correct = 0
    for u, i in zip(users.tolist(), items.tolist()):
        if i in preds[u][:k]:
            correct += 1
    return correct / len(users)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = build_graph(args.imdb_folder, args.interactions).to(device)

    # Load model
    model = MovieRecGNN(hidden_channels=args.hidden_dim,
                        num_relations=1,
                        num_entities=data['entity'].num_nodes).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Split test edges
    pos_edge = data['user','rates','movie'].edge_index
    _, test_pos = train_test_split(pos_edge, test_ratio=0.2)
    user_emb, movie_emb = model(data)

    # Prediction scores
    scores = torch.matmul(user_emb, movie_emb.t())
    _, idx = torch.topk(scores, k=args.k, dim=1)
    preds = {u: idx[u].tolist() for u in range(idx.size(0))}

    # Compute metrics
    hr = hit_rate(preds, test_pos, args.k)
    n10 = ndcg(preds, test_pos, args.k)
    prec = precision(preds, test_pos, args.k)
    rec = recall(preds, test_pos, args.k)

    print(f'HR@{args.k}: {hr:.4f}')
    print(f'NDCG@{args.k}: {n10:.4f}')
    print(f'Precision@{args.k}: {prec:.4f}')
    print(f'Recall@{args.k}: {rec:.4f}')

    # Plot metrics
    metrics = {
        f'HR@{args.k}': hr,
        f'NDCG@{args.k}': n10,
        f'Precision@{args.k}': prec,
        f'Recall@{args.k}': rec
    }

    plt.figure(figsize=(7, 5))
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
