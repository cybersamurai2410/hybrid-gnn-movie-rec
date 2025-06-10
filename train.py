"""
Training script for MovieRecGNN with training loss visualization.
"""

import torch
import argparse
import matplotlib.pyplot as plt
from build_graph import build_graph
from model import MovieRecGNN
from utils import negative_sampling, train_test_split

def main(args):
    # Load graph data
    data = build_graph(args.imdb_folder, args.interactions)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Prepare positive edges
    pos_edge = data['user','rates','movie'].edge_index.to(device)
    train_pos, _ = train_test_split(pos_edge, test_ratio=0.2)

    # Initialize model and optimizer
    model = MovieRecGNN(hidden_channels=args.hidden_dim,
                        num_relations=1,
                        num_entities=data['entity'].num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Training loop
    losses = []
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()
        user_emb, movie_emb = model(data)
        neg_edge = negative_sampling(train_pos, data['movie'].num_nodes, args.num_negatives).to(device)
        loss = model.loss(user_emb, movie_emb, train_pos, neg_edge)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % args.log_every == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f'Model saved to {args.save_path}')

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs+1), losses, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BPR Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("Saved training_loss.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdb_folder', type=str, default='data/imdb')
    parser.add_argument('--interactions', type=str, default=None)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_negatives', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='model.pt')
    args = parser.parse_args()
    main(args)
