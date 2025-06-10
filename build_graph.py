"""
Script to load IMDb data and construct a heterogeneous graph for MovieRecGNN.
Nodes: 'user', 'movie', 'entity'
Edges: user->movie, movie->entity.
Output: PyTorch Geometric HeteroData saved to disk.
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os

def build_graph(imdb_folder, interactions_file=None):
    """Load IMDb TSVs and build HeteroData graph."""
    # Load IMDb data files
    titles = pd.read_csv(os.path.join(imdb_folder, 'title.basics.tsv'), sep='\t', dtype=str, na_values='\\N')
    ratings = pd.read_csv(os.path.join(imdb_folder, 'title.ratings.tsv'), sep='\t', dtype={'averageRating': float, 'numVotes': int})
    principals = pd.read_csv(os.path.join(imdb_folder, 'title.principals.tsv'), sep='\t', dtype=str, na_values='\\N')

    # Filter only movies
    movies = titles[titles['titleType'] == 'movie'].copy()
    movies = movies.merge(ratings, on='tconst', how='left').fillna({'averageRating': 0, 'numVotes': 0})

    # Create ID mappings
    movie_ids = movies['tconst'].tolist()
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    # Process entities (actors, directors, etc.)
    entity_ids = principals['nconst'].unique().tolist()
    entity2idx = {eid: idx for idx, eid in enumerate(entity_ids)}

    # Genres extraction
    genres = set()
    for g in movies['genres'].dropna():
        genres.update(g.split(','))
    genre_ids = sorted(list(genres))
    genre2idx = {g: idx for idx, g in enumerate(genre_ids)}

    # Initialize HeteroData
    data = HeteroData()

    # Movie node features: one-hot genre vectors
    num_movies = len(movie_ids)
    num_genres = len(genre_ids)
    movie_feats = torch.zeros((num_movies, num_genres), dtype=torch.float)
    for m, g_str in zip(movie_ids, movies['genres']):
        if not pd.isna(g_str):
            for g in g_str.split(','):
                movie_feats[movie2idx[m], genre2idx[g]] = 1.0
    data['movie'].x = movie_feats

    # Entity node features: identity matrix (one-hot)
    num_entities = len(entity_ids)
    data['entity'].x = torch.eye(num_entities, dtype=torch.float)

    # User nodes and interactions 
    if interactions_file:
        # MovieLens 1M has '::' separator with no headers
        interactions = pd.read_csv(
            interactions_file,
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        interactions = interactions[interactions['rating'] >= 4]  # Liked movies with ratings >= 4 
    
        user_ids = interactions['user_id'].unique().tolist()
        user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
        data['user'].x = torch.ones((len(user_ids), 1), dtype=torch.float)

        # user->movie edge_index
        u_idx = [user2idx[u] for u in interactions['user_id']]
        m_idx = [movie2idx[m] for m in interactions['movie_id']]
        data['user', 'rates', 'movie'].edge_index = torch.tensor([u_idx, m_idx], dtype=torch.long)
    else:
        data['user'].x = torch.zeros((0, 1), dtype=torch.float)
        data['user', 'rates', 'movie'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # movie->entity edges from principals
    src, dst = [], []
    for _, row in principals.iterrows():
        m, e = row['tconst'], row['nconst']
        if m in movie2idx and e in entity2idx:
            src.append(movie2idx[m])
            dst.append(entity2idx[e])
    data['movie', 'to_entity', 'entity'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    return data

if __name__ == '__main__':
    imdb_folder = '../data/imdb'
    interactions_file = '../data/ml-1m/ratings.dat'
    data = build_graph(imdb_folder, interactions_file)
    torch.save(data, 'movie_rec_graph.pt')
    print('HeteroData graph saved to movie_rec_graph.pt')
