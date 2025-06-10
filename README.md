---

# MovieRecGNN - Hybrid Graph Neural Network for Movie Recommendation

---

## Project Overview

MovieRecGNN is a graph-based recommendation system that uses a hybrid Graph Neural Network architecture combining NGCF and R-GCN. It leverages both user–movie interaction data and movie–entity relationships (actors, directors, genres) to generate personalized movie rankings.

---

## Features

* Heterogeneous graph construction from IMDb and MovieLens data
* NGCF message passing for user–movie interactions
* R-GCN for message passing between movies and associated entities
* BPR loss function tailored for implicit feedback recommendation
* Training loss and evaluation metrics automatically visualized

---

## Dataset

### IMDb Data (in `data/imdb/`)

* `title.basics.tsv` – movie metadata
* `title.ratings.tsv` – average ratings and vote counts
* `title.principals.tsv` – actor, director, and crew associations

### MovieLens 1M (in `data/ml-1m/`)

* `ratings.dat` – implicit feedback from user–movie ratings (parsed with `::` separator)

---

## Usage

### 1. Build the Graph

```
python build_graph.py --imdb_folder data/imdb --interactions data/ml-1m/ratings.dat
```
*Outputs movie_rec_graph.pt*

### 2. Train the Model

```
python train.py --imdb_folder data/imdb --interactions data/ml-1m/ratings.dat \
                --epochs 50 --save_path model.pt
```
*Outputs training_loss.png and model.pt*

### 3. Evaluate the Model

```
python eval.py --imdb_folder data/imdb --interactions data/ml-1m/ratings.dat \
               --model_path model.pt --k 10
```
*Outputs evaluation_metrics.png and prints HR@10, NDCG@10, Precision@10, Recall@10*

---

## Model Details

* NGCF: Neural Graph Collaborative Filtering for user–movie message passing
* R-GCN: Relational Graph Convolutional Network for movie–entity connections
* Loss Function: BPR (Bayesian Personalized Ranking), optimized for implicit feedback

---

## Graph Schema

### Nodes

* user: Represented by index; feature = constant (1.0)
* movie: Represented by IMDb ID; feature = one-hot genre vector
* entity: People associated with movies (actors, directors, etc.); feature = identity matrix

### Edges

* user → movie (`rates`): Interaction (positive feedback)
* movie → entity (`to_entity`): Metadata linkage (all roles treated as one relation)

---

## Evaluation

The model is evaluated on held-out user–movie interactions using top-k ranking metrics.

Results (k = 10):

* **HR\@10**: 0.63 — the correct movie appeared in the top 10 recommendations 63% of the time
* **NDCG\@10**: 0.41 — rewards higher-ranked correct predictions
* **Recall\@10**: 0.58 — 58% of relevant movies were recommended within the top 10
* **Precision\@10**: 0.21 — 21% of the top 10 predictions were correct on average

Plots:

![training_loss_curve](https://github.com/user-attachments/assets/2a594f59-ad31-43fb-8e06-0bfdd1946263)
![evaluation_metrics_chart](https://github.com/user-attachments/assets/a62b6800-d373-4f7d-b337-e76d2d2552ab)

---

## Future Work

To improve the model and expand beyond the current implementation:

* Implement multi-relation edge types: separate `acted_by`, `directed_by`, and `has_genre` edge labels
* Add edge-type-aware R-GCN or use HeteroConv for better message passing
* Encode temporal interactions or rating strength for sequential recommendations
* Evaluate with larger test splits and hybrid metrics (MAP, MRR)
* Extend node features for users (demographics) and entities (roles, popularity)

---
