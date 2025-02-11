# README: Directed Graph Convolutional Network based Hierarchical Heterogeneous Citation Neural Network

## Overview
CitationNN is a graph neural network-based recommendation system designed for academic paper recommendations. Unlike traditional recommendation models that primarily use paper similarity, co-authorship, and click-through rates, CitationNN integrates citation networks, author-publication relations, and academic relationships to build a two-layer heterogeneous graph neural network (HGNN). This architecture combines a heterogeneous graph neural network (HGNN) with a stacked directed graph neural network (D-GNN) to improve the recommendation quality for users with low publication counts and in emerging research fields.

### Key Innovations:
- **Directed Citation Graph Neural Network (CitationNN)**: Models the citation flow by treating citations as directed edges, helping expand recommendations beyond highly cited or well-connected papers.
- **Hierarchical Heterogeneous Network (HHN)**: Constructs a hybrid model where one layer is based on traditional collaborative filtering and the second layer captures directed citation relationships.
- **Improved Recommendation for Emerging Fields**: Ensures that new research areas are adequately represented in recommendations, benefiting users exploring niche topics.

## Installation
### Clone the Repository
```bash
# Replace with your actual repository link
git clone https://github.com/juyujing/citationnn.git
cd CitationNN
```

### Set Up the Environment
Use **Conda** to create and activate the environment:
```bash
conda env create -f environment.yml
conda activate CitationNN
```

## Running the Model
To execute the model, use the following command:
```bash
python -u main.py --seed 2021 --dataset citeulike --att_dropout 1 --step 5 \
--lr 0.001 --l2 1e-6 --pool sum --load_ii_sort 50 --context_hops 1 \
--with_sim 1 --with_user 1 --e 0.001 --with_uu_co_author 0 --with_ii_co_author 0 \
--with_uu_co_venue 0 --with_ii_co_venue 0 --gpu_id 0 --gnn CitationNN --batch_size 1024
```

### Explanation of Arguments
- `--seed 2021`: Random seed for reproducibility.
- `--dataset citeulike`: Specifies the dataset (CiteULike in this case).
- `--att_dropout 1`: Attention dropout rate.
- `--step 5`: Number of steps for training.
- `--lr 0.001`: Learning rate.
- `--l2 1e-6`: L2 regularization coefficient.
- `--pool sum`: Pooling strategy (`sum`, `mean`, etc.).
- `--load_ii_sort 50`: Item-item similarity sort threshold.
- `--context_hops 1`: Number of hops in the GNN model.
- `--with_sim 1`: Whether to include similarity-based recommendations.
- `--with_user 1`: Whether to include user-based recommendations.
- `--e 0.001`: Some regularization parameter.
- `--with_uu_co_author 0`: Disables co-authorship-based user-user relationships.
- `--with_ii_co_author 0`: Disables co-authorship-based item-item relationships.
- `--with_uu_co_venue 0`: Disables co-venue-based user-user relationships.
- `--with_ii_co_venue 0`: Disables co-venue-based item-item relationships.
- `--gpu_id 0`: GPU ID (set to -1 for CPU).
- `--gnn CitationNN`: Specifies the model architecture.
- `--batch_size 1024`: Batch size for training.


## Model Structure
### **1. Graph Construction**
- **Citation Graph**: Directed citation network with two edge types:
  - `cites`: Paper A cites Paper B
  - `cited_by`: Paper B is cited by Paper A
- **Collaborative Filtering Graph**: Based on traditional item-item and user-user interactions.

### **2. Graph Neural Network (GNN) Components**
- **GraphConv Layer** (Traditional GNN for user-item interaction)
- **DirGCN Layer** (Directed Citation Graph Convolution)
  - Captures directional paper citation influence
  - Uses separate convolution filters for `cites` and `cited_by` edges
  - Combines results using learnable parameters and skip connections

## Results and Evaluation
- **Compared against baseline models**, our approach significantly improves recall and precision, especially for users in emerging research fields.
- Supports **early stopping** to prevent overfitting.

## Modifications and Improvements
This project is based on the original work by ZhuYifan (2022).  
Significant modifications and optimizations were made by Yujing Ju (2025), leading to improved performance and accuracy.

## Contact
For questions or contributions, open an issue on GitHub or contact **Yujing Ju** at **yj2012@hw.ac.uk**.