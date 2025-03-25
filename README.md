# Directed Graph Convolutional Network based Hierarchical Heterogeneous Citation Neural Network

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
git clone https://github.com/juyujing/CitationNN.git
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
--with_uu_co_venue 0 --with_ii_co_venue 0 --gpu_id 0 --gnn CitationNN --batch_size 1024 --alpha 0.8 --fast_test 1
```
or
```bash
python -u main.py --seed 2021 --context_hops 1 --dataset dblp --lr 0.001 --l2 1e-6  --att_dropout 1 --pool sum --with_sim 0 --with_user 0 --gpu_id 0 --gnn CitationNN --batch_size 1024 --alpha 0.95
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
- `--alpha 0.8`: Hyperparameters for control node aggregation.
- `--fast_test 1` : Activate fast test function.


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

## Evaluation
- **Compared against baseline models**, our approach significantly improves recall and precision, especially for users in emerging research fields.
- Supports **early stopping** to prevent overfitting.

## Results
### Average Results
- In the same task, our model is **better than SOTA models** within multiple metrics. The following is the average results of three experiments.


#### CiteULike

| Phase | Epoch        | Training Time (s) | Testing Time (s) | Loss | Recall                           | NDCG                              | Hit Ratio                        | Precision                         |
|-------|-------------|------------------|----------------|------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| Test  | 155/155/170 | 12.14            | 10.89         | 0.62 | [0.01825129 0.03211132 0.05438279] | [0.01516445 0.02093783 0.02855892] | [0.04485678 0.07728337 0.12928602] | [0.01006425 0.00912749 0.00794452] |

#### DBLP

| Phase | Training Time (s) | Testing Time (s) | Loss | Recall                           | NDCG                              | Hit Ratio                        | Precision                         |
|-------|-------------------|------------------|------|----------------------------------|-----------------------------------|----------------------------------|-----------------------------------|
| Test  | 128.15            | 241.22           | 0.08 | [0.15298944 0.22587824 0.31121094] | [0.12457147 0.15429006 0.18144566] | [0.26150776 0.37649007 0.49956339] | [0.05425459 0.04204725 0.03034261] |   

All the results are average values of three experiment.

### Comparison

#### CiteULike

| Metric    | SOTA Model(MCAP) | Ours  | Improvement (%) |
|-----------|------------------|-------|----------------|
| Recall@5  | 1.51             | 1.81  | +19.24%        |
| Recall@10 | 2.65             | 2.96  | +11.89%        |
| Recall@20 | 4.65             | 5.21  | +12.14%        |
| NDCG@5    | 1.23             | 1.45  | +17.57%        |
| NDCG@10   | 1.72             | 1.95  | +13.19%        |
| NDCG@20   | 2.43             | 2.77  | +13.77%        |
| HR@5      | 3.66             | 4.15  | +13.30%        |
| HR@10     | 6.38             | 6.93  | +8.54%         |
| HR@20     | 10.63            | 11.62 | +9.27%         |

#### DBLP

| Metric    | SOTA Model(MCAP) | Ours  | Improvement (%)|
|-----------|------------------|-------|----------------|
| Recall@5  | 14.17            | 1.81  | +7.9%          |
| Recall@10 | 21.79            | 2.96  | +3.6%          |
| Recall@20 | 31.11            | 5.21  | +0.0%          |
| NDCG@5    | 11.22            | 1.45  | +10.9%         |
| NDCG@10   | 14.24            | 1.95  | +8.3%          |
| NDCG@20   | 17.17            | 2.77  | +5.6%          |
| HR@5      | 25.86            | 4.15  | +1.12%         |
| HR@10     | 30.43            | 6.93  | +0.6%          |
| HR@20     | 50.05            | 11.62 | -0.1%          |


## Modifications and Improvements
This project is based on the original work by ZhuYifan (2022).  
Significant modifications and optimizations were made by Yujing Ju (2025), leading to improved performance and accuracy.

## Contact
For questions or contributions, open an issue on GitHub or contact **Yujing Ju** at **yj2012@hw.ac.uk**(Before May 2, 2025).
Yujing Ju will start his Doctor of Philosophy in Medical Science at University of Florida in fall 2025. Please contact him at **yujingju@ufl.edu**(After April 29, 2025).
