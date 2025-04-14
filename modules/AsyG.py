import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class AsymG(nn.Module):
    def __init__(self, in_dim, hidden_dim, node_embeddings, embed_dim):
        """
        in_dim: 輸入特徵維度
        hidden_dim: 隱藏層維度
        node_embeddings: 節點嵌入，形狀 (num_nodes, embed_dim)
        embed_dim: 嵌入維度
        """
        super(AsymG, self).__init__()
        self.node_embeddings = node_embeddings
        self.embed_dim = embed_dim
        
        # 線性變換
        self.pos_W = nn.Linear(in_dim, hidden_dim)
        self.neg_W = nn.Linear(in_dim, hidden_dim)
        self.self_W = nn.Linear(in_dim, hidden_dim)
        
        # 正向邊參數
        self.w_pos_beta = nn.Parameter(torch.randn(embed_dim) * 0.01)
        self.W_pos_u = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.alpha_pos = nn.Parameter(torch.tensor(1.0))
        
        # 反向邊參數
        self.w_neg_beta = nn.Parameter(torch.randn(embed_dim) * 0.01)
        self.W_neg_u = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.alpha_neg = nn.Parameter(torch.tensor(1.0))
        
        self.eps = 1e-6
    
    def compute_finsler_distance(self, src_idx, dst_idx, edge_type):
        """
        計算 d_Fk(j, i) = ||x_i - x_j||_2 + beta_k(x_j) <x_i - x_j, W_k^u x_j>
        """
        x_j = self.node_embeddings[src_idx]
        x_i = self.node_embeddings[dst_idx]
        diff = x_i - x_j
        
        # 歐氏距離
        euclidean = torch.norm(diff, dim=-1, p=2)
        
        # 參數選擇
        if edge_type == 'pos':
            w_beta = self.w_pos_beta
            W_u = self.W_pos_u
            alpha_k = torch.clamp(self.alpha_pos, min=0.1, max=10.0)
        else:
            w_beta = self.w_neg_beta
            W_u = self.W_neg_u
            alpha_k = torch.clamp(self.alpha_neg, min=0.1, max=10.0)
        
        # beta_k(x_j) = tanh(w_beta . x_j)
        beta_k = torch.tanh(torch.einsum('ij,j->i', x_j, w_beta))
        
        # u_k(x_j) = W_u x_j
        u_j = torch.matmul(x_j, W_u)
        
        # 非對稱項
        asymmetric = torch.einsum('ij,ij->i', diff, u_j)
        
        # Finsler 距離
        d_Fk = euclidean + beta_k * asymmetric
        
        # 權重
        weight = torch.exp(-alpha_k * d_Fk.clamp(min=0))
        return weight
    
    def forward(self, graph, h):
        """
        graph: 包含 'pos' 和 'neg' 子圖
        h: 輸入特徵 (num_nodes, in_dim)
        """
        pos_graph = graph['pos']
        neg_graph = graph['neg']
        
        # 正向邊
        pos_src, pos_dst = pos_graph.edges()
        pos_weights = self.compute_finsler_distance(pos_src, pos_dst, 'pos')
        pos_msg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        for u, v, w in zip(pos_src, pos_dst, pos_weights):
            pos_msg[v] += w * self.pos_W(h[u])
        
        # 反向邊
        neg_src, neg_dst = neg_graph.edges()
        neg_weights = self.compute_finsler_distance(neg_src, neg_dst, 'neg')
        neg_msg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        for u, v, w in zip(neg_src, neg_dst, neg_weights):
            neg_msg[v] += w * self.neg_W(h[u])
        
        # 自身特徵
        self_msg = self.self_W(h)
        
        # 聚合並激活
        out = F.relu(pos_msg + neg_msg + self_msg)
        return out
    
    def regularization_loss(self):
        """正則化損失"""
        reg = (torch.norm(self.w_pos_beta, p=2)**2 + torch.norm(self.w_neg_beta, p=2)**2 +
               torch.norm(self.W_pos_u, p='fro')**2 + torch.norm(self.W_neg_u, p='fro')**2)
        return reg
