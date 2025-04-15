import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

class FinslerAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, beta_pos=0.1, beta_neg=-0.1):
        """
        input_dim: 節點特徵維度
        embed_dim: 節點嵌入維度
        beta_pos, beta_neg: 正向和反向邊的非對稱參數
        """
        super(FinslerAttention, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 注意力參數
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim]))
        
        # 非對稱參數
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg
        # 方向向量 u_k
        self.u_pos = nn.Parameter(torch.randn(embed_dim))
        self.u_neg = nn.Parameter(torch.randn(embed_dim))
        
    def compute_finsler_distance(self, node_embeddings, src_idx, dst_idx, edge_type):
        """
        計算 Finsler 距離 d_Fk(j, i)
        """
        x_j = node_embeddings[src_idx]
        x_i = node_embeddings[dst_idx]
        diff = x_i - x_j
        euclidean = torch.norm(diff, dim=-1, p=2)
        beta = self.beta_pos if edge_type == 'pos' else self.beta_neg
        u_k = self.u_pos if edge_type == 'pos' else self.u_neg
        asymmetric = torch.einsum('ij,j->i', diff, u_k)
        d_Fk = euclidean + beta * asymmetric
        return d_Fk
    
    def forward(self, g, h, node_embeddings, edge_type):
        """
        g: DGLGraph ('pos' 或 'neg')
        h: 節點特徵 (num_nodes, input_dim)
        node_embeddings: 固定嵌入 (num_nodes, embed_dim)
        edge_type: 'pos' 或 'neg'
        返回注意力權重
        """
        with g.local_scope():
            src_idx, dst_idx = g.edges()
            Q = self.query(h)  # (num_nodes, input_dim)
            K = self.key(h)    # (num_nodes, input_dim)
            
            # 計算 Finsler 距離
            d_Fk = self.compute_finsler_distance(node_embeddings, src_idx, dst_idx, edge_type)
            f_d = torch.exp(-d_Fk)  # f(z) = exp(-z)
            
            # 計算注意力分數
            Q_src = Q[src_idx]  # (num_edges, input_dim)
            K_dst = K[dst_idx]  # (num_edges, input_dim)
            attention_scores = f_d * torch.sum(Q_src * K_dst, dim=-1) / self.scale
            
            g.edata['score'] = attention_scores
            # 按目標節點 softmax
            g.apply_edges(lambda edges: {
                'att_weight': torch.softmax(edges.data['score'], dim=0)
            })
            return g.edata['att_weight']

class AsymG(nn.Module):
    def __init__(self, in_dim, hidden_dim, node_embeddings, beta_pos=0.1, beta_neg=-0.1):
        """
        in_dim: 輸入特徵維度
        hidden_dim: 隱藏層維度
        node_embeddings: 節點固定嵌入 (num_nodes, embed_dim)
        """
        super(AsymG, self).__init__()
        self.node_embeddings = node_embeddings
        self.pos_att = FinslerAttention(in_dim, node_embeddings.size(1), beta_pos)
        self.neg_att = FinslerAttention(in_dim, node_embeddings.size(1), beta_neg)
        
        self.pos_W = nn.Linear(in_dim, hidden_dim)
        self.neg_W = nn.Linear(in_dim, hidden_dim)
        self.self_W = nn.Linear(in_dim, hidden_dim)
    
    def forward(self, graph, h):
        """
        graph: 包含 'pos' 和 'neg' 的字典
        h: 輸入特徵 (num_nodes, in_dim)
        """
        # 正向邊
        pos_graph = graph['pos']
        pos_att = self.pos_att(pos_graph, h, self.node_embeddings, 'pos')
        pos_msg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        for u, v, w in zip(*pos_graph.edges(), pos_att):
            pos_msg[v] += w * self.pos_W(h[u])
        
        # 反向邊
        neg_graph = graph['neg']
        neg_att = self.neg_att(neg_graph, h, self.node_embeddings, 'neg')
        neg_msg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        for u, v, w in zip(*neg_graph.edges(), neg_att):
            neg_msg[v] += w * self.neg_W(h[u])
        
        # 自身特徵
        self_msg = self.self_W(h)
        
        # 聚合並激活
        out = F.relu(pos_msg + neg_msg + self_msg)
        return out
