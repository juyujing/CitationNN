class AsymG(nn.Module):
    def __init__(self, in_dim, hidden_dim, node_embeddings, beta_pos=0.1, beta_neg=-0.1):
        super(AsymG, self).__init__()
        self.node_embeddings = node_embeddings
        self.pos_W = nn.Linear(in_dim, hidden_dim)
        self.neg_W = nn.Linear(in_dim, hidden_dim)
        self.self_W = nn.Linear(in_dim, hidden_dim)
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg
    
    def compute_finsler_distance(self, src_idx, dst_idx, edge_type):
        x_j = self.node_embeddings[src_idx]
        x_i = self.node_embeddings[dst_idx]
        diff = x_i - x_j
        euclidean = torch.norm(diff, dim=-1, p=2)
        u_k = torch.randn(self.node_embeddings.size(1), device=diff.device)
        asymmetric = torch.einsum('ij,j->i', diff, u_k)
        beta = self.beta_pos if edge_type == 'pos' else self.beta_neg
        d_Fk = euclidean + beta * asymmetric
        return torch.exp(-d_Fk)
    
    def forward(self, graph, h):
        pos_graph = graph['pos']
        neg_graph = graph['neg']
        
        pos_src, pos_dst = pos_graph.edges()
        neg_src, neg_dst = neg_graph.edges()
        pos_d = self.compute_finsler_distance(pos_src, pos_dst, 'pos')
        neg_d = self.compute_finsler_distance(neg_src, neg_dst, 'neg')
        
        # 特徵相似性
        pos_sim = torch.sigmoid(torch.sum(h[pos_src] * h[pos_dst], dim=-1))
        neg_sim = torch.sigmoid(torch.sum(h[neg_src] * h[neg_dst], dim=-1))
        
        pos_weights = pos_d * pos_sim
        neg_weights = neg_d * neg_sim
        
        pos_msg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        for u, v, w in zip(pos_src, pos_dst, pos_weights):
            pos_msg[v] += w * self.pos_W(h[u])
        
        neg_msg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        for u, v, w in zip(neg_src, neg_dst, neg_weights):
            neg_msg[v] += w * self.neg_W(h[u])
        
        self_msg = self.self_W(h)
        out = F.relu(pos_msg + neg_msg + self_msg)
        return out
