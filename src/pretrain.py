import torch
from torch import nn
from src.gats import GNN   
from src.funcs import givens_rotation, euc_distance
from torch.nn import functional as F

class gnn_kge(nn.Module):
    """
    gnn_kge: finish pretrained embeddings using R-GCN like network and KGE
    """
    def __init__(self, graph, num_nodes, num_rels, hidden_dim, score_func, 
                 layer_num, num_rel, num_head, gnn='rgat', att_drop=0.2, fea_drop=0.2,):
        super().__init__()
        self.graph = graph
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.hidden_dim = hidden_dim
        self.ent_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.rel_embedding = nn.Embedding(self.num_rels * 2, self.hidden_dim)
        nn.init.xavier_normal_(self.ent_embedding.weight, gain=1.414)
        nn.init.xavier_normal_(self.rel_embedding.weight, gain=1.414)
        self.score_func = score_func

        self.gnn_model = GNN(hidden_dim, hidden_dim, layer_num, num_rel, num_head, gnn, att_drop, fea_drop)

    def gnn_forward(self, ):
        total_e = F.normalize(self.ent_embedding(self.graph.ndata['id'].squeeze(1)))
        self.graph.edata['r_h'] = self.rel_embedding(self.graph.edata['type'])
        new_feature = self.gnn_model(self.graph, total_e)
        return new_feature
        
    def forward(self, triples, new_feature):
        h = new_feature[triples[:, 0]]
        r = self.rel_embedding(triples[:, 1])
        t = new_feature
        if self.score_func == "rotate":
            target = givens_rotation(r, h)
            score = -euc_distance(target, t, eval_mode=True)
        return score
        