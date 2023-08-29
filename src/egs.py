import torch
from torch import nn
import torch.nn.functional as F
from src.rrgcn import RGCNCell, RecurrentRGCN
from src.gats import GNN
from src.decoder import ConvTransE, ConvTransR


class EGS(nn.Module):
    def __init__(self, graph, global_gnn, global_layer_num, global_heads, num_nodes, num_rels, 
                hidden_dim, task, entity_prediction, relation_prediction, fuse, r_fuse, 
                num_bases, num_basis, evolve_layer_num, dropout, self_loop, skip_connect,
                encoder, decoder, opn, layer_norm, use_cuda, gpu, analysis):
        super().__init__()
        self.total_graph = graph
        self.global_gnn = global_gnn
        self.global_layer_num = global_layer_num
        self.num_nodes = num_nodes
        self.num_heads = global_heads
        self.num_rels = num_rels
        self.hidden_dim = hidden_dim
        self.task = task
        self.entity_prediction = entity_prediction
        self.relation_prediction = relation_prediction
        self.fuse = fuse
        self.r_fuse = r_fuse
        self.ent_global_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.ent_evolve_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.rel_global_embedding = nn.Embedding(self.num_rels * 2, self.hidden_dim)
        self.rel_evolve_embedding = nn.Embedding(self.num_rels * 2, self.hidden_dim)

        nn.init.xavier_normal_(self.ent_global_embedding.weight, gain=1.414)
        nn.init.xavier_normal_(self.ent_evolve_embedding.weight, gain=1.414)
        nn.init.xavier_normal_(self.rel_global_embedding.weight, gain=1.414)
        nn.init.xavier_normal_(self.rel_evolve_embedding.weight, gain=1.414)
        
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.evolve_layer_num = evolve_layer_num
        self.dropout = dropout
        self.encoder = encoder
        self.decoder = decoder
        self.layer_norm = layer_norm
        self.opn = opn
        self.use_cuda = use_cuda
        self.func = nn.CrossEntropyLoss()

        self.loss_r = nn.CrossEntropyLoss()
        self.loss_e = nn.CrossEntropyLoss()

        self.global_model = GNN(self.hidden_dim, self.hidden_dim, self.global_layer_num, self.num_heads, self.global_gnn, att_drop=0.2, fea_drop=0.2)
        self.evolve_model = RecurrentRGCN(
                decoder,
                encoder,
                num_nodes,
                num_rels,
                hidden_dim,
                opn,
                num_bases=num_bases,
                num_basis=num_basis,
                num_hidden_layers=evolve_layer_num,
                dropout=dropout,
                self_loop=self_loop,
                skip_connect=skip_connect,
                layer_norm=layer_norm,
                input_dropout=dropout,
                hidden_dropout=dropout,
                feat_dropout=dropout,
                entity_prediction=entity_prediction,
                relation_prediction=relation_prediction,
                use_cuda=use_cuda,
                gpu = gpu,
                analysis=analysis
            )
        
        self.time_gate_weight = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(self.hidden_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.relation_evolve_cell = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim)

        if self.fuse == 'con':
            self.linear_fuse = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)
        elif self.fuse == 'att':
            self.linear_l = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.linear_s = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        elif self.fuse == 'att1':
            self.linear_l = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.linear_s = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.fuse_f = nn.Linear(self.hidden_dim, 1, bias=True)
        elif self.fuse == 'gate':
            self.gate = GatingMechanism(self.num_nodes, self.hidden_dim)
        else:
            print('no fuse function')
        if self.r_fuse == 'con':
            self.linear_fuse_r = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)
        elif self.r_fuse == 'att1':
            self.linear_l_r = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.linear_s_r = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.fuse_f_r = nn.Linear(self.hidden_dim, 1, bias=True)
        elif self.r_fuse == 'gate':
            self.gate_r = GatingMechanism(self.num_rels *2 , self.hidden_dim)
        else:
            print('no fuse_r function')

        self.decoder_ob = ConvTransE(num_nodes, hidden_dim, self.dropout, self.dropout, self.dropout)
        self.rdecoder = ConvTransR(num_rels, hidden_dim, self.dropout, self.dropout, self.dropout)

    def global_forward(self, global_graph):
        total_e = F.normalize(self.ent_global_embedding(global_graph.ndata['id'].squeeze(1)))
        global_graph.edata['r_h'] = self.rel_global_embedding(global_graph.edata['type'])
        new_features = F.normalize(self.global_model(global_graph, total_e))
        return None


    def forward(self, input_list, global_graph, triples):
        """
        input_list: history_glist
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        total_e = F.normalize(self.ent_global_embedding(global_graph.ndata['id'].squeeze(1)))
        global_graph.edata['r_h'] = self.rel_global_embedding(global_graph.edata['type'])
        new_features = F.normalize(self.global_model(global_graph, total_e))

        
        self.evolve_model.dynamic_emb = self.ent_evolve_embedding.weight
        self.evolve_model.emb_rel = self.rel_evolve_embedding.weight
        # evolve embeddings forward
        evolve_embs, static_emb, r_emb, _, _ = self.evolve_model(input_list, use_cuda=True)
        last_snap_embs = F.normalize(evolve_embs[-1])
        ## Fuse evolve and global embeddings of entities and relations
        if self.fuse == 'con':
            ent_emb = self.linear_fuse(torch.cat((last_snap_embs, new_features), 1))
        elif self.fuse == 'att':
            ent_emb, e_cof = self.fuse_attention(last_snap_embs, new_features, self.ent_global_embedding.weight)
        elif self.fuse == 'att1':
            ent_emb, e_cof = self.fuse_attention1(last_snap_embs, new_features)
        elif self.fuse == 'gate':
            ent_emb, e_cof = self.gate(last_snap_embs, new_features)
        # relation embedding fusion
        if self.r_fuse == 'short':
            r_emb = r_emb
        elif self.r_fuse == 'long':
            r_emb = self.rel_global_embedding.weight
        elif self.r_fuse == 'con':
            r_emb = self.linear_fuse_r(torch.cat((r_emb, self.rel_global_embedding.weight), 1))
        elif self.r_fuse == 'att1':
            r_emb, r_cof = self.fuse_attention_r(r_emb, self.rel_global_embedding.weight)
        elif self.r_fuse == 'gate':
            r_emb, r_cof = self.gate_r(r_emb, self.rel_global_embedding.weight)

        loss_ent = torch.zeros(1).cuda().to(device)
        loss_rel = torch.zeros(1).cuda().to(device)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(device)

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(ent_emb, r_emb, all_triples).view(-1, self.num_nodes)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(ent_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        logits = last_snap_embs.mm(new_features.t())
        labels = torch.arange(self.num_nodes).to(device)
        contrastive_loss = self.func(logits, labels)
        loss = self.task * loss_ent + (1 - self.task) * loss_rel + 0.5 * contrastive_loss

        return ent_emb, r_emb, loss, loss_ent, loss_rel, contrastive_loss

    def predict(self, test_graph, num_rels, global_graph, test_triplets):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            ent_emb, r_emb, _, _, _, _ = self.forward(test_graph, global_graph, test_triplets)
            embedding = F.normalize(ent_emb) if self.layer_norm else ent_emb

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel


    def fuse_attention(self, s_embedding, l_embedding, o_embedding):
        w1 = (o_embedding * torch.tanh(self.linear_s(s_embedding))).sum(1)
        w2 = (o_embedding * torch.tanh(self.linear_l(l_embedding))).sum(1)
        aff = F.softmax(torch.cat((w1.unsqueeze(1),w2.unsqueeze(1)),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

    def fuse_attention1(self, s_embedding, l_embedding):
        w1 = self.fuse_f(torch.tanh(self.linear_s(s_embedding)))
        w2 = self.fuse_f(torch.tanh(self.linear_l(l_embedding)))
        aff = F.softmax(torch.cat((w1,w2),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

    def fuse_attention_r(self, s_embedding, l_embedding):
        w1 = self.fuse_f_r(torch.tanh(self.linear_s_r(s_embedding)))
        w2 = self.fuse_f_r(torch.tanh(self.linear_l_r(l_embedding)))
        aff = F.softmax(torch.cat((w1,w2),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

class GatingMechanism(nn.Module):
    def __init__(self, entity_num, hidden_dim):
        super(GatingMechanism, self).__init__()
        # gating 的参数
        self.gate_theta = nn.Parameter(torch.empty(entity_num, hidden_dim))
        nn.init.xavier_uniform_(self.gate_theta)
        # self.dropout = nn.Dropout(self.params.dropout)

    def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
        '''
        :param X:   LSTM 的输出tensor   |E| * H
        :param Y:   Entity 的索引 id    |E|,
        :return:    Gating后的结果      |E| * H
        '''
        gate = torch.sigmoid(self.gate_theta)
        output = torch.mul(gate, X) + torch.mul(-gate + 1, Y)
        return output, gate