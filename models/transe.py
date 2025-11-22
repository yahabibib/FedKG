import torch
import torch.nn as nn


class TransE(nn.Module):
    """ TransE 模型定义 """

    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, p_norm=2):
        super().__init__()
        self.entity_embeddings = nn.Embedding(
            num_entities, embedding_dim, max_norm=1.0)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.criterion = nn.MarginRankingLoss(margin=margin)
        self.p_norm = p_norm

    def score(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        return torch.norm(h_emb + r_emb - t_emb, p=self.p_norm, dim=1)

    def forward(self, pos_triples, neg_triples):
        pos_scores = self.score(
            pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self.score(
            neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        target = torch.full_like(pos_scores, -1.0)
        loss = self.criterion(pos_scores, neg_scores, target)
        return loss

    def get_entity_embeddings(self):
        return self.entity_embeddings.weight.data.detach().cpu()
