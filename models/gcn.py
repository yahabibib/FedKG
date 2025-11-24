# 📄 models/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# 📄 models/gcn.py (片段：仅替换 Layer 类)

class RelationGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

        # ⚡️ [优化] 维度减半：原来是 input*2 -> 1，现在 input -> 1
        self.attn_kernel = nn.Linear(input_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_kernel.weight)

    def forward(self, x, edge_index, edge_type, rel_emb):
        x_trans = self.linear(x)
        src_idx, tgt_idx = edge_index

        rel_emb = rel_emb.to(x.device)

        # 取出特征
        h_src = x[src_idx]
        h_rel = rel_emb[edge_type]

        # ⚡️ [优化] 轻量化计算: 使用加法代替拼接 (Save Memory!)
        # h_src: [E, D], h_rel: [E, D] -> sum_feat: [E, D]
        # 相比之前的 cat ([E, 2D]), 节省了一半内存
        sum_feat = h_src + h_rel

        # 计算 Attention
        attn_weights = torch.sigmoid(self.attn_kernel(sum_feat))

        # 消息传递
        msg = x_trans[src_idx] * attn_weights

        # 聚合
        out = torch.zeros(x.shape[0], x_trans.shape[1], device=x.device)
        out.index_add_(0, tgt_idx, msg)

        if self.activation:
            out = self.activation(out)
        return out


class RelationGCN(nn.Module):
    def __init__(self, num_entities, num_relations, feature_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        print(
            f"    [Model Init] Relation-Aware GCN: Utilizing {num_relations} relations.")

        # 实体初始特征
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        # 关系嵌入 (可学习，但用 SBERT 初始化)
        # 我们会在 forward 里接收 SBERT 初始值，或者在这里定义 Parameter
        # 为了灵活性，我们定义为 Parameter，初始化时加载 SBERT
        self.relation_embeddings = nn.Parameter(
            torch.randn(num_relations, feature_dim))

        # 层定义
        self.gc1 = RelationGCNLayer(feature_dim, hidden_dim)
        self.gc2 = RelationGCNLayer(
            hidden_dim, output_dim, activation=None)  # 最后一层通常不加激活

        self.dropout = dropout

    def init_relation_embeddings(self, sbert_rel_emb):
        """ 用 SBERT 初始化关系嵌入 """
        with torch.no_grad():
            for rid, emb in sbert_rel_emb.items():
                if rid < self.relation_embeddings.shape[0]:
                    # 假设 SBERT 是 768，feature_dim 是 300，需要投影或截断
                    # 如果 feature_dim != 768, 建议加个线性层投影，这里简单起见假设维度匹配
                    # 或者我们在外部做好投影。这里先只复制能复制的部分。
                    dim = min(self.relation_embeddings.shape[1], emb.shape[0])
                    self.relation_embeddings.data[rid, :dim] = emb[:dim]

    def forward(self, edge_index, edge_type):
        x = self.initial_features

        # Layer 1
        x = self.gc1(x, edge_index, edge_type, self.relation_embeddings)
        x = F.dropout(x, self.dropout, training=self.training)

        # Layer 2
        x = self.gc2(x, edge_index, edge_type, self.relation_embeddings)

        return x
