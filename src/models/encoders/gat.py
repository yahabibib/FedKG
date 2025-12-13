# src/models/encoders/gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGATLayer(nn.Module):
    """
    支持稀疏矩阵输入的 GAT 层 (Single Head)
    核心逻辑：学习边权重 alpha_ij，实现各向异性聚合，过滤噪声邻居。

    [MPS 兼容性修复]: 
    由于 MPS 对稀疏张量 (sparse_coo_tensor) 支持不完善，当检测到 adj 在 CPU 
    而输入 h 在 MPS 时，会自动将稀疏计算部分 Offload 到 CPU 进行，算完后再转回 MPS。
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(SparseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha_val = alpha  # 保存 alpha 值用于 functional 调用

        # 变换矩阵 W [In, Out]
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 注意力向量 a [2*Out, 1]
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        # 1. 线性变换 (Dense 运算，保留在 MPS/GPU 上以获得加速)
        Wh = self.W(h)  # [N, Out]

        # 2. 设备策略检查与分流
        # 如果 adj 在 CPU (通常为了存大图或避开 MPS bug)，而 Wh 在 MPS
        # 我们需要将涉及稀疏结构的操作 Offload 到 CPU
        if adj.device != h.device:
            target_device = h.device
            compute_device = adj.device

            # 将计算所需的 Dense 张量临时搬运到 CPU
            Wh_compute = Wh.to(compute_device)
            a_compute = self.a.to(compute_device)

            # 在 CPU 上执行复杂的 Attention 和 Aggregation
            out = self._compute_attention(
                Wh_compute, adj, a_compute, compute_device)

            # 结果搬回 MPS/GPU
            return out.to(target_device)
        else:
            # 设备一致 (纯 CPU 模式 或 CUDA 支持稀疏的场景)
            return self._compute_attention(Wh, adj, self.a, h.device)

    def _compute_attention(self, Wh, adj, a_param, device):
        """
        内部函数：执行具体的 GAT 注意力计算逻辑。
        确保传入的所有 Tensor 都在同一个 device 上。
        """
        indices = adj.indices()  # [2, E]
        rows, cols = indices[0], indices[1]

        # 3. 计算每条边的 Attention Score
        Wh_i = Wh[rows]
        Wh_j = Wh[cols]
        a_input = torch.cat([Wh_i, Wh_j], dim=1)  # [E, 2*Out]

        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # 使用 F.leaky_relu 避免 nn.Module 的设备依赖问题
        e = F.leaky_relu(torch.matmul(a_input, a_param).squeeze(1),
                         negative_slope=self.alpha_val)

        # 4. Sparse Softmax
        e = e - e.max()  # 数值稳定性
        exp_e = torch.exp(e)

        N = Wh.size(0)
        # 分母：对每个目标节点(row)的所有入边进行求和
        sum_exp = torch.zeros(N, device=device)
        sum_exp.index_add_(0, rows, exp_e)

        # 归一化 alpha_ij
        alpha = exp_e / (sum_exp[rows] + 1e-10)

        # Dropout (F.dropout 在 CPU 上运行无障碍)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # 5. 聚合
        # 构造带有 Attention 权重的稀疏矩阵
        # 注意：这里是在 CPU 上创建 sparse_coo，完全避开了 MPS 的不支持问题
        adj_att = torch.sparse_coo_tensor(
            indices,
            alpha,
            torch.Size([N, N]),
            device=device
        )

        # H_new = Adj_att * Wh
        h_prime = torch.sparse.mm(adj_att, Wh)

        return F.elu(h_prime)


class GATEncoder(nn.Module):
    """
    [Private Component]
    使用 GAT 替代 GCN 进行结构特征提取。
    """

    def __init__(self, num_entities, feature_dim, hidden_dim, dropout=0.3):
        super(GATEncoder, self).__init__()

        # 1. 初始节点特征 (可学习的 Embedding)
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        # 2. GAT 层
        self.layer = SparseGATLayer(feature_dim, hidden_dim, dropout=dropout)

        self.dropout = dropout

    def forward(self, adj):
        x = self.initial_features
        x = F.dropout(x, self.dropout, training=self.training)

        # GAT Forward
        x = self.layer(x, adj)

        return x
