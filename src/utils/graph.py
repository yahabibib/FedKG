# src/utils/graph.py
import torch
import numpy as np
import logging

log = logging.getLogger(__name__)


def build_adjacency_matrix(triples, num_entities, device='cpu'):
    """
    构建归一化的稀疏邻接矩阵 (D^-0.5 * A * D^-0.5)。
    注意：MPS 对稀疏矩阵的支持有限，通常建议 Adj 保持在 CPU。
    """
    # log.info(f"Building Adjacency Matrix for {num_entities} entities...")

    src, dst = [], []
    for h, r, t in triples:
        src.append(h)
        dst.append(t)
        # 添加逆向边 (无向图视角)
        src.append(t)
        dst.append(h)

    # 添加自环 (Self-loops)
    for i in range(num_entities):
        src.append(i)
        dst.append(i)

    src = np.array(src)
    dst = np.array(dst)

    indices = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    values = torch.ones(len(src))

    # 1. 构建稀疏矩阵
    adj_temp = torch.sparse_coo_tensor(
        indices, values, (num_entities, num_entities)
    )

    # 2. 计算度矩阵 (Row Sum)
    row_sum = torch.sparse.sum(adj_temp, dim=1).to_dense()

    # 3. 计算 D^-0.5
    deg_inv_sqrt = row_sum.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # 4. 归一化: D^-0.5 * A * D^-0.5
    # 利用广播机制：val[k] = deg[src[k]] * 1 * deg[dst[k]]
    norm_values = deg_inv_sqrt[src] * values * deg_inv_sqrt[dst]

    # 5. 重建归一化后的稀疏矩阵
    adj = torch.sparse_coo_tensor(
        indices, norm_values, (num_entities, num_entities)
    ).coalesce()

    return adj.to(device)
