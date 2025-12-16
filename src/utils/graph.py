# src/utils/graph.py
import torch
import numpy as np
import logging

log = logging.getLogger(__name__)


def build_adjacency_matrix(triples, num_entities, device='cpu', return_edge_types=False):
    """
    构建归一化的稀疏邻接矩阵 (D^-0.5 * A * D^-0.5)。

    :param triples: List of (h, r, t)
    :param num_entities: 实体总数
    :param device: 设备
    :param return_edge_types: [新增] 是否返回对应的关系类型 ID 张量
    :return: adj (如果 return_edge_types=False)
             adj, edge_types, total_rels (如果 return_edge_types=True)
    """
    # 1. 扫描关系数量
    max_rel = 0
    for h, r, t in triples:
        max_rel = max(max_rel, r)
    num_base_rels = max_rel + 1

    # 定义关系偏移量 (Base, Inverse, Self-loop)
    rel_offset_inv = num_base_rels
    rel_self_loop = 2 * num_base_rels
    total_rels = 2 * num_base_rels + 1

    # 2. 收集原始边信息
    # 使用字典来自动去重：(src, dst) -> relation_id
    # 策略：如果两点间有多条边，保留 ID 最大的那个（或最后出现的）
    edges_dict = {}

    for h, r, t in triples:
        # 正向边
        edges_dict[(h, t)] = r
        # 反向边
        edges_dict[(t, h)] = r + rel_offset_inv

    # 添加自环 (Self-loops)
    for i in range(num_entities):
        edges_dict[(i, i)] = rel_self_loop

    # 3. 关键步骤：确定性排序 (Deterministic Sorting)
    # 这一步保证了 indices 和 values/types 的严格对齐
    sorted_keys = sorted(edges_dict.keys())

    final_src = [k[0] for k in sorted_keys]
    final_dst = [k[1] for k in sorted_keys]

    # 提取排序后的 edge_types
    final_etypes = [edges_dict[k]
                    for k in sorted_keys] if return_edge_types else []

    # 4. 构建张量
    indices = torch.tensor(np.vstack((final_src, final_dst)), dtype=torch.long)
    values = torch.ones(len(final_src))

    # 5. 构建稀疏矩阵
    # 因为我们已经手动去重并排序了，这里的 indices 是唯一的且有序的
    # coalesce() 是安全的
    adj = torch.sparse_coo_tensor(
        indices, values, (num_entities, num_entities)
    ).coalesce()

    # 6. 归一化 (Standard GCN Normalization)
    # D^-0.5 * A * D^-0.5
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = row_sum.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    norm_values = deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]

    adj_norm = torch.sparse_coo_tensor(
        indices, norm_values, (num_entities, num_entities)
    ).coalesce()

    # 7. 返回结果
    adj_norm = adj_norm.to(device)

    if return_edge_types:
        edge_types_tensor = torch.tensor(
            final_etypes, dtype=torch.long).to(device)
        return adj_norm, edge_types_tensor, total_rels

    return adj_norm
