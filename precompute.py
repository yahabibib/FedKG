# 📄 precompute.py
# 负责离线计算 (构建邻接矩阵, 生成 SBERT 锚点)
# 【升级版】支持 SBERT 嵌入缓存，避免重复计算

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os  # 新增
import config

# --- 1. SBERT 嵌入生成 (带缓存) ---


@torch.no_grad()
def get_bert_embeddings(id_to_uri_map, attribute_descriptions, kg_name="KG", cache_file=None):
    """
    生成 SBERT 嵌入。
    如果提供了 cache_file 且文件存在，则直接加载，跳过计算。
    """
    # 1. 尝试加载缓存
    if cache_file and os.path.exists(cache_file):
        print(f"\n[Precompute] Found cached SBERT embeddings for {kg_name}!")
        print(f"             Loading from: {cache_file}")
        try:
            return torch.load(cache_file)
        except Exception as e:
            print(f"  [Warning] Failed to load cache ({e}). Re-computing...")

    # 2. 如果没有缓存，开始正常计算
    print(
        f"\n[Precompute] Computing SBERT embeddings for {kg_name} (No cache found)...")
    print(f"             Loading model: {config.BERT_MODEL_NAME}")

    sbert_model = SentenceTransformer(
        config.BERT_MODEL_NAME, device=config.DEVICE)
    sbert_model.eval()

    id_to_uri = id_to_uri_map[0]
    entity_ids = sorted(list(id_to_uri.keys()))

    all_texts = []
    used_desc_count = 0

    for ent_id in entity_ids:
        # 优先用描述，回退用名字
        description = attribute_descriptions.get(ent_id)
        if not description:
            ent_uri = id_to_uri[ent_id]
            description = ent_uri.split('/')[-1].replace('_', ' ')
        else:
            used_desc_count += 1
        all_texts.append(description)

    print(
        f"             > Used rich descriptions for {used_desc_count}/{len(entity_ids)} entities.")

    # 批量编码
    all_embeddings_list = []
    for i in tqdm(range(0, len(all_texts), config.BERT_BATCH_SIZE), desc=f"Encoding {kg_name}"):
        batch_texts = all_texts[i: i + config.BERT_BATCH_SIZE]
        embeddings = sbert_model.encode(
            batch_texts, convert_to_tensor=True, show_progress_bar=False, device=config.DEVICE)
        all_embeddings_list.append(embeddings.cpu())  # 存到 CPU 以便保存

    all_embeddings_tensor = torch.cat(all_embeddings_list, dim=0)

    # 转为字典
    result_dict = {ent_id: all_embeddings_tensor[i]
                   for i, ent_id in enumerate(entity_ids)}

    # 3. 保存缓存
    if cache_file:
        print(f"             Saving cache to: {cache_file}")
        torch.save(result_dict, cache_file)

    return result_dict

# --- 2. GCN 邻接矩阵构建 (核心) ---


def build_adjacency_matrix(triples, num_entities):
    """
    构建归一化的稀疏邻接矩阵 (D^-0.5 * A * D^-0.5)
    【MPS 兼容版】: 返回 CPU 张量
    """
    print(
        f"[Precompute] Building Adjacency Matrix for {num_entities} entities...")

    src, dst = [], []
    for h, r, t in triples:
        src.append(h)
        dst.append(t)
        src.append(t)
        dst.append(h)

    for i in range(num_entities):
        src.append(i)
        dst.append(i)

    src = np.array(src)
    dst = np.array(dst)

    indices = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    values = torch.ones(len(src))

    adj_temp = torch.sparse_coo_tensor(
        indices, values, (num_entities, num_entities))
    row_sum = torch.sparse.sum(adj_temp, dim=1).to_dense()

    deg_inv_sqrt = row_sum.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    norm_values = deg_inv_sqrt[src] * values * deg_inv_sqrt[dst]

    adj = torch.sparse_coo_tensor(
        indices, norm_values, (num_entities, num_entities))
    print(f"  Built sparse adjacency matrix with {indices.shape[1]} edges.")

    return adj.coalesce()
