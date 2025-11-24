# 📄 precompute.py
# 负责离线计算 (SBERT, 关系 SBERT, 构建图索引)
# 【升级版】支持关系感知 (Relation-Aware)

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import config

# --- 1. 实体 SBERT 嵌入 (带缓存) ---


@torch.no_grad()
def get_bert_embeddings(id_to_uri_map, attribute_descriptions, kg_name="KG", cache_file=None):
    if cache_file and os.path.exists(cache_file):
        print(f"\n[Precompute] Found cached SBERT embeddings for {kg_name}!")
        try:
            return torch.load(cache_file)
        except Exception as e:
            print(f"  [Warning] Failed to load cache ({e}). Re-computing...")

    print(f"\n[Precompute] Computing SBERT embeddings for {kg_name}...")
    sbert_model = SentenceTransformer(
        config.BERT_MODEL_NAME, device=config.DEVICE)
    sbert_model.eval()

    id_to_uri = id_to_uri_map[0]
    entity_ids = sorted(list(id_to_uri.keys()))

    all_texts = []
    for ent_id in entity_ids:
        description = attribute_descriptions.get(ent_id)
        if not description:
            ent_uri = id_to_uri[ent_id]
            description = ent_uri.split('/')[-1].replace('_', ' ')
        all_texts.append(description)

    # 批量编码
    all_embeddings_list = []
    for i in tqdm(range(0, len(all_texts), config.BERT_BATCH_SIZE), desc=f"Encoding {kg_name}"):
        batch_texts = all_texts[i: i + config.BERT_BATCH_SIZE]
        embeddings = sbert_model.encode(
            batch_texts, convert_to_tensor=True, show_progress_bar=False, device=config.DEVICE)
        all_embeddings_list.append(embeddings.cpu())

    all_embeddings_tensor = torch.cat(all_embeddings_list, dim=0)
    result_dict = {ent_id: all_embeddings_tensor[i]
                   for i, ent_id in enumerate(entity_ids)}

    if cache_file:
        torch.save(result_dict, cache_file)

    return result_dict


# --- 2. [新增] 关系 SBERT 嵌入 ---
@torch.no_grad()
def get_relation_embeddings(rel_id_map, kg_name="KG", cache_file=None):
    """
    计算关系的语义向量
    """
    if cache_file and os.path.exists(cache_file):
        print(f"[Precompute] Loading cached Relation SBERT for {kg_name}...")
        return torch.load(cache_file)

    print(f"\n[Precompute] Computing Relation SBERT for {kg_name}...")
    sbert_model = SentenceTransformer(
        config.BERT_MODEL_NAME, device=config.DEVICE)
    sbert_model.eval()

    id_to_uri = rel_id_map[0]
    sorted_ids = sorted(list(id_to_uri.keys()))

    texts = []
    for rid in sorted_ids:
        uri = id_to_uri[rid]
        # 简单清洗：从 URI 提取关系名
        name = uri.split('/')[-1].replace('_', ' ').replace(':', ' ')
        texts.append(name)

    embeddings = sbert_model.encode(
        texts, convert_to_tensor=True, device=config.DEVICE)
    res = {rid: embeddings[i].cpu() for i, rid in enumerate(sorted_ids)}

    if cache_file:
        torch.save(res, cache_file)

    return res


# --- 3. [修改] 构建图结构 (返回 Edge Index & Type) ---
def build_graph_data(triples, num_entities, num_relations):
    """
    替代原来的 build_adjacency_matrix。
    返回:
    - edge_index: [2, E]
    - edge_type: [E] (包含反向和自环)
    """
    print(
        f"[Precompute] Building Relation Graph for {num_entities} entities...")

    src, dst, rels = [], [], []

    # 1. 原始边 & 反向边
    for h, r, t in triples:
        # Forward
        src.append(h)
        dst.append(t)
        rels.append(r)

        # Inverse (r + num_relations)
        src.append(t)
        dst.append(h)
        rels.append(r + num_relations)

    # 2. 自环 (2 * num_relations)
    self_loop_rel = 2 * num_relations
    for i in range(num_entities):
        src.append(i)
        dst.append(i)
        rels.append(self_loop_rel)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rels, dtype=torch.long)

    print(f"  Constructed graph: {edge_index.shape[1]} edges.")
    return edge_index, edge_type

# 旧函数可以保留个空壳或者直接删掉，防止报错


def build_adjacency_matrix(*args, **kwargs):
    raise DeprecationWarning("Please use build_graph_data instead.")
