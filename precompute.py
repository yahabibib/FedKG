# 📄 precompute.py
# 【修复版】强制缓存 SBERT Embedding，避免重复计算

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import config
import re


@torch.no_grad()
def get_bert_embeddings(id_to_uri_map, attribute_descriptions, kg_name="KG", cache_file=None, custom_model_path=None):
    """
    计算实体 SBERT 向量
    """
    # 1. 尝试加载缓存
    if cache_file and os.path.exists(cache_file):
        print(f"\n[Precompute] Found cached SBERT embeddings for {kg_name}!")
        print(f"             Loading from: {cache_file}")
        try:
            return torch.load(cache_file)
        except Exception as e:
            print(f"  [Warning] Cache load failed ({e}). Re-computing...")

    # 2. 计算
    model_name_to_load = custom_model_path if custom_model_path else config.BERT_MODEL_NAME
    print(
        f"\n[Precompute] Computing Entity SBERT for {kg_name} using {model_name_to_load}...")

    sbert_model = SentenceTransformer(model_name_to_load, device=config.DEVICE)
    sbert_model.eval()

    id_to_uri = id_to_uri_map[0]
    entity_ids = sorted(list(id_to_uri.keys()))
    all_texts = []

    for ent_id in entity_ids:
        desc = attribute_descriptions.get(ent_id)
        if not desc:
            uri = id_to_uri[ent_id]
            desc = uri.split('/')[-1].replace('_', ' ')
        all_texts.append(desc)

    all_embeddings_list = []
    # 批量计算
    for i in tqdm(range(0, len(all_texts), config.BERT_BATCH_SIZE), desc=f"Encoding {kg_name}"):
        batch = all_texts[i: i + config.BERT_BATCH_SIZE]
        embs = sbert_model.encode(
            batch, convert_to_tensor=True, show_progress_bar=False, device=config.DEVICE)
        all_embeddings_list.append(embs.cpu())

    all_embeddings_tensor = torch.cat(all_embeddings_list, dim=0)
    result_dict = {ent_id: all_embeddings_tensor[i]
                   for i, ent_id in enumerate(entity_ids)}

    # 3. 【修复】强制保存缓存
    if cache_file:
        print(f"             Saving cache to: {cache_file}")
        torch.save(result_dict, cache_file)

    return result_dict


@torch.no_grad()
def get_relation_embeddings(rel_id_map, kg_name="KG", cache_file=None):
    """ 计算关系 SBERT 向量 """
    if cache_file and os.path.exists(cache_file):
        print(f"[Precompute] Found cached Relation embeddings for {kg_name}!")
        return torch.load(cache_file)

    print(f"\n[Precompute] Computing Relation SBERT for {kg_name}...")
    sbert_model = SentenceTransformer(
        config.BERT_MODEL_NAME, device=config.DEVICE)

    def clean_rel(uri):
        if not isinstance(uri, str):
            return str(uri)
        name = uri.split('/')[-1].replace('_',
                                          ' ').replace('<', '').replace('>', '')
        name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()
        return name.strip()

    sorted_ids = sorted(list(rel_id_map.keys()))
    rel_texts = [clean_rel(rel_id_map[rid]) for rid in sorted_ids]

    embs = sbert_model.encode(
        rel_texts, convert_to_tensor=True, show_progress_bar=True, device=config.DEVICE)
    result_tensor = embs.cpu()

    if cache_file:
        torch.save(result_tensor, cache_file)
    return result_tensor


def build_adjacency_matrix(triples, num_entities):
    print(f"[Precompute] Building Adj Matrix for {num_entities} entities...")
    src, dst = [], []
    for h, r, t in triples:
        src.append(h)
        dst.append(t)
        src.append(t)
        dst.append(h)
    for i in range(num_entities):
        src.append(i)
        dst.append(i)

    indices = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    values = torch.ones(len(src))

    adj = torch.sparse_coo_tensor(
        indices, values, (num_entities, num_entities))
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = row_sum.pow(-0.5)
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
    norm_val = d_inv_sqrt[indices[0]] * values * d_inv_sqrt[indices[1]]

    return torch.sparse_coo_tensor(indices, norm_val, (num_entities, num_entities)).coalesce()
