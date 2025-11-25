# 📄 precompute.py
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import config

# --- 1. 实体 SBERT 嵌入 ---


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

# --- 2. 关系 SBERT 嵌入 (工具函数，本次实验暂时不用，但保留功能) ---


@torch.no_grad()
def get_relation_embeddings(rel_id_map, kg_name="KG", cache_file=None):
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
        name = uri.split('/')[-1].replace('_', ' ').replace(':', ' ')
        texts.append(name)

    embeddings = sbert_model.encode(
        texts, convert_to_tensor=True, device=config.DEVICE)
    res = {rid: embeddings[i].cpu() for i, rid in enumerate(sorted_ids)}

    if cache_file:
        torch.save(res, cache_file)
    return res

# --- 3. 构建图结构 (Edge Index & Type) ---


def build_graph_data(triples, num_entities, num_relations):
    """
    构建包含反向边和自环的图结构
    """
    print(
        f"[Precompute] Building Relation Graph for {num_entities} entities...")

    src, dst, rels = [], [], []

    # 1. 原始边 (0 ~ R-1) & 反向边 (R ~ 2R-1)
    for h, r, t in triples:
        # Forward
        src.append(h)
        dst.append(t)
        rels.append(r)

        # Inverse
        src.append(t)
        dst.append(h)
        rels.append(r + num_relations)

    # 2. 自环 (2R)
    self_loop_rel = 2 * num_relations
    for i in range(num_entities):
        src.append(i)
        dst.append(i)
        rels.append(self_loop_rel)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rels, dtype=torch.long)

    print(f"  Constructed graph: {edge_index.shape[1]} edges.")
    return edge_index, edge_type


def build_adjacency_matrix(*args, **kwargs):
    raise DeprecationWarning("Please use build_graph_data instead.")
