# 📄 data_loader.py
# 【升级版】新增三元组文本生成，用于 MLM 预训练

import pandas as pd
from collections import defaultdict
import re
import pickle
import os

# ... (保持 load_id_map, load_triples, load_alignment_pairs, load_pickle_descriptions 不变) ...


def load_id_map(file_path):
    id_to_uri = {}
    uri_to_id = {}
    filename = file_path.split('/')[-1]
    print(f"  [Data Loader] Loading IDs from: {filename}")
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    parts = line.split()
                if len(parts) >= 2:
                    if parts[0].isdigit():
                        ent_id = int(parts[0])
                        uri = parts[1].strip()
                        id_to_uri[ent_id] = uri
                        uri_to_id[uri] = ent_id
                        count += 1
    except Exception as e:
        print(f"  [Error] Failed to load ID map {filename}: {e}")
    print(f"    > Loaded {count} IDs.")
    return id_to_uri, uri_to_id


def load_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                parts = line.split()
            if len(parts) >= 3:
                try:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append((h, r, t))
                except ValueError:
                    continue
    return triples


def load_alignment_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                parts = line.split()
            if len(parts) >= 2:
                try:
                    pairs.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    return pairs


def load_pickle_descriptions(file_path, ent_map):
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        uri_to_id = ent_map[1]
        id_to_uri = ent_map[0]
        final_descriptions = {}
        for key, desc in data.items():
            ent_id = None
            if isinstance(key, str):
                if key in uri_to_id:
                    ent_id = uri_to_id[key]
                elif key.strip('<>') in uri_to_id:
                    ent_id = uri_to_id[key.strip('<>')]
            elif isinstance(key, int):
                if key in id_to_uri:
                    ent_id = key
            if ent_id is not None and desc:
                name = id_to_uri[ent_id].split('/')[-1].replace('_', ' ')
                final_descriptions[ent_id] = f"{name}. {str(desc).strip()}"[
                    :500]
        return final_descriptions
    except Exception as e:
        return {}


def load_attribute_triples(file_path, ent_map):
    return {}

# --- 【新增】生成三元组句子 (Triple-to-Text) ---


def generate_triple_sentences(triples, ent_id_map, rel_id_map, lang='en'):
    """
    将三元组转化为自然语言句子，用于 MLM 训练
    Output: ["Steve Jobs founded Apple.", "Beijing is capital of China.", ...]
    """
    print(f"  [Data Loader] Generating triple sentences ({lang})...")
    sentences = []

    def clean_name(uri):
        if not isinstance(uri, str):
            return str(uri)
        name = uri.split('/')[-1].replace('_',
                                          ' ').replace('<', '').replace('>', '')
        name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()
        return name.strip()

    for h, r, t in triples:
        # 只处理实体都在 ID 列表里的三元组
        if h in ent_id_map and t in ent_id_map and r in rel_id_map:
            h_name = clean_name(ent_id_map[h])
            t_name = clean_name(ent_id_map[t])
            r_name = clean_name(rel_id_map[r])

            # 构造句子
            if lang == 'zh':
                sent = f"{h_name} 的 {r_name} 是 {t_name}。"
            else:
                sent = f"{h_name} {r_name} {t_name}."

            sentences.append(sent)

    return sentences


def _find_id_flexible(uri, mapping):
    if uri in mapping:
        return mapping[uri]
    if f"<{uri}>" in mapping:
        return mapping[f"<{uri}>"]
    if uri.split('/')[-1] in mapping:
        return mapping[uri.split('/')[-1]]
    if uri.strip('<>') in mapping:
        return mapping[uri.strip('<>')]
    return None


def _clean_value(val_str):
    val = val_str.split('^^')[0]
    if val.endswith('@en') or val.endswith('@zh') or val.endswith('@fr') or val.endswith('@ja'):
        val = val.rsplit('@', 1)[0]
    return val.strip('"')
