# 📄 data_loader.py
# 负责加载所有文件 (triples, ids, atts, pairs)
# 【终极修复版】增强了 ID 文件的解析能力，修复分隔符问题

import pandas as pd
from collections import defaultdict
import re
import pickle
import os


def load_id_map(file_path):
    """加载 ent_ids 或 rel_ids 文件"""
    id_to_uri = {}
    uri_to_id = {}
    filename = file_path.split('/')[-1]
    print(f"  [Data Loader] Loading IDs from: {filename}")

    count = 0
    try:
        # 使用 utf-8-sig 以处理可能的 BOM 头
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 1. 尝试 Tab 分隔
                parts = line.split('\t')
                # 2. 如果 Tab 分隔失败 (只有1列)，尝试空格分隔
                if len(parts) < 2:
                    parts = line.split()

                if len(parts) >= 2:
                    # 通常第一列是 ID，第二列是 URI
                    if parts[0].isdigit():
                        ent_id = int(parts[0])
                        uri = parts[1].strip()

                        id_to_uri[ent_id] = uri
                        uri_to_id[uri] = ent_id
                        count += 1
                    else:
                        # 可能是反过来的? (极少见，但防一手)
                        pass
    except Exception as e:
        print(f"  [Error] Failed to load ID map {filename}: {e}")

    print(f"    > Loaded {count} IDs. Example: {list(uri_to_id.items())[:2]}")
    return id_to_uri, uri_to_id


def load_triples(file_path):
    """加载 triples 文件 (head_id, rel_id, tail_id)"""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            # 同样尝试空格分隔
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
    """加载 ref_pairs (测试集)"""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                parts = line.split()

            if len(parts) >= 2:
                try:
                    e1, e2 = int(parts[0]), int(parts[1])
                    pairs.append((e1, e2))
                except ValueError:
                    continue
    return pairs


def load_attribute_triples(file_path, ent_map):
    """
    【全能加载器】支持多种格式，并自动尝试匹配 URI
    """
    filename = file_path.split('/')[-1]
    print(f"  [Data Loader] Reading attributes from: {filename}")

    uri_to_id = ent_map[1]

    # NT 格式正则: <Subject> <Pred> "Obj" .
    # 针对您提供的数据: <http://...> <...> "..."@en .
    nt_pattern = re.compile(r'^<([^>]+)>\s+<([^>]+)>\s+(.+?)\s*[\."]*$')

    entity_descriptions = defaultdict(list)
    skipped_count = 0
    valid_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # --- 1. 尝试 Tab 分隔 (ID 格式) ---
                parts = line.split('\t')
                if len(parts) >= 3 and parts[0].isdigit():
                    ent_id = int(parts[0])
                    attr_name = parts[1].split('/')[-1]
                    value = _clean_value(parts[2])
                    entity_descriptions[ent_id].append(
                        f"{attr_name} is {value}")
                    valid_count += 1
                    continue

                # --- 2. 尝试 N-Triples 正则 ---
                match = nt_pattern.match(line)
                if match:
                    # http://dbpedia.org/resource/Liverpool_F.C.
                    raw_subj = match.group(1)
                    raw_pred = match.group(2)
                    raw_obj = match.group(3)

                    # 【核心逻辑】智能查找 ID
                    # 这里的 raw_subj 应该能直接匹配到 ent_ids 里的 URI
                    ent_id = _find_id_flexible(raw_subj, uri_to_id)

                    if ent_id is None:
                        skipped_count += 1
                        # 调试：只打印前3个失败的，避免刷屏
                        if skipped_count <= 3:
                            print(f"    [Debug Fail] Unmapped URI: {raw_subj}")
                        continue

                    attr_name = raw_pred.split('/')[-1]
                    value = _clean_value(raw_obj)

                    entity_descriptions[ent_id].append(
                        f"{attr_name} is {value}")
                    valid_count += 1
                else:
                    skipped_count += 1

    except Exception as e:
        print(f"  [Error] reading {file_path}: {e}")
        return {}

    if skipped_count > 0:
        print(
            f"  [Data Loader] Skipped {skipped_count} lines (unmapped). Loaded {valid_count} descriptions.")

    final_descriptions = {
        ent_id: "; ".join(sentences)
        for ent_id, sentences in entity_descriptions.items()
    }
    return final_descriptions


def load_pickle_descriptions(file_path, ent_map):
    """
    【新功能】直接加载 .pkl 格式的高质量描述文件
    file_path: description.pkl 的路径
    ent_map: (id_to_uri, uri_to_id) 用于验证和对齐
    """
    filename = file_path.split('/')[-1]
    print(f"  [Data Loader] Loading descriptions from Pickle: {filename}")

    if not os.path.exists(file_path):
        print(f"  [Error] Pickle file not found: {file_path}")
        return {}

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        uri_to_id = ent_map[1]
        id_to_uri = ent_map[0]

        final_descriptions = {}
        mapped_count = 0

        # 遍历 Pickle 中的数据
        for key, desc in data.items():
            ent_id = None

            # 情况 A: Key 是 URI 字符串 (最常见)
            if isinstance(key, str):
                # 尝试直接匹配
                if key in uri_to_id:
                    ent_id = uri_to_id[key]
                # 尝试去掉尖括号匹配 <http...> -> http...
                elif key.strip('<>') in uri_to_id:
                    ent_id = uri_to_id[key.strip('<>')]

            # 情况 B: Key 已经是数字 ID (直接用)
            elif isinstance(key, int):
                if key in id_to_uri:
                    ent_id = key

            # 如果找到了对应的 ID，且描述有效
            if ent_id is not None and desc:
                # 清洗一下描述 (去掉多余空白)
                clean_desc = str(desc).strip()
                # 加上名字前缀，增强语义 (如 "Linux内核. Linux内核是...")
                name = id_to_uri[ent_id].split('/')[-1].replace('_', ' ')
                final_descriptions[ent_id] = f"{name}. {clean_desc}"[
                    :500]  # 截断防止过长
                mapped_count += 1

        print(
            f"  [Pickle Loader] Successfully mapped {mapped_count} descriptions.")
        return final_descriptions

    except Exception as e:
        print(f"  [Error] Failed to load pickle: {e}")
        return {}


def _find_id_flexible(uri, mapping):
    """尝试多种变体来查找 ID"""
    # 1. 精确匹配 (最可能的情况)
    if uri in mapping:
        return mapping[uri]

    # 2. 尝试加上尖括号 <uri>
    if f"<{uri}>" in mapping:
        return mapping[f"<{uri}>"]

    # 3. 尝试只取最后一部分 (Short Name)
    short_name = uri.split('/')[-1]
    if short_name in mapping:
        return mapping[short_name]

    # 4. 尝试去掉结尾可能的 > (容错)
    if uri.strip('<>') in mapping:
        return mapping[uri.strip('<>')]

    return None


def _clean_value(val_str):
    """清洗属性值：去掉类型后缀、语言标签、引号"""
    # 例子: "at FSV Frankfurt..."@en
    # 例子: "60"^^<http://...>

    # 1. 去掉类型后缀 ^^<...>
    val = val_str.split('^^')[0]

    # 2. 去掉语言标签 @en (从右边找最后一个@)
    # 注意：内容里可能有@，所以要小心，通常语言标签在引号外
    if val.endswith('@en') or val.endswith('@zh') or val.endswith('@fr') or val.endswith('@ja'):
        val = val.rsplit('@', 1)[0]

    # 3. 去掉首尾引号
    val = val.strip('"')
    return val
