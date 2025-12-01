# 📄 step1_build_noisy_mini.py
# 构建含噪音的迷你数据集：10 对齐 + 40 噪音/每侧
# 保持闭环子图结构，ID 重映射

import os
import random
import pickle
import config
import data_loader
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_pickle_robust(path, ent_map):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except:
        return {}

    uri2id = {v: k for k, v in ent_map.items()}
    clean_data = {}
    for k, v in data.items():
        if isinstance(k, int) and k in ent_map:
            clean_data[k] = v
        elif isinstance(k, str) and k.strip('<>') in uri2id:
            clean_data[uri2id[k.strip('<>')]] = v
    return clean_data


def run():
    print(f"{'='*60}")
    print("🏗️ 步骤一：构建含噪音的闭环 Mini-DBP15K (Noisy Subgraph)")
    print(f"   策略: 10 Core Pairs + 40 Noise Entities per KG")
    print(f"{'='*60}")

    save_path = "data/demo_mini/zh_en"
    ensure_dir(save_path)

    # 1. 加载全量数据
    print("📚 加载全量 DBP15K 数据...")
    ent_1, _ = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    ent_2, _ = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
    rel_1, _ = data_loader.load_id_map(config.BASE_PATH + "rel_ids_1")
    rel_2, _ = data_loader.load_id_map(config.BASE_PATH + "rel_ids_2")
    trip_1 = data_loader.load_triples(config.BASE_PATH + "triples_1")
    trip_2 = data_loader.load_triples(config.BASE_PATH + "triples_2")
    ref_pairs = data_loader.load_alignment_pairs(
        config.BASE_PATH + "ref_pairs")

    print("📝 加载描述文本...")
    attr_1 = load_pickle_robust(config.BASE_PATH + "description1.pkl", ent_1)
    attr_2 = load_pickle_robust(config.BASE_PATH + "description2.pkl", ent_2)

    # 2. 采样逻辑
    # 筛选"富实体"
    adj_1 = {h for h, r, t in trip_1}
    adj_2 = {h for h, r, t in trip_2}
    rich_pairs = [p for p in ref_pairs if p[0] in attr_1 and p[1]
                  in attr_2 and p[0] in adj_1 and p[1] in adj_2]

    # A. 核心对齐 (10对)
    core_pairs = random.sample(rich_pairs, 10)
    core_ids_1 = set([p[0] for p in core_pairs])
    core_ids_2 = set([p[1] for p in core_pairs])

    # B. 噪音实体 (各40个，互不重叠，且不包含核心)
    # 从剩余实体中选
    remain_1 = [
        e for e in ent_1 if e not in core_ids_1 and e in attr_1 and e in adj_1]
    remain_2 = [
        e for e in ent_2 if e not in core_ids_2 and e in attr_2 and e in adj_2]

    noise_ids_1 = set(random.sample(remain_1, 40))
    noise_ids_2 = set(random.sample(remain_2, 40))

    target_ids_1 = core_ids_1.union(noise_ids_1)
    target_ids_2 = core_ids_2.union(noise_ids_2)

    print(f"   ✅ KG1: 10 Core + 40 Noise = {len(target_ids_1)}")
    print(f"   ✅ KG2: 10 Core + 40 Noise = {len(target_ids_2)}")

    # 3. 子图提取 (包含尾实体描述的闭环构建)
    def process_subgraph(targets, full_trips, full_ents, full_rels, full_attr, suffix):
        print(f"   🔨 处理 KG{suffix} 子图...")
        mini_triples = []

        # 收集所有涉及的实体 (Head + Tail)
        used_ents = set(targets)  # 首先包含所有目标头实体
        used_rels = set()

        # 提取以 targets 为头的三元组
        for h, r, t in full_trips:
            if h in targets:
                mini_triples.append((h, r, t))
                used_ents.add(t)  # 尾实体必须加入，否则图是断的
                used_rels.add(r)

        # ID 重映射
        sorted_ents = sorted(list(used_ents))
        sorted_rels = sorted(list(used_rels))
        old2new_ent = {old: new for new, old in enumerate(sorted_ents)}
        old2new_rel = {old: new for new, old in enumerate(sorted_rels)}

        # 保存文件
        with open(os.path.join(save_path, f"ent_ids_{suffix}"), 'w', encoding='utf-8') as f:
            for old_id in sorted_ents:
                f.write(f"{old2new_ent[old_id]}\t{full_ents[old_id]}\n")

        with open(os.path.join(save_path, f"rel_ids_{suffix}"), 'w', encoding='utf-8') as f:
            for old_id in sorted_rels:
                f.write(f"{old2new_rel[old_id]}\t{full_rels[old_id]}\n")

        with open(os.path.join(save_path, f"triples_{suffix}"), 'w', encoding='utf-8') as f:
            for h, r, t in mini_triples:
                f.write(
                    f"{old2new_ent[h]}\t{old2new_rel[r]}\t{old2new_ent[t]}\n")

        # 保存描述 (所有涉及的实体，包括作为尾实体的叶子节点)
        mini_desc = {}
        count = 0
        for old_id in sorted_ents:
            if old_id in full_attr:
                mini_desc[old2new_ent[old_id]] = full_attr[old_id]
                count += 1

        with open(os.path.join(save_path, f"description{suffix}.pkl"), 'wb') as f:
            pickle.dump(mini_desc, f)

        print(
            f"     - 节点: {len(sorted_ents)} (含描述: {count}), 边: {len(mini_triples)}")
        return old2new_ent

    map_1 = process_subgraph(target_ids_1, trip_1, ent_1, rel_1, attr_1, "1")
    map_2 = process_subgraph(target_ids_2, trip_2, ent_2, rel_2, attr_2, "2")

    # 4. 保存对齐对 (只保存那 10 对 Core)
    print("💾 保存 ref_pairs (仅 10 对)...")
    with open(os.path.join(save_path, "ref_pairs"), 'w', encoding='utf-8') as f:
        for old_1, old_2 in core_pairs:
            f.write(f"{map_1[old_1]}\t{map_2[old_2]}\n")

    print(f"\n✅ 步骤一完成！数据集位于 {save_path}")


if __name__ == "__main__":
    run()
