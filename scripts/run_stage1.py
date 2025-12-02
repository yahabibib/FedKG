# scripts/run_stage1.py
from src.llm.finetuner import SBERTFinetuner
from src.llm.polisher import KnowledgePolisher
from src.data.loader import DataLoader
from src.utils.logger import setup_logger
from src.utils.config import Config
import sys
import os
import pickle
from collections import defaultdict

# 将项目根目录加入路径，确保能导入 src
sys.path.append(os.getcwd())


def prepare_triples_text(ent_map, triples, rel_map, attr_map):
    """辅助函数：将三元组整理为 Prompt 所需的列表"""
    adj = defaultdict(list)
    for h, r, t in triples:
        adj[h].append((r, t))

    tasks = []
    # 遍历所有实体
    for eid, uri in ent_map.items():
        # 如果没有邻居，跳过
        if eid not in adj:
            continue

        neighbors = adj[eid][:5]  # 限制邻居数量
        relations_str = []

        # 简单清洗名字的 lambda
        def clean(u): return u.split('/')[-1].replace('_', ' ')

        name = clean(uri)

        for r, t in neighbors:
            r_name = clean(rel_map.get(r, str(r)))
            t_name = clean(ent_map.get(t, str(t)))
            # 获取尾实体背景 (取前50字符)
            t_desc = str(attr_map.get(t, ""))[:50]

            line = f"- 关系: {r_name} -> 对象: {t_name}"
            if t_desc:
                line += f" (背景: {t_desc})"
            relations_str.append(line)

        tasks.append({
            "eid": eid,
            "name": name,
            "relations": relations_str,
            "raw_desc": attr_map.get(eid, name)  # 原始描述作为 Anchor
        })
    return tasks


def main():
    cfg = Config()
    logger = setup_logger("Stage1_Polishing")
    logger.info("🎬 Starting Stage 1: LLM Polishing & SBERT Fine-tuning")

    loader = DataLoader(cfg)

    # 1. 加载数据
    logger.info("Loading KG1 data...")
    ent1, _ = loader.load_id_map("ent_ids_1")
    rel1, _ = loader.load_id_map("rel_ids_1")
    trip1 = loader.load_triples("triples_1")
    attr1 = loader.load_pickle_descriptions("description1.pkl", ent1)

    # 2. 准备 LLM 任务
    tasks = prepare_triples_text(ent1, trip1, rel1, attr1)
    logger.info(f"Prepared {len(tasks)} entities for polishing.")

    # 3. 执行 LLM 润色
    # 检查是否已有润色结果，避免重复跑 (很慢)
    polished_file = os.path.join(
        cfg.project_root, cfg.relative_data_path, "polished_data_kg1.pkl")

    if os.path.exists(polished_file):
        logger.info(
            f"Found existing polished data: {polished_file}, skipping LLM inference.")
        with open(polished_file, 'rb') as f:
            polished_results = pickle.load(f)
    else:
        logger.info("Initializing LLM for inference...")
        polisher = KnowledgePolisher(cfg)

        prompts = []
        for t in tasks:
            p = polisher.construct_prompt(t['name'], t['relations'], lang='zh')
            prompts.append(p)

        # 批量生成
        # ⚠️ 注意：如果是 CPU 跑，建议把 tasks[:10] 先切片测试一下
        generated_texts = polisher.batch_generate(prompts, batch_size=4)

        polished_results = {}
        for task, text in zip(tasks, generated_texts):
            polished_results[task['eid']] = text

        # 保存
        with open(polished_file, 'wb') as f:
            pickle.dump(polished_results, f)

        polisher.clean_memory()

    # 4. 执行 SBERT 微调
    logger.info("Preparing Fine-tuning pairs...")
    train_pairs = []

    for t in tasks:
        eid = t['eid']
        if eid in polished_results:
            anchor = t['raw_desc']
            positive = polished_results[eid]
            # 只有当两个文本都足够长时才训练
            if len(anchor) > 5 and len(positive) > 5:
                train_pairs.append((anchor, positive))

    finetuner = SBERTFinetuner(cfg)
    finetuner.fine_tune(
        train_pairs,
        # 保存到 output/fine_tuned_models/exp4...
        output_path=cfg.sbert_model_path,
        epochs=3
    )

    logger.info("🎉 Stage 1 Completed!")


if __name__ == "__main__":
    main()
