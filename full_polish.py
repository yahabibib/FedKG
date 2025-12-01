# 📄 step2_local_llm.py
# 【批量加速版】利用 Batch Inference 榨干 CPU 性能
# 1. 支持 Batch Size > 1 (建议设为 4-8，视内存而定)
# 2. 保持所有逻辑一致 (强约束 Prompt + 智能摘要)

import os
import json
import pickle
import torch
import config
import data_loader
from tqdm import tqdm
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 🔧 模型配置
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cpu"

# 【关键】批量大小
# 建议从 4 开始试，如果内存够大可以加到 8 或 16
# CPU 推理虽然并行能力不如 GPU，但 Batching 依然能减少 Python 循环开销
BATCH_SIZE = 16

# 配置文件路径
PROMPT_FILE = "prompts.json"
PROGRESS_FILE_1 = "data/dbp15k/zh_en/progress_kg1.jsonl"
PROGRESS_FILE_2 = "data/dbp15k/zh_en/progress_kg2.jsonl"
FINAL_PKL_1 = "data/dbp15k/zh_en/desc_polish_1.pkl"
FINAL_PKL_2 = "data/dbp15k/zh_en/desc_polish_2.pkl"
# ==========================================

print(f"{'='*60}")
print(f"🤖 全量数据润色 (Batch Speedup Mode)")
print(f"   Model: {MODEL_ID}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"{'='*60}")

# --- 0. 加载 Prompt ---
if not os.path.exists(PROMPT_FILE):
    print(f"❌ 缺少 {PROMPT_FILE} 文件！")
    exit()
with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
    PROMPTS = json.load(f)

# --- 1. 加载模型 ---
try:
    print(f"📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 【关键】批量生成必须设置 padding_side='left'
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Load failed: {e}")
    exit()

# --- Dataset 类 ---


class PromptDataset(Dataset):
    def __init__(self, items):
        self.items = items  # List of (eid, prompt, mech_text, base_desc)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    # batch is list of tuples
    eids, prompts, mechs, descs = zip(*batch)
    return eids, prompts, mechs, descs

# --- 批量推理函数 ---


def batch_generate(prompts, system_prompt):
    # 构造对话格式
    batch_texts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        batch_texts.append(text)

    # Tokenize
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            do_sample=False
        )

    # Decode
    # 只提取新生成的部分
    outputs = []
    input_len = inputs.input_ids.shape[1]
    for i, gen_ids in enumerate(generated_ids):
        # 裁剪掉输入部分
        new_ids = gen_ids[input_len:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        outputs.append(response)

    return outputs

# --- 辅助函数 ---


def clean_name(uri):
    if not isinstance(uri, str):
        return str(uri)
    name = uri.split('/')[-1].replace('_',
                                      ' ').replace('<', '').replace('>', '')
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()
    return name.strip()


def get_smart_summary(text):
    if not text:
        return ""
    text = re.sub(r"\(.*?\)", "", text).replace("（", "").replace("）", "")
    if "。" in text:
        sentences = text.split("。")
    else:
        sentences = text.split(". ")
    summary = sentences[0].strip()
    if len(summary) < 10 and len(sentences) > 1:
        summary += "，" + sentences[1].strip()
    return summary[:80]

# --- 进度管理 ---


class ProgressManager:
    def __init__(self, log_file):
        self.log_file = log_file
        self.processed = {}
        self.load()

    def load(self):
        if not os.path.exists(self.log_file):
            return
        print(f"   🔄 Loading progress...")
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    self.processed[item['id']] = item['text']
                except:
                    pass
        print(f"   ✅ Resuming: {len(self.processed)} items done.")

    def is_done(self, eid):
        return eid in self.processed

    def save_batch(self, batch_results):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            for eid, text in batch_results:
                f.write(json.dumps({'id': eid, 'text': text},
                        ensure_ascii=False) + "\n")
                self.processed[eid] = text

# --- 3. 处理逻辑 ---


def process_kg(kg_name, ent_map, rel_map, triples, attr_map, progress_file, final_pkl, lang_code):
    print(f"\n🚀 Processing {kg_name} ({len(ent_map)} entities)...")
    pm = ProgressManager(progress_file)

    prompt_config = PROMPTS[lang_code]
    system_prompt = prompt_config["system"]
    user_template = prompt_config["user_template"]

    # 预处理：构建邻接表
    adj = defaultdict(list)
    for h, r, t in triples:
        adj[h].append((r, t))

    # 1. 准备待处理队列 (跳过已完成的)
    pending_items = []
    mech_dict = {}   # 用于最后汇总
    polish_dict = {}  # 用于最后汇总

    # 先把已完成的加载进来
    polish_dict.update(pm.processed)

    print("   Preparing task queue...")
    for eid, uri in ent_map.items():
        name = clean_name(uri)
        base_desc = attr_map.get(eid, "")

        # 如果已经处理过，且我们在 mech_dict 需要留档，这里也要生成 mech
        neighbors = adj.get(eid, [])[:5]
        simple_triples = []
        rich_triples = []

        for r, t in neighbors:
            r_n = clean_name(rel_map.get(r, "rel"))
            t_n = clean_name(ent_map.get(t, "ent"))
            t_ctx = get_smart_summary(attr_map.get(t, ""))

            if t_ctx:
                rich_item = f"- 关系: {r_n} -> 对象: {t_n} (背景: {t_ctx})"
            else:
                rich_item = f"- 关系: {r_n} -> 对象: {t_n}"

            rich_triples.append(rich_item)
            simple_triples.append(f"{r_n}: {t_n}")

        # 生成机械版 (Baseline)
        if simple_triples:
            mech_text = f"{base_desc} [SEP] Structure: {'; '.join(simple_triples)}"
        else:
            mech_text = base_desc
        mech_dict[eid] = mech_text

        # 如果已处理，跳过加入队列
        if pm.is_done(eid):
            continue

        # 如果无结构，直接保存结果
        if not simple_triples:
            pm.save_batch([(eid, base_desc)])  # 实时保存
            continue

        # 加入待处理队列
        data_str = "\n".join(rich_triples)
        prompt = user_template.format(name=name, data_str=data_str)
        pending_items.append((eid, prompt, mech_text, base_desc))

    print(f"   ⚡️ Pending tasks: {len(pending_items)}")
    if not pending_items:
        print("   ✅ All done.")
        return mech_dict, polish_dict

    # 2. 批量推理
    dataset = PromptDataset(pending_items)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            collate_fn=collate_fn, shuffle=False)

    pbar = tqdm(dataloader, desc="Batch Infer")
    for batch_eids, batch_prompts, batch_mechs, batch_descs in pbar:

        # 批量生成
        batch_outputs = batch_generate(batch_prompts, system_prompt)

        # 后处理与保存
        results_to_save = []

        for i, raw_out in enumerate(batch_outputs):
            eid = batch_eids[i]
            name = clean_name(ent_map[eid])  # 重新获取名字用于校验

            polished = raw_out.replace(
                "Output:", "").replace("结果:", "").strip()

            # 简单校验
            if len(polished) < len(name) + 5:
                polished = batch_mechs[i].split("Structure:")[-1].strip()  # 回退

            final_text = f"{batch_descs[i]} [SEP] {polished}"
            results_to_save.append((eid, final_text))

            # 监控
            # if i == 0: # 每个 Batch 打印第一条
            #    tqdm.write(f"   [Sample] {final_text[-50:]}...")

        pm.save_batch(results_to_save)

        # 更新内存中的字典
        for eid, text in results_to_save:
            polish_dict[eid] = text

    # 3. 导出
    print(f"💾 Exporting to {final_pkl}...")
    with open(final_pkl, 'wb') as f:
        pickle.dump(pm.processed, f)
    print(f"✅ {kg_name} Finished!")

    return mech_dict, pm.processed


def run():
    print("\n📚 Loading Data...")
    ent_1, _ = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    ent_2, _ = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
    rel_1, _ = data_loader.load_id_map(config.BASE_PATH + "rel_ids_1")
    rel_2, _ = data_loader.load_id_map(config.BASE_PATH + "rel_ids_2")
    trip_1 = data_loader.load_triples(config.BASE_PATH + "triples_1")
    trip_2 = data_loader.load_triples(config.BASE_PATH + "triples_2")

    attr_1 = data_loader.load_pickle_descriptions(
        config.BASE_PATH + "description1.pkl", (ent_1, {}))
    attr_2 = data_loader.load_pickle_descriptions(
        config.BASE_PATH + "description2.pkl", (ent_2, {}))

    m1, p1 = process_kg("KG1", ent_1, rel_1, trip_1, attr_1,
                        PROGRESS_FILE_1, FINAL_PKL_1, 'zh')
    m2, p2 = process_kg("KG2", ent_2, rel_2, trip_2, attr_2,
                        PROGRESS_FILE_2, FINAL_PKL_2, 'en')

    # 保存机械版 (润色版在 process_kg 里已经保存了)
    with open("data/demo_mini/zh_en/desc_mech_1.pkl", 'wb') as f:
        pickle.dump(m1, f)
    with open("data/demo_mini/zh_en/desc_mech_2.pkl", 'wb') as f:
        pickle.dump(m2, f)

    print("\n🎉 全量批量润色结束！")


if __name__ == "__main__":
    run()
