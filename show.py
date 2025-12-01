# 📄 inspect_empty_reason.py
# 诊断：为什么 desc_polish 里会有 4000+ 空值？是数据缺失还是匹配Bug？

import pickle
import os
import data_loader
import config
from tqdm import tqdm


def load_pkl_raw(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_diagnosis():
    print(f"{'='*60}")
    print("🕵️‍♂️ 空值原因深度诊断")
    print(f"{'='*60}")

    base_dir = "data/dbp15k/zh_en/"

    # 1. 加载 ID 映射 (ID -> URI)
    print("📚 加载 ID 映射...")
    ent_1, _ = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")

    # 2. 加载原始描述 (URI -> Text)
    print("📂 加载原始描述 (description1.pkl)...")
    raw_desc = load_pkl_raw(config.BASE_PATH + "description1.pkl")
    # 建立 URI -> Text 的快速查找表 (去除尖括号)
    uri_to_text = {}
    for k, v in raw_desc.items():
        if isinstance(k, str):
            uri_to_text[k.strip('<>')] = str(v).strip()

    # 3. 加载润色后的文件 (ID -> Text)
    print("📂 加载润色文件 (desc_polish_1.pkl)...")
    polish_desc = load_pkl_raw(base_dir + "desc_polish_1.pkl")

    # --- 诊断 1: 空值分布 ---
    print("\n📊 [诊断 1] 空值分布检查")
    empty_ids = [eid for eid, text in polish_desc.items() if not text.strip()]
    empty_ids.sort()

    print(f"   空值总数: {len(empty_ids)}")
    if empty_ids:
        print(f"   ID 范围: {min(empty_ids)} ~ {max(empty_ids)}")
        print(f"   前 10 个空 ID: {empty_ids[:10]}")

        # 检查是否集中在前部
        low_id_count = sum(1 for i in empty_ids if i < 5000)
        print(
            f"   ID < 5000 的空值数量: {low_id_count} (占比 {low_id_count/len(empty_ids)*100:.1f}%)")

    # --- 诊断 2: 丢失原因 ---
    print("\n📊 [诊断 2] 丢失原因分析 (抽样检查)")
    # 随机抽 10 个空 ID，看看它们在原始文件里有没有
    sample_check = empty_ids[:10]

    match_fail_count = 0
    no_source_count = 0

    for eid in sample_check:
        uri = ent_1.get(eid, "Unknown")
        clean_uri = uri.strip('<>')

        print(f"\n   🔹 ID: {eid}")
        print(f"      URI: {uri}")

        # 检查原始数据里有没有
        in_raw = clean_uri in uri_to_text
        if in_raw:
            print(f"      ✅ 原始文件中有描述！(长度: {len(uri_to_text[clean_uri])})")
            print(f"      ❌ 但 Step 2 没加载到 -> **匹配逻辑 Bug**")
            match_fail_count += 1
        else:
            print(f"      ⚠️ 原始文件中无描述 -> **本身缺失**")
            no_source_count += 1

    print("-" * 60)
    if match_fail_count > 0:
        print("🚨 结论：存在严重的匹配 Bug！原始文件有数据，但加载器没读出来。")
        print("   可能是 URI 编码问题 (比如 %E5%8C%97%E4%BA%AC vs 北京)")
    else:
        print("✅ 结论：代码没问题，这些实体本身就没有描述。")


if __name__ == "__main__":
    run_diagnosis()
