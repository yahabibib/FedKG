# 📄 plot_paper_figures.py
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# 设置风格 (类似论文风格)
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS',
                                   'SimHei', 'sans-serif']  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False

RESULT_FILE = "experiment_results.json"
OUTPUT_DIR = "figures"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_data():
    if not os.path.exists(RESULT_FILE):
        print("❌ 找不到结果文件，请先运行实验！")
        return []
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_method_comparison(df):
    """ 生成主实验对比图 (Bar Chart) """
    print("📊 正在生成方法对比图...")

    # 筛选最后一次出现的每个实验名 (避免重复运行导致数据堆积)
    df_latest = df.drop_duplicates(subset=['experiment'], keep='last')

    # 按照实验逻辑排序: Isolation -> FedKG -> Collection
    # 这里定义你想要的显示顺序
    sort_order = ["Isolation (SBERT)", "Isolation (Local)",
                  "FedKG (Proposed)", "Collection (Centralized)"]
    df_latest['experiment'] = pd.Categorical(
        df_latest['experiment'], categories=sort_order, ordered=True)
    df_latest = df_latest.sort_values('experiment')

    # 绘制 Hits@1 和 Hits@10
    # 需要把数据转换成长格式 (Long Format) 以便 Seaborn 绘图
    df_melt = df_latest.melt(id_vars=['experiment'], value_vars=[
                             'hits1', 'hits10'], var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melt, x='experiment', y='Score',
                     hue='Metric', palette="viridis")

    plt.title("FedKG 核心性能对比")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("")
    plt.ylim(0, 100)
    plt.legend(title="Metrics", loc='upper left')

    # 在柱子上标数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f')

    save_path = f"{OUTPUT_DIR}/comparison_bar.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"   ✅ 已保存: {save_path}")


def plot_ablation_alpha(data):
    """ 生成 Alpha 敏感性分析图 (Line Chart) """
    # 筛选出 FedKG 且有 alpha 参数的数据
    ablation_data = []
    for entry in data:
        if "alpha" in entry['params']:
            ablation_data.append({
                "Alpha": entry['params']['alpha'],
                "Hits@1": entry['metrics']['hits1']
            })

    if not ablation_data:
        print("⚠️ 没有检测到含 Alpha 参数的数据，跳过折线图。")
        return

    df = pd.DataFrame(ablation_data)
    # 去重，取最新的
    df = df.drop_duplicates(subset=['Alpha'], keep='last').sort_values('Alpha')

    print("📈 正在生成参数敏感性分析图...")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='Alpha', y='Hits@1',
                 marker='o', linewidth=2.5, markersize=8)

    plt.title("融合权重 (Alpha) 对 Hits@1 的影响")
    plt.xlabel("Structure Weight (Alpha)")
    plt.ylabel("Hits@1 (%)")
    plt.grid(True, linestyle='--')

    save_path = f"{OUTPUT_DIR}/ablation_alpha.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"   ✅ 已保存: {save_path}")


def export_results_table(df):
    """ 导出为 CSV 和 Markdown 表格 """
    df_latest = df.drop_duplicates(subset=['experiment'], keep='last')
    cols = ['experiment', 'hits1', 'hits10', 'mrr']
    df_export = df_latest[cols].copy()

    # 重命名列
    df_export.columns = ['Method', 'Hits@1 (%)', 'Hits@10 (%)', 'MRR']

    # 保存 CSV
    csv_path = f"{OUTPUT_DIR}/results_table.csv"
    df_export.to_csv(csv_path, index=False)

    print("\n📋 实验结果摘要:")
    print(df_export.to_markdown(index=False))
    print(f"\n   ✅ 表格已导出到 {OUTPUT_DIR}/")


def main():
    raw_data = load_data()
    if not raw_data:
        return

    # 展平数据结构以便 Pandas 处理
    flat_data = []
    for d in raw_data:
        item = d['metrics'].copy()
        item['experiment'] = d['experiment']
        flat_data.append(item)

    df = pd.DataFrame(flat_data)

    # 1. 画对比图
    plot_method_comparison(df)

    # 2. 画 Alpha 消融图 (传入原始数据以便提取 params)
    plot_ablation_alpha(raw_data)

    # 3. 导出表格
    export_results_table(df)


if __name__ == "__main__":
    main()
