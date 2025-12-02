import sys
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

RESULT_FILE = os.path.join(project_root, "experiment_results.json")
OUTPUT_DIR = os.path.join(project_root, "output/figures")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_data():
    if not os.path.exists(RESULT_FILE):
        print(f"❌ 数据文件不存在: {RESULT_FILE}")
        return []
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_ablation_bar(data):
    """
    绘制消融实验对比图 (Bar Chart)
    """
    # 提取需要的字段
    records = []
    for entry in data:
        rec = {
            "Method": entry['experiment'],
            "Hits@1": entry['metrics'].get('hits1', 0),
            "MRR": entry['metrics'].get('mrr', 0)
        }
        records.append(rec)

    df = pd.DataFrame(records)
    # 去重，保留同名实验的最后一次结果
    df = df.drop_duplicates(subset=['Method'], keep='last')

    # 定义我们期望的排序 (Full 放在最右边或最左边作为 Baseline)
    # 假设我们会有这三个实验名
    order = ["No LLM (Raw SBERT)", "No Mining (Iter=1)", "FedAnchor (Full)"]
    # 过滤掉不在 order 里的杂项，或者自动排序
    df = df[df['Method'].isin(order)]
    if df.empty:
        print("⚠️ 没有找到符合名称的消融实验数据，跳过绘图。")
        return

    # 转换格式用于绘图 (Melt)
    df_melt = df.melt(id_vars=['Method'], value_vars=[
                      'Hits@1'], var_name='Metric', value_name='Score')

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df_melt, x='Method', y='Score',
                     palette="viridis", order=order)

    plt.title("Ablation Study on DBP15K (ZH-EN)")
    plt.ylabel("Hits@1 (%)")
    plt.xlabel("")
    plt.ylim(0, 80)  # 根据你的最好结果 70% 设置上限

    # 标数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f')

    save_path = os.path.join(OUTPUT_DIR, "ablation_study.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 消融实验图已保存: {save_path}")


def main():
    data = load_data()
    if not data:
        return

    print(f"📚 加载了 {len(data)} 条实验记录。")
    plot_ablation_bar(data)


if __name__ == "__main__":
    main()
