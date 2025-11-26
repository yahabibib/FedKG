# 📄 result_logger.py
import json
import os
import datetime

RESULT_FILE = "experiment_results.json"


def log_experiment_result(exp_name, dataset, metrics, params=None):
    """
    记录实验结果到 JSON 文件
    :param exp_name: 实验名称 (如 'Isolation (SBERT)', 'FedKG')
    :param dataset: 数据集名称 (如 'dbp15k')
    :param metrics: 结果字典 (如 {'hits1': 45.2, 'mrr': 0.5})
    :param params: 额外参数 (如 {'alpha': 0.42})
    """
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": exp_name,
        "dataset": dataset,
        "metrics": metrics,
        "params": params or {}
    }

    data = []
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            data = []

    data.append(entry)

    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"📝 [Result Logger] 结果已保存到 {RESULT_FILE}")
