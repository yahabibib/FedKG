# src/utils/logger.py
import json
import os
import datetime
import logging
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def log_experiment_result(experiment_name, dataset_name, metrics, config=None, filename="experiment_results.json"):
    """
    将实验结果追加保存到 JSON 文件。

    :param experiment_name: 实验名称 (如 sbert_mixed_round5)
    :param dataset_name: 数据集名称 (如 dbp15k)
    :param metrics: 结果字典 (如 {'hits1': 65.2, 'mrr': 0.7})
    :param config: (可选) Hydra 配置对象，用于记录超参数
    :param filename: 保存的文件名
    """
    # 1. 准备数据条目
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": experiment_name,
        "dataset": dataset_name,
        "metrics": metrics,
        "params": {}
    }

    # 2. 如果传入了配置，提取关键超参数
    if config:
        if isinstance(config, DictConfig):
            # 将 OmegaConf 转为普通字典
            conf_dict = OmegaConf.to_container(config, resolve=True)
            entry["params"] = conf_dict.get('task', {})
        else:
            entry["params"] = config

    # 3. 读取现有数据 (Append Mode)
    data = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            log.warning(
                f"⚠️ Failed to load existing results: {e}. Starting new log.")
            data = []

    data.append(entry)

    # 4. 写入文件
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        log.info(f"📝 Results logged to {filename}")
    except Exception as e:
        log.error(f"❌ Failed to log results: {e}")
