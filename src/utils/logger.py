# src/utils/logger.py
import logging
import os
import sys
import json
from datetime import datetime


def setup_logger(name: str, save_dir: str = "logs") -> logging.Logger:
    """
    配置一个同时输出到控制台和文件的 Logger
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重复打印

    # 清除旧的 handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # 格式
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件 Handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class ResultRecorder:
    """
    【新增】实验结果记录器
    将实验的最终指标保存为 JSON，方便后续通过 plot_results.py 画图。
    """

    def __init__(self, filepath="experiment_results.json"):
        # 自动定位到项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils
        project_root = os.path.dirname(os.path.dirname(current_dir))  # root
        self.filepath = os.path.join(project_root, filepath)

    def add_record(self, exp_name: str, metrics: dict, config_diff: dict = None):
        """
        :param exp_name: 实验名称 (如 "FedAnchor (Full)")
        :param metrics: 结果字典 (如 {"hits1": 70.1, "mrr": 0.76})
        :param config_diff: 这一组实验的特殊配置
        """
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": exp_name,
            "metrics": metrics,
            "config": config_diff or {}
        }

        data = []
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = []

        data.append(record)

        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"📝 实验结果已追加到: {self.filepath}")
