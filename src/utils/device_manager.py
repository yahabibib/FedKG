import torch
import gc
import logging
import os

log = logging.getLogger(__name__)


class DeviceManager:
    """
    智能设备管理器：
    1. 自动适配 MPS/CUDA/CPU
    2. 提供统一的显存清理接口 (clean_memory)
    3. 管理 Offloading 策略状态
    """

    def __init__(self, cfg_system):
        self.cfg = cfg_system
        self.device = self._init_main_device()
        self._setup_env()

    def _init_main_device(self):
        """根据配置和硬件现状初始化主计算设备"""
        req = self.cfg.device.lower()

        # 1. 尝试 CUDA
        if req == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            log.warning("⚠️ Config requested CUDA but not available.")

        # 2. 尝试 MPS (Mac)
        if req == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            log.warning("⚠️ Config requested MPS but not available.")

        # 3. 回退
        log.info(f"Using fallback device: {self.cfg.fallback_device}")
        return torch.device(self.cfg.fallback_device)

    def _setup_env(self):
        """设置环境变量优化"""
        if self.device.type == 'mps':
            # Mac 显存优化关键环境变量，设为 0.0 解除上限限制，防止过早报错
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            log.info(
                "🍎 MPS Mode Detected: Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")

    @property
    def main_device(self):
        """返回用于计算的设备 (GPU/MPS)"""
        return self.device

    @property
    def cpu_device(self):
        """返回用于存储的设备"""
        return torch.device("cpu")

    def is_offload_enabled(self):
        """是否启用卸载策略"""
        return self.cfg.memory.offload_to_cpu

    def clean_memory(self):
        """
        强制垃圾回收和显存释放。
        在 Mac 上，这对于防止 OOM 至关重要。
        """
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Python 层的垃圾回收
        gc.collect()

    def get_safe_batch_size(self, requested_batch_size):
        """
        检查 task 请求的 batch_size 是否超过了 system 定义的安全上限
        """
        limit = self.cfg.get("max_batch_size", None)
        if limit and requested_batch_size > limit:
            log.warning(
                f"⚠️ Requested batch_size {requested_batch_size} exceeds system limit {limit}. Clamping to {limit}.")
            return limit
        return requested_batch_size
