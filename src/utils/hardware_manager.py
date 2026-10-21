"""Quản lý phần cứng và tự động điều chỉnh hồ sơ hiệu năng (Hardware Manager).
Tự động phát hiện:
- CUDA, tên GPU, dung lượng VRAM.
- Số nhân/luồng CPU, tổng dung lượng RAM.
- Gán hồ sơ: CPU_ONLY, GPU_LOW (<=6GB VRAM), GPU_MID+ (>6GB VRAM).
- Chạy benchmark khởi động (warmup) để ước tính FPS suy luận khả thi.
- Cảnh báo rõ ràng nếu hệ thống bị rơi về CPU.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
import os
import time
import cv2
import numpy as np
import torch

logger = logging.getLogger("FireSmokeAI.Hardware")


@dataclass
class HardwareProfile:
    """Hồ sơ cấu hình tương ứng với năng lực phần cứng."""
    name: str                 # "CPU_ONLY" | "GPU_LOW" | "GPU_MID+"
    device: str               # "cuda:0" | "cpu"
    use_fp16: bool            # True nếu GPU hỗ trợ FP16 half-precision
    max_models_active: int    # Số model chạy đồng thời tối đa
    recommended_stride: int   # Bước nhảy frame (1 = mọi frame, 2-3 = bỏ bớt frame trên CPU)
    enable_heavy_features: bool # Có bật các tính năng nặng (như CLAHE toàn cục, optical flow) không
    description: str


class HardwareManager:
    """Quản trị viên phát hiện phần cứng và thiết lập cấu hình runtime."""

    def __init__(self, override_profile: Optional[str] = None) -> None:
        self.is_cuda = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.is_cuda else 0
        self.gpu_name = torch.cuda.get_device_name(0) if self.is_cuda else "Không có GPU CUDA"
        self.vram_gb = 0.0

        if self.is_cuda:
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                self.vram_gb = float(total_mem / (1024 ** 3))
            except Exception:
                self.vram_gb = 0.0

        self.cpu_cores = os.cpu_count() or 4
        self.profile = self._determine_profile(override_profile)

        # Áp dụng cấu hình tối ưu luồng nếu là CPU
        if not self.is_cuda:
            torch.set_num_threads(max(1, min(self.cpu_cores, 8)))

        self._log_hardware_summary()

    def _determine_profile(self, override: Optional[str]) -> HardwareProfile:
        if override == "CPU_ONLY":
            return HardwareProfile("CPU_ONLY", "cpu", False, 2, 3, False, "Cưỡng bức chạy CPU")
        if override == "GPU_LOW":
            return HardwareProfile("GPU_LOW", "cuda:0", True, 2, 1, True, "Cưỡng bức GPU VRAM thấp")
        if override == "GPU_MID+":
            return HardwareProfile("GPU_MID+", "cuda:0", True, 3, 1, True, "Cưỡng bức GPU VRAM cao")

        if not self.is_cuda:
            return HardwareProfile(
                name="CPU_ONLY",
                device="cpu",
                use_fp16=False,
                max_models_active=2,
                recommended_stride=2,
                enable_heavy_features=False,
                description="Chế độ thuần CPU: Tiết kiệm tài nguyên, ưu tiên stride=2 để đảm bảo phát mượt"
            )

        if self.vram_gb <= 6.5:
            return HardwareProfile(
                name="GPU_LOW",
                device="cuda:0",
                use_fp16=True,
                max_models_active=2,
                recommended_stride=1,
                enable_heavy_features=True,
                description=f"GPU {self.gpu_name} ({self.vram_gb:.1f}GB VRAM): Bật FP16, tối ưu 2 model 640"
            )

        return HardwareProfile(
            name="GPU_MID+",
            device="cuda:0",
            use_fp16=True,
            max_models_active=3,
            recommended_stride=1,
            enable_heavy_features=True,
            description=f"GPU {self.gpu_name} ({self.vram_gb:.1f}GB VRAM): Bật FP16, hỗ trợ đầy đủ 3 model"
        )

    def _log_hardware_summary(self) -> None:
        logger.info("=" * 70)
        logger.info("🖥️  THÔNG TIN PHẦN CỨNG & HỒ SƠ THỰC THI (HARDWARE MANAGER)")
        logger.info(f"  • CPU Cores / Threads: {self.cpu_cores}")
        if self.is_cuda:
            logger.info(f"  • CUDA khả dụng: CÓ | GPU: {self.gpu_name} ({self.vram_gb:.1f} GB VRAM)")
            logger.info(f"  • Hồ sơ kích hoạt: [{self.profile.name}] - {self.profile.description}")
        else:
            logger.warning("  • CUDA khả dụng: KHÔNG (HỆ THỐNG ĐANG RƠI VỀ THUẦN CPU!)")
            logger.warning("  ⚠️  CHÚ Ý: Model PyTorch đang chạy trên CPU, tốc độ sẽ chậm hơn GPU 5-10 lần.")
            logger.warning("  ⚠️  Nếu máy có GPU NVIDIA, hãy cài lại PyTorch CUDA (ví dụ: cu121 hoặc cu124).")
            logger.info(f"  • Hồ sơ kích hoạt: [{self.profile.name}] - {self.profile.description}")
        logger.info("=" * 70)
