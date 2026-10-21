"""Tầng Thực thể & Kiểu dữ liệu trung tâm (Core Entities & Domain Models).
Định nghĩa tập trung các Dataclass bất biến cho toàn bộ hệ thống phát hiện Lửa & Khói,
tránh hoàn toàn hiện tượng import chéo (circular dependencies).
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from src.config import STATE_NORMAL, ALERT_MESSAGES


class EnvironmentType:
    """Định danh 6 môi trường quang học chuẩn hóa."""
    NIGHT_IR = "NIGHT_IR"          # Hồng ngoại ban đêm đơn sắc
    NIGHT_COLOR = "NIGHT_COLOR"    # Ban đêm có đèn đường / màu
    DAY_NORMAL = "DAY_NORMAL"      # Ban ngày tiêu chuẩn
    GLARE_SUN = "GLARE_SUN"        # Chói nắng gắt / phản quang mạnh
    DUSK_SUNSET = "DUSK_SUNSET"    # Hoàng hôn / chạng vạng
    FOG_HAZE = "FOG_HAZE"          # Sương mù / khói mờ diện rộng


@dataclass
class CandidateBox:
    """Hộp phát hiện thô từ một model AI riêng lẻ."""
    class_name: str                           # "fire" hoặc "smoke"
    confidence: float                         # 0.0 - 1.0
    bbox: Tuple[int, int, int, int]           # (x1, y1, x2, y2)
    model_name: str                           # "best.pt", "fire_yolo.pt", "best.onnx"
    timestamp_sec: float = 0.0
    frame_index: int = 0


@dataclass
class FusedBox:
    """Hộp phát hiện sau khi hợp nhất đa mô hình qua Bayesian WBF."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    n_votes: int                              # Số model đồng thuận (1, 2, 3)
    total_models: int                         # Tổng số model đang kích hoạt
    sources: List[str] = field(default_factory=list)
    threat_status: str = "xac_nhan"
    timestamp_sec: float = 0.0
    frame_index: int = 0
    roughness: float = 1.0
    core_sat_ratio: float = 0.0


@dataclass
class Detection:
    """Đại diện một đối tượng phát hiện đã qua thẩm định hoàn chỉnh."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]          # (x1, y1, x2, y2)
    timestamp_sec: float = 0.0
    frame_index: int = 0
    model_source: str = "gop"                # Nguồn phát hiện: "best.pt", "fire_yolo.pt", "best.onnx", hoặc "gop"
    n_votes: int = 1                         # Số model xác nhận
    threat_status: str = "xac_nhan"          # "cho_xac_nhan" | "xac_nhan" | "bung_lua"
    sources: List[str] = field(default_factory=list)
    environment: str = EnvironmentType.DAY_NORMAL
    is_static_suspect: bool = False
    track_id: int = -1
    evidence_score: float = 0.0
    track_state: str = "NONE"                # "SUSPECT" | "CONFIRMED" | "ALARM"
    roughness: float = 1.0
    core_sat_ratio: float = 0.0


@dataclass
class FrameAnalysisResult:
    """Kết quả phân tích hợp nhất toàn diện trên một khung hình camera."""
    frame_index: int
    timestamp_sec: float
    detections: List[Detection] = field(default_factory=list)
    has_fire: bool = False
    has_smoke: bool = False
    max_fire_conf: float = 0.0
    max_smoke_conf: float = 0.0
    alert_state: str = STATE_NORMAL
    alert_message: str = ALERT_MESSAGES[STATE_NORMAL]
    inference_time_ms: float = 0.0
    updated_monotonic: float = field(default_factory=time.monotonic)
    annotated_frame: Optional[np.ndarray] = None
    fire_ratio_in_window: str = "0/0"
    is_alert_confirmed: bool = False
    bypass_fast_alert: bool = False
    environment: str = EnvironmentType.DAY_NORMAL
    timing_ms: Dict[str, float] = field(default_factory=dict)
    scene_status_text: str = ""
    dynamic_conf_fire: float = 0.25
    dynamic_conf_smoke: float = 0.30
    camera_shake: float = 0.0
    noise_level: float = 0.0
