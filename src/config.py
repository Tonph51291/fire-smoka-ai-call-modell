"""Cấu hình toàn diện và tinh gọn cho ứng dụng AI Phát Hiện Lửa & Khói (Fire & Smoke AI).
Đã dọn dẹp các biến legacy thừa, chuẩn hóa cho kiến trúc đa mô hình WBF và thích ứng môi trường.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import torch

# 1. Đường dẫn thư mục gốc và phát hiện phần cứng
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

IS_CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda:0" if IS_CUDA_AVAILABLE else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if IS_CUDA_AVAILABLE else "CPU"

# ==========================================================
# 2. CẤU HÌNH CÁC MÔ HÌNH AI (MODELS)
# ==========================================================
# Model cũ (gốc)
MODEL_CU_PATH = PROJECT_ROOT / "models" / "best.pt"
MODEL_CU_CONF = 0.20
MODEL_CU_IMGSZ = 640
DEFAULT_MODEL_PATH = MODEL_CU_PATH
DEFAULT_IMGSZ = MODEL_CU_IMGSZ

# Model mới (seawsurf/fire_yolo - nhạy đêm/hồng ngoại)
MODEL_MOI_PATH = PROJECT_ROOT / "models" / "fire_yolo.pt"
MODEL_MOI_CONF = 0.20
MODEL_MOI_IMGSZ = 640

# Model ONNX (YOLOv8n 320x320 siêu nhẹ chạy cv2.dnn)
MODEL_ONNX_PATH = PROJECT_ROOT / "models" / "best.onnx"
MODEL_ONNX_CONF = 0.15
MODEL_ONNX_IMGSZ = 320

# Ngưỡng cổng thô để thu thập box ứng viên trước khi hợp nhất WBF
GATE_CU: float = 0.10
GATE_MOI: float = 0.10
GATE_ONNX: float = 0.10

# Tham số nội suy YOLO
PREDICT_IOU: float = 0.50
PREDICT_MAX_DET: int = 30

# Chế độ phối hợp ("both": 2 model 640, "tri": kèm ONNX, "cascade": chạy nhẹ trước)
ENSEMBLE_MODE: str = "tri"
HARDWARE_PROFILE_OVERRIDE: Optional[str] = None

# ==========================================================
# 3. HỢP NHẤT WEIGHTED BOXES FUSION (WBF) & MÔI TRƯỜNG QUANG HỌC
# ==========================================================
WBF_IOU_THRESH: float = 0.45
WBF_SINGLE_PENALTY_ALPHA: float = 0.35  # Phạt khi chỉ 1 model thấy box trong bối cảnh đa model
ENV_HYSTERESIS_SEC: float = 2.0         # Thời gian trễ chống rung lắc trạng thái quang học (giây)

# Cấu hình trọng số và ngưỡng phân tầng theo 6 môi trường quang học
ENVIRONMENT_CONFIGS: Dict[str, Dict] = {
    "NIGHT_IR": {
        "model_weights": {"best.pt": 0.70, "fire_yolo.pt": 1.40, "best.onnx": 0.60},
        "final_conf_fire": 0.18,
        "final_conf_smoke": 0.40,
    },
    "NIGHT_COLOR": {
        "model_weights": {"best.pt": 0.90, "fire_yolo.pt": 1.20, "best.onnx": 0.80},
        "final_conf_fire": 0.28,
        "final_conf_smoke": 0.35,
    },
    "DAY_NORMAL": {
        "model_weights": {"best.pt": 1.20, "fire_yolo.pt": 0.80, "best.onnx": 0.80},
        "final_conf_fire": 0.25,
        "final_conf_smoke": 0.30,
    },
    "GLARE_SUN": {
        "model_weights": {"best.pt": 1.00, "fire_yolo.pt": 1.00, "best.onnx": 0.60},
        "final_conf_fire": 0.35,
        "final_conf_smoke": 0.45,
    },
    "DUSK_SUNSET": {
        "model_weights": {"best.pt": 1.10, "fire_yolo.pt": 0.90, "best.onnx": 0.80},
        "final_conf_fire": 0.30,
        "final_conf_smoke": 0.35,
    },
    "FOG_HAZE": {
        "model_weights": {"best.pt": 1.00, "fire_yolo.pt": 1.00, "best.onnx": 0.70},
        "final_conf_fire": 0.25,
        "final_conf_smoke": 0.40,
    },
}

# Ngưỡng quyết định mặc định hệ thống
FINAL_CONF_FIRE: float = 0.25
FINAL_CONF_SMOKE: float = 0.30
DEFAULT_CONF_THRESHOLD: float = FINAL_CONF_FIRE

# Đồng nghĩa tên lớp (Class Synonyms)
TEN_LUA: Set[str] = {"fire", "flame", "lua", "chay"}
TEN_KHOI: Set[str] = {"smoke", "khoi"}

def is_fire(class_name: str) -> bool:
    """Kiểm tra tên lớp có thuộc nhóm Lửa hay không."""
    return class_name.strip().lower() in TEN_LUA

def is_smoke(class_name: str) -> bool:
    """Kiểm tra tên lớp có thuộc nhóm Khói hay không."""
    return class_name.strip().lower() in TEN_KHOI

DUNG_KHOI_DE_BAO: bool = True
HIEN_THI_NGUON_MODEL: bool = False

# ==========================================================
# 4. TIỀN XỬ LÝ QUANG HỌC & LỌC HÌNH DẠNG
# ==========================================================
UU_TIEN_CHO_TOI_VA_NHO: bool = True
CLAHE_CLIP_LIMIT: float = 1.5
BRIGHTNESS_THRESHOLD_DARK: float = 70.0
MIN_AREA_RATIO: float = 0.0005

BRIGHTNESS_THRESHOLD_NIGHT: float = 70.0
NIGHT_LUM_THRESHOLD: float = BRIGHTNESS_THRESHOLD_NIGHT
BRIGHTNESS_THRESHOLD_IR: float = 95.0
IR_SAT_THRESHOLD: float = 8.0
IR_COLOR_DIFF_THRESHOLD: float = 3.5

NIGHT_FIRE_CONF_SINGLE: float = 0.85
NIGHT_FIRE_CONF_DOUBLE: float = 0.25

LOC_DANG_NGUOI_DI_DUONG: bool = True
MAX_ASPECT_RATIO_LUA: float = 3.5

# ==========================================================
# 5. BỘ LỌC NGUỒN SÁNG TĨNH (STATIC LIGHT FILTER & MEMORY)
# ==========================================================
KICH_HOAT_LOC_NGUON_SANG_TINH: bool = True
CUA_SO_THEO_DOI_TINH: int = 30
NGUONG_DICH: float = 4.0
NGUONG_DONG: float = 5.0
IOU_TRACK_THRESH: float = 0.35
MIN_FRAMES_XAC_NHAN_TINH: int = 8
STATIC_CONFIRM_SEC: float = 1.0

STATIC_MEMORY_SEC: float = 600.0
STATIC_MEMORY_IOU_THRESH: float = 0.50
STATIC_RELEASE_AREA_GROWTH: float = 1.5
STATIC_RELEASE_FLICKER_STD: float = 8.0

SATURATION_PIXEL_THRESH: int = 240
SATURATION_RATIO_MIN: float = 0.05
FLICKER_STD_MAX: float = 5.0

# Vùng loại trừ thủ công (nếu người dùng cần vẽ vùng cấm cảnh báo)
EXCLUDE_ZONES: List[Tuple[float, float, float, float]] = []
EXCLUDE_ZONES_APPLY_SMOKE: bool = False

SAVE_HARD_NEGATIVES: bool = False
HARD_NEGATIVES_DIR = PROJECT_ROOT / "hard_negatives"
HARD_NEGATIVES_INTERVAL_SEC: float = 5.0
HARD_NEGATIVES_MAX_COUNT: int = 500

# ==========================================================
# 6. ĐIỀU PHỐI KHUNG HÌNH & XÁC NHẬN CHUỖI THỜI GIAN
# ==========================================================
FRAME_STRIDE: int = 1
DEQUE_WINDOW_SIZE: int = 5
MIN_FIRE_DETECTIONS_TO_ALERT: int = 3
SO_FRAME_COOLDOWN: int = DEQUE_WINDOW_SIZE
SO_FRAME_LUA_LIEN_TIEP: int = MIN_FIRE_DETECTIONS_TO_ALERT

# ==========================================================
# 7. PHÂN TÍCH NGUY CƠ, DẤU HIỆU SỐNG & BÙNG LỬA (ThreatAnalyzer)
# ==========================================================
NIGHT_LIVENESS_SEC: float = 3.0
NIGHT_FLICKER_STD_MIN: float = 3.5
NIGHT_AREA_STD_MIN: float = 15.0
NIGHT_GROWTH_MIN: float = 0.20
NIGHT_BYPASS_CONF: float = 0.85
NIGHT_BYPASS_WAIT_SEC: float = 1.0

FLARE_AREA_RATIO: float = 1.6
FLARE_MIN_SEC: float = 0.5
FLARE_JUMP_RATIO: float = 1.3
FLARE_SPREAD_RATIO: float = 0.25
FLARE_SMOKE_DIST_RATIO: float = 1.5
FLARE_HOLD_SEC: float = 60.0
FLARE_CLEAR_SEC: float = 20.0
FLARE_FAST_CONFIRM_SEC: float = 0.4
FLARE_REALERT_SEC: float = 10.0

# ==========================================================
# 8. BÁM VẾT KHÔNG-THỜI GIAN & TÍCH LŨY BẰNG CHỨNG (EvidenceAccumulator)
# ==========================================================
TAU_DECAY_FIRE: float = 2.5
TAU_DECAY_SMOKE: float = 8.0
E_SUSPECT: float = 25.0
E_CONFIRMED: float = 55.0
E_ALARM: float = 85.0
E_DROP_ALARM: float = 50.0
E_DROP_CONFIRMED: float = 30.0
E_DROP_SUSPECT: float = 10.0

TRACKER_SIM_THRESH: float = 0.25
TRACKER_MAX_MISSED_SEC: float = 1.5
SOFT_PENALTY_STATIC: float = 0.25
SOFT_PENALTY_VEHICLE: float = 0.10
EMERGENCY_GROWTH_RATIO: float = 1.30
EMERGENCY_ROUGHNESS_THRESH: float = 1.25

# ==========================================================
# 9. GHI LOG QUYẾT ĐỊNH & THƯ MỤC CẢNH BÁO
# ==========================================================
ENABLE_DECISION_LOGGER: bool = True
DECISION_LOG_DIR: Path = PROJECT_ROOT / "runs"
DECISION_LOG_PATH: Path = DECISION_LOG_DIR / "decision_log.jsonl"
CANH_BAO_DIR = PROJECT_ROOT / "canh_bao"
COOLDOWN_CANH_BAO_SEC: float = 30.0
DEFAULT_CAMERA_SOURCE = 0

# ==========================================================
# 10. GIAO DIỆN NGƯỜI DÙNG & TRÌNH DIỄN (UI & DISPLAY)
# ==========================================================
DEFAULT_DISPLAY_WIDTH = 960
DEFAULT_DISPLAY_HEIGHT = 540
UI_POLL_INTERVAL_MS = 20
BOX_PERSISTENCE_SEC = 0.8
ALERT_SMOOTHING_FRAMES = 1

SUPPORTED_VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".flv"
)

STATE_NORMAL = "NORMAL"
STATE_FIRE = "FIRE"
STATE_SMOKE = "SMOKE"
STATE_BOTH = "BOTH"

ALERT_MESSAGES = {
    STATE_NORMAL: "CHƯA PHÁT HIỆN LỬA HOẶC KHÓI",
    STATE_FIRE: "CẢNH BÁO: PHÁT HIỆN LỬA",
    STATE_SMOKE: "CẢNH BÁO: PHÁT HIỆN KHÓI",
    STATE_BOTH: "CẢNH BÁO: PHÁT HIỆN LỬA VÀ KHÓI",
}

UI_COLORS: Dict[str, str] = {
    "bg_dark": "#121824",
    "card_bg": "#1E2638",
    "card_border": "#2D3748",
    "text_primary": "#FFFFFF",
    "text_secondary": "#A0AEC0",
    "accent_primary": "#3182CE",
    "accent_hover": "#2B6CB0",
    STATE_NORMAL: "#27AE60",
    STATE_FIRE: "#E53E3E",
    STATE_SMOKE: "#ED8936",
    STATE_BOTH: "#9B2C2C",
}

CV_COLORS: Dict[str, Tuple[int, int, int]] = {
    "fire": (30, 30, 230),      # Đỏ (BGR)
    "smoke": (0, 140, 255),     # Cam (BGR)
    "default": (255, 255, 0),
}

# ==========================================================
# 11. TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBILITY ALIASES)
# ==========================================================
ACTIVE_PROFILE: str = "can_bang"
PROFILES: Dict[str, Dict] = {
    "can_bang": {
        "FINAL_CONF_FIRE": FINAL_CONF_FIRE,
        "FINAL_CONF_SMOKE": FINAL_CONF_SMOKE,
        "MIN_FIRE_DETECTIONS_TO_ALERT": MIN_FIRE_DETECTIONS_TO_ALERT,
        "DEQUE_WINDOW_SIZE": DEQUE_WINDOW_SIZE,
        "NIGHT_LIVENESS_SEC": NIGHT_LIVENESS_SEC,
        "FLARE_AREA_RATIO": FLARE_AREA_RATIO,
        "FLARE_FAST_CONFIRM_SEC": FLARE_FAST_CONFIRM_SEC,
    }
}
CHE_DO_DEM_TU_DONG: bool = True
NGUONG_CONF_DEM_LUA: float = NIGHT_FIRE_CONF_SINGLE
NGUONG_CONF_DEM_KHOI: float = 0.40
