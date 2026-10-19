"""Configuration settings for Fire & Smoke AI application."""

from pathlib import Path
from typing import Dict, Set, Tuple
import torch

# Project root directory (independent of current working directory)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Hardware device detection
IS_CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda:0" if IS_CUDA_AVAILABLE else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if IS_CUDA_AVAILABLE else "CPU"
HALF_PRECISION = IS_CUDA_AVAILABLE

# ==========================================================
# 1. MODEL CŨ (EXISTING MODEL) & MODEL MỚI (NEW MODEL)
# ==========================================================
# Model cũ (đang dùng)
MODEL_CU_PATH = PROJECT_ROOT / "models" / "best.pt"
MODEL_CU_CONF = 0.15
MODEL_CU_IMGSZ = 640
DEFAULT_MODEL_PATH = MODEL_CU_PATH
DEFAULT_CONF_THRESHOLD = MODEL_CU_CONF
DEFAULT_IMGSZ = MODEL_CU_IMGSZ

# Model mới (bổ sung - seawsurf/fire_yolo)
MODEL_MOI_PATH = PROJECT_ROOT / "models" / "fire_yolo.pt"
MODEL_MOI_REPO_ID = "seawsurf/fire_yolo"
MODEL_MOI_CONF = 0.15
MODEL_MOI_IMGSZ = 640

# Class Synonyms
TEN_LUA: Set[str] = {"fire", "flame", "lua", "chay"}
TEN_KHOI: Set[str] = {"smoke", "khoi"}

# Alert & Execution Policies
# Kích hoạt báo động cả cho Lửa và Khói (Lửa = Đỏ, Khói = Cam, Cả hai = Đỏ đậm)
DUNG_KHOI_DE_BAO: bool = True

# ==========================================================
# ƯU TIÊN CHỖ TỐI VÀ NGỌN LỬA NHỎ (DARK & SMALL FLAME ENHANCEMENT)
# ==========================================================
UU_TIEN_CHO_TOI_VA_NHO: bool = True     # Tự động tăng tương phản thích ứng (CLAHE) vùng tối
CLAHE_CLIP_LIMIT: float = 2.0           # Giới hạn tương phản vùng tối
NGUONG_CONF_LUA_NHO: float = 0.20       # Ngưỡng an toàn cho ngọn lửa nhỏ trong vùng tối (tránh bắt nhầm người/biển báo)
DIEN_TICH_LUA_NHO_MAX_PX: int = 4000    # Diện tích tối đa (px) được coi là đốm lửa nhỏ

# Hiển thị nhãn trên giao diện: CHỈ HIỆN LỬA GỘP (không hiện Mới hay Cũ)
HIEN_THI_NGUON_MODEL: bool = False      # False = Chỉ hiện 'LUA: xx%' / 'KHOI: xx%', ẩn hoàn toàn [Cu] hay [Moi]

# ==========================================================
# CHẾ ĐỘ ĐÊM (NIGHT / IR MODE)
# ==========================================================
# Tự động nhận biết ảnh không màu / camera hồng ngoại ban đêm
CHE_DO_DEM_TU_DONG: bool = True
# Ngưỡng lọc camera đêm: loại bỏ người đi đường áo trắng (<25%), biển báo (<20%) và quầng khói mặt đường (<45%)
NGUONG_CONF_DEM_LUA: float = 0.25
NGUONG_CONF_DEM_KHOI: float = 0.45
NGUONG_CONF_DEM_DON_LE: float = 0.45

# Lọc hình dạng người đi đường (aspect ratio = H / W > 2.0 dạng đứng)
LOC_DANG_NGUOI_DI_DUONG: bool = True
MAX_ASPECT_RATIO_LUA: float = 2.0

# ==========================================================
# BỘ LỌC NGUỒN SÁNG TĨNH (LOC NGUON SANG TINH)
# ==========================================================
# Bật bộ lọc theo dõi hộp qua 30 frame để loại bỏ đèn đường / đèn hiên cố định
KICH_HOAT_LOC_NGUON_SANG_TINH: bool = True
CUA_SO_THEO_DOI_TINH: int = 30         # Theo dõi hộp qua tối đa 30 frame
NGUONG_DICH: float = 4.0               # Ngưỡng dịch chuyển tâm (pixels): nếu < nguong_dich -> vị trí tĩnh
NGUONG_DONG: float = 5.0               # Ngưỡng biến thiên nội dung pixel: nếu < nguong_dong -> nội dung tĩnh
IOU_TRACK_THRESH: float = 0.35         # Ngưỡng IoU để ghép cặp hộp qua các frame
MIN_FRAMES_XAC_NHAN_TINH: int = 8      # Sau 8 frame đứng im là xác nhận vật thể tĩnh (rút ngắn thời gian loại trừ)

# Tùy chọn chạy OR: Nếu True, chỉ gọi model cũ khi model mới không thấy lửa để tiết kiệm tài nguyên
CHI_GOI_MODEL_CU_KHI_MODEL_MOI_KHONG_THAY: bool = False

# Phát hiện lửa sớm nhất: 2 frame liên tiếp (~0.06s - 0.08s) để khử nhiễu chớp 1 frame của xe cộ
DEQUE_WINDOW_SIZE: int = 5
MIN_FIRE_DETECTIONS_TO_ALERT: int = 2

# Frame stride: Xử lý từng frame để bắt lửa sớm nhất (stride=1)
FRAME_STRIDE: int = 1

# Cooldown & Alert Logging
CANH_BAO_DIR = PROJECT_ROOT / "canh_bao"
COOLDOWN_CANH_BAO_SEC: float = 30.0

# Camera source: 0 for webcam, or RTSP stream string
DEFAULT_CAMERA_SOURCE = 0

# ==========================================================
# 2. UI & DISPLAY CONFIGURATIONS
# ==========================================================
DEFAULT_DISPLAY_WIDTH = 960
DEFAULT_DISPLAY_HEIGHT = 540
UI_POLL_INTERVAL_MS = 20       # ~50 FPS UI poll rate
BOX_PERSISTENCE_SEC = 0.8      # Bounding boxes expire after 0.8s if no new AI result
ALERT_SMOOTHING_FRAMES = 1     # Immediate UI alert when detection confirmed by filter

# Supported video extensions
SUPPORTED_VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".webm",
    ".flv"
)

# Alert States
STATE_NORMAL = "NORMAL"
STATE_FIRE = "FIRE"
STATE_SMOKE = "SMOKE"
STATE_BOTH = "BOTH"

# Human readable alert messages
ALERT_MESSAGES = {
    STATE_NORMAL: "CHƯA PHÁT HIỆN LỬA HOẶC KHÓI",
    STATE_FIRE: "CẢNH BÁO: PHÁT HIỆN LỬA",
    STATE_SMOKE: "CẢNH BÁO: PHÁT HIỆN KHÓI",
    STATE_BOTH: "CẢNH BÁO: PHÁT HIỆN LỬA VÀ KHÓI",
}

# UI Theme & Colors
UI_COLORS: Dict[str, str] = {
    "bg_dark": "#121824",
    "card_bg": "#1E2638",
    "card_border": "#2D3748",
    "text_primary": "#FFFFFF",
    "text_secondary": "#A0AEC0",
    "accent_primary": "#3182CE",
    "accent_hover": "#2B6CB0",
    
    # State colors (Hex)
    STATE_NORMAL: "#27AE60",   # Soft green
    STATE_FIRE: "#E53E3E",     # Crimson red
    STATE_SMOKE: "#ED8936",    # Vivid orange
    STATE_BOTH: "#9B2C2C",     # Deep intense red
}

# OpenCV Bounding Box Colors (BGR format)
CV_COLORS: Dict[str, Tuple[int, int, int]] = {
    "fire": (30, 30, 230),      # Bright Red (BGR)
    "smoke": (0, 140, 255),     # Bright Orange (BGR)
    "default": (255, 255, 0),   # Cyan/Yellow
}
