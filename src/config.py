"""Cấu hình toàn diện cho ứng dụng AI Phát Hiện Lửa & Khói.
Bao gồm thông số mô hình kép (Ensemble), hồ sơ hoạt động (Profiles),
chế độ hồng ngoại ban đêm, bộ lọc nguồn sáng tĩnh và vùng loại trừ.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import torch

# Project root directory (independent of current working directory)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Hardware device detection
IS_CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda:0" if IS_CUDA_AVAILABLE else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if IS_CUDA_AVAILABLE else "CPU"

# ==========================================================
# 1. MODEL CŨ (EXISTING MODEL) & MODEL MỚI (NEW MODEL)
# ==========================================================
# Model cũ (đang dùng)
MODEL_CU_PATH = PROJECT_ROOT / "models" / "best.pt"
MODEL_CU_CONF = 0.20           # Ngưỡng tin cậy mặc định của model cũ (dùng bởi test_anh.py)
MODEL_CU_IMGSZ = 640
DEFAULT_MODEL_PATH = MODEL_CU_PATH
DEFAULT_IMGSZ = MODEL_CU_IMGSZ

# Model mới (bổ sung - seawsurf/fire_yolo)
MODEL_MOI_PATH = PROJECT_ROOT / "models" / "fire_yolo.pt"
MODEL_MOI_CONF = 0.20          # Ngưỡng tin cậy mặc định của model mới (dùng bởi test_anh.py)
MODEL_MOI_IMGSZ = 640

# ==========================================================
# CẤU HÌNH PROFILE & NGƯỠNG HỆ THỐNG
# ==========================================================
# 3 mức profile: "chinh_xac" (ưu tiên giảm báo giả), "can_bang" (mặc định), "nhay" (ưu tiên bắt sớm)
ACTIVE_PROFILE: str = "can_bang"

PROFILES: Dict[str, Dict] = {
    "chinh_xac": {
        "FINAL_CONF_FIRE": 0.40,
        "FINAL_CONF_SMOKE": 0.45,
        "SINGLE_MODEL_PENALTY": 0.60,
        "AGREE_BONUS": 0.10,
        "MIN_FIRE_DETECTIONS_TO_ALERT": 3,
        "DEQUE_WINDOW_SIZE": 5,
        "REQUIRE_TWO_VOTES": True,
    },
    "can_bang": {
        "FINAL_CONF_FIRE": 0.30,
        "FINAL_CONF_SMOKE": 0.35,
        "SINGLE_MODEL_PENALTY": 0.70,
        "AGREE_BONUS": 0.10,
        "MIN_FIRE_DETECTIONS_TO_ALERT": 3,
        "DEQUE_WINDOW_SIZE": 5,
        "REQUIRE_TWO_VOTES": False,
    },
    "nhay": {
        "FINAL_CONF_FIRE": 0.22,
        "FINAL_CONF_SMOKE": 0.30,
        "SINGLE_MODEL_PENALTY": 0.85,
        "AGREE_BONUS": 0.12,
        "MIN_FIRE_DETECTIONS_TO_ALERT": 2,
        "DEQUE_WINDOW_SIZE": 5,
        "REQUIRE_TWO_VOTES": False,
    },
}

# Ngưỡng cổng thấp cho từng model trước khi gộp (để có đủ box ứng viên)
GATE_MOI: float = 0.20
GATE_CU: float = 0.20

# Tham số khi gọi model.predict
PREDICT_IOU: float = 0.50
PREDICT_MAX_DET: int = 30

# Tham số thuật toán hợp nhất (Fusion)
FUSE_IOU: float = 0.40
WEIGHT_MOI: float = 1.0
WEIGHT_CU: float = 1.0
AGREE_BONUS: float = PROFILES[ACTIVE_PROFILE]["AGREE_BONUS"]
SINGLE_MODEL_PENALTY: float = PROFILES[ACTIVE_PROFILE]["SINGLE_MODEL_PENALTY"]

# Ngưỡng quyết định cuối cùng (sau khi gộp)
FINAL_CONF_FIRE: float = PROFILES[ACTIVE_PROFILE]["FINAL_CONF_FIRE"]
FINAL_CONF_SMOKE: float = PROFILES[ACTIVE_PROFILE]["FINAL_CONF_SMOKE"]
DEFAULT_CONF_THRESHOLD = FINAL_CONF_FIRE

# Chế độ phối hợp Ensemble:
# - "both": Chạy cả 2 model mọi frame (mặc định)
# - "cascade": Chạy fire_yolo trước, chỉ chạy best.pt nếu fire_yolo có box ứng viên (tiết kiệm CPU)
ENSEMBLE_MODE: str = "both"

# Class Synonyms
TEN_LUA: Set[str] = {"fire", "flame", "lua", "chay"}
TEN_KHOI: Set[str] = {"smoke", "khoi"}


def is_fire(class_name: str) -> bool:
    """Kiểm tra tên lớp có thuộc nhóm Lửa hay không (không phân biệt hoa thường)."""
    return class_name.strip().lower() in TEN_LUA


def is_smoke(class_name: str) -> bool:
    """Kiểm tra tên lớp có thuộc nhóm Khói hay không (không phân biệt hoa thường)."""
    return class_name.strip().lower() in TEN_KHOI


# Alert & Execution Policies
# Kích hoạt báo động cả cho Lửa và Khói (Lửa = Đỏ, Khói = Cam, Cả hai = Đỏ đậm)
DUNG_KHOI_DE_BAO: bool = True

# ==========================================================
# ƯU TIÊN CHỖ TỐI VÀ NGỌN LỬA NHỎ (DARK & SMALL FLAME ENHANCEMENT)
# ==========================================================
UU_TIEN_CHO_TOI_VA_NHO: bool = True     # Tự động tăng tương phản thích ứng (CLAHE) vùng tối
CLAHE_CLIP_LIMIT: float = 1.5           # Giới hạn tương phản vùng tối (nhẹ hơn, tránh kích nhiễu hạt)
BRIGHTNESS_THRESHOLD_DARK: float = 70.0 # Ngưỡng độ sáng thực sự tối để bật CLAHE
MIN_AREA_RATIO: float = 0.0005          # Bỏ box có diện tích < 0.05% khung hình nếu chỉ 1 model thấy

# Hiển thị nhãn trên giao diện: CHỈ HIỆN LỬA GỘP (không hiện Mới hay Cũ)
HIEN_THI_NGUON_MODEL: bool = False      # False = Chỉ hiện 'LUA: xx%' / 'KHOI: xx%', ẩn hoàn toàn [Cu] hay [Moi]

# ==========================================================
# CHẾ ĐỘ ĐÊM (NIGHT / IR MODE)
# ==========================================================
CHE_DO_DEM_TU_DONG: bool = True
BRIGHTNESS_THRESHOLD_NIGHT: float = 70.0 # Độ sáng trung bình (Gray) < 70 mới coi là ban đêm
BRIGHTNESS_THRESHOLD_IR: float = 95.0    # Độ sáng tối đa của camera hồng ngoại IR ban đêm
IR_SAT_THRESHOLD: float = 18.0           # Ngưỡng bão hòa màu rất thấp đặc trưng của ảnh hồng ngoại (chấp nhận nhiễu nén video)
IR_COLOR_DIFF_THRESHOLD: float = 6.0     # Độ chênh lệch màu giữa các kênh gần bằng 0 ở ảnh IR

# Ngưỡng phát hiện đêm phân tầng theo số model đồng thuận:
# Box chỉ 1 model thấy cần đạt NIGHT_FIRE_CONF_SINGLE để triệt tiêu báo giả đèn đường (52% -> chặn nếu < 50%)
NIGHT_FIRE_CONF_SINGLE: float = 0.50
NIGHT_FIRE_CONF_DOUBLE: float = 0.30

# Ngưỡng lọc camera đêm cũ (giữ tương thích)
NGUONG_CONF_DEM_LUA: float = NIGHT_FIRE_CONF_SINGLE  # Ngưỡng đêm cho lửa tương thích cũ (đọc bởi evaluate.py)
NGUONG_CONF_DEM_KHOI: float = 0.45                   # Ngưỡng đêm cho khói tương thích cũ

# Lọc hình dạng người đi đường (aspect ratio = H / W >= 2.0 dạng đứng)
LOC_DANG_NGUOI_DI_DUONG: bool = True
MAX_ASPECT_RATIO_LUA: float = 2.0

# ==========================================================
# BỘ LỌC NGUỒN SÁNG TĨNH (LOC NGUON SANG TINH & STATIC MEMORY)
# ==========================================================
KICH_HOAT_LOC_NGUON_SANG_TINH: bool = True
CUA_SO_THEO_DOI_TINH: int = 30         # Theo dõi hộp qua tối đa 30 frame
NGUONG_DICH: float = 4.0               # Ngưỡng dịch chuyển tâm (pixels): nếu < nguong_dich -> vị trí tĩnh
NGUONG_DONG: float = 5.0               # Ngưỡng biến thiên nội dung pixel: nếu < nguong_dong -> nội dung tĩnh
IOU_TRACK_THRESH: float = 0.35         # Ngưỡng IoU để ghép cặp hộp qua các frame
MIN_FRAMES_XAC_NHAN_TINH: int = 8      # Số frame tối thiểu để bắt đầu đánh giá tính tĩnh
STATIC_CONFIRM_SEC: float = 1.0        # Xác nhận nguồn sáng tĩnh sau tối thiểu 1.0 giây theo dõi liên tục

# Bộ nhớ nguồn sáng tĩnh tự học (Static Memory)
STATIC_MEMORY_SEC: float = 600.0       # Lưu vị trí đèn tĩnh đã xác nhận trong 600 giây (10 phút)
STATIC_MEMORY_IOU_THRESH: float = 0.50 # Ngưỡng IoU so khớp với bộ nhớ đèn tĩnh để loại bỏ ngay tức thì
STATIC_RELEASE_AREA_GROWTH: float = 1.5# Mở khóa vùng đèn tĩnh khi diện tích box tăng >= 1.5 lần (dấu hiệu cháy lan)
STATIC_RELEASE_FLICKER_STD: float = 8.0# Mở khóa vùng đèn tĩnh khi độ lệch chuẩn nhấp nháy tăng vọt do lửa bập bùng

# Đặc trưng bão hòa ánh sáng và nhấp nháy
SATURATION_PIXEL_THRESH: int = 240     # Ngưỡng pixel bão hòa trắng trong lõi bóng đèn
SATURATION_RATIO_MIN: float = 0.05     # Tỉ lệ pixel bão hòa tối thiểu trong lõi đèn
FLICKER_STD_MAX: float = 5.0           # Độ lệch chuẩn nhấp nháy độ sáng tối đa để coi là đèn tĩnh

# ==========================================================
# VÙNG LOẠI TRỪ CẤU HÌNH TAY (EXCLUDE ZONES)
# ==========================================================
# Danh sách vùng cấm cảnh báo theo tọa độ chuẩn hóa 0-1: [(x1, y1, x2, y2), ...]
# Ví dụ: [(0.35, 0.05, 0.50, 0.30)] sẽ bỏ qua box lửa có tâm nằm trong vùng này
EXCLUDE_ZONES: List[Tuple[float, float, float, float]] = []
EXCLUDE_ZONES_APPLY_SMOKE: bool = False # False = chỉ áp dụng cho Lửa, Khói vẫn báo động bình thường

# ==========================================================
# THU THẬP HARD NEGATIVES ĐỂ FINE-TUNE
# ==========================================================
SAVE_HARD_NEGATIVES: bool = False       # Tùy chọn lưu frame bị static filter loại vào thư mục hard_negatives/
HARD_NEGATIVES_DIR = PROJECT_ROOT / "hard_negatives"
HARD_NEGATIVES_INTERVAL_SEC: float = 5.0# Khoảng cách tối thiểu giữa 2 lần lưu ảnh hard negative
HARD_NEGATIVES_MAX_COUNT: int = 500    # Giới hạn tối đa số ảnh lưu

# Xác nhận phát hiện qua chuỗi frame theo Profile
DEQUE_WINDOW_SIZE: int = PROFILES[ACTIVE_PROFILE]["DEQUE_WINDOW_SIZE"]             # Kích thước cửa sổ trượt (frames) của profile hiện hành
MIN_FIRE_DETECTIONS_TO_ALERT: int = PROFILES[ACTIVE_PROFILE]["MIN_FIRE_DETECTIONS_TO_ALERT"] # Số lần phát hiện tối thiểu để cảnh báo của profile hiện hành

# Tùy chọn kiểm tra trùng khớp vị trí không gian (Spatial Consistency)
ENABLE_SPATIAL_CONSISTENCY: bool = False
SPATIAL_CONSISTENCY_IOU: float = 0.20

# Frame stride: Xử lý từng frame để bắt lửa sớm nhất (stride=1)
FRAME_STRIDE: int = 1

# Cooldown & Alert Logging
CANH_BAO_DIR = PROJECT_ROOT / "canh_bao"
COOLDOWN_CANH_BAO_SEC: float = 30.0

# Camera source: 0 for webcam, or RTSP stream string (dùng bởi detect_camera.py)
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
