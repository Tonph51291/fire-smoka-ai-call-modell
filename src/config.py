"""Configuration settings for Fire & Smoke AI application."""

from pathlib import Path
from typing import Dict, Tuple
import torch

# Project root directory (independent of current working directory)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Hardware device detection
IS_CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda:0" if IS_CUDA_AVAILABLE else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if IS_CUDA_AVAILABLE else "CPU"

# Model settings
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_CONF_THRESHOLD = 0.25

# Inference resolution: 640 on GPU, 416 on CPU for optimal speed/accuracy trade-off
DEFAULT_IMGSZ = 640 if IS_CUDA_AVAILABLE else 416
HALF_PRECISION = IS_CUDA_AVAILABLE

# Timing and performance settings
DEFAULT_DISPLAY_WIDTH = 960
DEFAULT_DISPLAY_HEIGHT = 540
UI_POLL_INTERVAL_MS = 20       # ~50 FPS UI poll rate
BOX_PERSISTENCE_SEC = 0.5      # Bounding boxes expire after 0.5s if no new AI result
ALERT_SMOOTHING_FRAMES = 2     # Consecutive detections before confirming alert

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
