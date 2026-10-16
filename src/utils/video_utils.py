"""Video utility functions for Fire & Smoke AI application."""

from pathlib import Path
from typing import Dict, Tuple, Union
import cv2
import numpy as np
from PIL import Image


def format_timestamp(seconds: float, include_millis: bool = False) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    if seconds < 0:
        seconds = 0.0
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if include_millis:
        millis = int((seconds - total_seconds) * 100)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:02d}"
        return f"{minutes:02d}:{secs:02d}.{millis:02d}"
    else:
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
    """Retrieve metadata from a video file safely."""
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở file video: {path_str}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        duration_sec = total_frames / fps if fps > 0 and total_frames > 0 else 0.0

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "duration_str": format_timestamp(duration_sec),
            "filename": Path(path_str).name
        }
    finally:
        cap.release()


def resize_and_letterbox(
    frame: np.ndarray,
    target_width: int,
    target_height: int
) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize image to fit within target bounding box with black padding (letterboxing).
    Returns (letterboxed_frame, scale, pad_x, pad_y).
    """
    h, w = frame.shape[:2]
    if w == 0 or h == 0 or target_width <= 0 or target_height <= 0:
        return frame, 1.0, 0, 0

    scale = min(target_width / w, target_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Use INTER_AREA when downscaling, INTER_LINEAR when upscaling
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    # Place resized onto black canvas
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    pad_x = (target_width - new_w) // 2
    pad_y = (target_height - new_h) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def scale_bbox(
    bbox: Tuple[int, int, int, int],
    scale: float,
    pad_x: int,
    pad_y: int
) -> Tuple[int, int, int, int]:
    """Scale a bounding box (x1, y1, x2, y2) to letterbox canvas coordinates."""
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * scale + pad_x)),
        int(round(y1 * scale + pad_y)),
        int(round(x2 * scale + pad_x)),
        int(round(y2 * scale + pad_y))
    )


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR image to a PIL RGB Image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
