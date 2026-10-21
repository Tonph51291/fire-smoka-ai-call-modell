"""Tiền xử lý và tăng cường chất lượng hình ảnh camera (Image Enhancements & Preprocessing).
Tách biệt toàn bộ các thuật toán xử lý không gian màu LAB, HSV, CLAHE ra khỏi DetectionService.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from src.config import (
    BRIGHTNESS_THRESHOLD_IR,
    CLAHE_CLIP_LIMIT,
    IR_COLOR_DIFF_THRESHOLD,
    IR_SAT_THRESHOLD,
    NIGHT_LUM_THRESHOLD,
)


from src.utils.geometry import scale_bbox


def compute_brightness_and_contrast(gray_img: np.ndarray) -> Tuple[float, float]:
    """Tính toán nhanh độ sáng trung bình (mean luminance) và độ tương phản (std deviation)."""
    if gray_img is None or gray_img.size == 0:
        return 0.0, 0.0
    h, w = gray_img.shape[:2]
    step_y = max(1, h // 120)
    step_x = max(1, w // 160)
    sub = gray_img[::step_y, ::step_x]
    mean_val = float(np.mean(sub))
    std_val = float(np.std(sub))
    return mean_val, std_val


def compute_dynamic_color_mask(frame: np.ndarray, is_night: bool = False) -> np.ndarray:
    """Tạo mặt nạ nhị phân lọc vùng màu nghi vấn lửa (HSV & YCrCb) thích ứng theo ánh sáng."""
    if frame is None or frame.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    if is_night:
        lower_hsv = np.array([0, 40, 140], dtype=np.uint8)
        upper_hsv = np.array([38, 255, 255], dtype=np.uint8)
    else:
        lower_hsv = np.array([0, 70, 100], dtype=np.uint8)
        upper_hsv = np.array([35, 255, 255], dtype=np.uint8)

    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = (ycrcb[:, :, 1] >= 140) & (ycrcb[:, :, 2] <= 135)
    combined = cv2.bitwise_and(mask_hsv, mask_hsv, mask=mask_ycrcb.astype(np.uint8) * 255)
    return combined


def is_night_or_ir_frame(frame: np.ndarray) -> bool:
    """
    Nhận diện nhanh khung hình ban đêm hoặc camera hồng ngoại đơn sắc:
    - Ban đêm màu: độ sáng trung bình kênh V trong HSV < NIGHT_LUM_THRESHOLD (80).
    - Camera hồng ngoại IR: độ bão hòa màu S cực thấp (< 12) VÀ chênh lệch kênh màu R-G-B < 5.0.
    """
    if frame is None or frame.size == 0:
        return False
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        lum = float(np.mean(val))

        if lum < NIGHT_LUM_THRESHOLD:
            return True

        sat_mean = float(np.mean(sat))
        b, g, r = cv2.split(frame)
        rg_diff = float(np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32))))
        gb_diff = float(np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32))))
        color_diff = (rg_diff + gb_diff) / 2.0

        is_ir = (sat_mean < IR_SAT_THRESHOLD) and (color_diff < IR_COLOR_DIFF_THRESHOLD) and (lum < BRIGHTNESS_THRESHOLD_IR)
        return is_ir
    except Exception:
        return False


def enhance_dark_frame_clahe(img: np.ndarray, clip_limit: float = CLAHE_CLIP_LIMIT) -> np.ndarray:
    """
    Tăng cường tương phản cục bộ thích ứng (CLAHE) trong không gian màu LAB trên kênh L:
    - Làm rõ các chi tiết đốm lửa, tia lửa nhỏ trong các vùng tối / ban đêm.
    - clip_limit nhẹ hơn (1.5) để tránh phóng đại nhiễu hạt (sensor noise).
    """
    if img is None or img.size == 0:
        return img
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_enhanced = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    except Exception:
        return img


def suppress_specular_highlights(
    frame: np.ndarray,
    sat_thresh: int = 240,
    suppression_ratio: float = 0.20
) -> np.ndarray:
    """
    Nén dải động cục bộ quanh các tâm sáng bão hòa (Highlight / Specular Glare Suppression):
    - Đèn pha xe máy/ô tô hoặc chao đèn bão hòa sáng (>= sat_thresh).
    - Tạo quầng làm dịu (halo suppression) giảm độ chói lóa 15-20% quanh tâm đèn,
      giúp model phát hiện được chi tiết đám cháy hoặc khói xung quanh mà không bị lóa.
    """
    if frame is None or frame.size == 0:
        return frame
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, sat_mask = cv2.threshold(gray, sat_thresh, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(sat_mask) == 0:
            return frame

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        halo_mask = cv2.dilate(sat_mask, kernel, iterations=2)

        attenuated = frame.astype(np.float32)
        mask_f = (halo_mask.astype(np.float32) / 255.0)
        if len(frame.shape) == 3:
            mask_f = mask_f[:, :, np.newaxis]
        attenuated = attenuated * (1.0 - mask_f * suppression_ratio)
        return np.clip(attenuated, 0, 255).astype(np.uint8)
    except Exception:
        return frame


def balance_sunset_illumination(
    frame: np.ndarray,
    strength: float = 0.50
) -> np.ndarray:
    """
    Cân bằng trắng thích ứng cho cảnh hoàng hôn / ngược sáng (Dynamic Gray-World Correction):
    - Khi hoàng hôn, ánh cam/đỏ bao phủ toàn cảnh làm model dễ nhận nhầm tường/đường thành lửa.
    - Nắn chỉnh độ lệch sắc đỏ nhẹ nhàng (strength=0.50) để giữ tự nhiên nhưng khử sắc cam giả.
    """
    if frame is None or frame.size == 0 or len(frame.shape) != 3:
        return frame
    try:
        b, g, r = cv2.split(frame.astype(np.float32))
        r_mean = float(np.mean(r))
        g_mean = float(np.mean(g))
        b_mean = float(np.mean(b))

        if r_mean <= 1.0 or b_mean <= 1.0:
            return frame

        # Kiểm tra nếu kênh R áp đảo kênh B (hoàng hôn / ánh đèn natri)
        if r_mean / max(1.0, b_mean) > 1.30:
            gray_target = (r_mean + g_mean + b_mean) / 3.0
            scale_r = 1.0 - strength * (1.0 - gray_target / r_mean)
            scale_b = 1.0 + strength * (gray_target / b_mean - 1.0)
            scale_g = 1.0 - strength * (1.0 - gray_target / max(1.0, g_mean))

            r = np.clip(r * scale_r, 0, 255)
            g = np.clip(g * scale_g, 0, 255)
            b = np.clip(b * scale_b, 0, 255)
            return cv2.merge((b, g, r)).astype(np.uint8)
        return frame
    except Exception:
        return frame


def compute_dark_channel_prior(
    frame: np.ndarray,
    patch_size: int = 15
) -> float:
    """
    Đo độ tán xạ sương mù theo kênh tối (Dark Channel Prior):
    - Sương mù hoặc mưa phùn làm kênh tối cao đều toàn cảnh.
    - Khói cháy cục bộ có kênh tối thấp ở nền và cao đột ngột tại gốc khói.
    """
    if frame is None or frame.size == 0 or len(frame.shape) != 3:
        return 0.0
    try:
        min_channel = np.min(frame, axis=2)
        h, w = min_channel.shape[:2]
        small_min = cv2.resize(min_channel, (min(160, w), min(120, h)), interpolation=cv2.INTER_AREA)
        k_sz = max(3, patch_size // 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz))
        dark_channel = cv2.erode(small_min, kernel)
        return float(np.mean(dark_channel))
    except Exception:
        return 0.0
