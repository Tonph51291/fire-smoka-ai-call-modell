"""Module trích xuất đặc trưng quang học và hình thái (Optical & Morphological Features).
Gộp các thao tác cắt ảnh (crop), tính độ sáng (luminance), bão hòa lõi (saturation),
ngưỡng Otsu (Otsu threshold) và độ nhám quầng viền (boundary jitter roughness)
thành một lượt xử lý (single-pass) duy nhất để tối ưu hiệu năng CPU.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class OpticalMetrics:
    """Các chỉ số quang học trích xuất từ một vùng ROI."""
    crop: Optional[np.ndarray] = None
    thresh: Optional[np.ndarray] = None
    mean_lum: float = 0.0
    bright_area: float = 0.0
    core_sat_ratio: float = 0.0
    roughness: float = 1.0


def crop_roi_gray(
    gray_img: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    """Cắt vùng ROI an toàn từ ảnh xám (hoặc BGR)."""
    if gray_img is None or gray_img.size == 0:
        return None
    h, w = gray_img.shape[:2]
    x1 = max(0, min(w - 1, int(bbox[0])))
    y1 = max(0, min(h - 1, int(bbox[1])))
    x2 = max(0, min(w, int(bbox[2])))
    y2 = max(0, min(h, int(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = gray_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop


def extract_optical_metrics(
    raw_gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    sat_thresh: int = 235
) -> OpticalMetrics:
    """
    Trích xuất toàn diện các chỉ số quang học chỉ qua một lần crop và một lần threshold Otsu:
    - mean_lum: Độ sáng trung bình của ROI
    - bright_area: Số lượng pixel sáng sau ngưỡng Otsu
    - core_sat_ratio: Tỷ lệ pixel bão hòa cao (>= sat_thresh)
    - roughness: Độ nhám quầng viền (chu vi thực / chu vi bao lồi)
    """
    crop = crop_roi_gray(raw_gray, bbox)
    if crop is None or crop.size == 0:
        return OpticalMetrics()

    mean_lum = float(np.mean(crop))
    sat_count = np.sum(crop >= sat_thresh)
    core_sat_ratio = float(sat_count / crop.size)

    # Đơn luồng Otsu threshold
    try:
        _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_area = float(np.count_nonzero(thresh))
    except Exception:
        bright_area = float(crop.size)
        thresh = np.ones_like(crop, dtype=np.uint8) * 255

    # Tính độ nhám viền (Roughness)
    roughness = 1.0
    h_c, w_c = crop.shape[:2]
    if h_c >= 10 and w_c >= 10:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_c)
            if area >= 30:
                peri = cv2.arcLength(largest_c, True)
                hull = cv2.convexHull(largest_c)
                hull_peri = cv2.arcLength(hull, True)
                if hull_peri > 0:
                    roughness = float(peri / hull_peri)

    return OpticalMetrics(
        crop=crop,
        thresh=thresh,
        mean_lum=mean_lum,
        bright_area=bright_area,
        core_sat_ratio=core_sat_ratio,
        roughness=roughness
    )


def analyze_spark_dispersion(
    crop_gray: Optional[np.ndarray],
    bright_thresh: int = 200,
    min_spark_particles: int = 15,
    max_fill_ratio: float = 0.55
) -> Tuple[bool, int, float]:
    """
    Phân tích tính chất phân tán hạt đạn đạo của tia lửa cơ khí (hàn, cắt, mài kim loại)
    so với đám cháy chất lưu liên tục (Solid Fluid Combustion):
    - Đám cháy thật: Có thân lửa liên tục kết dính, số hạt rời rạc ít (<= 5 ở ngưỡng 200).
    - Tia lửa gia công cơ khí: Bắn ra chùm tia đạn đạo dạng nón, bao gồm hàng chục đến hàng trăm
      hạt/vệt sáng li ti rời rạc (spark_particles >= 15 ở ngưỡng 200).
    
    Returns:
        Tuple[bool, int, float]: (is_grinding_sparks, spark_particles, fill_ratio)
    """
    if crop_gray is None or crop_gray.size == 0:
        return False, 0, 1.0
    try:
        h, w = crop_gray.shape[:2]
        if h < 15 or w < 15:
            return False, 0, 1.0

        _, thresh = cv2.threshold(crop_gray, bright_thresh, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)

        # Đếm các hạt/vệt sáng nhỏ li ti (diện tích từ 2px đến 350px)
        spark_particles = 0
        total_bright_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            total_bright_area += area
            if 2 <= area <= 350:
                spark_particles += 1

        if total_bright_area == 0:
            return False, 0, 1.0

        # Tính tỷ lệ lấp đầy bao lồi (Convex Hull Fill Ratio)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fill_ratio = 1.0
        if contours:
            all_pts = np.vstack(contours)
            hull = cv2.convexHull(all_pts)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                fill_ratio = float(total_bright_area / hull_area)

        is_sparks = bool(spark_particles >= min_spark_particles or (spark_particles >= 10 and fill_ratio < max_fill_ratio))
        return is_sparks, spark_particles, fill_ratio
    except Exception:
        return False, 0, 1.0


def verify_fire_color_chromaticity(
    bgr_crop: np.ndarray,
    min_fire_ratio: float = 0.08
) -> Tuple[bool, float]:
    """
    Xác thực quang phổ ngọn lửa nhiệt động học trong không gian màu YCbCr và HSV:
    - Ngọn lửa thật có tính chất phân bố nhiệt:
      Y(x, y) > Cb(x, y) VÀ Cr(x, y) > Cb(x, y) VÀ |Cb - Cr| >= 15
    - Kênh HSV: Hue trong khoảng [0, 42] (đỏ - cam - vàng) và Saturation >= 40.
    Returns: (is_valid_fire, fire_pixel_ratio)
    """
    if bgr_crop is None or bgr_crop.size == 0 or len(bgr_crop.shape) != 3:
        return True, 1.0
    try:
        ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

        y = ycrcb[:, :, 0].astype(np.int16)
        cr = ycrcb[:, :, 1].astype(np.int16)
        cb = ycrcb[:, :, 2].astype(np.int16)

        rule_ycrcb = (y > cb) & (cr > cb) & (np.abs(cr - cb) >= 15)

        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        rule_hsv = (hue <= 42) & (sat >= 40) & (val >= 60)

        fire_pixels = np.count_nonzero(rule_ycrcb & rule_hsv)
        ratio = float(fire_pixels / bgr_crop[:, :, 0].size)
        return (ratio >= min_fire_ratio), ratio
    except Exception:
        return True, 1.0


def is_sunlight_glare(
    bgr_crop: np.ndarray,
    glare_thresh: int = 225,
    max_chroma_diff: float = 28.0
) -> bool:
    """
    Phát hiện vệt nắng mặt trời phản quang trên cửa kính, gương xe, mái tôn bóng:
    - Vệt nắng chói có độ sáng cực cao (Y >= 225) nhưng bản chất là ánh sáng trắng bạc,
      chênh lệch sắc độ màu |R - B| rất thấp (< 28).
    - Ngọn lửa thật luôn có sắc ấm rõ rệt: R áp đảo B (|R - B| >= 45) và Saturation đậm.
    """
    if bgr_crop is None or bgr_crop.size == 0 or len(bgr_crop.shape) != 3:
        return False
    try:
        b, g, r = cv2.split(bgr_crop.astype(np.float32))
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        bright_mask = gray >= glare_thresh
        bright_count = np.count_nonzero(bright_mask)
        if bright_count < 25:
            return False

        diff_rb = np.abs(r[bright_mask] - b[bright_mask])
        mean_diff = float(np.mean(diff_rb)) if diff_rb.size > 0 else 999.0

        glare_ratio = float(bright_count / gray.size)
        return bool(glare_ratio >= 0.25 and mean_diff < max_chroma_diff)
    except Exception:
        return False


def is_floor_sunlight_patch(
    bgr_crop: np.ndarray,
    bbox: Tuple[int, int, int, int],
    frame_h: int,
    core_sat_ratio: float = 0.0
) -> bool:
    """
    Phát hiện vệt nắng chiếu trên sàn nhà (gỗ, gạch men, thảm) ban ngày:
    - Vị trí: Nằm ở nửa dưới khung hình (y2 > 0.65 * H).
    - Không có lõi lửa bão hòa trắng nhiệt cao (core_sat_ratio < 0.03).
    - Cường độ sáng trung bình ở mức phản quang bề mặt, không đạt mức chói lòa của ngọn lửa (mean_lum < 165).
    """
    if bgr_crop is None or bgr_crop.size == 0 or len(bgr_crop.shape) != 3:
        return False
    try:
        y2 = bbox[3]
        if y2 <= 0.65 * frame_h:
            return False
        if core_sat_ratio >= 0.03:
            return False
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        if mean_lum > 165.0:
            return False
        return True
    except Exception:
        return False

