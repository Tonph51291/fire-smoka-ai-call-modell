"""Tầng Hiển thị & Trình diễn Cảnh báo (Alert Rendering & Visualization).
Chuyên trách vẽ bounding box, nhãn text, phân loại màu sắc theo mức độ nguy cơ,
vẽ header banner cảnh báo và lưu trữ ảnh báo động vào thư mục canh_bao/.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.config import (
    CANH_BAO_DIR,
    CV_COLORS,
    is_fire,
    is_smoke,
)
from src.core.entities import Detection
from src.utils.image_enhancer import scale_bbox

logger = logging.getLogger("FireSmokeAI.Renderer")


class AlertRenderer:
    """Quản lý hiển thị trực quan và lưu ảnh cảnh báo."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or CANH_BAO_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def annotate_alert_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        timestamp_sec: float,
        max_fire_conf: float,
        fire_ratio_str: str,
    ) -> np.ndarray:
        """
        Vẽ bounding box, nhãn độ tin cậy và banner tóm tắt thông tin trên đầu ảnh cảnh báo.
        """
        annotated = frame.copy()
        h_img, w_img = annotated.shape[:2]

        for det in detections:
            cname_lower = det.class_name.lower()
            threat_st = getattr(det, "threat_status", "xac_nhan")

            if is_fire(cname_lower):
                if threat_st == "bung_lua":
                    color = (0, 0, 255)      # Đỏ thẫm: Bùng lửa đột biến
                    label_prefix = "BUNG LUA"
                elif threat_st == "cho_xac_nhan":
                    color = (0, 255, 255)    # Vàng: Đang theo dõi
                    label_prefix = "THEO DOI LUA"
                else:
                    color = CV_COLORS.get("fire", (30, 30, 230))
                    label_prefix = "LUA"
            elif is_smoke(cname_lower):
                color = CV_COLORS.get("smoke", (0, 140, 255))
                label_prefix = "KHOI"
            else:
                color = CV_COLORS.get("default", (255, 255, 0))
                label_prefix = det.class_name.upper()

            x1, y1, x2, y2 = det.bbox
            x1 = max(0, min(w_img - 1, int(x1)))
            y1 = max(0, min(h_img - 1, int(y1)))
            x2 = max(0, min(w_img, int(x2)))
            y2 = max(0, min(h_img, int(y2)))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text = f"{label_prefix}: {det.confidence:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            bg_y1 = max(0, y1 - th - 6)
            bg_y2 = y1
            bg_x2 = min(w_img, x1 + tw + 8)

            cv2.rectangle(annotated, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(
                annotated,
                text,
                (x1 + 4, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA
            )

        # Vẽ banner tóm tắt thông tin trên đầu ảnh cảnh báo
        banner_h = 36
        cv2.rectangle(annotated, (0, 0), (w_img, banner_h), (0, 0, 180), -1)
        header_text = f"CANH BAO CHAY! | Time: {timestamp_sec:.2f}s | Conf: {max_fire_conf:.1%} | Ratio: {fire_ratio_str}"
        cv2.putText(
            annotated,
            header_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        return annotated

    def save_alert_image(
        self,
        annotated_frame: np.ndarray,
        timestamp_sec: float
    ) -> Optional[Path]:
        """Lưu ảnh cảnh báo đã khoanh box vào thư mục output_dir."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            save_path = self.output_dir / f"canh_bao_{ts_str}_{int(timestamp_sec)}s.jpg"
            cv2.imwrite(str(save_path), annotated_frame)
            logger.info(f"Đã lưu ảnh cảnh báo cháy vào: {save_path.resolve()}")
            return save_path
        except Exception as e:
            logger.error(f"Lỗi khi lưu ảnh cảnh báo cháy: {e}")
            return None

    def draw_annotations_on_letterbox(
        self,
        canvas: np.ndarray,
        detections: List[Detection],
        scale: float,
        pad_x: int,
        pad_y: int
    ) -> np.ndarray:
        """Vẽ bounding box của các phát hiện lên canvas letterbox dùng hiển thị GUI."""
        for det in detections:
            x1, y1, x2, y2 = scale_bbox(det.bbox, scale, pad_x, pad_y)
            cname_lower = det.class_name.lower()
            threat_st = getattr(det, "threat_status", "xac_nhan")

            if is_fire(cname_lower):
                if threat_st == "cho_xac_nhan":
                    color = (0, 255, 255)
                    label_prefix = "THEO DOI LUA"
                elif threat_st == "bung_lua":
                    color = (0, 0, 255)
                    label_prefix = "BUNG LUA"
                else:
                    color = CV_COLORS.get("fire", (30, 30, 230))
                    label_prefix = "LUA"
            elif is_smoke(cname_lower):
                color = CV_COLORS.get("smoke", (0, 140, 255))
                label_prefix = "KHOI"
            else:
                color = CV_COLORS.get("default", (255, 255, 0))
                label_prefix = det.class_name.upper()

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{label_prefix}: {det.confidence:.1%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA
            )

        return canvas
