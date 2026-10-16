"""AI Detection Service for Fire and Smoke detection using Ultralytics YOLO."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
import torch

from src.config import (
    ALERT_MESSAGES,
    CV_COLORS,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    DEFAULT_MODEL_PATH,
    HALF_PRECISION,
    STATE_BOTH,
    STATE_FIRE,
    STATE_NORMAL,
    STATE_SMOKE,
)
from src.utils.video_utils import scale_bbox

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single detected object in a frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original frame coords
    timestamp_sec: float = 0.0
    frame_index: int = 0


@dataclass
class FrameAnalysisResult:
    """Consolidated detection and alert information for a processed frame."""
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


class DetectionService:
    """Manages YOLO model lifecycle, inference, and visualization."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = DEFAULT_DEVICE
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self.half = HALF_PRECISION and ("cuda" in str(self.device))
        self.model = None
        self.class_names: Dict[int, str] = {}
        
        # Categorized class IDs
        self.fire_class_ids: Set[int] = set()
        self.smoke_class_ids: Set[int] = set()

        self._load_model()

    def _load_model(self) -> None:
        """Verify model file existence and load YOLO weights."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model weights tại: {self.model_path.resolve()}.\n"
                f"Vui lòng đảm bảo file model tồn tại trước khi chạy."
            )

        logger.info(f"Đang nạp model YOLO từ {self.model_path} trên thiết bị '{self.device}'...")
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            raise RuntimeError(
                f"Lỗi khi nạp model {self.model_path.name}: {e}\n"
                f"Nếu model sử dụng kiến trúc YOLO11 mới, hãy đảm bảo thư viện ultralytics đã được nâng cấp."
            ) from e

        # Extract model classes
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            self.class_names = {int(k): str(v).strip() for k, v in self.model.names.items()}
        else:
            self.class_names = {}

        logger.info(f"Model đã tải thành công. Danh sách classes: {self.class_names}")

        # Map fire and smoke classes dynamically based on actual class names
        for cid, name in self.class_names.items():
            name_lower = name.lower()
            if "fire" in name_lower or "lua" in name_lower or "chay" in name_lower:
                self.fire_class_ids.add(cid)
            elif "smoke" in name_lower or "khoi" in name_lower:
                self.smoke_class_ids.add(cid)

        # Warm-up model on dummy frame to eliminate first-frame latency
        self._warmup()

    def _warmup(self) -> None:
        """Run warm-up inference so first video frame isn't delayed."""
        try:
            logger.info("Đang khởi động (warm-up) model...")
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            with torch.inference_mode():
                _ = self.model.predict(
                    source=dummy,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )
            logger.info("Warm-up hoàn tất.")
        except Exception as e:
            logger.warning(f"Lỗi trong quá trình warm-up: {e}")

    def set_conf_threshold(self, threshold: float) -> None:
        """Dynamically update confidence threshold (e.g. from UI slider)."""
        self.conf_threshold = max(0.01, min(1.0, float(threshold)))

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
        conf_threshold: Optional[float] = None
    ) -> FrameAnalysisResult:
        """Perform object detection on a single video frame."""
        if self.model is None:
            raise RuntimeError("Model chưa được khởi tạo.")

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        t0 = time.perf_counter()

        # Run inference using torch.inference_mode for maximum speed & lowest memory
        with torch.inference_mode():
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=conf,
                device=self.device,
                verbose=False
            )

        inf_time_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        has_fire = False
        has_smoke = False
        max_fire_conf = 0.0
        max_smoke_conf = 0.0

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cid = int(box.cls[0].item())
                    conf_val = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    cname = self.class_names.get(cid, str(cid))
                    cname_lower = cname.lower()

                    is_fire = cid in self.fire_class_ids or "fire" in cname_lower or "lua" in cname_lower
                    is_smoke = cid in self.smoke_class_ids or "smoke" in cname_lower or "khoi" in cname_lower

                    if is_fire:
                        has_fire = True
                        if conf_val > max_fire_conf:
                            max_fire_conf = conf_val
                    elif is_smoke:
                        has_smoke = True
                        if conf_val > max_smoke_conf:
                            max_smoke_conf = conf_val

                    detections.append(Detection(
                        class_id=cid,
                        class_name=cname,
                        confidence=conf_val,
                        bbox=(x1, y1, x2, y2),
                        timestamp_sec=timestamp_sec,
                        frame_index=frame_index
                    ))

        # Determine alert state
        if has_fire and has_smoke:
            alert_state = STATE_BOTH
            alert_msg = f"{ALERT_MESSAGES[STATE_BOTH]} (Lửa: {max_fire_conf:.1%}, Khói: {max_smoke_conf:.1%})"
        elif has_fire:
            alert_state = STATE_FIRE
            alert_msg = f"{ALERT_MESSAGES[STATE_FIRE]} (Độ tin cậy: {max_fire_conf:.1%})"
        elif has_smoke:
            alert_state = STATE_SMOKE
            alert_msg = f"{ALERT_MESSAGES[STATE_SMOKE]} (Độ tin cậy: {max_smoke_conf:.1%})"
        else:
            alert_state = STATE_NORMAL
            alert_msg = ALERT_MESSAGES[STATE_NORMAL]

        return FrameAnalysisResult(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            detections=detections,
            has_fire=has_fire,
            has_smoke=has_smoke,
            max_fire_conf=max_fire_conf,
            max_smoke_conf=max_smoke_conf,
            alert_state=alert_state,
            alert_message=alert_msg,
            inference_time_ms=inf_time_ms,
            updated_monotonic=time.monotonic()
        )

    def draw_annotations_on_letterbox(
        self,
        canvas: np.ndarray,
        detections: List[Detection],
        scale: float,
        pad_x: int,
        pad_y: int
    ) -> np.ndarray:
        """Fast drawing of scaled bounding boxes directly onto the resized display frame."""
        for det in detections:
            x1, y1, x2, y2 = scale_bbox(det.bbox, scale, pad_x, pad_y)
            cname_lower = det.class_name.lower()

            if "fire" in cname_lower or "lua" in cname_lower or det.class_id in self.fire_class_ids:
                color = CV_COLORS["fire"]
                label_prefix = "LUA"
            elif "smoke" in cname_lower or "khoi" in cname_lower or det.class_id in self.smoke_class_ids:
                color = CV_COLORS["smoke"]
                label_prefix = "KHOI"
            else:
                color = CV_COLORS["default"]
                label_prefix = det.class_name.upper()

            # Bounding rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label banner
            text = f"{label_prefix}: {det.confidence:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            bg_y1 = max(0, y1 - text_h - 6)
            bg_y2 = y1
            bg_x2 = min(canvas.shape[1], x1 + text_w + 8)

            cv2.rectangle(canvas, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(
                canvas,
                text,
                (x1 + 4, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA
            )

        return canvas
