"""Tầng Động cơ Suy luận Đa mô hình AI (Multi-Model AI Inference Engine).
Chuyên trách nạp, khởi tạo, cấu hình phần cứng (CPU/CUDA/FP16) và chạy suy luận:
1. Primary Detector: models/best.pt (YOLO26s 640x640 qua PyTorch)
2. LowLight Detector: models/fire_yolo.pt (YOLO26n 640x640 qua PyTorch)
3. Edge Auxiliary Detector: models/best.onnx (YOLOv8n 320x320 qua OpenCV DNN)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
import numpy as np

from src.config import (
    GATE_CU,
    GATE_MOI,
    GATE_ONNX,
    MODEL_CU_PATH,
    MODEL_MOI_PATH,
    MODEL_ONNX_PATH,
    MODEL_MOI_IMGSZ,
    MODEL_ONNX_IMGSZ,
    DEFAULT_IMGSZ,
    PREDICT_IOU,
    PREDICT_MAX_DET,
    is_fire,
)
from src.core.entities import CandidateBox
from src.utils.geometry import LetterboxTransform

logger = logging.getLogger("FireSmokeAI.Engine")


class MultiModelDetector:
    """Quản lý nạp và suy luận 3 mô hình AI phát hiện Lửa & Khói."""

    def __init__(
        self,
        model_primary_path: Optional[Union[str, Path]] = None,
        model_lowlight_path: Optional[Union[str, Path]] = None,
        model_onnx_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        use_fp16: bool = False,
        imgsz_primary: int = DEFAULT_IMGSZ,
        imgsz_lowlight: int = MODEL_MOI_IMGSZ,
        imgsz_onnx: int = MODEL_ONNX_IMGSZ,
    ) -> None:
        self.device = device
        self.use_fp16 = use_fp16

        self.path_primary = Path(model_primary_path) if model_primary_path else MODEL_CU_PATH
        self.path_lowlight = Path(model_lowlight_path) if model_lowlight_path else MODEL_MOI_PATH
        self.path_onnx = Path(model_onnx_path) if model_onnx_path else MODEL_ONNX_PATH

        self.imgsz_primary = imgsz_primary
        self.imgsz_lowlight = imgsz_lowlight
        self.imgsz_onnx = imgsz_onnx

        self.gate_primary = GATE_CU
        self.gate_lowlight = GATE_MOI
        self.gate_onnx = GATE_ONNX

        self.model_primary = None
        self.model_lowlight = None
        self.net_onnx = None

        self.names_primary: Dict[int, str] = {}
        self.names_lowlight: Dict[int, str] = {}
        self.names_onnx: Dict[int, str] = {0: "fire", 1: "smoke"}

        self._load_models()
        self._warmup_models()

    @staticmethod
    def parse_names(names_obj: Any) -> Dict[int, str]:
        """Trích xuất chuẩn xác dictionary {id: name} từ model.names."""
        if not names_obj:
            return {}
        if isinstance(names_obj, dict):
            return {int(k): str(v).strip() for k, v in names_obj.items()}
        if isinstance(names_obj, (list, tuple)):
            return {int(i): str(v).strip() for i, v in enumerate(names_obj)}
        return {}

    def _get_class_name(self, names_dict: Dict[int, str], model_obj: Any, cid: int) -> str:
        """Đọc tên lớp chính xác từ model.names."""
        if cid in names_dict:
            return names_dict[cid]
        if model_obj and hasattr(model_obj, "names") and model_obj.names:
            if isinstance(model_obj.names, dict) and cid in model_obj.names:
                return str(model_obj.names[cid]).strip()
            elif isinstance(model_obj.names, (list, tuple)) and 0 <= cid < len(model_obj.names):
                return str(model_obj.names[cid]).strip()
        return f"class_{cid}"

    def _load_models(self) -> None:
        """Nạp các mô hình PyTorch và ONNX."""
        from ultralytics import YOLO

        # 1. Tải tự động model mới nếu chưa có
        if not self.path_lowlight.exists() and self.path_lowlight.name == MODEL_MOI_PATH.name:
            logger.warning(f"Chưa có {self.path_lowlight}, tự động tải từ Hugging Face...")
            try:
                from download_model import download_fire_yolo_model
                download_fire_yolo_model()
            except Exception as e:
                logger.error(f"Không thể tải model lowlight: {e}")

        # 2. Nạp LowLight Detector (fire_yolo.pt)
        if self.path_lowlight.exists():
            try:
                logger.info(f"[LowLight Detector] Nạp {self.path_lowlight.name} trên {self.device}...")
                self.model_lowlight = YOLO(str(self.path_lowlight))
                if hasattr(self.model_lowlight, "names"):
                    self.names_lowlight = self.parse_names(self.model_lowlight.names)
            except Exception as e:
                logger.error(f"Lỗi khi nạp model lowlight: {e}")

        # 3. Nạp Primary Detector (best.pt)
        if self.path_primary.exists():
            try:
                logger.info(f"[Primary Detector] Nạp {self.path_primary.name} trên {self.device}...")
                self.model_primary = YOLO(str(self.path_primary))
                if hasattr(self.model_primary, "names"):
                    self.names_primary = self.parse_names(self.model_primary.names)
            except Exception as e:
                logger.error(f"Lỗi khi nạp model primary: {e}")

        # 4. Nạp Edge Auxiliary Detector (best.onnx) qua OpenCV DNN
        if self.path_onnx.exists():
            try:
                logger.info(f"[Edge ONNX] Nạp {self.path_onnx.name} qua cv2.dnn...")
                self.net_onnx = cv2.dnn.readNetFromONNX(str(self.path_onnx))
                self.net_onnx.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net_onnx.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as e:
                logger.error(f"Lỗi khi nạp model ONNX: {e}")

    def _warmup_models(self) -> None:
        """Chạy warmup benchmark với frame giả lập 0 để khởi động bộ nhớ đệm."""
        dummy = np.zeros((self.imgsz_primary, self.imgsz_primary, 3), dtype=np.uint8)
        if self.model_primary is not None:
            try:
                self.predict_primary(dummy, conf=0.5)
            except Exception:
                pass
        if self.model_lowlight is not None:
            try:
                self.predict_lowlight(dummy, conf=0.5)
            except Exception:
                pass
        if self.net_onnx is not None:
            try:
                self.predict_onnx(dummy, conf=0.5)
            except Exception:
                pass

    def _predict_pytorch_model(
        self,
        model: Any,
        imgsz: int,
        default_gate: float,
        names_dict: Dict[int, str],
        model_name: str,
        frame: np.ndarray,
        conf: Optional[float] = None
    ) -> List[CandidateBox]:
        """Thực thi suy luận an toàn qua Ultralytics YOLO PyTorch."""
        if model is None:
            return []
        gate = conf if conf is not None else default_gate
        try:
            predict_kwargs = {
                "source": frame,
                "conf": gate,
                "iou": PREDICT_IOU,
                "imgsz": imgsz,
                "device": self.device,
                "max_det": PREDICT_MAX_DET,
                "verbose": False,
            }
            if self.use_fp16 and str(self.device).lower() not in ("cpu", "intel"):
                predict_kwargs["half"] = True

            results = model.predict(**predict_kwargs)
            candidates: List[CandidateBox] = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cid = int(box.cls[0].cpu().numpy())
                        cname = self._get_class_name(names_dict, model, cid)
                        conf_val = float(box.conf[0].cpu().numpy())
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = [int(v) for v in xyxy]

                        norm_cname = "fire" if is_fire(cname) else "smoke"
                        candidates.append(CandidateBox(
                            class_name=norm_cname,
                            confidence=conf_val,
                            bbox=(x1, y1, x2, y2),
                            model_name=model_name
                        ))
            return candidates
        except Exception as e:
            logger.error(f"Lỗi khi gọi predict ({model_name}): {e}", exc_info=True)
            return []

    def predict_primary(self, frame: np.ndarray, conf: Optional[float] = None) -> List[CandidateBox]:
        """Suy luận bằng Primary Detector (best.pt)."""
        return self._predict_pytorch_model(
            model=self.model_primary,
            imgsz=self.imgsz_primary,
            default_gate=self.gate_primary,
            names_dict=self.names_primary,
            model_name="best.pt",
            frame=frame,
            conf=conf
        )

    def predict_lowlight(self, frame: np.ndarray, conf: Optional[float] = None) -> List[CandidateBox]:
        """Suy luận bằng LowLight Detector (fire_yolo.pt)."""
        return self._predict_pytorch_model(
            model=self.model_lowlight,
            imgsz=self.imgsz_lowlight,
            default_gate=self.gate_lowlight,
            names_dict=self.names_lowlight,
            model_name="fire_yolo.pt",
            frame=frame,
            conf=conf
        )

    def predict_onnx(self, frame: np.ndarray, conf: Optional[float] = None) -> List[CandidateBox]:
        """Suy luận bằng Edge Auxiliary Detector (best.onnx 320x320) qua cv2.dnn và Letterbox."""
        if self.net_onnx is None:
            return []
        c_thresh = conf if conf is not None else self.gate_onnx
        h_orig, w_orig = frame.shape[:2]
        try:
            padded, scale, pad = LetterboxTransform.apply(frame, (self.imgsz_onnx, self.imgsz_onnx), fill_color=114)
            blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (self.imgsz_onnx, self.imgsz_onnx), swapRB=True, crop=False)
            self.net_onnx.setInput(blob)
            preds = self.net_onnx.forward()

            if len(preds.shape) == 3 and preds.shape[1] == 6:
                out = preds[0].T
            else:
                out = preds[0] if len(preds.shape) == 2 else preds

            boxes = []
            confidences = []
            class_ids = []

            for row in out:
                classes_scores = row[4:]
                cls_id = int(np.argmax(classes_scores))
                score = float(classes_scores[cls_id])
                if score >= c_thresh:
                    cx, cy, bw, bh = row[0:4]
                    bx1 = cx - bw / 2.0
                    by1 = cy - bh / 2.0
                    boxes.append([int(bx1), int(by1), int(bw), int(bh)])
                    confidences.append(score)
                    class_ids.append(cls_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, c_thresh, PREDICT_IOU)
            candidates: List[CandidateBox] = []

            if len(indices) > 0:
                for idx in indices:
                    i = int(idx)
                    bx = boxes[i]
                    bx_ltrb = (bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3])
                    orig_x1, orig_y1, orig_x2, orig_y2 = LetterboxTransform.inverse_bbox(
                        bbox=bx_ltrb,
                        scale=scale,
                        pad=pad,
                        orig_shape=(h_orig, w_orig)
                    )
                    cid = class_ids[i]
                    cname = self.names_onnx.get(cid, "fire" if cid == 0 else "smoke")
                    norm_cname = "fire" if is_fire(cname) else "smoke"
                    candidates.append(CandidateBox(
                        class_name=norm_cname,
                        confidence=confidences[i],
                        bbox=(orig_x1, orig_y1, orig_x2, orig_y2),
                        model_name="best.onnx"
                    ))
            return candidates
        except Exception as e:
            logger.error(f"Lỗi khi gọi ONNX predict: {e}", exc_info=True)
            return []

    def predict_all(
        self,
        frame: np.ndarray,
        use_cascade: bool = False,
        include_onnx: bool = True
    ) -> Tuple[List[CandidateBox], Dict[str, float]]:
        """
        Thực hiện toàn bộ quy trình suy luận trên frame, trả về danh sách ứng viên và thời gian đo đạc.
        """
        timing_ms: Dict[str, float] = {}
        all_candidates: List[CandidateBox] = []

        if use_cascade and self.net_onnx is not None:
            t_onnx = time.perf_counter()
            dets_onnx = self.predict_onnx(frame)
            timing_ms["best_onnx"] = (time.perf_counter() - t_onnx) * 1000.0
            all_candidates.extend(dets_onnx)

            if len(dets_onnx) > 0:
                t_low = time.perf_counter()
                dets_low = self.predict_lowlight(frame)
                timing_ms["fire_yolo_pt"] = (time.perf_counter() - t_low) * 1000.0
                all_candidates.extend(dets_low)

                t_pri = time.perf_counter()
                dets_pri = self.predict_primary(frame)
                timing_ms["best_pt"] = (time.perf_counter() - t_pri) * 1000.0
                all_candidates.extend(dets_pri)
            else:
                timing_ms["fire_yolo_pt"] = 0.0
                timing_ms["best_pt"] = 0.0
        else:
            t_low = time.perf_counter()
            dets_low = self.predict_lowlight(frame)
            timing_ms["fire_yolo_pt"] = (time.perf_counter() - t_low) * 1000.0
            all_candidates.extend(dets_low)

            t_pri = time.perf_counter()
            dets_pri = self.predict_primary(frame)
            timing_ms["best_pt"] = (time.perf_counter() - t_pri) * 1000.0
            all_candidates.extend(dets_pri)

            if include_onnx and self.net_onnx is not None:
                t_onnx = time.perf_counter()
                dets_onnx = self.predict_onnx(frame)
                timing_ms["best_onnx"] = (time.perf_counter() - t_onnx) * 1000.0
                all_candidates.extend(dets_onnx)
            else:
                timing_ms["best_onnx"] = 0.0

        return all_candidates, timing_ms
