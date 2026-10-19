"""AI Detection Service supporting dual-model parallel inference (OR logic) with temporal smoothing."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np
import torch

from src.config import (
    ALERT_MESSAGES,
    CANH_BAO_DIR,
    CHE_DO_DEM_TU_DONG,
    CHI_GOI_MODEL_CU_KHI_MODEL_MOI_KHONG_THAY,
    CLAHE_CLIP_LIMIT,
    COOLDOWN_CANH_BAO_SEC,
    CV_COLORS,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    DEQUE_WINDOW_SIZE,
    DIEN_TICH_LUA_NHO_MAX_PX,
    DUNG_KHOI_DE_BAO,
    HIEN_THI_NGUON_MODEL,
    KICH_HOAT_LOC_NGUON_SANG_TINH,
    LOC_DANG_NGUOI_DI_DUONG,
    MAX_ASPECT_RATIO_LUA,
    MIN_FIRE_DETECTIONS_TO_ALERT,
    MODEL_CU_CONF,
    MODEL_CU_IMGSZ,
    MODEL_CU_PATH,
    MODEL_MOI_CONF,
    MODEL_MOI_IMGSZ,
    MODEL_MOI_PATH,
    NGUONG_CONF_DEM_DON_LE,
    NGUONG_CONF_DEM_KHOI,
    NGUONG_CONF_DEM_LUA,
    NGUONG_CONF_LUA_NHO,
    STATE_BOTH,
    STATE_FIRE,
    STATE_NORMAL,
    STATE_SMOKE,
    TEN_KHOI,
    TEN_LUA,
    UU_TIEN_CHO_TOI_VA_NHO,
)
from src.services.static_light_filter import LocNguonSangTinh
from src.utils.video_utils import scale_bbox

logger = logging.getLogger(__name__)


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Tính Intersection-over-Union (IoU) giữa 2 bounding box (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return (inter / union) if union > 0 else 0.0


def is_night_or_ir_frame(img: np.ndarray) -> bool:
    """
    Phát hiện nhanh khung hình ban đêm hoặc camera hồng ngoại (ảnh đen trắng / ít màu):
    - Resize nhỏ để kiểm tra cực nhanh (< 1ms trên CPU).
    - Độ bão hòa HSV trung bình < 25 hoặc độ chênh lệch màu giữa các kênh < 12.
    """
    if img is None or img.size == 0:
        return False
    try:
        small = cv2.resize(img, (80, 60))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        diff_rg = np.abs(small[:, :, 2].astype(np.float32) - small[:, :, 1].astype(np.float32))
        diff_gb = np.abs(small[:, :, 1].astype(np.float32) - small[:, :, 0].astype(np.float32))
        color_diff = float((np.mean(diff_rg) + np.mean(diff_gb)) / 2.0)
        return (sat_mean < 25.0) or (color_diff < 12.0)
    except Exception:
        return False


def enhance_dark_frame_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Tăng cường tương phản cục bộ thích ứng (CLAHE) trong không gian màu LAB:
    - Làm rõ các chi tiết đốm lửa, tia lửa nhỏ trong các vùng tối / ban đêm.
    - Giúp YOLO phát hiện lửa sớm nhất ngay cả khi ngọn lửa còn rất nhỏ và mờ.
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


def fuse_model_detections(
    dets_moi: List["Detection"],
    dets_cu: List["Detection"],
    iou_thresh: float = 0.35
) -> List["Detection"]:
    """
    Hợp nhất thông minh phát hiện từ Model Mới và Model Cũ thành Model Lửa Gộp:
    1. Ghép cặp các box cùng loại có IoU >= iou_thresh.
    2. Box gộp:
       - Tọa độ: Trung bình có trọng số theo độ tin cậy.
       - Độ tin cậy: Cộng hưởng theo xác suất P(A u B) = 1 - (1 - P_A)*(1 - P_B).
       - Nguồn: 'gop'.
    3. Box đơn lẻ (chỉ 1 model thấy): Giữ nguyên 100% confidence để đạt Recall tối đa, gắn nhãn 'gop'.
    """
    if not dets_moi:
        for d in dets_cu:
            d.model_source = "gop"
        return list(dets_cu)
    if not dets_cu:
        for d in dets_moi:
            d.model_source = "gop"
        return list(dets_moi)

    fused_results: List["Detection"] = []
    used_cu_indices = set()

    for d_moi in dets_moi:
        matched_cu_idx = -1
        best_iou = 0.0

        for i, d_cu in enumerate(dets_cu):
            if i in used_cu_indices:
                continue
            is_same_class = (
                (d_moi.class_name.lower() in TEN_LUA and d_cu.class_name.lower() in TEN_LUA) or
                (d_moi.class_name.lower() in TEN_KHOI and d_cu.class_name.lower() in TEN_KHOI) or
                (d_moi.class_name.lower() == d_cu.class_name.lower())
            )
            if not is_same_class:
                continue

            iou = compute_iou(d_moi.bbox, d_cu.bbox)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                matched_cu_idx = i

        if matched_cu_idx >= 0:
            d_cu = dets_cu[matched_cu_idx]
            used_cu_indices.add(matched_cu_idx)

            w1, w2 = d_moi.confidence, d_cu.confidence
            w_total = w1 + w2 if (w1 + w2) > 0 else 1.0

            x1 = int((d_moi.bbox[0] * w1 + d_cu.bbox[0] * w2) / w_total)
            y1 = int((d_moi.bbox[1] * w1 + d_cu.bbox[1] * w2) / w_total)
            x2 = int((d_moi.bbox[2] * w1 + d_cu.bbox[2] * w2) / w_total)
            y2 = int((d_moi.bbox[3] * w1 + d_cu.bbox[3] * w2) / w_total)

            fused_conf = 1.0 - (1.0 - d_moi.confidence) * (1.0 - d_cu.confidence)
            fused_conf = min(0.99, max(fused_conf, max(d_moi.confidence, d_cu.confidence)))

            fused_results.append(Detection(
                class_id=d_moi.class_id,
                class_name=d_moi.class_name,
                confidence=fused_conf,
                bbox=(x1, y1, x2, y2),
                timestamp_sec=d_moi.timestamp_sec,
                frame_index=d_moi.frame_index,
                model_source="gop"
            ))
        else:
            d_moi.model_source = "gop"
            fused_results.append(d_moi)

    for i, d_cu in enumerate(dets_cu):
        if i not in used_cu_indices:
            d_cu.model_source = "gop"
            fused_results.append(d_cu)

    return fused_results


@dataclass
class Detection:
    """Represents a single detected object in a frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original frame coords
    timestamp_sec: float = 0.0
    frame_index: int = 0
    model_source: str = "moi"  # 'moi' (fire_yolo) hoặc 'cu' (best.pt)


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
    fire_ratio_in_window: str = "0/0"
    is_alert_confirmed: bool = False


class DetectionService:
    """
    Manages dual-model inference:
    1. Model Mới: seawsurf/fire_yolo (models/fire_yolo.pt) - tối ưu cho camera đêm/hồng ngoại
    2. Model Cũ: best.pt (models/best.pt)
    Combined with OR logic, temporal deque filter (5/10), and cooldown alerting.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_moi_path: Union[str, Path] = MODEL_MOI_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = DEFAULT_DEVICE
    ) -> None:
        self.device = device
        self.model_cu_path = Path(model_path) if model_path else MODEL_CU_PATH
        self.model_moi_path = Path(model_moi_path)

        self.conf_threshold_cu = conf_threshold
        self.conf_threshold_moi = MODEL_MOI_CONF
        self.imgsz_cu = imgsz
        self.imgsz_moi = MODEL_MOI_IMGSZ

        self.model_cu = None
        self.model_moi = None
        self.names_cu: Dict[int, str] = {}
        self.names_moi: Dict[int, str] = {}

        # Temporal filter: deque lưu kết quả phát hiện lửa (1/0) trong N lần gần nhất
        self.fire_history: deque = deque(maxlen=DEQUE_WINDOW_SIZE)
        self.last_alert_time: float = 0.0

        # Bộ lọc nguồn sáng tĩnh (đèn đường, đèn hiên)
        self.loc_nguon_sang_tinh: Optional[LocNguonSangTinh] = (
            LocNguonSangTinh() if KICH_HOAT_LOC_NGUON_SANG_TINH else None
        )

        # Ensure alert output directory exists
        CANH_BAO_DIR.mkdir(parents=True, exist_ok=True)

        self._load_models()

    def reset_tracking(self) -> None:
        """Reset bộ lọc nguồn sáng tĩnh và lịch sử làm mịn khi đổi video hoặc tua vị trí."""
        self.fire_history.clear()
        if self.loc_nguon_sang_tinh:
            self.loc_nguon_sang_tinh.reset()

    @staticmethod
    def _parse_names(names_obj) -> Dict[int, str]:
        """Trích xuất chuẩn xác dictionary {id: name} từ model.names (hỗ trợ cả dict và list/tuple)."""
        if not names_obj:
            return {}
        if isinstance(names_obj, dict):
            return {int(k): str(v).strip() for k, v in names_obj.items()}
        if isinstance(names_obj, (list, tuple)):
            return {int(i): str(v).strip() for i, v in enumerate(names_obj)}
        return {}

    def _get_class_name(self, names_dict: Dict[int, str], model_obj, cid: int) -> str:
        """Đọc tên lớp chính xác từ model.names, bảo đảm không bị mất tên lớp."""
        if cid in names_dict:
            return names_dict[cid]
        if model_obj and hasattr(model_obj, "names") and model_obj.names:
            if isinstance(model_obj.names, dict) and cid in model_obj.names:
                return str(model_obj.names[cid]).strip()
            elif isinstance(model_obj.names, (list, tuple)) and 0 <= cid < len(model_obj.names):
                return str(model_obj.names[cid]).strip()
        return f"class_{cid}"

    def _load_models(self) -> None:
        """Load both models and print model.names."""
        from ultralytics import YOLO

        # 1. Nạp Model Mới (seawsurf/fire_yolo)
        if not self.model_moi_path.exists() and self.model_moi_path.name == MODEL_MOI_PATH.name:
            logger.warning(
                f"Chưa tìm thấy model mới tại {self.model_moi_path}. "
                f"Đang tiến hành tải tự động từ Hugging Face..."
            )
            from download_model import download_fire_yolo_model
            try:
                download_fire_yolo_model()
            except Exception as e:
                logger.error(f"Không thể tải model mới: {e}")

        if self.model_moi_path.exists():
            try:
                logger.info(f"Đang nạp Model Mới từ {self.model_moi_path} trên {self.device}...")
                self.model_moi = YOLO(str(self.model_moi_path))
                if hasattr(self.model_moi, "names"):
                    self.names_moi = self._parse_names(self.model_moi.names)
                print(f"[MODEL MỚI] {self.model_moi_path.name} loaded. Classes: {self.names_moi}")
                logger.info(f"[MODEL MỚI] Classes: {self.names_moi}")
            except Exception as e:
                logger.error(f"Lỗi khi nạp model mới {self.model_moi_path}: {e}")

        # 2. Nạp Model Cũ (best.pt)
        if self.model_cu_path.exists():
            try:
                logger.info(f"Đang nạp Model Cũ từ {self.model_cu_path} trên {self.device}...")
                self.model_cu = YOLO(str(self.model_cu_path))
                if hasattr(self.model_cu, "names"):
                    self.names_cu = self._parse_names(self.model_cu.names)
                print(f"[MODEL CŨ] {self.model_cu_path.name} loaded. Classes: {self.names_cu}")
                logger.info(f"[MODEL CŨ] Classes: {self.names_cu}")
            except Exception as e:
                logger.error(f"Lỗi khi nạp model cũ {self.model_cu_path}: {e}")
        else:
            logger.warning(f"Không tìm thấy model cũ tại {self.model_cu_path}.")

        if self.model_moi is None and self.model_cu is None:
            raise FileNotFoundError("Không thể nạp được bất kỳ model nào (cả model mới và model cũ).")

        self._warmup()

    def _warmup(self) -> None:
        """Warm-up models on dummy frame."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.inference_mode():
            if self.model_moi:
                try:
                    _ = self.model_moi.predict(source=dummy, imgsz=self.imgsz_moi, device=self.device, verbose=False)
                except Exception:
                    pass
            if self.model_cu:
                try:
                    _ = self.model_cu.predict(source=dummy, imgsz=self.imgsz_cu, device=self.device, verbose=False)
                except Exception:
                    pass

    def set_conf_threshold(self, threshold: float) -> None:
        """Dynamically update confidence threshold for models."""
        self.conf_threshold_cu = max(0.01, min(1.0, float(threshold)))
        self.conf_threshold_moi = max(0.01, min(1.0, float(threshold) * 0.6))  # model mới nhạy hơn

    def model_cu_phat_hien(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """
        Gọi model cũ phát hiện trên frame.
        Bọc try/except an toàn, nếu lỗi thì ghi log và tiếp tục, không làm sập chương trình.
        """
        if self.model_cu is None:
            return []

        c_thresh = conf if conf is not None else self.conf_threshold_cu
        try:
            with torch.inference_mode():
                results = self.model_cu.predict(
                    source=frame,
                    imgsz=self.imgsz_cu,
                    conf=c_thresh,
                    device=self.device,
                    verbose=False
                )

            detections: List[Detection] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cid = int(box.cls[0].item())
                    cf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cname = self._get_class_name(self.names_cu, self.model_cu, cid)
                    detections.append(Detection(
                        class_id=cid,
                        class_name=cname,
                        confidence=cf,
                        bbox=(x1, y1, x2, y2),
                        model_source="cu"
                    ))
            return detections
        except Exception as e:
            logger.error(f"Lỗi khi gọi model_cu_phat_hien: {e}", exc_info=True)
            return []

    def model_moi_phat_hien(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """Gọi model mới (fire_yolo) phát hiện trên frame."""
        if self.model_moi is None:
            return []

        c_thresh = conf if conf is not None else self.conf_threshold_moi
        try:
            with torch.inference_mode():
                results = self.model_moi.predict(
                    source=frame,
                    imgsz=self.imgsz_moi,
                    conf=c_thresh,
                    device=self.device,
                    verbose=False
                )

            detections: List[Detection] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cid = int(box.cls[0].item())
                    cf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cname = self._get_class_name(self.names_moi, self.model_moi, cid)
                    detections.append(Detection(
                        class_id=cid,
                        class_name=cname,
                        confidence=cf,
                        bbox=(x1, y1, x2, y2),
                        model_source="moi"
                    ))
            return detections
        except Exception as e:
            logger.error(f"Lỗi khi gọi model_moi_phat_hien: {e}", exc_info=True)
            return []

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
        conf_threshold: Optional[float] = None
    ) -> FrameAnalysisResult:
        """
        Chạy phát hiện kết hợp thông minh cả 2 model (Fusion + OR), áp dụng Chế độ đêm và Bộ lọc nguồn sáng tĩnh.
        """
        t0 = time.perf_counter()

        if conf_threshold is not None:
            self.set_conf_threshold(conf_threshold)

        # 1. Nhận biết ảnh không màu / thiếu sáng và tăng cường vùng tối
        is_night = is_night_or_ir_frame(frame) if CHE_DO_DEM_TU_DONG else False
        frame_mean_lum = float(np.mean(frame))
        is_dark_frame = (frame_mean_lum < 110.0) or is_night

        # Tăng cường tương phản cục bộ thích ứng (CLAHE) cho vùng tối để bắt đốm lửa nhỏ sớm nhất
        if UU_TIEN_CHO_TOI_VA_NHO and is_dark_frame:
            infer_frame = enhance_dark_frame_clahe(frame, clip_limit=CLAHE_CLIP_LIMIT)
        else:
            infer_frame = frame

        conf_moi_effective = self.conf_threshold_moi
        conf_cu_effective = self.conf_threshold_cu

        # 2. Gọi Model Mới (fire_yolo.pt)
        detections_moi = self.model_moi_phat_hien(infer_frame, conf=conf_moi_effective)
        has_fire_moi = any(det.class_name.lower() in TEN_LUA for det in detections_moi)

        # 3. Gọi Model Cũ (best.pt) theo policy cấu hình
        detections_cu: List[Detection] = []
        if CHI_GOI_MODEL_CU_KHI_MODEL_MOI_KHONG_THAY:
            if not has_fire_moi:
                detections_cu = self.model_cu_phat_hien(infer_frame, conf=conf_cu_effective)
        else:
            detections_cu = self.model_cu_phat_hien(infer_frame, conf=conf_cu_effective)

        # 4. Hợp nhất thông minh phát hiện từ 2 model thành Model Lửa Gộp duy nhất
        all_detections = fuse_model_detections(detections_moi, detections_cu, iou_thresh=0.35)
        for det in all_detections:
            det.frame_index = frame_index
            det.timestamp_sec = timestamp_sec
            det.model_source = "gop"

        # 5. LỌC HÌNH DẠNG NGƯỜI ĐI ĐƯỜNG & CHẾ ĐỘ ĐÊM:
        valid_detections: List[Detection] = []
        for det in all_detections:
            cname = det.class_name.lower()
            w = max(1, det.bbox[2] - det.bbox[0])
            h = max(1, det.bbox[3] - det.bbox[1])
            aspect_ratio = float(h) / float(w)

            # Lọc người đi đường / xe máy (hình chữ nhật đứng hẹp cao H/W >= 2.0)
            if LOC_DANG_NGUOI_DI_DUONG and cname in TEN_LUA:
                if aspect_ratio >= MAX_ASPECT_RATIO_LUA and det.confidence < 0.45:
                    logger.debug(
                        f"[LocNguoi] Loại bỏ box nghi vấn người/xe máy đứng: {det.bbox}, "
                        f"H/W={aspect_ratio:.2f}, conf={det.confidence:.1%}"
                    )
                    continue

            # Lọc camera đêm / hồng ngoại (ảnh đen trắng):
            # - Lửa: cần đạt tối thiểu NGUONG_CONF_DEM_LUA (0.25) để khử biển báo (<20%) và áo trắng (<20%)
            # - Khói: cần đạt tối thiểu NGUONG_CONF_DEM_KHOI (0.45) để khử mặt đường sương mờ và đèn chóa
            if is_night and CHE_DO_DEM_TU_DONG:
                if cname in TEN_LUA:
                    if det.confidence < NGUONG_CONF_DEM_LUA:
                        logger.debug(
                            f"[CheDoDem] Loại bỏ box lửa dưới ngưỡng đêm: {det.confidence:.1%} < {NGUONG_CONF_DEM_LUA:.1%}"
                        )
                        continue
                elif cname in TEN_KHOI:
                    if det.confidence < NGUONG_CONF_DEM_KHOI:
                        logger.debug(
                            f"[CheDoDem] Loại bỏ box khói dưới ngưỡng đêm: {det.confidence:.1%} < {NGUONG_CONF_DEM_KHOI:.1%}"
                        )
                        continue

            valid_detections.append(det)

        all_detections = valid_detections

        # 6. BỘ LỌC NGUỒN SÁNG TĨNH (LocNguonSangTinh):
        # Theo dõi từng hộp qua 30 frame bằng IoU, loại hộp có vị trí và nội dung gần như không đổi
        if KICH_HOAT_LOC_NGUON_SANG_TINH and self.loc_nguon_sang_tinh is not None:
            all_detections, static_lights = self.loc_nguon_sang_tinh.filter_detections(
                frame=frame,
                detections=all_detections,
                frame_index=frame_index
            )

        # Xác định Lửa và Khói
        has_fire = any(det.class_name.lower() in TEN_LUA for det in all_detections)
        has_smoke = any(det.class_name.lower() in TEN_KHOI for det in all_detections)

        max_fire_conf = max(
            [d.confidence for d in all_detections if d.class_name.lower() in TEN_LUA],
            default=0.0
        )
        max_smoke_conf = max(
            [d.confidence for d in all_detections if d.class_name.lower() in TEN_KHOI],
            default=0.0
        )

        # 7. Bộ lọc thời gian: deque(maxlen=6), lưu vết phát hiện nguy cơ (Lửa hoặc Khói)
        is_hazard = has_fire or (has_smoke and DUNG_KHOI_DE_BAO)
        self.fire_history.append(1 if is_hazard else 0)
        hazard_detections_in_window = sum(self.fire_history)
        history_len = len(self.fire_history)
        fire_ratio_str = f"{hazard_detections_in_window}/{history_len}"

        is_alert_confirmed = hazard_detections_in_window >= MIN_FIRE_DETECTIONS_TO_ALERT

        # 8. Phát cảnh báo và cooldown (chỉ khi có lửa thật)
        now = time.time()
        if is_alert_confirmed and has_fire and (now - self.last_alert_time >= COOLDOWN_CANH_BAO_SEC):
            self.last_alert_time = now
            self._trigger_fire_alarm(frame, max_fire_conf, timestamp_sec, fire_ratio_str)

        # 9. Xác định trạng thái cảnh báo giao diện
        if is_alert_confirmed:
            if has_fire and has_smoke and DUNG_KHOI_DE_BAO:
                alert_state = STATE_BOTH
                alert_msg = f"{ALERT_MESSAGES[STATE_BOTH]} ({fire_ratio_str} lần | Lửa: {max_fire_conf:.1%} | Khói: {max_smoke_conf:.1%})"
            elif has_fire:
                alert_state = STATE_FIRE
                alert_msg = f"{ALERT_MESSAGES[STATE_FIRE]} ({fire_ratio_str} lần | Lửa: {max_fire_conf:.1%})"
            elif has_smoke and DUNG_KHOI_DE_BAO:
                alert_state = STATE_SMOKE
                alert_msg = f"{ALERT_MESSAGES[STATE_SMOKE]} ({fire_ratio_str} lần | Khói: {max_smoke_conf:.1%})"
            else:
                alert_state = STATE_NORMAL
                alert_msg = ALERT_MESSAGES[STATE_NORMAL]
        elif has_fire:
            alert_state = STATE_NORMAL
            alert_msg = f"Đang theo dõi đốm lửa nghi vấn ({max_fire_conf:.1%})..."
        elif has_smoke and DUNG_KHOI_DE_BAO:
            alert_state = STATE_NORMAL
            alert_msg = f"Đang theo dõi làn khói nghi vấn ({max_smoke_conf:.1%})..."
        else:
            alert_state = STATE_NORMAL
            alert_msg = ALERT_MESSAGES[STATE_NORMAL]

        inf_time_ms = (time.perf_counter() - t0) * 1000.0

        return FrameAnalysisResult(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            detections=all_detections,
            has_fire=has_fire,
            has_smoke=has_smoke,
            max_fire_conf=max_fire_conf,
            max_smoke_conf=max_smoke_conf,
            alert_state=alert_state,
            alert_message=alert_msg,
            inference_time_ms=inf_time_ms,
            updated_monotonic=time.monotonic(),
            fire_ratio_in_window=fire_ratio_str,
            is_alert_confirmed=is_alert_confirmed
        )

    def _trigger_fire_alarm(
        self,
        frame: np.ndarray,
        max_fire_conf: float,
        timestamp_sec: float,
        fire_ratio_str: str
    ) -> None:
        """In CẢNH BÁO CHÁY ra console và lưu ảnh cảnh báo vào thư mục canh_bao/."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("\n" + "=" * 55)
        print(f"🔥 CẢNH BÁO CHÁY! [Tỉ lệ: {fire_ratio_str}] - Độ tin cậy: {max_fire_conf:.1%}")
        print(f"   Thời điểm video: {timestamp_sec:.2f}s | Thời gian hệ thống: {ts_str}")
        print("=" * 55 + "\n")

        # Lưu ảnh cảnh báo
        try:
            save_path = CANH_BAO_DIR / f"canh_bao_{ts_str}_{int(timestamp_sec)}s.jpg"
            cv2.imwrite(str(save_path), frame)
            logger.info(f"Đã lưu ảnh cảnh báo cháy vào: {save_path.resolve()}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu ảnh cảnh báo cháy: {e}")

    def draw_annotations_on_letterbox(
        self,
        canvas: np.ndarray,
        detections: List[Detection],
        scale: float,
        pad_x: int,
        pad_y: int
    ) -> np.ndarray:
        """Vẽ bounding box của Model Lửa Gộp: chỉ hiện 'LUA: xx%' hoặc 'KHOI: xx%', không hiện Mới hay Cũ."""
        for det in detections:
            x1, y1, x2, y2 = scale_bbox(det.bbox, scale, pad_x, pad_y)
            cname_lower = det.class_name.lower()

            # Chọn màu và nhãn hiển thị thống nhất cho Model Lửa Gộp
            if cname_lower in TEN_LUA:
                color = CV_COLORS["fire"]
                label_prefix = "LUA"
            elif cname_lower in TEN_KHOI:
                color = CV_COLORS["smoke"]
                label_prefix = "KHOI"
            else:
                color = CV_COLORS["default"]
                label_prefix = det.class_name.upper()

            # Nếu bật cờ hiển thị nguồn (mặc định False), chỉ hiện [GOP], tuyệt đối không hiện Mới hay Cũ
            if HIEN_THI_NGUON_MODEL:
                label_prefix = f"{label_prefix} [GOP]"

            # Vẽ khung bao
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Vẽ nhãn
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
