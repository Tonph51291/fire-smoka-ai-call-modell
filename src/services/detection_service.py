"""Dịch vụ AI phát hiện lửa và khói kết hợp 2 mô hình (Dual-Model Ensemble).
Hỗ trợ tiền xử lý ban đêm/IR, hợp nhất weighted agreement, lọc hậu xử lý đa tầng,
và làm mịn cảnh báo theo chuỗi thời gian (Temporal Smoothing).
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from src.config import (
    ACTIVE_PROFILE,
    AGREE_BONUS,
    ALERT_MESSAGES,
    BRIGHTNESS_THRESHOLD_DARK,
    BRIGHTNESS_THRESHOLD_IR,
    BRIGHTNESS_THRESHOLD_NIGHT,
    CANH_BAO_DIR,
    CHE_DO_DEM_TU_DONG,
    CLAHE_CLIP_LIMIT,
    COOLDOWN_CANH_BAO_SEC,
    CV_COLORS,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    DUNG_KHOI_DE_BAO,
    ENABLE_SPATIAL_CONSISTENCY,
    ENSEMBLE_MODE,
    EXCLUDE_ZONES,
    EXCLUDE_ZONES_APPLY_SMOKE,
    FUSE_IOU,
    GATE_CU,
    GATE_MOI,
    HIEN_THI_NGUON_MODEL,
    IR_COLOR_DIFF_THRESHOLD,
    IR_SAT_THRESHOLD,
    KICH_HOAT_LOC_NGUON_SANG_TINH,
    LOC_DANG_NGUOI_DI_DUONG,
    MAX_ASPECT_RATIO_LUA,
    MIN_AREA_RATIO,
    MODEL_CU_PATH,
    MODEL_MOI_IMGSZ,
    MODEL_MOI_PATH,
    NIGHT_FIRE_CONF_DOUBLE,
    NIGHT_FIRE_CONF_SINGLE,
    NGUONG_CONF_DEM_KHOI,
    PREDICT_IOU,
    PREDICT_MAX_DET,
    PROFILES,
    SINGLE_MODEL_PENALTY,
    SPATIAL_CONSISTENCY_IOU,
    STATE_BOTH,
    STATE_FIRE,
    STATE_NORMAL,
    STATE_SMOKE,
    STATIC_RELEASE_AREA_GROWTH,
    TEN_KHOI,
    TEN_LUA,
    UU_TIEN_CHO_TOI_VA_NHO,
    WEIGHT_CU,
    WEIGHT_MOI,
    is_fire,
    is_smoke,
)
from src.services.static_light_filter import LocNguonSangTinh
from src.utils.video_utils import compute_iou, scale_bbox

logger = logging.getLogger(__name__)


def is_in_exclude_zones(
    bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    exclude_zones: List[Any],
) -> Tuple[bool, int]:
    """
    Kiểm tra tâm của bounding box có nằm trong vùng loại trừ (EXCLUDE_ZONES) hay không.
    Trả về (is_in, zone_idx).
    Zone có thể là:
    - Bbox chuẩn hóa (x1, y1, x2, y2) với giá trị 0.0 -> 1.0
    - Hoặc polygon [(x1, y1), (x2, y2), ...]
    """
    if not exclude_zones or frame_w <= 0 or frame_h <= 0:
        return False, -1

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    norm_x = cx / frame_w
    norm_y = cy / frame_h

    for idx, zone in enumerate(exclude_zones):
        if len(zone) == 4 and isinstance(zone[0], (int, float)):
            zx1, zy1, zx2, zy2 = zone
            min_x, max_x = min(zx1, zx2), max(zx1, zx2)
            min_y, max_y = min(zy1, zy2), max(zy1, zy2)
            if min_x <= norm_x <= max_x and min_y <= norm_y <= max_y:
                return True, idx
        elif len(zone) >= 3:
            poly = np.array([[int(p[0] * frame_w), int(p[1] * frame_h)] for p in zone], dtype=np.int32)
            dist = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False)
            if dist >= 0:
                return True, idx

    return False, -1


def is_night_or_ir_frame(img: np.ndarray) -> bool:
    """
    Phát hiện khung hình ban đêm hoặc camera hồng ngoại (IR):
    - Đêm thật sự: độ sáng trung bình (Gray) < BRIGHTNESS_THRESHOLD_NIGHT (70.0).
    - Camera hồng ngoại IR: độ bão hòa rất thấp (< 10.0), chênh lệch màu gần 0 (< 6.0)
      VÀ độ sáng < BRIGHTNESS_THRESHOLD_IR (95.0).
    - Ảnh ban ngày (kể cả tông xám/sương mù/bê tông có độ sáng >= 95 hoặc có sắc màu) không bị coi là đêm.
    """
    if img is None or img.size == 0:
        return False
    try:
        small = cv2.resize(img, (80, 60))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lum = float(np.mean(gray))

        # 1. Đêm thực sự: độ sáng tối rõ rệt
        if lum < BRIGHTNESS_THRESHOLD_NIGHT:
            return True

        # 2. Camera hồng ngoại (IR ban đêm / thiếu màu đặc trưng)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        diff_rg = np.abs(small[:, :, 2].astype(np.float32) - small[:, :, 1].astype(np.float32))
        diff_gb = np.abs(small[:, :, 1].astype(np.float32) - small[:, :, 0].astype(np.float32))
        color_diff = float((np.mean(diff_rg) + np.mean(diff_gb)) / 2.0)

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


def fuse_model_detections(
    dets_moi: List["Detection"],
    dets_cu: List["Detection"],
    iou_thresh: float = FUSE_IOU,
    weight_moi: float = WEIGHT_MOI,
    weight_cu: float = WEIGHT_CU,
    agree_bonus: float = AGREE_BONUS,
    single_penalty: float = SINGLE_MODEL_PENALTY
) -> List["Detection"]:
    """
    Hợp nhất thông minh phát hiện từ Model Mới và Model Cũ:
    1. Ghép cặp tham lam các box cùng loại có IoU >= iou_thresh (ưu tiên IoU cao nhất).
    2. Box được CẢ 2 MODEL ĐỒNG Ý (n_votes = 2):
       - Tọa độ: Trung bình có trọng số theo (confidence x model_weight).
       - Conf: Trung bình có trọng số + agree_bonus (khởi đầu 0.10), giới hạn tối đa 0.99.
         (KHÔNG dùng công thức noisy-OR).
    3. Box CHỈ 1 MODEL THẤY (n_votes = 1):
       - Nhân conf với single_penalty (khởi đầu 0.70).
    """
    if not dets_moi and not dets_cu:
        return []

    def _make_penalized(d: "Detection") -> "Detection":
        penalized_conf = max(0.01, min(0.99, d.confidence * single_penalty))
        return Detection(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=penalized_conf,
            bbox=d.bbox,
            timestamp_sec=d.timestamp_sec,
            frame_index=d.frame_index,
            model_source="gop",
            n_votes=1
        )

    if not dets_moi:
        return [_make_penalized(d) for d in dets_cu]

    if not dets_cu:
        return [_make_penalized(d) for d in dets_moi]

    # Tìm tất cả các cặp có thể ghép nối
    candidates = []
    for i, d_moi in enumerate(dets_moi):
        cname_moi = d_moi.class_name.lower()
        for j, d_cu in enumerate(dets_cu):
            cname_cu = d_cu.class_name.lower()
            is_same_class = (
                (is_fire(cname_moi) and is_fire(cname_cu)) or
                (is_smoke(cname_moi) and is_smoke(cname_cu)) or
                (cname_moi == cname_cu)
            )
            if not is_same_class:
                continue
            iou = compute_iou(d_moi.bbox, d_cu.bbox)
            if iou >= iou_thresh:
                candidates.append((iou, i, j))

    # Ghép tham lam theo IoU lớn nhất
    candidates.sort(key=lambda x: x[0], reverse=True)
    used_moi = set()
    used_cu = set()
    fused_results: List["Detection"] = []

    for iou, i, j in candidates:
        if i in used_moi or j in used_cu:
            continue
        used_moi.add(i)
        used_cu.add(j)

        d_moi = dets_moi[i]
        d_cu = dets_cu[j]

        w1 = d_moi.confidence * weight_moi
        w2 = d_cu.confidence * weight_cu
        w_total = w1 + w2 if (w1 + w2) > 0 else 1.0

        x1 = int((d_moi.bbox[0] * w1 + d_cu.bbox[0] * w2) / w_total)
        y1 = int((d_moi.bbox[1] * w1 + d_cu.bbox[1] * w2) / w_total)
        x2 = int((d_moi.bbox[2] * w1 + d_cu.bbox[2] * w2) / w_total)
        y2 = int((d_moi.bbox[3] * w1 + d_cu.bbox[3] * w2) / w_total)

        # Trung bình có trọng số + agree_bonus (không dùng noisy-OR)
        avg_conf = (d_moi.confidence * w1 + d_cu.confidence * w2) / w_total
        fused_conf = min(0.99, avg_conf + agree_bonus)

        fused_results.append(Detection(
            class_id=d_moi.class_id,
            class_name=d_moi.class_name,
            confidence=fused_conf,
            bbox=(x1, y1, x2, y2),
            timestamp_sec=d_moi.timestamp_sec,
            frame_index=d_moi.frame_index,
            model_source="gop",
            n_votes=2
        ))

    # Các box của Model Mới chỉ 1 model thấy
    for i, d_moi in enumerate(dets_moi):
        if i not in used_moi:
            fused_results.append(_make_penalized(d_moi))

    # Các box của Model Cũ chỉ 1 model thấy
    for j, d_cu in enumerate(dets_cu):
        if j not in used_cu:
            fused_results.append(_make_penalized(d_cu))

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
    model_source: str = "moi"  # 'moi' (fire_yolo) hoặc 'cu' (best.pt) hoặc 'gop'
    n_votes: int = 1  # Số model xác nhận (2 nếu cả 2 model đồng ý, 1 nếu chỉ 1 model)


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
    Combined with weighted agreement fusion, penalty for single-model hits, temporal deque, and static filter.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_moi_path: Union[str, Path] = MODEL_MOI_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = DEFAULT_DEVICE,
        profile: str = ACTIVE_PROFILE
    ) -> None:
        self.device = device
        self.model_cu_path = Path(model_path) if model_path else MODEL_CU_PATH
        self.model_moi_path = Path(model_moi_path)

        self.profile = profile if profile in PROFILES else ACTIVE_PROFILE
        self.profile_cfg = PROFILES[self.profile]

        # Ngưỡng quyết định cuối cùng
        self.final_conf_fire = conf_threshold if conf_threshold is not None else self.profile_cfg["FINAL_CONF_FIRE"]
        self.final_conf_smoke = self.profile_cfg["FINAL_CONF_SMOKE"]

        # Ngưỡng cổng thấp cho từng model trước khi gộp
        self.gate_cu = GATE_CU
        self.gate_moi = GATE_MOI
        self.imgsz_cu = imgsz
        self.imgsz_moi = MODEL_MOI_IMGSZ

        self.model_cu = None
        self.model_moi = None
        self.names_cu: Dict[int, str] = {}
        self.names_moi: Dict[int, str] = {}

        # Temporal filter: deque lưu kết quả phát hiện nguy cơ (1/0) trong N lần gần nhất
        self.fire_history: deque = deque(maxlen=self.profile_cfg["DEQUE_WINDOW_SIZE"])
        self.min_fire_detections_to_alert = self.profile_cfg["MIN_FIRE_DETECTIONS_TO_ALERT"]
        self.last_alert_time: float = 0.0

        # Bộ nhớ kiểm tra tính nhất quán không gian (Spatial Consistency)
        self.recent_fire_boxes: deque = deque(maxlen=self.profile_cfg["DEQUE_WINDOW_SIZE"])

        # Bộ lọc nguồn sáng tĩnh (đèn đường, đèn hiên)
        self.loc_nguon_sang_tinh: Optional[LocNguonSangTinh] = (
            LocNguonSangTinh() if KICH_HOAT_LOC_NGUON_SANG_TINH else None
        )

        # Bộ nhớ vùng loại trừ (Exclude Zones) để theo dõi diện tích ban đầu
        self.exclude_zone_baselines: Dict[int, float] = {}

        # Ensure alert output directory exists
        CANH_BAO_DIR.mkdir(parents=True, exist_ok=True)

        self._load_models()

    def reset_tracking(self) -> None:
        """Reset bộ lọc nguồn sáng tĩnh và lịch sử làm mịn khi đổi video hoặc tua vị trí."""
        self.fire_history.clear()
        self.exclude_zone_baselines.clear()
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
                except Exception as e:
                    logger.debug(f"Warmup model_moi: {e}")
            if self.model_cu:
                try:
                    _ = self.model_cu.predict(source=dummy, imgsz=self.imgsz_cu, device=self.device, verbose=False)
                except Exception as e:
                    logger.debug(f"Warmup model_cu: {e}")

    def set_conf_threshold(self, threshold: float) -> None:
        """
        Cập nhật ngưỡng quyết định cuối cùng từ thanh trượt UI (bỏ hệ số * 0.6).
        Thanh trượt điều khiển ngưỡng phân loại CUỐI; 2 model dùng ngưỡng cổng thấp riêng.
        """
        th = max(0.05, min(0.95, float(threshold)))
        self.final_conf_fire = th
        # Giữ khoảng lệch giữa khói và lửa tương ứng với cấu hình profile
        delta_smoke = self.profile_cfg["FINAL_CONF_SMOKE"] - self.profile_cfg["FINAL_CONF_FIRE"]
        self.final_conf_smoke = max(0.05, min(0.95, th + delta_smoke))

    def _predict_single_model(
        self,
        model,
        imgsz: int,
        default_gate: float,
        names_dict: Dict[int, str],
        model_source: str,
        frame: np.ndarray,
        conf: Optional[float] = None
    ) -> List[Detection]:
        """Phương thức dự đoán chung cho cả Model Cũ và Model Mới."""
        if model is None:
            return []

        c_thresh = conf if conf is not None else default_gate
        try:
            with torch.inference_mode():
                results = model.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=c_thresh,
                    iou=PREDICT_IOU,
                    max_det=PREDICT_MAX_DET,
                    device=self.device,
                    verbose=False
                )

            detections: List[Detection] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cid = int(box.cls[0].item())
                    cf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cname = self._get_class_name(names_dict, model, cid)
                    detections.append(Detection(
                        class_id=cid,
                        class_name=cname,
                        confidence=cf,
                        bbox=(x1, y1, x2, y2),
                        model_source=model_source,
                        n_votes=1
                    ))
            return detections
        except Exception as e:
            logger.error(f"Lỗi khi gọi predict ({model_source}): {e}", exc_info=True)
            return []

    def model_cu_phat_hien(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """Gọi model cũ phát hiện trên frame với ngưỡng cổng thấp và tham số NMS chuẩn hóa."""
        return self._predict_single_model(
            model=self.model_cu,
            imgsz=self.imgsz_cu,
            default_gate=self.gate_cu,
            names_dict=self.names_cu,
            model_source="cu",
            frame=frame,
            conf=conf
        )

    def model_moi_phat_hien(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """Gọi model mới (fire_yolo) phát hiện trên frame."""
        return self._predict_single_model(
            model=self.model_moi,
            imgsz=self.imgsz_moi,
            default_gate=self.gate_moi,
            names_dict=self.names_moi,
            model_source="moi",
            frame=frame,
            conf=conf
        )

    def _post_filter_candidates(
        self,
        all_detections: List[Detection],
        frame_w: int,
        frame_h: int,
        is_night: bool,
    ) -> List[Detection]:
        """
        Lọc hậu xử lý các box ứng viên sau khi gộp:
        1. Ngưỡng quyết định cuối cùng (Final Thresholding).
        2. Yêu cầu 2 model đồng thuận (nếu cấu hình REQUIRE_TWO_VOTES).
        3. Vùng loại trừ thủ công (EXCLUDE_ZONES) kèm cơ chế mở lại (Safety Release).
        4. Diện tích tối thiểu (MIN_AREA_RATIO) cho box 1 model.
        5. Hình dạng người đi đường (H/W >= 2.0 dạng đứng) cho box 1 model.
        6. Ngưỡng ban đêm / IR phân tầng (NIGHT_FIRE_CONF_SINGLE / DOUBLE).
        """
        frame_area = max(1, frame_h * frame_w)
        require_two_votes = self.profile_cfg.get("REQUIRE_TWO_VOTES", False)

        valid_detections: List[Detection] = []
        for det in all_detections:
            cname = det.class_name.lower()
            w = max(1, det.bbox[2] - det.bbox[0])
            h = max(1, det.bbox[3] - det.bbox[1])
            box_area = w * h
            aspect_ratio = float(h) / float(w)

            # Ngưỡng quyết định cuối cùng (Final Thresholding)
            req_conf = self.final_conf_fire if is_fire(cname) else self.final_conf_smoke
            if det.confidence < req_conf:
                continue

            # Yêu cầu 2 model đồng ý nếu cấu hình profile quy định (ví dụ profile chinh_xac)
            if require_two_votes and det.n_votes < 2:
                continue

            # Lọc vùng loại trừ thủ công (EXCLUDE_ZONES)
            if EXCLUDE_ZONES and (is_fire(cname) or (is_smoke(cname) and EXCLUDE_ZONES_APPLY_SMOKE)):
                in_zone, z_idx = is_in_exclude_zones(det.bbox, frame_w, frame_h, EXCLUDE_ZONES)
                if in_zone:
                    base_area = self.exclude_zone_baselines.get(z_idx)
                    if base_area is None:
                        self.exclude_zone_baselines[z_idx] = float(box_area)
                        logger.debug(f"[ExcludeZones] Bỏ qua box nằm trong vùng loại trừ #{z_idx}: bbox={det.bbox}")
                        continue
                    elif box_area >= STATIC_RELEASE_AREA_GROWTH * base_area:
                        logger.warning(
                            f"[ExcludeZones-Release] ⚠️ MỞ LẠI VÙNG LOẠI TRỪ #{z_idx} do diện tích tăng "
                            f"{box_area/base_area:.2f}x >= {STATIC_RELEASE_AREA_GROWTH} | bbox={det.bbox}"
                        )
                    else:
                        logger.debug(f"[ExcludeZones] Bỏ qua box nằm trong vùng loại trừ #{z_idx}")
                        continue

            # Lọc diện tích tối thiểu (MIN_AREA_RATIO):
            # Box quá nhỏ (< 0.05% diện tích khung) bị loại trừ khi CẢ 2 MODEL ĐỒNG Ý
            if det.n_votes == 1 and (box_area / frame_area) < MIN_AREA_RATIO:
                logger.debug(
                    f"[LocDienTich] Bỏ box quá nhỏ n_votes=1: area={box_area}px ({box_area/frame_area:.4%})"
                )
                continue

            # Lọc hình dạng người đi đường (H/W >= 2.0 dạng đứng):
            # CHỈ áp dụng cho box có n_votes == 1 và conf < 0.45. Cả 2 model đồng ý thì KHÔNG bị loại
            if LOC_DANG_NGUOI_DI_DUONG and is_fire(cname):
                if det.n_votes == 1 and aspect_ratio >= MAX_ASPECT_RATIO_LUA and det.confidence < 0.45:
                    logger.debug(
                        f"[LocNguoi] Loại bỏ box nghi vấn người/xe máy đứng n_votes=1: "
                        f"H/W={aspect_ratio:.2f}, conf={det.confidence:.1%}"
                    )
                    continue

            # Lọc camera đêm / hồng ngoại:
            if is_night and CHE_DO_DEM_TU_DONG:
                if is_fire(cname):
                    if det.n_votes == 1 and det.confidence < NIGHT_FIRE_CONF_SINGLE:
                        continue
                    elif det.n_votes >= 2 and det.confidence < NIGHT_FIRE_CONF_DOUBLE:
                        continue
                elif is_smoke(cname):
                    if det.n_votes == 1 and det.confidence < NGUONG_CONF_DEM_KHOI:
                        continue

            valid_detections.append(det)

        return valid_detections

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
        conf_threshold: Optional[float] = None
    ) -> FrameAnalysisResult:
        """
        Chạy phát hiện kết hợp thông minh 2 model:
        1. Tiền xử lý (nhận diện đêm/IR, CLAHE nhẹ khi thực sự tối).
        2. Trích xuất box ứng viên ở ngưỡng cổng thấp GATE_MOI, GATE_CU.
        3. Hợp nhất (Fusion) với thỏa thuận 2 model + phạt box 1 model.
        4. Lọc diện tích tối thiểu, dáng đứng, ngưỡng đêm, và nguồn sáng tĩnh.
        5. Xác nhận theo chuỗi frame (temporal confirmation) và phát cảnh báo.
        """
        t0 = time.perf_counter()

        if conf_threshold is not None:
            self.set_conf_threshold(conf_threshold)

        # 1. Nhận biết ảnh đêm/hồng ngoại và kiểm tra độ sáng thực tế
        is_night = is_night_or_ir_frame(frame) if CHE_DO_DEM_TU_DONG else False
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_mean_lum = float(np.mean(frame_gray))
        is_dark_frame = (frame_mean_lum < BRIGHTNESS_THRESHOLD_DARK)

        # Tăng cường tương phản cục bộ thích ứng (CLAHE) chỉ khi khung hình thật sự tối
        if UU_TIEN_CHO_TOI_VA_NHO and is_dark_frame:
            infer_frame = enhance_dark_frame_clahe(frame, clip_limit=CLAHE_CLIP_LIMIT)
        else:
            infer_frame = frame

        # 2. Gọi Model Mới (fire_yolo.pt) ở ngưỡng cổng thấp
        detections_moi = self.model_moi_phat_hien(infer_frame, conf=self.gate_moi)

        # 3. Gọi Model Cũ (best.pt) theo chế độ ENSEMBLE_MODE:
        # - "both": luôn chạy cả 2 model (mặc định)
        # - "cascade": chỉ gọi best.pt khi fire_yolo có box ứng viên
        detections_cu: List[Detection] = []
        if ENSEMBLE_MODE == "cascade":
            if len(detections_moi) > 0:
                detections_cu = self.model_cu_phat_hien(infer_frame, conf=self.gate_cu)
        else:
            detections_cu = self.model_cu_phat_hien(infer_frame, conf=self.gate_cu)

        # 4. Hợp nhất thông minh phát hiện từ 2 model
        all_detections = fuse_model_detections(
            dets_moi=detections_moi,
            dets_cu=detections_cu,
            iou_thresh=FUSE_IOU,
            weight_moi=WEIGHT_MOI,
            weight_cu=WEIGHT_CU,
            agree_bonus=self.profile_cfg["AGREE_BONUS"],
            single_penalty=self.profile_cfg["SINGLE_MODEL_PENALTY"]
        )
        for det in all_detections:
            det.frame_index = frame_index
            det.timestamp_sec = timestamp_sec

        # 5. LỌC HẬU XỬ LÝ (Ngưỡng cuối, diện tích, dáng người/xe, chế độ đêm)
        frame_h, frame_w = frame.shape[:2]
        all_detections = self._post_filter_candidates(
            all_detections=all_detections,
            frame_w=frame_w,
            frame_h=frame_h,
            is_night=is_night
        )

        # 6. BỘ LỌC NGUỒN SÁNG TĨNH (LocNguonSangTinh)
        if KICH_HOAT_LOC_NGUON_SANG_TINH and self.loc_nguon_sang_tinh is not None:
            all_detections, static_lights = self.loc_nguon_sang_tinh.filter_detections(
                frame=frame,
                detections=all_detections,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec
            )

        # Xác định Lửa và Khói
        has_fire = any(is_fire(det.class_name) for det in all_detections)
        has_smoke = any(is_smoke(det.class_name) for det in all_detections)

        # Kiểm tra tính nhất quán không gian (Tùy chọn Spatial Consistency)
        if ENABLE_SPATIAL_CONSISTENCY and has_fire and len(self.recent_fire_boxes) > 0:
            current_fire_boxes = [d.bbox for d in all_detections if is_fire(d.class_name)]
            has_spatial_match = any(
                compute_iou(b_curr, b_past) >= SPATIAL_CONSISTENCY_IOU
                for b_curr in current_fire_boxes
                for past_list in self.recent_fire_boxes
                for b_past in past_list
            )
            if not has_spatial_match and all(d.n_votes == 1 for d in all_detections if is_fire(d.class_name)):
                has_fire = False

        self.recent_fire_boxes.append([d.bbox for d in all_detections if is_fire(d.class_name)])

        max_fire_conf = max(
            [d.confidence for d in all_detections if is_fire(d.class_name)],
            default=0.0
        )
        max_smoke_conf = max(
            [d.confidence for d in all_detections if is_smoke(d.class_name)],
            default=0.0
        )

        # 7. Bộ lọc thời gian: deque lưu vết phát hiện nguy cơ (Lửa hoặc Khói)
        is_hazard = has_fire or (has_smoke and DUNG_KHOI_DE_BAO)
        self.fire_history.append(1 if is_hazard else 0)
        hazard_detections_in_window = sum(self.fire_history)
        history_len = len(self.fire_history)
        fire_ratio_str = f"{hazard_detections_in_window}/{history_len}"

        is_alert_confirmed = hazard_detections_in_window >= self.min_fire_detections_to_alert

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
            if is_fire(cname_lower):
                color = CV_COLORS["fire"]
                label_prefix = "LUA"
            elif is_smoke(cname_lower):
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

        # Vẽ các vùng loại trừ thủ công (EXCLUDE_ZONES) bằng viền mờ màu cam
        if EXCLUDE_ZONES:
            img_w = canvas.shape[1] - 2 * pad_x
            img_h = canvas.shape[0] - 2 * pad_y
            for z_idx, zone in enumerate(EXCLUDE_ZONES):
                if len(zone) == 4 and isinstance(zone[0], (int, float)):
                    zx1, zy1, zx2, zy2 = zone
                    cx1 = int(pad_x + min(zx1, zx2) * img_w)
                    cy1 = int(pad_y + min(zy1, zy2) * img_h)
                    cx2 = int(pad_x + max(zx1, zx2) * img_w)
                    cy2 = int(pad_y + max(zy1, zy2) * img_h)

                    cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 165, 255), 1, cv2.LINE_AA)
                    cv2.putText(
                        canvas,
                        f"VUNG LOAI TRU #{z_idx + 1}",
                        (cx1 + 2, max(12, cy1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (0, 165, 255),
                        1,
                        lineType=cv2.LINE_AA
                    )

        return canvas
