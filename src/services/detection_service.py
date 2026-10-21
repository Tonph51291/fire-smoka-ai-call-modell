"""Dịch vụ Điều phối Phát hiện Lửa & Khói (Detection Service Orchestrator).
Tái cấu trúc theo Clean Architecture & Single Responsibility Principle:
- Đóng vai trò là Nhạc trưởng (Facade Orchestrator) điều phối dòng chảy dữ liệu.
- Bóc tách toàn bộ suy luận AI sang src.engine.model_detector (MultiModelDetector).
- Bóc tách thuật toán gộp sang src.engine.bayesian_wbf (BayesianWBF).
- Bóc tách kiểm định quang học sang src.analytics.candidate_filter (CandidateFilter).
- Bóc tách hiển thị đồ họa & lưu file sang src.rendering.alert_renderer (AlertRenderer).
- Bóc tách tăng cường ảnh sang src.utils.image_enhancer.
"""

import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
import numpy as np

from src.config import (
    ACTIVE_PROFILE,
    ALERT_MESSAGES,
    COOLDOWN_CANH_BAO_SEC,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DUNG_KHOI_DE_BAO,
    ENABLE_DECISION_LOGGER,
    ENSEMBLE_MODE,
    ENV_HYSTERESIS_SEC,
    ENVIRONMENT_CONFIGS,
    FLARE_REALERT_SEC,
    HARDWARE_PROFILE_OVERRIDE,
    MODEL_CU_PATH,
    MODEL_MOI_PATH,
    MODEL_ONNX_PATH,
    MODEL_MOI_IMGSZ,
    MODEL_ONNX_IMGSZ,
    PROFILES,
    SO_FRAME_COOLDOWN,
    SO_FRAME_LUA_LIEN_TIEP,
    STATE_FIRE,
    STATE_NORMAL,
    STATE_SMOKE,
    WBF_IOU_THRESH,
    WBF_SINGLE_PENALTY_ALPHA,
    is_fire,
    is_smoke,
)
from src.utils.geometry import compute_iou
from src.core.entities import (
    CandidateBox,
    Detection,
    EnvironmentType,
    FrameAnalysisResult,
    FusedBox,
)
from src.engine.bayesian_wbf import BayesianWBF
from src.engine.model_detector import MultiModelDetector
from src.analytics.candidate_filter import CandidateFilter
from src.rendering.alert_renderer import AlertRenderer
from src.utils.decision_logger import DecisionLogger
from src.utils.hardware_manager import HardwareManager
from src.services.scene_auto_configurator import DynamicSceneProfile, SceneAutoConfigurator
from src.services.static_light_filter import LocNguonSangTinh
from src.services.threat_analyzer import ThreatAnalyzer
from src.utils.image_enhancer import (
    balance_sunset_illumination,
    compute_brightness_and_contrast,
    compute_dynamic_color_mask,
    enhance_dark_frame_clahe,
    is_night_or_ir_frame,
    scale_bbox,
    suppress_specular_highlights,
)

logger = logging.getLogger("FireSmokeAI.DetectionService")


def fuse_model_detections(
    dets_moi: List[Detection],
    dets_cu: List[Detection],
    iou_thresh: float = 0.40,
    weight_moi: float = 1.0,
    weight_cu: float = 1.0,
    agree_bonus: float = 0.10,
    single_penalty: float = 0.70
) -> List[Detection]:
    """Hàm hợp nhất cũ giữ lại để tương thích ngược 100% với các bài test."""
    if not dets_moi and not dets_cu:
        return []

    def _make_penalized(d: Detection) -> Detection:
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

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_moi = set()
    used_cu = set()
    fused_results: List[Detection] = []

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

    for i, d_moi in enumerate(dets_moi):
        if i not in used_moi:
            fused_results.append(_make_penalized(d_moi))
    for j, d_cu in enumerate(dets_cu):
        if j not in used_cu:
            fused_results.append(_make_penalized(d_cu))

    return fused_results


class DetectionService:
    """
    Bộ điều phối trung tâm:
    - Tiếp nhận frame
    - Phân tích ngoại cảnh
    - Điều phối động cơ AI
    - Hợp nhất Bayesian WBF
    - Thẩm định đe dọa không-thời gian
    - Kích hoạt cảnh báo & xuất hình
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_moi_path: Union[str, Path] = MODEL_MOI_PATH,
        model_onnx_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        imgsz: int = DEFAULT_IMGSZ,
        device: Optional[str] = None,
        profile: str = ACTIVE_PROFILE,
        hardware_profile_override: Optional[str] = HARDWARE_PROFILE_OVERRIDE,
        enable_decision_logger: bool = ENABLE_DECISION_LOGGER,
        enable_adaptive_cascade: bool = False,
    ) -> None:
        # 1. Quản lý phần cứng
        self.hw_mgr = HardwareManager(override_profile=hardware_profile_override)
        self.device = device or self.hw_mgr.profile.device
        self.use_fp16 = self.hw_mgr.profile.use_fp16
        self.enable_adaptive_cascade = enable_adaptive_cascade
        self.cascade_scan_interval: int = 25

        self.profile = profile if profile in PROFILES else ACTIVE_PROFILE
        self.profile_cfg = PROFILES[self.profile]
        self.final_conf_fire = conf_threshold if conf_threshold is not None else self.profile_cfg["FINAL_CONF_FIRE"]
        self.final_conf_smoke = self.profile_cfg["FINAL_CONF_SMOKE"]

        # Kiểm tra file model tồn tại (Raise lỗi nếu cả 2 model chính đều thiếu)
        p_cu = Path(model_path) if model_path else MODEL_CU_PATH
        p_moi = Path(model_moi_path)
        if not p_cu.exists() and not p_moi.exists():
            raise FileNotFoundError(f"Không tìm thấy cả model cũ ({p_cu}) và model mới ({p_moi}).")

        # 2. Khởi tạo Động cơ suy luận đa mô hình (Engine)
        self.model_detector = MultiModelDetector(
            model_primary_path=p_cu,
            model_lowlight_path=p_moi,
            model_onnx_path=model_onnx_path,
            device=self.device,
            use_fp16=self.use_fp16,
            imgsz_primary=imgsz,
            imgsz_lowlight=MODEL_MOI_IMGSZ,
            imgsz_onnx=MODEL_ONNX_IMGSZ
        )

        # 3. Bộ tự động cấu hình ngoại cảnh (Environment Layer)
        self.scene_configurator = SceneAutoConfigurator(
            warmup_frames=30,
            env_hysteresis_sec=ENV_HYSTERESIS_SEC
        )
        self.env_classifier = self.scene_configurator.env_classifier

        # 4. Bộ hợp nhất Bayesian WBF (Engine Layer)
        self.wbf_fusion = BayesianWBF(
            iou_thresh=WBF_IOU_THRESH,
            single_penalty_alpha=WBF_SINGLE_PENALTY_ALPHA
        )

        # 5. Bộ lọc ứng viên quang học (Analytics Layer)
        self.candidate_filter = CandidateFilter(
            final_conf_fire=self.final_conf_fire,
            final_conf_smoke=self.final_conf_smoke,
            require_two_votes=self.profile_cfg.get("REQUIRE_TWO_VOTES", False)
        )

        # 6. Bộ phân tích mối nguy & dấu hiệu sống (Analytics Layer)
        self.threat_analyzer = ThreatAnalyzer(static_filter=LocNguonSangTinh())
        self.loc_nguon_sang_tinh = self.threat_analyzer.static_filter

        # 7. Bộ xuất ảnh & trình diễn đồ họa (Rendering Layer)
        self.alert_renderer = AlertRenderer()

        # 8. Bộ lọc thời gian (Temporal Smoothing Deque)
        self.fire_history: deque = deque(maxlen=self.profile_cfg.get("TEMPORAL_WINDOW_SIZE", SO_FRAME_COOLDOWN))
        self.min_fire_detections_to_alert = self.profile_cfg.get("MIN_FIRE_DETECTIONS_TO_ALERT", SO_FRAME_LUA_LIEN_TIEP)
        self.last_alert_time: float = 0.0

        # 9. Bộ ghi log quyết định (Decision Logger)
        self.decision_logger = DecisionLogger() if enable_decision_logger else None
        self.exclude_zone_baselines: Dict[int, float] = {}

    # Thuộc tính tương thích ngược cho các đoạn mã bên ngoài
    @property
    def model_cu(self):
        return self.model_detector.model_primary

    @property
    def model_moi(self):
        return self.model_detector.model_lowlight

    @property
    def net_onnx(self):
        return self.model_detector.net_onnx

    @property
    def names_cu(self):
        return self.model_detector.names_primary

    @property
    def names_moi(self):
        return self.model_detector.names_lowlight

    @property
    def names_onnx(self):
        return self.model_detector.names_onnx

    def model_cu_phat_hien(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """Wrapper gọi Primary Detector, ánh xạ sang Detection object."""
        cands = self.model_detector.predict_primary(frame, conf=conf)
        return [Detection(class_id=0 if c.class_name == "fire" else 1, class_name=c.class_name,
                          confidence=c.confidence, bbox=c.bbox, model_source="best.pt", sources=["best.pt"]) for c in cands]

    def model_moi_phat_hien(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """Wrapper gọi LowLight Detector, ánh xạ sang Detection object."""
        cands = self.model_detector.predict_lowlight(frame, conf=conf)
        return [Detection(class_id=0 if c.class_name == "fire" else 1, class_name=c.class_name,
                          confidence=c.confidence, bbox=c.bbox, model_source="fire_yolo.pt", sources=["fire_yolo.pt"]) for c in cands]

    def model_onnx_phat_hien(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Detection]:
        """Wrapper gọi ONNX Detector, ánh xạ sang Detection object."""
        cands = self.model_detector.predict_onnx(frame, conf=conf)
        return [Detection(class_id=0 if c.class_name == "fire" else 1, class_name=c.class_name,
                          confidence=c.confidence, bbox=c.bbox, model_source="best.onnx", sources=["best.onnx"]) for c in cands]

    def reset(self) -> None:
        """Đặt lại trạng thái toàn bộ hệ thống."""
        self.fire_history.clear()
        self.last_alert_time = 0.0
        if self.threat_analyzer:
            self.threat_analyzer.reset()
        if self.env_classifier:
            self.env_classifier.reset()
        self.exclude_zone_baselines.clear()

    def reset_tracking(self) -> None:
        """Đặt lại trạng thái theo dõi và bộ nhớ lọc thời gian."""
        self.reset()

    def has_active_hazard(self) -> bool:
        """Kiểm tra xem hệ thống có đối tượng nghi vấn hoặc báo động nào đang hoạt động không."""
        if hasattr(self, "threat_analyzer") and self.threat_analyzer:
            for t in self.threat_analyzer.tracker.tracks.values():
                if t.state in ("SUSPECT", "CONFIRMED", "ALARM"):
                    return True
        if len(self.fire_history) > 0 and self.fire_history[-1] == 1:
            return True
        return False

    def get_dynamic_stride(self) -> int:
        """Lấy bước nhảy khung hình động: nếu có nguy cơ thì siết về 1, bình thường dùng recommended_stride."""
        if self.has_active_hazard():
            return 1
        return self.hw_mgr.profile.recommended_stride

    def detect_frame(
        self,
        frame: np.ndarray,
        timestamp_sec: float = 0.0,
        frame_index: int = 0
    ) -> FrameAnalysisResult:
        """
        Quy trình xử lý khung hình camera chuẩn hóa qua 6 bước.
        """
        t_start = time.perf_counter()
        timing_ms: Dict[str, float] = {}

        # 1. Đánh giá môi trường & Tối ưu tiền xử lý
        t_env = time.perf_counter()
        dyn_profile = self.scene_configurator.update(frame, timestamp_sec)
        env = dyn_profile.environment
        env_cfg = ENVIRONMENT_CONFIGS.get(env, ENVIRONMENT_CONFIGS[EnvironmentType.DAY_NORMAL])
        timing_ms["environment"] = (time.perf_counter() - t_env) * 1000.0

        is_night = (env in (EnvironmentType.NIGHT_IR, EnvironmentType.NIGHT_COLOR))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if env == EnvironmentType.NIGHT_COLOR:
            infer_frame = enhance_dark_frame_clahe(frame)
        elif env == EnvironmentType.DUSK_SUNSET:
            infer_frame = balance_sunset_illumination(frame)
        else:
            infer_frame = frame

        # 2. Suy luận AI đa mô hình
        is_cpu = (self.device == "cpu")
        has_hazard = (len(self.fire_history) > 0 and self.fire_history[-1] == 1)
        is_periodic_scan = (frame_index % self.cascade_scan_interval == 0)
        use_cascade = (
            (ENSEMBLE_MODE == "cascade") or
            (self.enable_adaptive_cascade and is_cpu and not has_hazard and not is_periodic_scan)
        )
        include_onnx = (self.net_onnx is not None and ENSEMBLE_MODE in ("tri", "all"))

        candidates_wbf, infer_timings = self.model_detector.predict_all(
            frame=infer_frame,
            use_cascade=use_cascade,
            include_onnx=include_onnx
        )
        timing_ms.update(infer_timings)

        # 3. Hợp nhất Bayesian WBF
        t_wbf = time.perf_counter()
        total_active_models = 3 if include_onnx else 2
        fused_boxes = self.wbf_fusion.fuse(
            candidates=candidates_wbf,
            model_weights=dyn_profile.model_weights,
            total_models_active=total_active_models
        )
        timing_ms["wbf_fusion"] = (time.perf_counter() - t_wbf) * 1000.0

        all_detections: List[Detection] = [
            Detection(
                class_id=0 if fb.class_name == "fire" else 1,
                class_name=fb.class_name,
                confidence=fb.confidence,
                bbox=fb.bbox,
                timestamp_sec=timestamp_sec,
                frame_index=frame_index,
                model_source="gop",
                n_votes=fb.n_votes,
                threat_status="xac_nhan",
                sources=fb.sources,
                environment=env
            )
            for fb in fused_boxes
        ]

        # 4. Lọc hậu xử lý hình học & biển báo
        all_detections, suppressed_list = self.candidate_filter.filter(
            all_detections=all_detections,
            frame_shape=frame.shape[:2],
            environment=env,
            env_cfg=env_cfg,
            dyn_profile=dyn_profile,
            frame_gray=frame_gray,
            frame_bgr=frame
        )

        # 5. Phân tích mối nguy & dấu hiệu sống (ThreatAnalyzer)
        t_ta = time.perf_counter()
        all_detections, bypass_fast_alert = self.threat_analyzer.analyze(
            candidates=all_detections,
            raw_gray=frame_gray,
            timestamp_monotonic=time.monotonic(),
            is_night=is_night,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec
        )
        timing_ms["threat_analyzer"] = (time.perf_counter() - t_ta) * 1000.0

        # 6. Đánh giá trạng thái nguy cơ & Cảnh báo
        has_fire = any(is_fire(d.class_name) and getattr(d, "threat_status", "xac_nhan") != "cho_xac_nhan" for d in all_detections)
        has_smoke = any(is_smoke(d.class_name) for d in all_detections)
        has_flare = any(getattr(d, "threat_status", "") == "bung_lua" for d in all_detections)

        max_fire_conf = max([d.confidence for d in all_detections if is_fire(d.class_name)], default=0.0)
        max_smoke_conf = max([d.confidence for d in all_detections if is_smoke(d.class_name)], default=0.0)

        is_hazard = has_fire or (has_smoke and DUNG_KHOI_DE_BAO)
        self.fire_history.append(1 if is_hazard else 0)
        hazard_in_win = sum(self.fire_history)
        fire_ratio_str = f"{hazard_in_win}/{len(self.fire_history)}"

        is_alert_confirmed = (hazard_in_win >= self.min_fire_detections_to_alert)
        has_alarm_track = any(
            (getattr(d, "track_state", "") == "ALARM" and getattr(d, "evidence_score", 0.0) >= 85.0)
            for d in all_detections if (is_fire(d.class_name) or (is_smoke(d.class_name) and DUNG_KHOI_DE_BAO))
        )
        if (bypass_fast_alert and has_flare) or has_alarm_track:
            is_alert_confirmed = True

        # Kích hoạt chuông còi và lưu ảnh cảnh báo
        now = time.time()
        effective_cooldown = FLARE_REALERT_SEC if (has_flare and bypass_fast_alert) else COOLDOWN_CANH_BAO_SEC
        is_hazard_active = has_fire or (has_smoke and DUNG_KHOI_DE_BAO)
        if is_alert_confirmed and is_hazard_active and (now - self.last_alert_time >= effective_cooldown):
            self.last_alert_time = now
            effective_conf = max_fire_conf if has_fire else max_smoke_conf
            self._trigger_fire_alarm(frame, all_detections, timestamp_sec, effective_conf, fire_ratio_str)

        # Tổng hợp trạng thái
        if has_fire:
            alert_state = STATE_FIRE
            alert_msg = ALERT_MESSAGES[STATE_FIRE]
        elif has_smoke:
            alert_state = STATE_SMOKE
            alert_msg = ALERT_MESSAGES[STATE_SMOKE]
        else:
            alert_state = STATE_NORMAL
            alert_msg = ALERT_MESSAGES[STATE_NORMAL]

        total_time_ms = (time.perf_counter() - t_start) * 1000.0

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
            inference_time_ms=total_time_ms,
            fire_ratio_in_window=fire_ratio_str,
            is_alert_confirmed=is_alert_confirmed,
            bypass_fast_alert=bypass_fast_alert,
            environment=env,
            timing_ms=timing_ms,
            scene_status_text=f"{env} | Conf: {max_fire_conf:.1%}",
            dynamic_conf_fire=dyn_profile.confidence_fire,
            dynamic_conf_smoke=dyn_profile.confidence_smoke,
            camera_shake=dyn_profile.camera_shake,
            noise_level=dyn_profile.noise_level
        )

    def _trigger_fire_alarm(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        timestamp_sec: float,
        max_fire_conf: float,
        fire_ratio_str: str,
    ) -> None:
        """Kích hoạt chuông còi phần cứng và lưu ảnh cảnh báo."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("\n" + "=" * 55)
        print(f"🔥 CẢNH BÁO CHÁY! [Tỉ lệ: {fire_ratio_str}] - Độ tin cậy: {max_fire_conf:.1%}")
        print(f"   Thời điểm video: {timestamp_sec:.2f}s | Thời gian hệ thống: {ts_str}")
        print("=" * 55 + "\n")

        # Vẽ và lưu ảnh cảnh báo qua AlertRenderer
        annotated = self.alert_renderer.annotate_alert_frame(
            frame=frame,
            detections=detections,
            timestamp_sec=timestamp_sec,
            max_fire_conf=max_fire_conf,
            fire_ratio_str=fire_ratio_str
        )
        self.alert_renderer.save_alert_image(annotated, timestamp_sec)

    def draw_annotations_on_letterbox(
        self,
        canvas: np.ndarray,
        detections: List[Detection],
        scale: float,
        pad_x: int,
        pad_y: int
    ) -> np.ndarray:
        """Vẽ bounding box của các phát hiện lên canvas letterbox dùng hiển thị GUI."""
        return self.alert_renderer.draw_annotations_on_letterbox(canvas, detections, scale, pad_x, pad_y)
