"""Bộ phân tích nguy cơ (ThreatAnalyzer): Cổng dấu hiệu sống ban đêm, Mở khóa bùng lửa và Báo động nhanh.

1. Cổng "dấu hiệu sống" (Night Liveness Gate - CHỈ áp dụng ban đêm/IR):
   - Theo dõi box lửa qua các frame bằng IoU trong cửa sổ NIGHT_LIVENESS_SEC.
   - Đo độ lệch chuẩn độ sáng (flicker_std), độ lệch chuẩn diện tích Otsu (area_std), và xu hướng tăng diện tích trên ảnh xám GỐC.
   - Box lửa chỉ 'xac_nhan' khi có dấu hiệu sống; chưa đủ dữ liệu thì 'cho_xac_nhan' (hiển thị viền vàng, không còi, không lưu ảnh).
   - Ngoại lệ: Cả 2 model đồng ý và conf >= NIGHT_BYPASS_CONF thì rút ngắn thời gian chờ.

2. Mở khóa khi lửa bùng lên (Flare-up Unlock - Áp dụng cho cả EXCLUDE_ZONES và Static Memory):
   - Duy trì baseline cho từng vùng/đèn tĩnh, cập nhật chậm và dừng cập nhật khi có dấu hiệu bùng.
   - Kích hoạt 'bung_lua' ngay khi thỏa một trong 6 điều kiện:
     a: diện tích >= 1.6x baseline kéo dài >= 0.5s
     b: nhảy vọt diện tích >= 1.3x trong <= 0.5s
     c: lan tràn ra ngoài baseline >= 0.25x diện tích
     d: nhấp nháy mạnh (std >= 3.5)
     e: khói xuất hiện gần vùng (khoảng cách tâm < 1.5x width)
     f: box lửa mới được 2 model đồng thuận sát vùng (IoU < 0.5).

3. Hysteresis:
   - Sau khi mở khóa, duy trì trạng thái mở tối thiểu FLARE_HOLD_SEC (60s).
   - Chỉ khóa lại khi mục tiêu biến mất liên tục >= FLARE_CLEAR_SEC (20s) rồi đo baseline mới.

4. Báo động nhanh khi bùng lên (Fast Alert):
   - Trạng thái 'bung_lua' (điều kiện a, b hoặc e) được phát cảnh báo sau FLARE_FAST_CONFIRM_SEC (0.4s) thay vì chờ đầy deque.
   - Cooldown rút ngắn còn FLARE_REALERT_SEC (10s) khi vẫn đang bùng lửa.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np

from src.config import (
    EXCLUDE_ZONES,
    EXCLUDE_ZONES_APPLY_SMOKE,
    FLARE_AREA_RATIO,
    FLARE_CLEAR_SEC,
    FLARE_FAST_CONFIRM_SEC,
    FLARE_HOLD_SEC,
    FLARE_JUMP_RATIO,
    FLARE_MIN_SEC,
    FLARE_REALERT_SEC,
    FLARE_SMOKE_DIST_RATIO,
    FLARE_SPREAD_RATIO,
    IOU_TRACK_THRESH,
    NIGHT_AREA_STD_MIN,
    NIGHT_BYPASS_CONF,
    NIGHT_BYPASS_WAIT_SEC,
    NIGHT_FLICKER_STD_MIN,
    NIGHT_GROWTH_MIN,
    NIGHT_LIVENESS_SEC,
    TRACKER_SIM_THRESH,
    TRACKER_MAX_MISSED_SEC,
    STATIC_RELEASE_FLICKER_STD,
    EMERGENCY_GROWTH_RATIO,
    EMERGENCY_ROUGHNESS_THRESH,
    is_fire,
    is_smoke,
)
from src.utils.geometry import compute_iou, is_reflective_traffic_sign
from src.utils.optical_features import extract_optical_metrics, OpticalMetrics, analyze_spark_dispersion, crop_roi_gray
from src.services.spatio_temporal_tracker import SpatioTemporalTracker, TrackedObject
from src.services.motion_feature_extractor import MotionFeatureExtractor
from src.services.evidence_accumulator import EvidenceAccumulator, AccumulatorConfig

logger = logging.getLogger(__name__)


@dataclass
class FlareTracker:
    """Theo dõi một vùng hoặc nguồn sáng tĩnh để phát hiện dấu hiệu bùng lửa."""
    tracker_id: Union[int, str]
    is_exclude_zone: bool = False
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    baseline_bright_area: float = 0.0
    baseline_mean_lum: float = 0.0
    baseline_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    baseline_initialized: bool = False
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0

    # Lịch sử đo ngắn hạn (time, bright_area, mean_lum, bbox)
    history_measurements: deque = field(default_factory=deque)

    # Trạng thái mở khóa & Hysteresis
    is_unlocked: bool = False
    unlocked_time: float = 0.0
    unlock_reason: str = ""
    last_seen_fire_time: float = 0.0


class ThreatAnalyzer:
    """
    Bộ phân tích nguy cơ tập trung: Lọc vùng loại trừ, Bộ lọc tĩnh,
    Cổng dấu hiệu sống ban đêm và Mở khóa khi bùng lửa.
    """

    def __init__(
        self,
        static_filter=None,
        profile_cfg: Optional[Dict] = None,
        exclude_zones: Optional[List] = None,
    ) -> None:
        self.static_filter = static_filter
        self.profile_cfg = profile_cfg or {}
        self.exclude_zones = exclude_zones if exclude_zones is not None else EXCLUDE_ZONES

        # Tham số cấu hình
        self.night_liveness_sec = float(self.profile_cfg.get("NIGHT_LIVENESS_SEC", NIGHT_LIVENESS_SEC))
        self.flare_area_ratio = float(self.profile_cfg.get("FLARE_AREA_RATIO", FLARE_AREA_RATIO))
        self.flare_fast_confirm_sec = float(self.profile_cfg.get("FLARE_FAST_CONFIRM_SEC", FLARE_FAST_CONFIRM_SEC))

        # Quản lý bùng lửa cho EXCLUDE_ZONES
        self.zone_trackers: Dict[int, FlareTracker] = {}
        for z_idx in range(len(self.exclude_zones)):
            self.zone_trackers[z_idx] = FlareTracker(tracker_id=z_idx, is_exclude_zone=True)

        # Quản lý bùng lửa cho các nguồn sáng tĩnh (Static Flare Trackers)
        self._next_flare_id: int = 1
        self.light_flare_trackers: Dict[int, FlareTracker] = {}

        # Theo dõi thời gian bắt đầu bùng lửa để kích hoạt báo động nhanh
        self.flare_fast_start_time: float = 0.0

        # Stage 2: Tracker không-thời gian, trích xuất đặc trưng động học & bộ tích lũy bằng chứng
        self.tracker = SpatioTemporalTracker(
            max_time_lost_sec=float(self.profile_cfg.get("TRACKER_MAX_MISSED_SEC", TRACKER_MAX_MISSED_SEC)),
            min_composite_similarity=float(self.profile_cfg.get("TRACKER_SIM_THRESH", TRACKER_SIM_THRESH)),
        )
        self.motion_extractor = MotionFeatureExtractor()
        self.accumulator = EvidenceAccumulator()

    def reset(self) -> None:
        """Reset toàn bộ trạng thái tracking khi đổi nguồn video."""
        self.tracker.reset()
        self.light_flare_trackers.clear()
        self.flare_fast_start_time = 0.0
        for tracker in self.zone_trackers.values():
            tracker.is_unlocked = False
            tracker.unlocked_time = 0.0
            tracker.baseline_initialized = False
            tracker.history_measurements.clear()

    @staticmethod
    def _extract_crop_metrics(raw_gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float, np.ndarray]:
        """Trích xuất diện tích sáng Otsu và độ sáng trung bình từ khung hình xám GỐC (single-pass)."""
        opt = extract_optical_metrics(raw_gray, bbox)
        thresh = opt.thresh if opt.thresh is not None else np.zeros((1, 1), dtype=np.uint8)
        return opt.bright_area, opt.mean_lum, thresh

    def _check_point_in_zone(self, cx: float, cy: float, frame_w: int, frame_h: int, zone) -> bool:
        """Kiểm tra tâm điểm có nằm trong vùng loại trừ polygon hay bbox chuẩn hóa 0-1 hay không."""
        norm_x = cx / max(1, frame_w)
        norm_y = cy / max(1, frame_h)

        if len(zone) == 4 and isinstance(zone[0], (int, float)):
            zx1, zy1, zx2, zy2 = zone
            min_x, max_x = min(zx1, zx2), max(zx1, zx2)
            min_y, max_y = min(zy1, zy2), max(zy1, zy2)
            return (min_x <= norm_x <= max_x) and (min_y <= norm_y <= max_y)
        elif len(zone) >= 3:
            poly = np.array([[int(p[0] * frame_w), int(p[1] * frame_h)] for p in zone], dtype=np.int32)
            dist = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False)
            return dist >= 0
        return False

    def _evaluate_flare_triggers(
        self,
        tracker: FlareTracker,
        det,
        raw_gray: np.ndarray,
        now_time: float,
        all_candidates: List,
    ) -> Optional[Tuple[str, str]]:
        """
        Đánh giá 6 điều kiện mở khóa bùng lửa (a-f):
        a. Diện tích >= 1.6x baseline kéo dài >= 0.5s.
        b. Tăng đột ngột >= 1.3x trong <= 0.5s.
        c. Vùng sáng tràn ra ngoài bbox baseline >= 0.25x baseline.
        d. Nhấp nháy mạnh (std >= NIGHT_FLICKER_STD_MIN).
        e. Khói xuất hiện sát vùng (khoảng cách tâm < 1.5x width).
        f. Box lửa mới sát vùng (IoU < 0.5, 2 model đồng ý).
        """
        opt = extract_optical_metrics(raw_gray, det.bbox)
        bright_area = opt.bright_area
        mean_lum = opt.mean_lum
        thresh = opt.thresh if opt.thresh is not None else np.zeros((1, 1), dtype=np.uint8)
        det.roughness = opt.roughness
        det.core_sat_ratio = opt.core_sat_ratio

        if not tracker.baseline_initialized:
            tracker.baseline_bright_area = max(1.0, bright_area)
            tracker.baseline_mean_lum = mean_lum
            tracker.baseline_bbox = det.bbox
            tracker.baseline_initialized = True
            tracker.history_measurements.append((now_time, bright_area, mean_lum, det.bbox))
            return None

        tracker.history_measurements.append((now_time, bright_area, mean_lum, det.bbox))
        while tracker.history_measurements and (now_time - tracker.history_measurements[0][0] > 5.0):
            tracker.history_measurements.popleft()

        base_area = max(1.0, tracker.baseline_bright_area)
        base_bbox = tracker.baseline_bbox
        base_w = max(10, base_bbox[2] - base_bbox[0])
        base_cx = (base_bbox[0] + base_bbox[2]) / 2.0
        base_cy = (base_bbox[1] + base_bbox[3]) / 2.0
        max_smoke_dist = FLARE_SMOKE_DIST_RATIO * base_w

        # Nếu box là biển báo hoặc chao đèn hình học cứng: không kích hoạt bùng lửa chỉ vì nhấp nháy hoặc dao động diện tích
        if is_reflective_traffic_sign(raw_gray, det.bbox):
            # Biển báo / chao đèn chỉ có thể là bùng lửa THẬT nếu có KHÓI sát vùng (điều kiện e)
            for cand in all_candidates:
                if is_smoke(cand.class_name):
                    scx = (cand.bbox[0] + cand.bbox[2]) / 2.0
                    scy = (cand.bbox[1] + cand.bbox[3]) / 2.0
                    dist = math.hypot(scx - base_cx, scy - base_cy)
                    if dist < max_smoke_dist:
                        return "e", f"Phát hiện khói sát vùng khoảng cách={dist:.1f}px < {max_smoke_dist:.1f}px"
            return None

        det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
        det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
        drift_from_base = math.hypot(det_cx - base_cx, det_cy - base_cy)

        # Nếu là chùm tia lửa gia công cơ khí phân mảnh li ti (mài, cắt, hàn kim loại):
        # Không kích hoạt bùng lửa trừ khi có khói dày xuất hiện (cháy lan thực sự)
        if raw_gray is not None:
            crop_g = crop_roi_gray(raw_gray, det.bbox)
            is_sparks, spark_count, _ = analyze_spark_dispersion(crop_g)
            if is_sparks:
                det.threat_status = "cho_xac_nhan"
                for cand in all_candidates:
                    if is_smoke(cand.class_name):
                        scx = (cand.bbox[0] + cand.bbox[2]) / 2.0
                        scy = (cand.bbox[1] + cand.bbox[3]) / 2.0
                        dist = math.hypot(scx - base_cx, scy - base_cy)
                        if dist < max_smoke_dist:
                            return "e", f"Tia lửa cơ khí bắt cháy sinh khói khoảng cách={dist:.1f}px < {max_smoke_dist:.1f}px"
                return None

        # Điều kiện a: Diện tích >= FLARE_AREA_RATIO so với baseline kéo dài >= FLARE_MIN_SEC (tại vị trí tĩnh)
        if drift_from_base <= 25.0 and bright_area >= self.flare_area_ratio * base_area:
            persisting_duration = 0.0
            for t_meas, a_meas, _, _ in reversed(tracker.history_measurements):
                if a_meas >= self.flare_area_ratio * base_area:
                    persisting_duration = now_time - t_meas
                else:
                    break
            if persisting_duration >= FLARE_MIN_SEC:
                return "a", (
                    f"Diện tích {bright_area:.0f}px >= {self.flare_area_ratio:.1f}x baseline ({base_area:.0f}px) "
                    f"kéo dài {persisting_duration:.2f}s >= {FLARE_MIN_SEC}s"
                )

        # Điều kiện b: Tăng đột ngột >= FLARE_JUMP_RATIO tại vị trí tĩnh (loại trừ xe cộ di chuyển)
        if drift_from_base <= 15.0:
            for t_prev, a_prev, _, _ in tracker.history_measurements:
                dt = now_time - t_prev
                if 0.05 <= dt <= 0.50:
                    # Yêu cầu diện tích tăng tuyệt đối tối thiểu 150px để loại trừ rung lắc pixel viền
                    if a_prev > 0 and (bright_area - a_prev >= 150.0) and (bright_area / a_prev) >= FLARE_JUMP_RATIO:
                        return "b", (
                            f"Tăng vọt {bright_area / a_prev:.2f}x >= {FLARE_JUMP_RATIO} "
                            f"(+{bright_area - a_prev:.0f}px) trong khoảng thời gian {dt:.2f}s <= 0.5s"
                        )

        # Điều kiện c: Vùng sáng lan ra ngoài bbox baseline với phần vượt >= FLARE_SPREAD_RATIO * baseline
        # Chỉ áp dụng khi tâm vùng không bị tịnh tiến di chuyển (drift <= 20px) và có viền bập bùng hữu cơ (roughness >= 1.15)
        # Loại trừ dứt điểm đèn xe máy / ô tô di chuyển ngang qua khung hình
        if drift_from_base <= 20.0 and det.roughness >= 1.15:
            x1, y1, x2, y2 = det.bbox
            bx1, by1, bx2, by2 = base_bbox
            if thresh is not None and thresh.size > 0:
                h_crop, w_crop = thresh.shape[:2]
                gy, gx = np.ogrid[0:h_crop, 0:w_crop]
                abs_x = x1 + gx
                abs_y = y1 + gy
                outside_mask = (abs_x < bx1) | (abs_x > bx2) | (abs_y < by1) | (abs_y > by2)
                spread_pixels = float(np.count_nonzero((thresh > 0) & outside_mask))
                if spread_pixels >= 150.0 and spread_pixels >= 0.75 * base_area:
                    return "c", (
                        f"Vùng sáng lan tràn ra ngoài baseline {spread_pixels:.0f}px "
                        f">= 0.75x baseline ({base_area * 0.75:.0f}px)"
                    )

        # Điều kiện d: Nhấp nháy bập bùng mạnh KÈM diện tích tăng (cháy lan thực sự từ đèn tĩnh/biển báo)
        if drift_from_base <= 25.0 and len(tracker.history_measurements) >= 3 and bright_area >= 1.40 * base_area:
            lum_values = [m[2] for m in tracker.history_measurements]
            flicker_std = float(np.std(lum_values))
            if flicker_std >= STATIC_RELEASE_FLICKER_STD:
                return "d", f"Nhấp nháy bập bùng mạnh std={flicker_std:.2f} >= {STATIC_RELEASE_FLICKER_STD} kèm diện tích tăng {bright_area/base_area:.2f}x"

        # Điều kiện e: Có box KHÓI trong hoặc sát vùng (khoảng cách tâm < FLARE_SMOKE_DIST_RATIO * width)
        max_smoke_dist = FLARE_SMOKE_DIST_RATIO * base_w
        for cand in all_candidates:
            if is_smoke(cand.class_name):
                scx = (cand.bbox[0] + cand.bbox[2]) / 2.0
                scy = (cand.bbox[1] + cand.bbox[3]) / 2.0
                dist = math.hypot(scx - base_cx, scy - base_cy)
                if dist < max_smoke_dist:
                    return "e", f"Phát hiện khói sát vùng khoảng cách={dist:.1f}px < {max_smoke_dist:.1f}px"

        # Điều kiện f: Có box lửa MỚI sát vùng, không khớp baseline (IoU < 0.5) và được 2 model đồng ý
        for cand in all_candidates:
            if cand is not det and is_fire(cand.class_name):
                iou_base = compute_iou(cand.bbox, base_bbox)
                if iou_base < 0.50 and cand.n_votes >= 2:
                    fcx = (cand.bbox[0] + cand.bbox[2]) / 2.0
                    fcy = (cand.bbox[1] + cand.bbox[3]) / 2.0
                    dist = math.hypot(fcx - base_cx, fcy - base_cy)
                    if dist < 2.0 * base_w:
                        return "f", (
                            f"Box lửa mới cạnh vùng (IoU={iou_base:.2f} < 0.5, 2 model đồng ý, "
                            f"khoảng cách={dist:.1f}px)"
                        )

        # Nếu không có dấu hiệu bùng lửa: Cập nhật baseline rất chậm (EMA 0.05)
        if bright_area < 1.15 * base_area:
            tracker.baseline_bright_area = 0.95 * tracker.baseline_bright_area + 0.05 * bright_area
            tracker.baseline_mean_lum = 0.95 * tracker.baseline_mean_lum + 0.05 * mean_lum

        return None

    def analyze(
        self,
        candidates: List,
        raw_gray: np.ndarray,
        timestamp_monotonic: float,
        is_night: bool,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
    ) -> Tuple[List, bool]:
        """
        Hàm phân tích điều phối duy nhất được detect_frame() gọi:
        Đầu vào:
          - candidates: danh sách box ứng viên đã qua _post_filter_candidates
          - raw_gray: khung hình xám GỐC (chưa qua CLAHE)
          - timestamp_monotonic: mốc thời gian thời gian thực
          - is_night: cờ chế độ đêm / hồng ngoại IR
        Đầu ra:
          - confirmed_detections: danh sách box kèm threat_status ('cho_xac_nhan', 'xac_nhan', 'bung_lua')
          - bypass_fast_alert: cờ phát còi cảnh báo nhanh tức thì
        """
        now = timestamp_monotonic
        # Đảm bảo mốc thời gian chuyển cho static filter đồng bộ
        effective_timestamp = timestamp_sec if timestamp_sec > 0.0 else timestamp_monotonic
        h, w = raw_gray.shape[:2]
        bypass_fast_alert = False

        if not candidates:
            self._maintain_hysteresis_relock(now)
            self.flare_fast_start_time = 0.0
            return [], False

        # ==========================================================
        # BƯỚC 1: KIỂM TRA MỞ KHÓA BÙNG LỬA (FLARE-UP UNLOCK)
        # ==========================================================
        flare_active_conditions: Set[str] = set()

        # 1.1. Đánh giá trên EXCLUDE_ZONES
        for det in candidates:
            cname = det.class_name.lower()
            if not is_fire(cname) and not (is_smoke(cname) and EXCLUDE_ZONES_APPLY_SMOKE):
                continue

            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0

            for z_idx, zone in enumerate(self.exclude_zones):
                if self._check_point_in_zone(cx, cy, w, h, zone):
                    tracker = self.zone_trackers.get(z_idx)
                    if not tracker:
                        tracker = FlareTracker(tracker_id=z_idx, is_exclude_zone=True)
                        self.zone_trackers[z_idx] = tracker

                    trigger = self._evaluate_flare_triggers(tracker, det, raw_gray, now, candidates)
                    if trigger:
                        cond_code, cond_msg = trigger
                        flare_active_conditions.add(cond_code)
                        tracker.is_unlocked = True
                        tracker.unlocked_time = now
                        tracker.last_seen_fire_time = now
                        tracker.unlock_reason = f"[{cond_code}] {cond_msg}"
                        det.threat_status = "bung_lua"
                        logger.warning(
                            f"[FlareUnlock-ExcludeZone #{z_idx}] ⚠️ BÙNG LỬA: {cond_msg} | bbox={det.bbox}"
                        )
                    elif tracker.is_unlocked:
                        tracker.last_seen_fire_time = now
                        det.threat_status = "bung_lua"

        # 1.2. Đánh giá trên từng nguồn sáng lửa (Static Flare Trackers)
        matched_flare_ids: Set[int] = set()
        for det in candidates:
            if not is_fire(det.class_name):
                continue

            det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
            det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
            det_w = max(10, det.bbox[2] - det.bbox[0])

            # Ghép cặp với tracker có sẵn (IoU >= 0.20 hoặc tâm gần <= 1.2x width)
            best_id = -1
            best_iou = 0.0
            for fid, trk in self.light_flare_trackers.items():
                if fid in matched_flare_ids:
                    continue
                iou = compute_iou(det.bbox, trk.bbox)
                tcx = (trk.bbox[0] + trk.bbox[2]) / 2.0
                tcy = (trk.bbox[1] + trk.bbox[3]) / 2.0
                dist = math.hypot(det_cx - tcx, det_cy - tcy)
                if (iou >= 0.20 and iou > best_iou) or (dist < 1.2 * det_w and best_id == -1):
                    best_id = fid
                    best_iou = max(best_iou, iou)

            if best_id >= 0:
                matched_flare_ids.add(best_id)
                tracker = self.light_flare_trackers[best_id]
                tracker.bbox = det.bbox
                tracker.last_seen_time = now
            else:
                best_id = self._next_flare_id
                self._next_flare_id += 1
                tracker = FlareTracker(
                    tracker_id=best_id,
                    is_exclude_zone=False,
                    bbox=det.bbox,
                    first_seen_time=now,
                    last_seen_time=now
                )
                self.light_flare_trackers[best_id] = tracker
                matched_flare_ids.add(best_id)

            # Đánh giá 6 điều kiện bùng lửa
            trigger = self._evaluate_flare_triggers(tracker, det, raw_gray, now, candidates)
            if trigger:
                cond_code, cond_msg = trigger
                flare_active_conditions.add(cond_code)
                tracker.is_unlocked = True
                tracker.unlocked_time = now
                tracker.last_seen_fire_time = now
                tracker.unlock_reason = f"[{cond_code}] {cond_msg}"
                det.threat_status = "bung_lua"

                # Nếu box nằm trong static memory của LocNguonSangTinh: Mở khóa ngay
                if self.static_filter and hasattr(self.static_filter, "unlock_region"):
                    self.static_filter.unlock_region(det.bbox, det_cx, det_cy, det_w)

                logger.warning(
                    f"[FlareUnlock-StaticLight #{tracker.tracker_id}] ⚠️ BÙNG LỬA: {cond_msg} | bbox={det.bbox}"
                )
            elif tracker.is_unlocked:
                tracker.last_seen_fire_time = now
                det.threat_status = "bung_lua"

        # Dọn dẹp flare trackers mất dấu > 10s
        dead_fids = [fid for fid, ft in self.light_flare_trackers.items() if now - ft.last_seen_time > 10.0 and not ft.is_unlocked]
        for fid in dead_fids:
            del self.light_flare_trackers[fid]

        # ==========================================================
        # BƯỚC 2: DUY TRÌ HYSTERESIS VÀ KHÓA LẠI (RE-LOCKING)
        # ==========================================================
        self._maintain_hysteresis_relock(now)

        # ==========================================================
        # BƯỚC 3: LỌC VÙNG LOẠI TRỪ (EXCLUDE_ZONES)
        # ==========================================================
        unlocked_candidates: List = []
        for det in candidates:
            cname = det.class_name.lower()
            is_fire_box = is_fire(cname)
            is_smoke_box = is_smoke(cname)

            if self.exclude_zones and (is_fire_box or (is_smoke_box and EXCLUDE_ZONES_APPLY_SMOKE)):
                cx = (det.bbox[0] + det.bbox[2]) / 2.0
                cy = (det.bbox[1] + det.bbox[3]) / 2.0
                in_zone = False
                in_unlocked_zone = False

                for z_idx, zone in enumerate(self.exclude_zones):
                    if self._check_point_in_zone(cx, cy, w, h, zone):
                        in_zone = True
                        tracker = self.zone_trackers.get(z_idx)
                        if tracker and tracker.is_unlocked:
                            in_unlocked_zone = True
                        break

                if in_zone and not in_unlocked_zone:
                    logger.debug(f"[ThreatAnalyzer-Exclude] Bỏ qua box nằm trong vùng loại trừ: {det.bbox}")
                    continue

            unlocked_candidates.append(det)

        # ==========================================================
        # BƯỚC 4: BỘ LỌC NGUỒN SÁNG TĨNH (LocNguonSangTinh) - PHẠT MỀM
        # ==========================================================
        if self.static_filter is not None:
            flare_boxes = [d for d in unlocked_candidates if getattr(d, "threat_status", "") == "bung_lua"]
            normal_boxes = [d for d in unlocked_candidates if getattr(d, "threat_status", "") != "bung_lua"]

            filtered_normal, _ = self.static_filter.filter_detections(
                frame=raw_gray,
                detections=normal_boxes,
                frame_index=frame_index,
                timestamp_sec=effective_timestamp,
                soft_filter=False
            )
            filtered_candidates = flare_boxes + filtered_normal
        else:
            filtered_candidates = unlocked_candidates

        # ==========================================================
        # BƯỚC 5: TRACKER KHÔNG-THỜI GIAN & TÍCH LŨY BẰNG CHỨNG (GIAI ĐOẠN 2)
        # ==========================================================
        active_tracks = self.tracker.update(filtered_candidates, frame_index, effective_timestamp)

        confirmed_detections: List = []
        for det in filtered_candidates:
            # Tìm track phù hợp nhất
            best_t: Optional[TrackedObject] = None
            best_sim = -1.0
            for t in active_tracks:
                if t.class_name.lower() == det.class_name.lower():
                    sim = self.tracker.compute_similarity(det.bbox, t.bbox)
                    if sim > best_sim:
                        best_sim = sim
                        best_t = t

            if best_t is None:
                confirmed_detections.append(det)
                continue

            # 1. Trích xuất đặc trưng động học & độ nhám viền (tái sử dụng từ lượt đo trước nếu có)
            kinematics = self.motion_extractor.compute_kinematic_features(best_t)
            roughness = getattr(det, "roughness", 1.0)
            if roughness == 1.0:
                roughness = self.motion_extractor.extract_boundary_jitter(raw_gray, det.bbox)
                det.roughness = roughness

            # 2. Cờ bùng lửa khẩn cấp (Emergency flare escalation)
            is_flare = (getattr(det, "threat_status", "") == "bung_lua")

            # Trường hợp khẩn cấp 1: Có cột khói bao trùm hoặc bốc lên từ đốm lửa
            if not is_flare and is_fire(det.class_name):
                for other in filtered_candidates:
                    if is_smoke(other.class_name):
                        scx = (other.bbox[0] + other.bbox[2]) / 2.0
                        fcx = (det.bbox[0] + det.bbox[2]) / 2.0
                        scy = (other.bbox[1] + other.bbox[3]) / 2.0
                        fcy = (det.bbox[1] + det.bbox[3]) / 2.0
                        fw = max(10, det.bbox[2] - det.bbox[0])
                        # Khói bao trùm hoặc bốc lên sát đốm lửa
                        is_near_smoke = (
                            compute_iou(other.bbox, det.bbox) > 0.0 or
                            (abs(scx - fcx) < 2.0 * fw and scy <= fcy + 40)
                        )
                        if is_near_smoke:
                            is_flare = True
                            det.threat_status = "bung_lua"
                            flare_active_conditions.add("e")
                            break

            # Trường hợp khẩn cấp 2: Diện tích tăng vọt >= 1.3x kèm viền bập bùng hữu cơ >= 1.25 (loại trừ tia lửa cơ khí)
            if not is_flare and is_fire(det.class_name):
                crop_g = crop_roi_gray(raw_gray, det.bbox) if raw_gray is not None else None
                is_sparks = False
                if crop_g is not None:
                    is_sparks, _, _ = analyze_spark_dispersion(crop_g)
                if not is_sparks and kinematics.get("area_growth_ratio", 1.0) >= EMERGENCY_GROWTH_RATIO and roughness >= EMERGENCY_ROUGHNESS_THRESH:
                    is_flare = True
                    det.threat_status = "bung_lua"
                    flare_active_conditions.add("b")

            # Đồng bộ cờ static suspect từ static_filter
            if getattr(det, "is_static_suspect", False):
                best_t.is_static_suspect = True

            # 3. Cập nhật điểm tích lũy bằng chứng E(t)
            e_score, state, reason = self.accumulator.update_track_evidence(
                track=best_t,
                timestamp_sec=effective_timestamp,
                kinematic_feats=kinematics,
                roughness=roughness,
                is_flare=is_flare
            )

            # 4. Hỗ trợ ảnh tĩnh / frame đầu tiên (Single-frame bootstrapping)
            is_smooth_lamp = (roughness < 1.08 and getattr(det, "core_sat_ratio", 0.0) >= 0.40)
            if best_t.total_seen_frames <= 3:
                if is_smooth_lamp:
                    # Chao đèn pha xe máy / bóng đèn đường viền trơn: Phạt điểm, không cho phép boost
                    best_t.evidence_score = max(0.0, best_t.evidence_score - 15.0)
                    best_t.state = "SUSPECT"
                elif det.n_votes >= 2:
                    # Đám cháy ban đầu 2 model đồng thuận: cho phép khởi động CONFIRMED (tối đa 75đ)
                    seeded_e = max(best_t.evidence_score, min(75.0, det.confidence * 80.0 + 10.0))
                    best_t.evidence_score = round(seeded_e, 2)
                    best_t.state = "CONFIRMED" if seeded_e >= 50.0 else "SUSPECT"
                elif is_night and det.confidence >= 0.20 and roughness >= 1.08:
                    # Đốm lửa ban đầu ban đêm: nạp điểm khởi động ở mức SUSPECT (30đ), cần tích lũy thời gian để lên CONFIRMED/ALARM
                    seeded_e = max(best_t.evidence_score, 30.0)
                    best_t.evidence_score = round(seeded_e, 2)
                    best_t.state = "SUSPECT"
                elif det.confidence >= 0.40:
                    # 1 model ban ngày: khởi động ở mức SUSPECT (tối đa 40đ), cần tích lũy qua các frame để lên CONFIRMED
                    seeded_e = max(best_t.evidence_score, min(40.0, det.confidence * 50.0))
                    best_t.evidence_score = round(seeded_e, 2)
                    best_t.state = "SUSPECT"

            # Phân loại vật thể chuyển động ngoại cảnh (xe cộ, người mặc áo phản quang)
            is_vehicle = self.motion_extractor.is_vehicle_motion(kinematics)
            is_vest = self.motion_extractor.is_reflective_vest_or_rigid_worker(kinematics, roughness)

            # 5. Gán dữ liệu track vào detection
            det.track_id = best_t.track_id
            det.evidence_score = best_t.evidence_score
            det.track_state = best_t.state
            if best_t.state in ("CONFIRMED", "ALARM"):
                posterior_conf = max(det.confidence, min(0.95, best_t.evidence_score / 100.0))
                det.confidence = round(posterior_conf, 2)

            # 6. Ánh xạ trạng thái mối nguy (threat_status)
            if getattr(det, "threat_status", "") == "bung_lua":
                det.threat_status = "bung_lua"
            elif (is_vehicle or is_vest or is_smooth_lamp) and not is_flare:
                # Xe cộ, người mặc áo phản quang hoặc đèn pha xe máy: Giữ ở trạng thái chờ xác nhận
                det.threat_status = "cho_xac_nhan"
                best_t.evidence_score = max(0.0, best_t.evidence_score - 10.0)
            elif is_night:
                if is_fire(det.class_name):
                    # Ban đêm: Xác nhận khi 2 model đồng thuận HOẶC đạt CONFIRMED/ALARM và đã theo dõi >= 2 frame HOẶC conf cao
                    if det.n_votes >= 2 or (best_t.state in ("CONFIRMED", "ALARM") and best_t.total_seen_frames >= 2) or det.confidence >= 0.50:
                        det.threat_status = "xac_nhan"
                    else:
                        det.threat_status = "cho_xac_nhan"
                else:
                    det.threat_status = "xac_nhan"
            else:
                # Ban ngày:
                if best_t.state in ("ALARM", "CONFIRMED") or det.confidence >= 0.45 or det.n_votes >= 2:
                    det.threat_status = "xac_nhan"
                else:
                    det.threat_status = "cho_xac_nhan"

            confirmed_detections.append(det)

        # ==========================================================
        # BƯỚC 6: BÁO ĐỘNG NHANH BÙNG LỬA (FAST ALERT BYPASS)
        # ==========================================================
        has_fast_flare = any(
            cond in flare_active_conditions for cond in ("a", "b", "e")
        ) or any(
            getattr(d, "threat_status", "") == "bung_lua" for d in confirmed_detections
        )

        if has_fast_flare:
            if self.flare_fast_start_time == 0.0:
                self.flare_fast_start_time = now
            flare_duration = now - self.flare_fast_start_time
            if flare_duration >= self.flare_fast_confirm_sec:
                bypass_fast_alert = True
        else:
            self.flare_fast_start_time = 0.0

        return confirmed_detections, bypass_fast_alert

    def _maintain_hysteresis_relock(self, now: float) -> None:
        """Quản lý thời gian mở khóa tối thiểu (FLARE_HOLD_SEC) và khóa lại (FLARE_CLEAR_SEC)."""
        # Cho EXCLUDE_ZONES
        for z_idx, tracker in self.zone_trackers.items():
            if tracker.is_unlocked:
                time_since_unlock = now - tracker.unlocked_time
                time_since_fire = now - tracker.last_seen_fire_time
                if (time_since_unlock >= FLARE_HOLD_SEC) and (time_since_fire >= FLARE_CLEAR_SEC):
                    tracker.is_unlocked = False
                    tracker.baseline_initialized = False
                    tracker.history_measurements.clear()
                    logger.info(
                        f"[Hysteresis-Relock] 🔒 Khóa lại Vùng loại trừ #{z_idx} "
                        f"sau {time_since_fire:.1f}s không thấy lửa."
                    )

        # Cho Light Flare Trackers
        for fid, tracker in list(self.light_flare_trackers.items()):
            if tracker.is_unlocked:
                time_since_unlock = now - tracker.unlocked_time
                time_since_fire = now - tracker.last_seen_fire_time
                if (time_since_unlock >= FLARE_HOLD_SEC) and (time_since_fire >= FLARE_CLEAR_SEC):
                    tracker.is_unlocked = False
                    tracker.baseline_initialized = False
                    tracker.history_measurements.clear()
                    del self.light_flare_trackers[fid]
                    logger.info(
                        f"[Hysteresis-Relock] 🔒 Khóa lại Nguồn sáng #{fid} "
                        f"sau {time_since_fire:.1f}s không thấy lửa."
                    )
