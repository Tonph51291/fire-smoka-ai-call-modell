"""Bộ lọc nguồn sáng tĩnh (LocNguonSangTinh) và Bộ nhớ tự học (Static Memory).

Loại bỏ các nguồn sáng cố định gây báo giả như đèn đường, đèn hiên, biển quảng cáo:
1. Theo dõi các hộp phát hiện qua IoU và thời gian thực (giây).
2. Đo đạc độ dịch chuyển tâm (spatial drift), độ biến thiên nội dung pixel (motion) trên ảnh xám GỐC.
3. Đánh giá đặc trưng bão hòa ánh sáng (lõi đèn >= 240) và độ lệch chuẩn nhấp nháy (flicker std).
4. Bộ nhớ nguồn sáng tĩnh (Static Memory): Lưu vị trí đèn tĩnh đã xác nhận trong 600s.
   Các frame tiếp theo nếu box trùng vị trí thì loại bỏ NGAY TỨC THÌ.
5. Cơ chế an toàn (Safety Release): Tự động MỞ LẠI vùng bị chặn khi diện tích box tăng >= 1.5 lần
   hoặc xuất hiện nhấp nháy lớn (dấu hiệu ngọn lửa bùng phát hoặc cháy lan).
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import (
    CUA_SO_THEO_DOI_TINH,
    FLICKER_STD_MAX,
    HARD_NEGATIVES_DIR,
    HARD_NEGATIVES_INTERVAL_SEC,
    HARD_NEGATIVES_MAX_COUNT,
    IOU_TRACK_THRESH,
    MIN_FRAMES_XAC_NHAN_TINH,
    NGUONG_DICH,
    NGUONG_DONG,
    SATURATION_PIXEL_THRESH,
    SATURATION_RATIO_MIN,
    SAVE_HARD_NEGATIVES,
    STATIC_CONFIRM_SEC,
    STATIC_MEMORY_IOU_THRESH,
    STATIC_MEMORY_SEC,
    STATIC_RELEASE_AREA_GROWTH,
    STATIC_RELEASE_FLICKER_STD,
    is_fire,
)
from src.utils.video_utils import compute_iou

logger = logging.getLogger(__name__)

# Giữ alias cho tương thích ngược với các bài kiểm thử
compute_iou_boxes = compute_iou


@dataclass
class StaticMemoryItem:
    """Mục lưu trữ trong bộ nhớ nguồn sáng tĩnh tự học."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    first_confirmed_time: float
    last_seen_time: float
    baseline_area: float
    baseline_mean_lum: float
    core_sat_ratio: float


@dataclass
class TrackedLightBox:
    """Theo dõi một bounding box qua các frame liên tiếp."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    last_frame_idx: int
    first_seen_time: float
    last_seen_time: float
    history_centers: deque = field(default_factory=lambda: deque(maxlen=CUA_SO_THEO_DOI_TINH))
    history_crops: deque = field(default_factory=lambda: deque(maxlen=CUA_SO_THEO_DOI_TINH))
    history_timestamps: deque = field(default_factory=lambda: deque(maxlen=CUA_SO_THEO_DOI_TINH))
    history_areas: deque = field(default_factory=lambda: deque(maxlen=CUA_SO_THEO_DOI_TINH))
    history_mean_lum: deque = field(default_factory=lambda: deque(maxlen=CUA_SO_THEO_DOI_TINH))
    total_seen: int = 1
    is_static: bool = False
    last_drift: float = 0.0
    last_motion: float = 0.0
    core_saturation_ratio: float = 0.0
    flicker_std: float = 0.0


class LocNguonSangTinh:
    """
    Bộ lọc nguồn sáng tĩnh nâng cao kết hợp Static Memory và cơ chế Safety Release.
    """

    def __init__(
        self,
        cua_so_frame: int = CUA_SO_THEO_DOI_TINH,
        nguong_dich: float = NGUONG_DICH,
        nguong_dong: float = NGUONG_DONG,
        iou_thresh: float = IOU_TRACK_THRESH,
        min_frames_xac_nhan: int = MIN_FRAMES_XAC_NHAN_TINH,
        static_confirm_sec: float = STATIC_CONFIRM_SEC,
        max_missed_frames: int = 15,
    ) -> None:
        self.cua_so_frame = cua_so_frame
        self.nguong_dich = nguong_dich
        self.nguong_dong = nguong_dong
        self.iou_thresh = iou_thresh
        self.min_frames_xac_nhan = min_frames_xac_nhan
        self.static_confirm_sec = static_confirm_sec
        self.max_missed_frames = max_missed_frames

        self._next_track_id: int = 1
        self.tracks: Dict[int, TrackedLightBox] = {}

        # Bộ nhớ nguồn sáng tĩnh tự học (Static Memory)
        self.static_memory: Dict[int, StaticMemoryItem] = {}

        # Quản lý lưu Hard Negatives
        self._last_hard_negative_save: float = 0.0
        self._hard_negative_count: int = 0

    def reset(self) -> None:
        """Xóa toàn bộ lịch sử theo dõi (khi chuyển video hoặc mở stream mới)."""
        self.tracks.clear()
        self.static_memory.clear()
        self._next_track_id = 1

    def _extract_crop_raw_gray(self, raw_frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop vùng bbox trên ảnh xám GỐC (chưa qua CLAHE) để đo biến thiên thực sự."""
        if raw_frame is None or raw_frame.size == 0:
            return None
        h, w = raw_frame.shape[:2]
        x1 = max(0, min(w - 1, bbox[0]))
        y1 = max(0, min(h - 1, bbox[1]))
        x2 = max(0, min(w, bbox[2]))
        y2 = max(0, min(h, bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = raw_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if len(crop.shape) == 3:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop

        return cv2.resize(crop_gray, (32, 32), interpolation=cv2.INTER_AREA)

    def _compute_core_saturation(self, crop_gray: np.ndarray) -> float:
        """
        Tính tỉ lệ pixel bão hòa trắng (>= 240) ở vùng lõi trung tâm (50% diện tích).
        Đèn đường/đèn pha thường có bóng đèn sáng bão hòa trắng ở tâm.
        """
        if crop_gray is None or crop_gray.size == 0:
            return 0.0
        # Vùng lõi trung tâm 16x16 ở giữa 32x32
        core = crop_gray[8:24, 8:24]
        saturated_pixels = np.sum(core >= SATURATION_PIXEL_THRESH)
        return float(saturated_pixels / core.size)

    def filter_detections(
        self,
        frame: np.ndarray,
        detections: List,  # List of Detection objects
        frame_index: int = 0,
        timestamp_sec: Optional[float] = None
    ) -> Tuple[List, List]:
        """
        Lọc danh sách detection:
        - Sử dụng Static Memory để loại ngay các đèn tĩnh đã biết từ trước.
        - Theo dõi các box mới để xác nhận đèn tĩnh mới sau STATIC_CONFIRM_SEC.
        - Kiểm tra Safety Release để tự động mở lại cảnh báo nếu có cháy lan lớn.
        """
        use_time = (timestamp_sec is not None)
        now_time = float(timestamp_sec) if use_time else time.time()

        if not detections:
            self._prune_dead_tracks(frame_index)
            self._prune_expired_memory(now_time)
            return [], []

        # Dọn dẹp bộ nhớ tĩnh đã hết hạn (> 600s)
        self._prune_expired_memory(now_time)

        matched_track_ids = set()
        det_to_track_map: Dict[int, int] = {}

        # 1. Ghép cặp detection với các track hiện có bằng IoU
        for d_idx, det in enumerate(detections):
            best_iou = 0.0
            best_tid = -1

            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = compute_iou_boxes(det.bbox, track.bbox)
                if iou >= self.iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid >= 0:
                matched_track_ids.add(best_tid)
                det_to_track_map[d_idx] = best_tid

        # 2. Cập nhật hoặc tạo mới track
        for d_idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            box_area = float((x2 - x1) * (y2 - y1))
            crop_raw = self._extract_crop_raw_gray(frame, det.bbox)
            mean_lum = float(np.mean(crop_raw)) if crop_raw is not None else 0.0

            if d_idx in det_to_track_map:
                tid = det_to_track_map[d_idx]
                track = self.tracks[tid]
                track.bbox = det.bbox
                track.class_name = det.class_name
                track.confidence = det.confidence
                track.last_frame_idx = frame_index
                track.last_seen_time = now_time
                track.total_seen += 1
                track.history_centers.append(center)
                track.history_timestamps.append(now_time)
                track.history_areas.append(box_area)
                track.history_mean_lum.append(mean_lum)
                if crop_raw is not None:
                    track.history_crops.append(crop_raw)
                    track.core_saturation_ratio = self._compute_core_saturation(crop_raw)
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                new_track = TrackedLightBox(
                    track_id=tid,
                    bbox=det.bbox,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    last_frame_idx=frame_index,
                    first_seen_time=now_time,
                    last_seen_time=now_time,
                    total_seen=1
                )
                new_track.history_centers.append(center)
                new_track.history_timestamps.append(now_time)
                new_track.history_areas.append(box_area)
                new_track.history_mean_lum.append(mean_lum)
                if crop_raw is not None:
                    new_track.history_crops.append(crop_raw)
                    new_track.core_saturation_ratio = self._compute_core_saturation(crop_raw)
                self.tracks[tid] = new_track
                det_to_track_map[d_idx] = tid

        # 3. Đánh giá tính tĩnh (static) cho từng track hiện tại
        for tid in matched_track_ids:
            track = self.tracks[tid]
            self._evaluate_static_nature(track, now_time, use_time=use_time)

        # 4. Phân loại detections thành hợp lệ hoặc bị loại bỏ
        valid_dets = []
        static_dets = []

        for d_idx, det in enumerate(detections):
            tid = det_to_track_map.get(d_idx, -1)
            track = self.tracks.get(tid)
            is_fire_box = is_fire(det.class_name)

            if not is_fire_box:
                # Không áp dụng lọc đèn tĩnh cho khói
                valid_dets.append(det)
                continue

            # (A) Kiểm tra với Bộ nhớ nguồn sáng tĩnh (Static Memory)
            matched_memory_tid = -1
            memory_release_reason = None
            det_area = float((det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]))

            for mem_tid, mem_item in self.static_memory.items():
                iou = compute_iou_boxes(det.bbox, mem_item.bbox)
                if iou >= STATIC_MEMORY_IOU_THRESH:
                    matched_memory_tid = mem_tid

                    # Kiểm tra điều kiện Safety Release (Mở lại khi cháy lan thật)
                    area_growth = det_area / max(1.0, mem_item.baseline_area)
                    if area_growth >= STATIC_RELEASE_AREA_GROWTH:
                        memory_release_reason = f"Diện tích tăng mạnh {area_growth:.2f}x >= {STATIC_RELEASE_AREA_GROWTH}"
                    elif track and track.flicker_std >= STATIC_RELEASE_FLICKER_STD:
                        memory_release_reason = f"Độ nhấp nháy bập bùng tăng vọt std={track.flicker_std:.1f} >= {STATIC_RELEASE_FLICKER_STD}"
                    break

            if matched_memory_tid >= 0:
                if memory_release_reason:
                    # Mở lại cảnh báo vì có dấu hiệu cháy lan thật
                    logger.warning(
                        f"[StaticMemory-Release] ⚠️ MỞ LẠI VÙNG ĐÈN TĨNH ID={matched_memory_tid} | "
                        f"BBox: {det.bbox} | Lý do: {memory_release_reason}"
                    )
                    # Xóa khỏi bộ nhớ tĩnh để không chặn nữa
                    del self.static_memory[matched_memory_tid]
                    valid_dets.append(det)
                    continue
                else:
                    # Chặn ngay tức thì nhờ bộ nhớ tĩnh
                    static_dets.append(det)
                    self._maybe_save_hard_negative(frame, det.bbox)
                    continue

            # (B) Kiểm tra trạng thái tĩnh hiện tại của track
            if track and track.is_static:
                static_dets.append(det)
                # Ghi nhớ vào bộ nhớ tĩnh để bảo vệ các frame sau
                if tid not in self.static_memory:
                    baseline_area = float(np.median(track.history_areas)) if track.history_areas else det_area
                    baseline_lum = float(np.median(track.history_mean_lum)) if track.history_mean_lum else 100.0
                    self.static_memory[tid] = StaticMemoryItem(
                        track_id=tid,
                        bbox=track.bbox,
                        first_confirmed_time=now_time,
                        last_seen_time=now_time,
                        baseline_area=baseline_area,
                        baseline_mean_lum=baseline_lum,
                        core_sat_ratio=track.core_saturation_ratio
                    )
                    logger.info(
                        f"[StaticMemory] 🔒 ĐÃ GHI NHỚ ĐÈN TĨNH MỚI ID={tid} | BBox: {track.bbox} | "
                        f"Area={baseline_area:.0f}px | SatCore={track.core_saturation_ratio:.1%}"
                    )
                else:
                    self.static_memory[tid].last_seen_time = now_time

                self._maybe_save_hard_negative(frame, det.bbox)
            else:
                valid_dets.append(det)

        # 5. Dọn dẹp track chết
        self._prune_dead_tracks(frame_index)

        return valid_dets, static_dets

    def _evaluate_static_nature(self, track: TrackedLightBox, current_time: float, use_time: bool = False) -> None:
        """Đo đạc độ dịch chuyển tâm, độ biến thiên pixel và nhấp nháy."""
        num_frames = len(track.history_centers)
        if num_frames < self.min_frames_xac_nhan:
            track.is_static = False
            return

        if use_time:
            time_elapsed = current_time - track.first_seen_time
            if time_elapsed < self.static_confirm_sec:
                track.is_static = False
                return

        # 1. Tính độ dịch chuyển tâm (Spatial Drift)
        centers = np.array(track.history_centers)
        centroid = np.mean(centers, axis=0)
        distances = np.linalg.norm(centers - centroid, axis=1)
        pos_drift = float(np.mean(distances))
        track.last_drift = pos_drift

        # 2. Tính độ biến thiên nội dung pixel trên ảnh GỐC (Raw Content Motion)
        crops = list(track.history_crops)
        if len(crops) >= self.min_frames_xac_nhan:
            diffs = [
                float(np.mean(np.abs(crops[i].astype(np.float32) - crops[i - 1].astype(np.float32))))
                for i in range(1, len(crops))
            ]
            motion_score = float(np.mean(diffs)) if diffs else 0.0
        else:
            motion_score = 999.0
        track.last_motion = motion_score

        # 3. Tính độ lệch chuẩn nhấp nháy độ sáng (Flicker Std)
        lums = list(track.history_mean_lum)
        flicker_std = float(np.std(lums)) if len(lums) >= 5 else 0.0
        track.flicker_std = flicker_std

        # 4. Tiêu chí xác nhận nguồn sáng tĩnh:
        # - Vị trí đứng yên (pos_drift < nguong_dich)
        # - Nội dung không biến động mạnh (motion_score < nguong_dong)
        # - Ánh sáng ít nhấp nháy (flicker_std < FLICKER_STD_MAX)
        # - Hoặc có lõi bão hòa trắng >= 240 rõ nét của bóng đèn
        is_spatially_static = (pos_drift < self.nguong_dich)
        is_motion_static = (motion_score < self.nguong_dong)
        is_flicker_static = (flicker_std < FLICKER_STD_MAX)
        has_bulb_saturation = (track.core_saturation_ratio >= SATURATION_RATIO_MIN)

        if is_spatially_static and is_motion_static and (is_flicker_static or has_bulb_saturation):
            track.is_static = True
        else:
            track.is_static = False

    def _maybe_save_hard_negative(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Tự động lưu ảnh hard negative khi phát hiện đèn tĩnh (nếu bật cờ SAVE_HARD_NEGATIVES)."""
        if not SAVE_HARD_NEGATIVES:
            return
        now = time.time()
        if now - self._last_hard_negative_save < HARD_NEGATIVES_INTERVAL_SEC:
            return
        if self._hard_negative_count >= HARD_NEGATIVES_MAX_COUNT:
            return

        try:
            HARD_NEGATIVES_DIR.mkdir(parents=True, exist_ok=True)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = HARD_NEGATIVES_DIR / f"hard_neg_{ts_str}_{self._hard_negative_count}.jpg"
            cv2.imwrite(str(save_path), frame)
            self._last_hard_negative_save = now
            self._hard_negative_count += 1
            logger.info(f"[HardNegative] Đã lưu mẫu âm tính vào: {save_path.name}")
        except Exception as e:
            logger.debug(f"Lỗi khi lưu hard negative: {e}")

    def _prune_dead_tracks(self, current_frame_idx: int) -> None:
        """Xóa các track không còn xuất hiện trong nhiều frame."""
        dead_tids = [
            tid for tid, trk in self.tracks.items()
            if (current_frame_idx - trk.last_frame_idx) > self.max_missed_frames
        ]
        for tid in dead_tids:
            del self.tracks[tid]

    def _prune_expired_memory(self, current_time: float) -> None:
        """Xóa các mục bộ nhớ tĩnh đã hết hạn (> STATIC_MEMORY_SEC)."""
        expired_tids = [
            tid for tid, mem in self.static_memory.items()
            if (current_time - mem.last_seen_time) > STATIC_MEMORY_SEC
        ]
        for tid in expired_tids:
            del self.static_memory[tid]
