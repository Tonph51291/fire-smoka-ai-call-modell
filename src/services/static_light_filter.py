"""Bộ lọc nguồn sáng tĩnh (LocNguonSangTinh) - Loại bỏ đèn đường và đèn hiên cố định."""

from collections import deque
from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import (
    CUA_SO_THEO_DOI_TINH,
    IOU_TRACK_THRESH,
    MIN_FRAMES_XAC_NHAN_TINH,
    NGUONG_DICH,
    NGUONG_DONG,
    TEN_LUA,
)

logger = logging.getLogger(__name__)


def compute_iou_boxes(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Tính IoU giữa 2 bounding box (x1, y1, x2, y2)."""
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


@dataclass
class TrackedLightBox:
    """Theo dõi một bounding box qua các frame liên tiếp."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    last_frame_idx: int
    history_centers: deque = field(default_factory=lambda: deque(maxlen=CUA_SO_THEO_DOI_TINH))
    history_crops: deque = field(default_factory=lambda: deque(maxlen=CUA_SO_THEO_DOI_TINH))
    total_seen: int = 1
    is_static: bool = False
    last_drift: float = 0.0
    last_motion: float = 0.0


class LocNguonSangTinh:
    """
    Bộ lọc nguồn sáng tĩnh:
    - Theo dõi từng hộp qua 30 frame bằng IoU.
    - Loại bỏ các hộp có vị trí gần như không đổi (dịch chuyển < nguong_dich)
      VÀ nội dung bên trong gần như không thay đổi (biến thiên pixel < nguong_dong).
    - Ngọn lửa thật luôn bập bùng (motion cao) hoặc mở rộng/dịch chuyển (drift cao)
      nên sẽ không bị loại bỏ.
    """

    def __init__(
        self,
        cua_so_frame: int = CUA_SO_THEO_DOI_TINH,
        nguong_dich: float = NGUONG_DICH,
        nguong_dong: float = NGUONG_DONG,
        iou_thresh: float = IOU_TRACK_THRESH,
        min_frames_xac_nhan: int = MIN_FRAMES_XAC_NHAN_TINH,
        max_missed_frames: int = 10,
    ) -> None:
        self.cua_so_frame = cua_so_frame
        self.nguong_dich = nguong_dich
        self.nguong_dong = nguong_dong
        self.iou_thresh = iou_thresh
        self.min_frames_xac_nhan = min_frames_xac_nhan
        self.max_missed_frames = max_missed_frames

        self._next_track_id: int = 1
        self.tracks: Dict[int, TrackedLightBox] = {}

    def reset(self) -> None:
        """Xóa toàn bộ lịch sử theo dõi (khi chuyển video hoặc mở stream mới)."""
        self.tracks.clear()
        self._next_track_id = 1

    def _extract_crop_gray(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop vùng bbox và resize về chuẩn (32, 32) grayscale để đo biến thiên."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, bbox[0]))
        y1 = max(0, min(h - 1, bbox[1]))
        x2 = max(0, min(w, bbox[2]))
        y2 = max(0, min(h, bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if len(crop.shape) == 3:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop

        return cv2.resize(crop_gray, (32, 32), interpolation=cv2.INTER_AREA)

    def filter_detections(
        self,
        frame: np.ndarray,
        detections: List,  # List of Detection objects
        frame_index: int = 0
    ) -> Tuple[List, List]:
        """
        Lọc danh sách detection:
        Trả về (valid_detections, static_filtered_detections).
        Các detection bị coi là nguồn sáng tĩnh (đèn đường/đèn hiên) sẽ bị loại khỏi valid_detections.
        """
        if not detections:
            # Vẫn xóa các track đã mất dấu quá lâu
            self._prune_dead_tracks(frame_index)
            return [], []

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
            crop_std = self._extract_crop_gray(frame, det.bbox)

            if d_idx in det_to_track_map:
                tid = det_to_track_map[d_idx]
                track = self.tracks[tid]
                track.bbox = det.bbox
                track.class_name = det.class_name
                track.confidence = det.confidence
                track.last_frame_idx = frame_index
                track.total_seen += 1
                track.history_centers.append(center)
                if crop_std is not None:
                    track.history_crops.append(crop_std)
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                new_track = TrackedLightBox(
                    track_id=tid,
                    bbox=det.bbox,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    last_frame_idx=frame_index,
                    total_seen=1
                )
                new_track.history_centers.append(center)
                if crop_std is not None:
                    new_track.history_crops.append(crop_std)
                self.tracks[tid] = new_track
                det_to_track_map[d_idx] = tid

        # 3. Đánh giá tính tĩnh (static) cho từng track
        for tid in matched_track_ids:
            track = self.tracks[tid]
            self._evaluate_static_nature(track)

        # 4. Phân loại detections thành hợp lệ hoặc bị loại bỏ
        valid_dets = []
        static_dets = []

        for d_idx, det in enumerate(detections):
            tid = det_to_track_map.get(d_idx, -1)
            track = self.tracks.get(tid)

            # Chỉ lọc với lớp Lửa (TEN_LUA)
            is_fire = det.class_name.lower() in TEN_LUA
            if is_fire and track and track.is_static:
                static_dets.append(det)
                logger.debug(
                    f"[LocNguonSangTinh] Đã loại bỏ đèn tĩnh ID={tid} | "
                    f"Drift={track.last_drift:.2f}px < {self.nguong_dich} | "
                    f"Motion={track.last_motion:.2f} < {self.nguong_dong}"
                )
            else:
                valid_dets.append(det)

        # 5. Dọn dẹp track chết
        self._prune_dead_tracks(frame_index)

        return valid_dets, static_dets

    def _evaluate_static_nature(self, track: TrackedLightBox) -> None:
        """Đo đạc độ dịch chuyển tâm và độ biến thiên pixel nội dung."""
        # Cần tối thiểu min_frames_xac_nhan để đánh giá
        if len(track.history_centers) < self.min_frames_xac_nhan:
            track.is_static = False
            return

        # 1. Tính độ dịch chuyển tâm (Spatial Drift)
        centers = np.array(track.history_centers)
        centroid = np.mean(centers, axis=0)
        distances = np.linalg.norm(centers - centroid, axis=1)
        pos_drift = float(np.mean(distances))
        track.last_drift = pos_drift

        # 2. Tính độ biến thiên nội dung pixel qua các frame (Temporal Content Motion)
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

        # 3. Kết luận: Nếu cả vị trí và nội dung đều đứng yên -> NGUỒN SÁNG TĨNH
        if pos_drift < self.nguong_dich and motion_score < self.nguong_dong:
            track.is_static = True
        else:
            track.is_static = False

    def _prune_dead_tracks(self, current_frame_idx: int) -> None:
        """Xóa các track không còn xuất hiện."""
        dead_tids = [
            tid for tid, trk in self.tracks.items()
            if (current_frame_idx - trk.last_frame_idx) > self.max_missed_frames
        ]
        for tid in dead_tids:
            del self.tracks[tid]
