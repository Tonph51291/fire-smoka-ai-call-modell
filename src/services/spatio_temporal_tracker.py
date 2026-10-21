"""Bộ bám vết không-thời gian (Spatio-Temporal Tracker) cho vật thể biến dạng, giãn nở và trôi dạt.
Được thiết kế tối ưu cho ngọn lửa bập bùng và làn khói bốc lên:
- Sử dụng hàm tương đồng đa tiêu chí: IoU + Độ bao hàm (Containment) + Khoảng cách tâm chuẩn hóa.
- Quản lý vòng đời Track ID ổn định, dung thứ cho việc mất dấu tạm thời trong vài frame.
- Cung cấp dữ liệu chuỗi thời gian (timestamps, bounding boxes, centers, areas) cho bộ trích xuất đặc trưng.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
import numpy as np

from src.utils.geometry import compute_iou, compute_containment


@dataclass
class TrackedObject:
    """Đại diện cho một vật thể đang được theo dõi qua các frame liên tiếp."""
    track_id: int
    class_name: str                           # "fire" hoặc "smoke"
    bbox: Tuple[int, int, int, int]           # (x1, y1, x2, y2) hiện tại
    confidence: float                         # Điểm tự tin hiện tại
    first_seen_time: float                    # Timestamp lần đầu phát hiện (giây)
    last_seen_time: float                     # Timestamp lần cuối phát hiện (giây)
    last_frame_idx: int                       # Chỉ số frame gần nhất
    total_seen_frames: int = 1                # Tổng số frame nhìn thấy
    missed_frames: int = 0                    # Số frame liên tiếp bị mất dấu
    
    # Lịch sử chuỗi thời gian
    history_timestamps: List[float] = field(default_factory=list)
    history_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    history_confs: List[float] = field(default_factory=list)
    history_n_votes: List[int] = field(default_factory=list)
    
    # Trạng thái bằng chứng và phân loại nguy cơ
    evidence_score: float = 0.0               # Điểm tích lũy E(t) từ 0 đến 100
    state: str = "NONE"                       # "NONE" | "SUSPECT" | "CONFIRMED" | "ALARM"
    state_enter_time: float = 0.0             # Thời điểm bắt đầu bước vào trạng thái hiện tại
    is_static_suspect: bool = False           # Cờ nghi vấn là nguồn sáng tĩnh (đèn/biển báo)
    is_vehicle_suspect: bool = False          # Cờ nghi vấn là đèn xe chạy qua
    is_flare: bool = False                    # Cờ bùng lửa khẩn cấp


class SpatioTemporalTracker:
    """Bộ điều phối bám vết đa mục tiêu cho Lửa và Khói."""

    def __init__(
        self,
        max_time_lost_sec: float = 1.5,       # [GIẢ ĐỊNH CẦN ĐO] - Xóa track nếu mất dấu quá 1.5s
        min_composite_similarity: float = 0.25,# [GIẢ ĐỊNH CẦN ĐO] - Ngưỡng tương đồng tối thiểu để ghép cặp
        w_iou: float = 0.50,
        w_contain: float = 0.30,
        w_dist: float = 0.20,
    ) -> None:
        self.max_time_lost_sec = max_time_lost_sec
        self.min_similarity = min_composite_similarity
        self.w_iou = w_iou
        self.w_contain = w_contain
        self.w_dist = w_dist
        
        self._next_track_id: int = 1
        self.tracks: Dict[int, TrackedObject] = {}

    def reset(self) -> None:
        """Xóa toàn bộ track khi chuyển đổi luồng video hoặc tua frame."""
        self.tracks.clear()
        self._next_track_id = 1

    def compute_similarity(
        self,
        det_bbox: Tuple[int, int, int, int],
        track_bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Tính điểm tương đồng đa tiêu chí giữa box mới và track cũ:
        1. IoU hình học.
        2. Containment (độ bao hàm khi khói nở to hoặc bao bọc box cũ).
        3. Khoảng cách tâm chuẩn hóa theo kích thước vật thể.
        """
        iou = compute_iou(det_bbox, track_bbox)
        contain = compute_containment(det_bbox, track_bbox)
        
        # Khoảng cách tâm chuẩn hóa
        c_det_x = (det_bbox[0] + det_bbox[2]) / 2.0
        c_det_y = (det_bbox[1] + det_bbox[3]) / 2.0
        c_trk_x = (track_bbox[0] + track_bbox[2]) / 2.0
        c_trk_y = (track_bbox[1] + track_bbox[3]) / 2.0
        
        trk_w = max(1.0, float(track_bbox[2] - track_bbox[0]))
        trk_h = max(1.0, float(track_bbox[3] - track_bbox[1]))
        sigma = 0.5 * max(trk_w, trk_h)
        
        dist = math.hypot(c_det_x - c_trk_x, c_det_y - c_trk_y)
        dist_score = math.exp(-dist / max(1.0, sigma))
        
        composite = self.w_iou * iou + self.w_contain * contain + self.w_dist * dist_score
        return composite

    def update(
        self,
        detections: List,  # Danh sách đối tượng Detection từ bước WBF
        frame_index: int,
        timestamp_sec: float
    ) -> List[TrackedObject]:
        """
        Cập nhật chuỗi bám vết từ các detection của frame hiện tại:
        - Ghép cặp detection với track hiện có theo độ tương đồng cực đại.
        - Tạo mới track cho detection chưa được gán.
        - Loại bỏ các track bị mất dấu quá max_time_lost_sec.
        """
        matched_track_ids = set()
        matched_det_indices = set()
        
        # 1. Ghép cặp tham lam theo độ tương đồng cao nhất
        candidates = []
        for d_idx, det in enumerate(detections):
            cname = det.class_name.lower()
            for tid, track in self.tracks.items():
                if track.class_name.lower() != cname:
                    continue  # Không ghép khác lớp (lửa không ghép với khói)
                sim = self.compute_similarity(det.bbox, track.bbox)
                if sim >= self.min_similarity:
                    candidates.append((sim, d_idx, tid))
                    
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for sim, d_idx, tid in candidates:
            if d_idx in matched_det_indices or tid in matched_track_ids:
                continue
            matched_det_indices.add(d_idx)
            matched_track_ids.add(tid)
            
            # Cập nhật track
            det = detections[d_idx]
            track = self.tracks[tid]
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.last_seen_time = timestamp_sec
            track.last_frame_idx = frame_index
            track.total_seen_frames += 1
            track.missed_frames = 0
            
            track.history_timestamps.append(timestamp_sec)
            track.history_bboxes.append(det.bbox)
            track.history_confs.append(det.confidence)
            track.history_n_votes.append(getattr(det, "n_votes", 1))
            
            # Giữ lịch sử tối đa 60 điểm (khoảng 3-4 giây video)
            if len(track.history_timestamps) > 60:
                track.history_timestamps.pop(0)
                track.history_bboxes.pop(0)
                track.history_confs.pop(0)
                track.history_n_votes.pop(0)

        # 2. Tạo mới Track cho các detection chưa được ghép
        for d_idx, det in enumerate(detections):
            if d_idx in matched_det_indices:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            
            new_track = TrackedObject(
                track_id=tid,
                class_name=det.class_name,
                bbox=det.bbox,
                confidence=det.confidence,
                first_seen_time=timestamp_sec,
                last_seen_time=timestamp_sec,
                last_frame_idx=frame_index,
                total_seen_frames=1,
                missed_frames=0,
                history_timestamps=[timestamp_sec],
                history_bboxes=[det.bbox],
                history_confs=[det.confidence],
                history_n_votes=[getattr(det, "n_votes", 1)],
                state_enter_time=timestamp_sec
            )
            self.tracks[tid] = new_track
            matched_track_ids.add(tid)

        # 3. Tăng bộ đếm mất dấu và dọn dẹp track đã hết hạn
        dead_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_track_ids:
                track.missed_frames += 1
                if (timestamp_sec - track.last_seen_time) > self.max_time_lost_sec:
                    dead_ids.append(tid)
                    
        for tid in dead_ids:
            del self.tracks[tid]

        return list(self.tracks.values())
