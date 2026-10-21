"""Bộ hợp nhất Weighted Boxes Fusion kết hợp Thẩm định Đồng thuận Bayesian (Bayesian WBF).
Tối ưu hóa phản ứng trên mọi điều kiện ánh sáng:
- Khi có >= 2 model đồng thuận: Cộng hưởng độ tin cậy (+8% đến +12%), giữ trần 90% - 99%.
- Khi chỉ 1 model thấy: Phạt mềm thích ứng dựa trên độ mạnh ngọn lửa (giữ nguyên cháy rõ, triệt tiêu đèn xe).
"""

from typing import List, Dict, Optional
import numpy as np

from src.core.entities import CandidateBox, FusedBox
from src.utils.geometry import compute_iou


class BayesianWBF:
    """Bộ gộp phát hiện đa mô hình sử dụng Weighted Boxes Fusion & Bayesian Consensus."""

    def __init__(
        self,
        iou_thresh: float = 0.45,
        single_penalty_alpha: float = 0.35,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.single_penalty_alpha = single_penalty_alpha

    def fuse(
        self,
        candidates: List[CandidateBox],
        model_weights: Dict[str, float],
        total_models_active: int = 2,
    ) -> List[FusedBox]:
        """
        Gộp danh sách CandidateBox thành danh sách FusedBox.
        """
        if not candidates:
            return []

        sorted_candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        clusters: List[List[CandidateBox]] = []

        for cand in sorted_candidates:
            matched_cluster = None
            best_iou = 0.0

            for cluster in clusters:
                if cluster[0].class_name != cand.class_name:
                    continue
                if any(m.model_name == cand.model_name for m in cluster):
                    continue

                for item in cluster:
                    iou = compute_iou(cand.bbox, item.bbox)
                    if iou >= self.iou_thresh and iou > best_iou:
                        best_iou = iou
                        matched_cluster = cluster

            if matched_cluster is not None:
                matched_cluster.append(cand)
            else:
                clusters.append([cand])

        fused_boxes: List[FusedBox] = []

        for cluster in clusters:
            cname = cluster[0].class_name
            n_votes = len(cluster)
            sources = [c.model_name for c in cluster]
            ref_ts = cluster[0].timestamp_sec
            ref_frame = cluster[0].frame_index

            # 1. Tính tọa độ trung bình có trọng số theo (confidence * model_weight)
            weights = np.array([c.confidence * model_weights.get(c.model_name, 1.0) for c in cluster], dtype=np.float32)
            total_w = float(np.sum(weights))
            if total_w <= 0.0:
                total_w = 1.0
                weights = np.ones(len(cluster), dtype=np.float32)

            boxes_arr = np.array([c.bbox for c in cluster], dtype=np.float32)
            fused_coords = np.sum(boxes_arr * weights[:, None], axis=0) / total_w
            x1, y1, x2, y2 = [int(round(v)) for v in fused_coords]

            # 2. Tính độ tin cậy gộp (Fused Confidence)
            raw_weights = np.array([model_weights.get(c.model_name, 1.0) for c in cluster], dtype=np.float32)
            cluster_confs = np.array([c.confidence for c in cluster], dtype=np.float32)
            base_conf = float(np.sum(cluster_confs * raw_weights) / np.sum(raw_weights))
            max_conf = float(np.max(cluster_confs))

            m = max(1, total_models_active)
            agreement_ratio = n_votes / float(m)

            if n_votes >= 2:
                # ĐỒNG THUẬN TỪ 2 MODEL TRỞ LÊN:
                # Không bao giờ kéo tụt độ tin cậy của model tự tin nhất;
                # thưởng thêm điểm đồng thuận: +0.08 cho 2 model, +0.12 cho 3 model.
                agree_bonus = 0.12 if n_votes >= 3 else 0.08
                fused_conf = min(0.99, max(max_conf, base_conf) + agree_bonus)
            else:
                # CHỈ 1 MODEL PHÁT HIỆN (n_votes == 1):
                # Phân biệt rõ theo mức độ chắc chắn:
                # - Ngọn lửa rõ ràng / độ tin cậy cao (>= 0.75): Giữ nguyên, phạt rất nhẹ (scale 0.08)
                # - Ngọn lửa trung bình (0.60 - 0.75): Phạt vừa phải (scale 0.35)
                # - Dấu hiệu mập mờ / phản quang / đèn xe (< 0.50): Phạt đầy đủ (scale 1.0) để triệt tiêu báo giả
                single_conf = cluster_confs[0]
                if single_conf >= 0.80:
                    scale = 0.08
                elif single_conf >= 0.60:
                    scale = 0.35
                else:
                    scale = 1.0

                max_model_w = float(np.max(raw_weights)) if len(raw_weights) > 0 else 1.0
                eff_alpha = self.single_penalty_alpha * 0.4 if max_model_w >= 1.20 else self.single_penalty_alpha
                penalty_factor = 1.0 - eff_alpha * (1.0 - agreement_ratio) * scale
                penalty_factor = max(0.1, min(1.0, penalty_factor))
                fused_conf = float(np.clip(single_conf * penalty_factor, 0.01, 0.99))

            fused_boxes.append(FusedBox(
                class_name=cname,
                confidence=round(fused_conf, 4),
                bbox=(x1, y1, x2, y2),
                n_votes=n_votes,
                total_models=total_models_active,
                sources=sources,
                timestamp_sec=ref_ts,
                frame_index=ref_frame
            ))

        return fused_boxes
