"""Máy trạng thái và Bộ tích lũy bằng chứng thời gian thực (Evidence Accumulator).
Cơ chế cốt lõi:
- 4 Trạng thái: NONE -> SUSPECT -> CONFIRMED -> ALARM.
- Điểm bằng chứng E(t) tích lũy trong khoảng [0, 100] với phân rã liên tục theo thời gian thực:
  E(t) = E(t - dt) * exp(-dt / tau) + delta_E.
- Tách biệt tau_decay cho Lửa (2.5s) và Khói (8.0s).
- Không chặn cứng: Khói độ tin cậy thấp và vật thể tĩnh bị phạt vẫn leo thang lên ALARM
  nếu tồn tại bền vững qua chuỗi thời gian.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import logging

from src.services.spatio_temporal_tracker import TrackedObject
from src.config import (
    TAU_DECAY_FIRE,
    TAU_DECAY_SMOKE,
    E_SUSPECT,
    E_CONFIRMED,
    E_ALARM,
    E_DROP_ALARM,
    E_DROP_CONFIRMED,
    E_DROP_SUSPECT,
    SOFT_PENALTY_STATIC,
    SOFT_PENALTY_VEHICLE,
    EMERGENCY_GROWTH_RATIO,
    EMERGENCY_ROUGHNESS_THRESH,
)

logger = logging.getLogger(__name__)


@dataclass
class AccumulatorConfig:
    """Cấu hình tham số máy trạng thái và bộ tích lũy."""
    tau_fire: float = TAU_DECAY_FIRE              # [GIẢ ĐỊNH CẦN ĐO] - Thời gian phân rã của Lửa (giây)
    tau_smoke: float = TAU_DECAY_SMOKE            # [GIẢ ĐỊNH CẦN ĐO] - Thời gian phân rã của Khói (giây)
    
    # Ngưỡng leo thang trạng thái
    e_suspect: float = E_SUSPECT                  # [GIẢ ĐỊNH CẦN ĐO]
    e_confirmed: float = E_CONFIRMED              # [GIẢ ĐỊNH CẦN ĐO]
    e_alarm: float = E_ALARM                      # [GIẢ ĐỊNH CẦN ĐO]
    
    # Ngưỡng tụt trạng thái (Hysteresis)
    e_drop_alarm: float = E_DROP_ALARM            # [GIẢ ĐỊNH CẦN ĐO]
    e_drop_confirmed: float = E_DROP_CONFIRMED    # [GIẢ ĐỊNH CẦN ĐO]
    e_drop_suspect: float = E_DROP_SUSPECT        # [GIẢ ĐỊNH CẦN ĐO]
    
    # Thời gian tối thiểu ở từng trạng thái (chống nhiễu nháy)
    min_time_fire_suspect: float = 0.4        # Giây
    min_time_fire_confirm: float = 0.5        # Giây
    min_time_smoke_suspect: float = 2.0       # Giây
    min_time_smoke_confirm: float = 3.0       # Giây


class EvidenceAccumulator:
    """Điều phối cập nhật điểm bằng chứng và chuyển đổi trạng thái Track."""

    def __init__(self, config: Optional[AccumulatorConfig] = None) -> None:
        self.cfg = config or AccumulatorConfig()

    def update_track_evidence(
        self,
        track: TrackedObject,
        timestamp_sec: float,
        kinematic_feats: Dict[str, float],
        roughness: float,
        is_flare: bool = False
    ) -> Tuple[float, str, str]:
        """
        Cập nhật điểm bằng chứng E(t) và quyết định trạng thái mới cho Track:
        
        Args:
            track: Đối tượng TrackedObject cần cập nhật.
            timestamp_sec: Dấu thời gian frame hiện tại (giây).
            kinematic_feats: Đặc trưng động học từ MotionFeatureExtractor.
            roughness: Độ dao động viền quầng sáng.
            is_flare: Cờ bùng lửa khẩn cấp từ ThreatAnalyzer.

        Returns:
            (new_evidence_score, new_state, transition_reason)
        """
        is_fire_obj = (track.class_name.lower() == "fire")
        tau = self.cfg.tau_fire if is_fire_obj else self.cfg.tau_smoke

        # 1. Tính dt thời gian thực so với lần cập nhật gần nhất
        last_ts = track.history_timestamps[-2] if len(track.history_timestamps) >= 2 else (timestamp_sec - 0.04)
        dt = max(0.001, min(2.0, float(timestamp_sec - last_ts)))

        # 2. Phân rã theo hàm mũ liên tục
        current_e = track.evidence_score * math.exp(-dt / tau)

        # 3. Tính lượng bằng chứng nạp thêm (delta_E)
        n_votes = track.history_n_votes[-1] if track.history_n_votes else 1
        vote_multiplier = 1.4 if n_votes >= 2 else 1.0

        # Hệ số động học quầng viền:
        # Lửa thật có roughness >= 1.20 -> thưởng điểm; đèn/biển báo viền trơn hình học có roughness < 1.08 -> giảm điểm
        if is_fire_obj:
            if roughness >= 1.20:
                liveness_mult = 1.3
            elif roughness < 1.08:
                liveness_mult = 0.5
            else:
                liveness_mult = 1.0
        else:
            # Khói: thưởng điểm nếu có xu hướng bốc lên cao
            upward_ratio = kinematic_feats.get("upward_ratio", 0.0)
            liveness_mult = 1.3 if upward_ratio > 0.10 else 1.0

        # Hệ số phạt nếu nghi vấn là nguồn sáng tĩnh (biển báo / bóng đèn)
        v_drift = kinematic_feats.get("v_drift", 0.0)
        track_age = kinematic_feats.get("track_age_sec", 0.0)
        
        # Nếu đứng yên qua > 2.0s và viền trơn hình học (roughness < 1.08) -> gán cờ static suspect
        if track_age >= 2.0 and v_drift < 0.05 and roughness < 1.08:
            track.is_static_suspect = True
        elif v_drift >= 0.10 or roughness >= 1.15 or kinematic_feats.get("area_growth_ratio", 1.0) >= EMERGENCY_GROWTH_RATIO:
            track.is_static_suspect = False

        static_penalty = SOFT_PENALTY_STATIC if track.is_static_suspect else 1.0

        # Hệ số phạt nếu là phương tiện giao thông lướt qua (drift nhanh và chuyển động thẳng)
        linear_score = kinematic_feats.get("linear_motion_score", 0.0)
        if v_drift > 1.2 and linear_score >= 0.75:
            track.is_vehicle_suspect = True
            vehicle_penalty = SOFT_PENALTY_VEHICLE
        else:
            track.is_vehicle_suspect = False
            vehicle_penalty = 1.0

        # Hệ số thưởng bền vững theo thời gian (Persistence Factor):
        # Vật thể tồn tại thực sự càng lâu, xác suất là đám cháy thật càng cao
        persistence_mult = 1.0 + min(2.5, track_age / 3.0)

        # Tích lũy delta_E
        base_rate = 52.0 if is_fire_obj else 32.0
        conf_term = (0.25 + 0.75 * track.confidence) if is_fire_obj else track.confidence
        delta_e = (
            conf_term
            * vote_multiplier
            * liveness_mult
            * static_penalty
            * vehicle_penalty
            * persistence_mult
            * base_rate
            * dt
        )
        
        # Trường hợp bùng lửa khẩn cấp: nạp xung năng lượng cực lớn
        if is_flare:
            delta_e += 50.0

        new_e = float(min(100.0, max(0.0, current_e + delta_e)))
        if track.is_static_suspect:
            # Khóa trần điểm bằng chứng tối đa 35.0 điểm cho nguồn sáng tĩnh, không bao giờ leo lên ALARM (85)
            new_e = min(new_e, 35.0)
        track.evidence_score = round(new_e, 2)

        # 4. Máy trạng thái chuyển đổi
        old_state = track.state
        new_state = old_state
        reason = "DUY_TRÌ"
        time_in_state = float(timestamp_sec - track.state_enter_time)

        min_suspect_time = self.cfg.min_time_fire_suspect if is_fire_obj else self.cfg.min_time_smoke_suspect
        min_confirm_time = self.cfg.min_time_fire_confirm if is_fire_obj else self.cfg.min_time_smoke_confirm

        if old_state == "NONE":
            if new_e >= self.cfg.e_suspect:
                new_state = "SUSPECT"
                reason = f"VƯỢT_NGƯỠNG_SUSPECT ({new_e:.1f} >= {self.cfg.e_suspect})"

        elif old_state == "SUSPECT":
            if new_e >= self.cfg.e_confirmed and time_in_state >= min_suspect_time:
                new_state = "CONFIRMED"
                reason = f"XÁC_NHẬN_NGUY_CƠ ({new_e:.1f} >= {self.cfg.e_confirmed})"
            elif new_e < self.cfg.e_drop_suspect:
                new_state = "NONE"
                reason = f"PHÂN_RÃ_VỀ_NONE ({new_e:.1f} < {self.cfg.e_drop_suspect})"

        elif old_state == "CONFIRMED":
            if (new_e >= self.cfg.e_alarm and time_in_state >= min_confirm_time) or is_flare:
                new_state = "ALARM"
                reason = f"KÍCH_HOẠT_BÁO_ĐỘNG ({new_e:.1f} >= {self.cfg.e_alarm})" if not is_flare else "BÙNG_LỬA_KHẨN_CẤP"
            elif new_e < self.cfg.e_drop_confirmed:
                new_state = "SUSPECT"
                reason = f"HẠ_CẤP_VỀ_SUSPECT ({new_e:.1f} < {self.cfg.e_drop_confirmed})"

        elif old_state == "ALARM":
            if new_e < self.cfg.e_drop_alarm:
                new_state = "CONFIRMED"
                reason = f"HẠ_CẤP_VỀ_CONFIRMED ({new_e:.1f} < {self.cfg.e_drop_alarm})"

        if new_state != old_state:
            track.state = new_state
            track.state_enter_time = timestamp_sec

        return track.evidence_score, track.state, reason
