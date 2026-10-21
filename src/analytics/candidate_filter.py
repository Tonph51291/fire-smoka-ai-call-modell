"""Tầng Thẩm định Ứng viên (Candidate Validation & Optical Post-Filtering).
Chuyên trách áp dụng các quy tắc kiểm định vật lý, hình học và quang học theo từng môi trường ánh sáng:
- Sử dụng ngưỡng quyết định final_conf_fire / smoke từ dyn_profile hoặc env_cfg.
- Lọc biển báo phản quang và chao đèn hình học cứng bằng is_reflective_traffic_sign.
- Lọc diện tích tối thiểu, dáng đứng người đi đường.
- Ràng buộc đồng thuận 2 model trên camera hồng ngoại đen trắng (NIGHT_IR).
- Trả về danh sách box hợp lệ và danh sách box bị triệt tiêu kèm lý do.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.config import (
    LOC_DANG_NGUOI_DI_DUONG,
    MAX_ASPECT_RATIO_LUA,
    MIN_AREA_RATIO,
    NIGHT_FIRE_CONF_SINGLE,
    is_fire,
    is_smoke,
)
from src.core.entities import Detection, EnvironmentType
from src.utils.geometry import is_reflective_traffic_sign
from src.utils.optical_features import (
    analyze_spark_dispersion,
    crop_roi_gray,
    extract_optical_metrics,
    is_floor_sunlight_patch,
    is_sunlight_glare,
    verify_fire_color_chromaticity,
)


class CandidateFilter:
    """Bộ lọc hậu xử lý ứng viên phát hiện theo môi trường quang học."""

    def __init__(
        self,
        final_conf_fire: float = 0.25,
        final_conf_smoke: float = 0.30,
        require_two_votes: bool = False,
    ) -> None:
        self.final_conf_fire = final_conf_fire
        self.final_conf_smoke = final_conf_smoke
        self.require_two_votes = require_two_votes

    def filter(
        self,
        all_detections: List[Detection],
        frame_shape: Tuple[int, int],
        environment: str,
        env_cfg: Optional[Dict[str, Any]] = None,
        dyn_profile: Optional[Any] = None,
        frame_gray: Optional[np.ndarray] = None,
        frame_bgr: Optional[np.ndarray] = None
    ) -> Tuple[List[Detection], List[Dict[str, Any]]]:
        """
        Lọc hậu xử lý các box sau khi gộp WBF.

        Returns:
            Tuple[List[Detection], List[Dict[str, Any]]]: (valid_detections, suppressed_list)
        """
        frame_h, frame_w = frame_shape[:2]
        frame_area = max(1, frame_h * frame_w)
        cfg = env_cfg or {}

        valid_detections: List[Detection] = []
        suppressed_list: List[Dict[str, Any]] = []

        if dyn_profile is not None:
            req_conf_fire = dyn_profile.confidence_fire
            req_conf_smoke = dyn_profile.confidence_smoke
        else:
            req_conf_fire = cfg.get("final_conf_fire", self.final_conf_fire)
            req_conf_smoke = cfg.get("final_conf_smoke", self.final_conf_smoke)

        for det in all_detections:
            cname = det.class_name.lower()
            w = max(1, det.bbox[2] - det.bbox[0])
            h = max(1, det.bbox[3] - det.bbox[1])
            box_area = w * h
            aspect_ratio = float(h) / float(w)

            # Trích xuất quang học đơn luồng
            if frame_gray is not None:
                opt = extract_optical_metrics(frame_gray, det.bbox)
                det.roughness = opt.roughness
                det.core_sat_ratio = opt.core_sat_ratio

            # 1. Ngưỡng quyết định theo môi trường
            req_conf = req_conf_fire if is_fire(cname) else req_conf_smoke
            if det.confidence < req_conf:
                suppressed_list.append({
                    "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                    "reason": f"DƯỚI_NGƯỠNG_{environment} ({det.confidence:.2f} < {req_conf:.2f})"
                })
                continue

            # 2. Lọc biển báo giao thông phản quang / chao đèn hình học cứng
            # Nếu có khói bốc lên sát cạnh hoặc bao trùm: Chắc chắn là đám cháy, bỏ qua kiểm tra biển báo
            has_smoke_nearby = False
            for other in all_detections:
                if is_smoke(other.class_name) and other.confidence >= req_conf_smoke:
                    scx = (other.bbox[0] + other.bbox[2]) / 2.0
                    scy = (other.bbox[1] + other.bbox[3]) / 2.0
                    fcx = (det.bbox[0] + det.bbox[2]) / 2.0
                    fcy = (det.bbox[1] + det.bbox[3]) / 2.0
                    if abs(scx - fcx) < 2.5 * w and (scy <= fcy + 60):
                        has_smoke_nearby = True
                        break

            if frame_gray is not None and is_fire(cname) and not has_smoke_nearby:
                if is_reflective_traffic_sign(frame_gray, det.bbox):
                    suppressed_list.append({
                        "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                        "reason": "BIỂN_BÁO_HOẶC_CHAO_ĐÈN_HÌNH_HỌC_CỨNG"
                    })
                    continue

                # 2b. Lọc tia lửa gia công cơ khí (máy mài, cắt đĩa, hàn xì):
                # Vùng sáng gồm hàng chục hạt/tia sáng li ti phân tán rỗng và KHÔNG có cột khói bốc lên
                crop_g = crop_roi_gray(frame_gray, det.bbox)
                if crop_g is not None:
                    is_sparks, spark_count, fill_ratio = analyze_spark_dispersion(crop_g)
                    if is_sparks and not (det.n_votes >= 3 and det.confidence >= 0.85):
                        suppressed_list.append({
                            "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                            "reason": f"TIA_LỬA_GIA_CÔNG_CƠ_KHÍ (Mài/Cắt/Hàn: {spark_count} hạt, Fill={fill_ratio:.1%})"
                        })
                        continue

                # 2c. Lọc máng đèn trần / ô kính lấy sáng nóc nhà xưởng (Ceiling Luminaire / Skylight):
                # Nằm sát mép trần (y1 <= 2% frame height, y2 <= 15% frame height), không có khói bốc lên
                y1_norm = det.bbox[1] / max(1, frame_h)
                y2_norm = det.bbox[3] / max(1, frame_h)
                if y1_norm <= 0.02 and y2_norm <= 0.15:
                    suppressed_list.append({
                        "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                        "reason": "MÁNG_ĐÈN_TRẦN_Ô_KÍNH_NÓC_XƯỞNG"
                    })
                    continue

            # 2b. Lọc vệt nắng phản quang trên kính/mái tôn & kiểm định quang phổ YCbCr
            if frame_bgr is not None and is_fire(cname) and det.n_votes == 1:
                x1, y1, x2, y2 = det.bbox
                cx1, cy1 = max(0, min(frame_w - 1, x1)), max(0, min(frame_h - 1, y1))
                cx2, cy2 = max(0, min(frame_w, x2)), max(0, min(frame_h, y2))
                if cx2 > cx1 and cy2 > cy1:
                    crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
                    if environment not in (EnvironmentType.NIGHT_IR, EnvironmentType.NIGHT_COLOR):
                        if is_sunlight_glare(crop_bgr):
                            suppressed_list.append({
                                "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                                "reason": "VỆT_NẮNG_PHẢN_QUANG_KÍNH_TÔN"
                            })
                            continue

                        if not has_smoke_nearby and is_floor_sunlight_patch(crop_bgr, det.bbox, frame_h, getattr(det, "core_sat_ratio", 0.0)):
                            suppressed_list.append({
                                "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                                "reason": "VỆT_NẮNG_CHIẾU_SÀN_NHÀ"
                            })
                            continue

                        is_chroma, chroma_ratio = verify_fire_color_chromaticity(crop_bgr)
                        if not is_chroma and chroma_ratio < 0.03:
                            suppressed_list.append({
                                "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                                "reason": f"KHÔNG_ĐẠT_QUANG_PHỔ_LỬA_YCBCR ({chroma_ratio:.1%})"
                            })
                            continue

            # 2c. Lọc vật thể tối không có lõi sáng bão hòa (tường, máy móc rỉ sét màu đỏ cam)
            if is_fire(cname) and det.n_votes == 1 and det.confidence < 0.35 and getattr(det, "core_sat_ratio", 0.0) == 0.0:
                suppressed_list.append({
                    "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                    "reason": "VẬT_THỂ_TỐI_KHÔNG_CÓ_LÕI_SÁNG (Tường/máy móc gỉ sét)"
                })
                continue

            # 3. Yêu cầu 2 model đồng thuận nếu cấu hình profile quy định
            if self.require_two_votes and det.n_votes < 2:
                suppressed_list.append({
                    "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                    "reason": "YÊU_CẦU_2_MODEL_ĐỒNG_THUẬN"
                })
                continue

            # 4. Lọc diện tích tối thiểu cho box 1 model
            if det.n_votes == 1 and (box_area / frame_area) < MIN_AREA_RATIO:
                suppressed_list.append({
                    "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                    "reason": f"DIỆN_TÍCH_QUÁ_NHỎ ({(box_area/frame_area):.4%})"
                })
                continue

            # 5. Lọc hình dạng người đi đường (H/W >= 3.5) cho box 1 model conf yếu (< 0.35)
            if LOC_DANG_NGUOI_DI_DUONG and is_fire(cname):
                if det.n_votes == 1 and aspect_ratio >= MAX_ASPECT_RATIO_LUA and det.confidence < 0.35:
                    suppressed_list.append({
                        "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                        "reason": f"DÁNG_ĐỨNG_NGHI_VẤN_NGƯỜI (H/W={aspect_ratio:.2f})"
                    })
                    continue

            # 6. Lọc đơn lẻ trong môi trường IR (NIGHT_IR):
            # Nếu có khói bốc lên lân cận: chỉ cần conf >= 0.18 để bắt lửa sớm
            # Nếu là chao đèn trơn/biển báo (roughness < 1.08): yêu cầu ít nhất 2 model hoặc conf đơn >= 0.45
            # Nếu là đốm lửa bập bùng tự nhiên (roughness >= 1.08): cho phép conf >= 0.18 để bắt nhen nhóm sớm
            if environment == EnvironmentType.NIGHT_IR and is_fire(cname):
                is_smooth = (getattr(det, "roughness", 1.0) < 1.08)
                req_ir_single = 0.45 if is_smooth and not has_smoke_nearby else 0.18
                if det.n_votes < 2 and det.confidence < req_ir_single:
                    suppressed_list.append({
                        "bbox": list(det.bbox), "cls": cname, "conf": det.confidence,
                        "reason": f"IR_YÊU_CẦU_ĐỒNG_THUẬN ({det.n_votes}/2 votes, conf={det.confidence:.2f} < {req_ir_single:.2f})"
                    })
                    continue

            valid_detections.append(det)

        return valid_detections, suppressed_list
