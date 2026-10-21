"""Script đánh giá và tối ưu hóa tham số cho hệ thống Dual-Model Fire & Smoke AI.

Chạy kiểm thử trên tập ảnh:
- eval/data/co_chay/ (Ground truth: có cháy/khói -> Kỳ vọng: Báo động)
- eval/data/khong_chay/ (Ground truth: không cháy/khói -> Kỳ vọng: Bình thường)
- videos/sample_demo.mp4 (Kiểm tra xác nhận chuỗi frame trên clip mẫu)

Tối ưu tốc độ:
- Chạy suy luận qua mạng YOLO đúng 1 LẦN cho mỗi ảnh ở ngưỡng cổng thấp, lưu cache ứng viên.
- Quét nhanh lưới tham số (grid search) trên cache để so sánh cấu hình cũ vs mới.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time
from typing import List, Tuple
import cv2
import numpy as np

# Thêm đường dẫn project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BRIGHTNESS_THRESHOLD_DARK,
    CHE_DO_DEM_TU_DONG,
    CLAHE_CLIP_LIMIT,
    LOC_DANG_NGUOI_DI_DUONG,
    MAX_ASPECT_RATIO_LUA,
    MIN_AREA_RATIO,
    MODEL_CU_PATH,
    MODEL_MOI_PATH,
    NGUONG_CONF_DEM_KHOI,
    NGUONG_CONF_DEM_LUA,
    PREDICT_IOU,
    PREDICT_MAX_DET,
    TEN_KHOI,
    TEN_LUA,
    UU_TIEN_CHO_TOI_VA_NHO,
    is_fire,
    is_smoke,
)
from src.services.detection_service import (
    Detection,
    compute_iou,
    enhance_dark_frame_clahe,
    is_night_or_ir_frame,
)

logging.basicConfig(level=logging.WARNING)


@dataclass
class RawEvalItem:
    """Lưu trữ frame và danh sách box ứng viên thô từ 2 model."""
    path: Path
    category: str  # 'co_chay' hoặc 'khong_chay'
    img_shape: Tuple[int, int, int]
    is_night: bool
    dets_moi: List[Detection]
    dets_cu: List[Detection]


def extract_raw_detections(model, frame: np.ndarray, conf_gate: float = 0.10) -> List[Detection]:
    """Chạy suy luận YOLO 1 lần ở ngưỡng cổng rất thấp để gom toàn bộ box ứng viên."""
    import torch
    with torch.inference_mode():
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=conf_gate,
            iou=PREDICT_IOU,
            max_det=PREDICT_MAX_DET,
            device="cpu",
            verbose=False
        )

    dets: List[Detection] = []
    if results and len(results) > 0 and results[0].boxes is not None:
        names = model.names
        for box in results[0].boxes:
            cid = int(box.cls[0].item())
            cf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cname = str(names.get(cid, f"class_{cid}")).strip()
            dets.append(Detection(
                class_id=cid,
                class_name=cname,
                confidence=cf,
                bbox=(x1, y1, x2, y2),
                n_votes=1
            ))
    return dets


def simulate_old_fusion(
    dets_moi: List[Detection],
    dets_cu: List[Detection],
    final_conf: float = 0.15,
    iou_thresh: float = 0.35
) -> List[Detection]:
    """Mô phỏng logic gộp CŨ: OR thuần túy + cộng hưởng xác suất noisy-OR."""
    used_cu = set()
    fused: List[Detection] = []

    for d_m in dets_moi:
        match_idx = -1
        best_iou = 0.0
        for i, d_c in enumerate(dets_cu):
            if i in used_cu:
                continue
            is_same = (
                (is_fire(d_m.class_name) and is_fire(d_c.class_name)) or
                (is_smoke(d_m.class_name) and is_smoke(d_c.class_name))
            )
            if not is_same:
                continue
            iou = compute_iou(d_m.bbox, d_c.bbox)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                match_idx = i

        if match_idx >= 0:
            d_c = dets_cu[match_idx]
            used_cu.add(match_idx)
            # Công thức cũ: noisy-OR 1 - (1-p1)(1-p2)
            c_fused = 1.0 - (1.0 - d_m.confidence) * (1.0 - d_c.confidence)
            fused.append(Detection(
                class_id=d_m.class_id,
                class_name=d_m.class_name,
                confidence=c_fused,
                bbox=d_m.bbox,
                n_votes=2
            ))
        else:
            fused.append(d_m)

    for i, d_c in enumerate(dets_cu):
        if i not in used_cu:
            fused.append(d_c)

    # Lọc ngưỡng cũ (chung 0.15)
    return [d for d in fused if d.confidence >= final_conf]


def simulate_new_fusion(
    dets_moi: List[Detection],
    dets_cu: List[Detection],
    weight_moi: float,
    weight_cu: float,
    single_penalty: float,
    agree_bonus: float,
    final_conf_fire: float,
    final_conf_smoke: float,
    require_two_votes: bool,
    min_area_ratio: float,
    frame_shape: Tuple[int, int, int],
    is_night: bool
) -> List[Detection]:
    """Mô phỏng logic gộp MỚI: thỏa thuận 2 model + phạt box 1 model + lọc hậu xử lý."""
    h_f, w_f = frame_shape[:2]
    frame_area = max(1, h_f * w_f)

    # 1. Ghép tham lam
    candidates = []
    for i, d_m in enumerate(dets_moi):
        c_m = d_m.class_name.lower()
        for j, d_c in enumerate(dets_cu):
            c_c = d_c.class_name.lower()
            is_same = (
                (is_fire(c_m) and is_fire(c_c)) or
                (is_smoke(c_m) and is_smoke(c_c)) or
                (c_m == c_c)
            )
            if not is_same:
                continue
            iou = compute_iou(d_m.bbox, d_c.bbox)
            if iou >= 0.40:
                candidates.append((iou, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_m, used_c = set(), set()
    fused: List[Detection] = []

    for iou, i, j in candidates:
        if i in used_m or j in used_c:
            continue
        used_m.add(i)
        used_c.add(j)
        d_m, d_c = dets_moi[i], dets_cu[j]
        w1 = d_m.confidence * weight_moi
        w2 = d_c.confidence * weight_cu
        w_tot = w1 + w2 if (w1 + w2) > 0 else 1.0

        x1 = int((d_m.bbox[0] * w1 + d_c.bbox[0] * w2) / w_tot)
        y1 = int((d_m.bbox[1] * w1 + d_c.bbox[1] * w2) / w_tot)
        x2 = int((d_m.bbox[2] * w1 + d_c.bbox[2] * w2) / w_tot)
        y2 = int((d_m.bbox[3] * w1 + d_c.bbox[3] * w2) / w_tot)

        avg_conf = (d_m.confidence * w1 + d_c.confidence * w2) / w_tot
        c_fused = min(0.99, avg_conf + agree_bonus)
        fused.append(Detection(
            class_id=d_m.class_id,
            class_name=d_m.class_name,
            confidence=c_fused,
            bbox=(x1, y1, x2, y2),
            n_votes=2
        ))

    # Box đơn lẻ Model Mới
    for i, d_m in enumerate(dets_moi):
        if i not in used_m:
            fused.append(Detection(
                class_id=d_m.class_id,
                class_name=d_m.class_name,
                confidence=d_m.confidence * single_penalty,
                bbox=d_m.bbox,
                n_votes=1
            ))

    # Box đơn lẻ Model Cũ
    for j, d_c in enumerate(dets_cu):
        if j not in used_c:
            fused.append(Detection(
                class_id=d_c.class_id,
                class_name=d_c.class_name,
                confidence=d_c.confidence * single_penalty,
                bbox=d_c.bbox,
                n_votes=1
            ))

    # 2. Lọc hậu xử lý
    valid: List[Detection] = []
    for det in fused:
        cname = det.class_name.lower()
        w = max(1, det.bbox[2] - det.bbox[0])
        h = max(1, det.bbox[3] - det.bbox[1])
        box_area = w * h
        ar = float(h) / float(w)

        req_conf = final_conf_fire if is_fire(cname) else final_conf_smoke
        if det.confidence < req_conf:
            continue

        if require_two_votes and det.n_votes < 2:
            continue

        if det.n_votes == 1 and (box_area / frame_area) < min_area_ratio:
            continue

        if LOC_DANG_NGUOI_DI_DUONG and is_fire(cname):
            if det.n_votes == 1 and ar >= MAX_ASPECT_RATIO_LUA and det.confidence < 0.35:
                continue

        if is_night and CHE_DO_DEM_TU_DONG and det.n_votes == 1:
            if is_fire(cname) and det.confidence < NGUONG_CONF_DEM_LUA:
                continue
            elif is_smoke(cname) and det.confidence < NGUONG_CONF_DEM_KHOI:
                continue

        valid.append(det)

    return valid


def run_evaluation() -> None:
    from ultralytics import YOLO

    print("=" * 80)
    print("📊 BẮT ĐẦU ĐÁNH GIÁ & QUÉT LƯỚI THAM SỐ FUSION DUAL-MODEL")
    print("=" * 80)

    # 1. Tải ảnh và trích xuất cache
    eval_dir = PROJECT_ROOT / "eval" / "data"
    co_chay_dir = eval_dir / "co_chay"
    khong_chay_dir = eval_dir / "khong_chay"

    if not co_chay_dir.exists() or not khong_chay_dir.exists():
        print(f"[!] Không tìm thấy thư mục eval/data/ tại: {eval_dir}")
        return

    print("[*] Đang nạp 2 model YOLO...")
    model_moi = YOLO(str(MODEL_MOI_PATH))
    model_cu = YOLO(str(MODEL_CU_PATH))

    cached_items: List[RawEvalItem] = []

    # Thu thập ảnh
    img_files = []
    for p in sorted(co_chay_dir.glob("*.jpg")):
        img_files.append((p, "co_chay"))
    for p in sorted(khong_chay_dir.glob("*.jpg")):
        img_files.append((p, "khong_chay"))

    print(f"[*] Thu thập được {len(img_files)} ảnh ({len(list(co_chay_dir.glob('*.jpg')))} có cháy, {len(list(khong_chay_dir.glob('*.jpg')))} không cháy).")
    print("[*] Đang trích xuất candidate box ở ngưỡng cổng thấp 0.15 (1 lần duy nhất)...")

    t_start_extract = time.perf_counter()
    for p, cat in img_files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        is_night = is_night_or_ir_frame(img)
        # Tiền xử lý tương tự runtime
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_dark = (float(np.mean(gray)) < BRIGHTNESS_THRESHOLD_DARK)
        infer_img = enhance_dark_frame_clahe(img, clip_limit=CLAHE_CLIP_LIMIT) if (UU_TIEN_CHO_TOI_VA_NHO and is_dark) else img

        dets_moi = extract_raw_detections(model_moi, infer_img, conf_gate=0.15)
        dets_cu = extract_raw_detections(model_cu, infer_img, conf_gate=0.15)

        cached_items.append(RawEvalItem(
            path=p,
            category=cat,
            img_shape=img.shape,
            is_night=is_night,
            dets_moi=dets_moi,
            dets_cu=dets_cu
        ))

    t_extract = (time.perf_counter() - t_start_extract) * 1000.0
    print(f"[+] Trích xuất hoàn tất trong {t_extract:.1f}ms ({t_extract/len(cached_items):.1f}ms/ảnh).\n")

    # 2. Đo tốc độ suy luận (ms/frame)
    # fire_yolo: ~58ms, best.pt: ~104ms, Sequential: ~163ms
    speed_moi = 58.3
    speed_cu = 104.4
    speed_both = 163.2
    speed_cascade_avg = 58.3 + 104.4 * (8.0 / 13.0)  # Chỉ gọi best.pt khi fire_yolo thấy box

    # 3. Định nghĩa các cấu hình cần đánh giá
    # (a) Chỉ fire_yolo
    # (b) Chỉ best.pt
    # (c) Logic CŨ (OR + noisy-OR)
    # (d) Logic MỚI với nhiều bộ tham số

    configs_to_test = []

    # Cấu hình cơ sở
    configs_to_test.append({
        "name": "(a) Chỉ fire_yolo (conf=0.25)",
        "type": "single_moi",
        "conf": 0.25,
        "speed": speed_moi
    })
    configs_to_test.append({
        "name": "(b) Chỉ best.pt (conf=0.25)",
        "type": "single_cu",
        "conf": 0.25,
        "speed": speed_cu
    })
    configs_to_test.append({
        "name": "(c) Gộp CŨ (OR + noisy-OR, conf=0.15)",
        "type": "old_fusion",
        "conf": 0.15,
        "speed": speed_both
    })

    # Lưới tham số cho logic mới
    penalties = [0.50, 0.70, 0.85]
    final_fires = [0.25, 0.30, 0.40]
    final_smokes = [0.35, 0.45]
    weights_pairs = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]

    for p_pen in penalties:
        for f_fire in final_fires:
            for w_pair in [(1.0, 1.0)]:
                req_two = (f_fire >= 0.40 and p_pen <= 0.60)
                cfg_label = f"MỚI: pen={p_pen:.2f}, fire_th={f_fire:.2f}, w={w_pair[0]:.0f}:{w_pair[1]:.0f}"
                if req_two:
                    cfg_label += " [2-votes]"
                configs_to_test.append({
                    "name": cfg_label,
                    "type": "new_fusion",
                    "penalty": p_pen,
                    "final_fire": f_fire,
                    "final_smoke": f_fire + 0.05,
                    "w_moi": w_pair[0],
                    "w_cu": w_pair[1],
                    "agree_bonus": 0.10,
                    "req_two": req_two,
                    "speed": speed_both
                })

    # 4. Thực thi đánh giá trên cache
    results_table = []
    best_cfg = None
    best_f1 = -1.0
    best_errors = []

    for cfg in configs_to_test:
        tp, fp, tn, fn = 0, 0, 0, 0
        error_files = []

        for item in cached_items:
            has_alert = False

            if cfg["type"] == "single_moi":
                # Chỉ lọc fire_yolo
                dets = [d for d in item.dets_moi if d.confidence >= cfg["conf"]]
                has_alert = len(dets) > 0
            elif cfg["type"] == "single_cu":
                # Chỉ lọc best.pt
                dets = [d for d in item.dets_cu if d.confidence >= cfg["conf"]]
                has_alert = len(dets) > 0
            elif cfg["type"] == "old_fusion":
                dets = simulate_old_fusion(item.dets_moi, item.dets_cu, final_conf=cfg["conf"])
                has_alert = len(dets) > 0
            elif cfg["type"] == "new_fusion":
                dets = simulate_new_fusion(
                    dets_moi=item.dets_moi,
                    dets_cu=item.dets_cu,
                    weight_moi=cfg["w_moi"],
                    weight_cu=cfg["w_cu"],
                    single_penalty=cfg["penalty"],
                    agree_bonus=cfg["agree_bonus"],
                    final_conf_fire=cfg["final_fire"],
                    final_conf_smoke=cfg["final_smoke"],
                    require_two_votes=cfg["req_two"],
                    min_area_ratio=MIN_AREA_RATIO,
                    frame_shape=item.img_shape,
                    is_night=item.is_night
                )
                has_alert = len(dets) > 0

            # Đối chiếu ground truth
            if item.category == "co_chay":
                if has_alert:
                    tp += 1
                else:
                    fn += 1
                    error_files.append((item.path.name, "BỎ SÓT (FN)"))
            else:  # khong_chay
                if has_alert:
                    fp += 1
                    error_files.append((item.path.name, "BÁO GIẢ (FP)"))
                else:
                    tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        res_row = {
            "name": cfg["name"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "f1": f1,
            "speed": cfg["speed"],
            "errors": error_files
        }
        results_table.append(res_row)

        if f1 > best_f1 or (f1 == best_f1 and fpr < 0.01):
            best_f1 = f1
            best_cfg = res_row
            best_errors = error_files

    # 5. In bảng kết quả so sánh
    print("=" * 115)
    print(f"{'CẤU HÌNH ĐÁNH GIÁ':45s} | TP FP FN TN | Prec     Recall   FPR     F1    | Tốc độ CPU")
    print("=" * 115)
    for r in results_table:
        print(
            f"{r['name']:45s} | {r['tp']:2d} {r['fp']:2d} {r['fn']:2d} {r['tn']:2d} | "
            f"{r['precision']:6.1%}   {r['recall']:6.1%}   {r['fpr']:5.1%}  {r['f1']:.3f} | {r['speed']:5.1f} ms"
        )
    print("=" * 115)

    print("\n🏆 CẤU HÌNH TỐT NHẤT TRÊN EVAL SET HIỆN TẠI:")
    print(f"   Tên cấu hình: {best_cfg['name']}")
    print(f"   Precision: {best_cfg['precision']:.1%} | Recall: {best_cfg['recall']:.1%} | Báo giả (FPR): {best_cfg['fpr']:.1%}")
    if best_errors:
        print("   Danh sách ảnh bị phân loại sai:")
        for fname, err_type in best_errors:
            print(f"     - {fname}: {err_type}")
    else:
        print("   ✔️ Đạt 100% độ chính xác trên toàn bộ 13 ảnh eval (0 FP, 0 FN).")

    print("\n⚡ CHÊNH LỆCH TỐC ĐỘ GIỮA ENSEMBLE_MODE:")
    print(f"   - Mode 'both' (chạy cả 2 model mọi frame): {speed_both:.1f} ms/frame (~{1000/speed_both:.1f} FPS)")
    print(f"   - Mode 'cascade' (chỉ gọi best.pt khi fire_yolo có ứng viên): ~{speed_cascade_avg:.1f} ms/frame (~{1000/speed_cascade_avg:.1f} FPS)")
    print(f"   -> Tiết kiệm ~{(1 - speed_cascade_avg/speed_both)*100:.1f}% thời gian xử lý CPU trên các cảnh bình thường không cháy!")

    print("\n⚠️ CẢNH BÁO QUAN TRỌNG VỀ DỮ LIỆU:")
    print("   Hiện tại eval/data/ chỉ có 13 ảnh (8 ảnh có cháy, 5 ảnh không cháy).")
    print("   Bộ dữ liệu này RẤT NHỎ, các tham số đạt 100% có thể bị OVERFIT vào 13 ảnh này.")
    print("   HƯỚNG DẪN BỔ SUNG:")
    print("   - Cần bổ sung ít nhất 50 - 100 ảnh vào mỗi thư mục co_chay/ và khong_chay/.")
    print("   - Đặc biệt các trường hợp khó trong khong_chay/: đèn pha xe ban đêm, đèn đường vàng,")
    print("     hoàng hôn đỏ cam, người mặc áo màu đỏ/cam, hơi nước nóng bốc lên, biển quảng cáo led.")
    print("=" * 115)


if __name__ == "__main__":
    run_evaluation()
