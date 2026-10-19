"""Script kiểm thử phát hiện cháy/khói độc lập trên ảnh tĩnh cho từng model.

Hỗ trợ kiểm tra ảnh camera đêm (hồng ngoại, đen trắng):
- Chạy Model Cũ (models/best.pt)
- Chạy Model Mới (models/fire_yolo.pt) ở conf=0.15
- Nếu không thấy, tự động thử: conf=0.05, Grayscale 3 kênh, Tăng Gamma
- Lưu kết quả vào thư mục runs_test/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import (
    MODEL_CU_CONF,
    MODEL_CU_PATH,
    MODEL_MOI_CONF,
    MODEL_MOI_PATH,
    TEN_KHOI,
    TEN_LUA,
)


def adjust_gamma(image: np.ndarray, gamma: float = 1.8) -> np.ndarray:
    """Tăng độ tương phản / độ sáng vùng tối bằng biến đổi Gamma."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def draw_boxes(img: np.ndarray, boxes_data: List[dict], model_name: str) -> np.ndarray:
    """Vẽ bounding box lên ảnh kết quả."""
    annotated = img.copy()
    for item in boxes_data:
        x1, y1, x2, y2 = item["bbox"]
        cname = item["class_name"].lower()
        conf = item["confidence"]

        if cname in TEN_LUA:
            color = (0, 0, 255)       # Đỏ BGR
            prefix = "LUA"
        elif cname in TEN_KHOI:
            color = (0, 140, 255)     # Cam BGR
            prefix = "KHOI"
        else:
            color = (255, 255, 0)     # Vàng BGR
            prefix = item["class_name"].upper()

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix} [{model_name}]: {conf:.1%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    return annotated


def run_single_inference(
    model: YOLO,
    img: np.ndarray,
    conf_thresh: float,
    imgsz: int = 640
) -> List[dict]:
    """Chạy inference YOLO và trả về danh sách detection."""
    with torch.inference_mode():
        results = model.predict(source=img, imgsz=imgsz, conf=conf_thresh, verbose=False)

    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            cid = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cname = model.names.get(cid, str(cid))
            detections.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })
    return detections


def test_image(image_path_str: str) -> None:
    """Hàm kiểm thử chính trên file ảnh."""
    img_path = Path(image_path_str)
    if not img_path.exists():
        print(f"[!] Lỗi: Không tìm thấy file ảnh tại: {img_path.resolve()}")
        sys.exit(1)

    print("=" * 70)
    print(f"🔥 BẮT ĐẦU KIỂM THỬ PHÁT HIỆN CHÁY/KHÓI TRÊN ẢNH: {img_path.name}")
    print(f"   Đường dẫn: {img_path.resolve()}")
    print("=" * 70)

    orig_img = cv2.imread(str(img_path))
    if orig_img is None:
        print("[!] Không thể đọc ảnh bằng OpenCV.")
        sys.exit(1)

    out_dir = Path("runs_test")
    out_dir.mkdir(exist_ok=True)
    stem = img_path.stem

    # ----------------------------------------------------
    # 1. KIỂM THỬ MODEL CŨ (best.pt)
    # ----------------------------------------------------
    print(f"\n--- [1] KIỂM THỬ MODEL CŨ ({MODEL_CU_PATH.name}) ---")
    if MODEL_CU_PATH.exists():
        model_cu = YOLO(str(MODEL_CU_PATH))
        print(f"[*] Model Cũ classes: {model_cu.names}")
        dets_cu = run_single_inference(model_cu, orig_img, conf_thresh=MODEL_CU_CONF, imgsz=640)
        print(f"[*] Ngưỡng conf={MODEL_CU_CONF} | Số đối tượng phát hiện: {len(dets_cu)}")
        for d in dets_cu:
            print(f"    - Lớp: {d['class_name']} | Conf: {d['confidence']:.2%} | Box: {d['bbox']}")

        res_cu = draw_boxes(orig_img, dets_cu, "Model_Cu")
        out_cu = out_dir / f"{stem}_model_cu.jpg"
        cv2.imwrite(str(out_cu), res_cu)
        print(f"[+] Đã lưu ảnh kết quả: {out_cu.resolve()}")
    else:
        print(f"[!] Không tìm thấy {MODEL_CU_PATH}")

    # ----------------------------------------------------
    # 2. KIỂM THỬ MODEL MỚI (fire_yolo.pt)
    # ----------------------------------------------------
    print(f"\n--- [2] KIỂM THỬ MODEL MỚI ({MODEL_MOI_PATH.name}) ---")
    if not MODEL_MOI_PATH.exists():
        print(f"[!] Chưa có {MODEL_MOI_PATH}. Đang tự động tải...")
        from download_model import download_fire_yolo_model
        download_fire_yolo_model()

    model_moi = YOLO(str(MODEL_MOI_PATH))
    print(f"[*] Model Mới classes: {model_moi.names}")

    # 2.1. Thử ở conf mặc định (0.15)
    dets_moi = run_single_inference(model_moi, orig_img, conf_thresh=MODEL_MOI_CONF, imgsz=640)
    print(f"\n[*] [Cách 1] Chạy ảnh gốc ở conf={MODEL_MOI_CONF}: Phát hiện {len(dets_moi)} đối tượng")
    for d in dets_moi:
        print(f"    - Lớp: {d['class_name']} | Conf: {d['confidence']:.2%} | Box: {d['bbox']}")

    res_moi = draw_boxes(orig_img, dets_moi, "Fire_YOLO")
    out_moi = out_dir / f"{stem}_model_moi_conf015.jpg"
    cv2.imwrite(str(out_moi), res_moi)
    print(f"[+] Đã lưu ảnh kết quả: {out_moi.resolve()}")

    # 2.2. Nếu không thấy (hoặc để kiểm tra độ nhạy ban đêm): Thử hạ conf=0.05
    print(f"\n[*] [Cách 2] Thử nghiệm hạ ngưỡng nhạy conf=0.05:")
    dets_conf005 = run_single_inference(model_moi, orig_img, conf_thresh=0.05, imgsz=640)
    print(f"    -> Phát hiện {len(dets_conf005)} đối tượng")
    for d in dets_conf005:
        print(f"       - Lớp: {d['class_name']} | Conf: {d['confidence']:.2%} | Box: {d['bbox']}")
    res_conf005 = draw_boxes(orig_img, dets_conf005, "Fire_YOLO_0.05")
    cv2.imwrite(str(out_dir / f"{stem}_model_moi_conf005.jpg"), res_conf005)

    # 2.3. Thử nghiệm chuyển sang Grayscale 3 kênh (giả lập camera hồng ngoại ban đêm)
    print(f"\n[*] [Cách 3] Thử nghiệm Grayscale 3 kênh (Hồng ngoại ban đêm):")
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dets_gray = run_single_inference(model_moi, gray_3ch, conf_thresh=MODEL_MOI_CONF, imgsz=640)
    print(f"    -> Phát hiện {len(dets_gray)} đối tượng ở conf={MODEL_MOI_CONF}")
    for d in dets_gray:
        print(f"       - Lớp: {d['class_name']} | Conf: {d['confidence']:.2%} | Box: {d['bbox']}")
    res_gray = draw_boxes(gray_3ch, dets_gray, "Fire_YOLO_Gray")
    cv2.imwrite(str(out_dir / f"{stem}_model_moi_gray3ch.jpg"), res_gray)

    # 2.4. Thử nghiệm tăng Gamma (tăng tương phản đốm lửa sáng trong đêm)
    print(f"\n[*] [Cách 4] Thử nghiệm tăng tương phản Gamma (gamma=2.0):")
    img_gamma = adjust_gamma(orig_img, gamma=2.0)
    dets_gamma = run_single_inference(model_moi, img_gamma, conf_thresh=MODEL_MOI_CONF, imgsz=640)
    print(f"    -> Phát hiện {len(dets_gamma)} đối tượng ở conf={MODEL_MOI_CONF}")
    for d in dets_gamma:
        print(f"       - Lớp: {d['class_name']} | Conf: {d['confidence']:.2%} | Box: {d['bbox']}")
    res_gamma = draw_boxes(img_gamma, dets_gamma, "Fire_YOLO_Gamma")
    cv2.imwrite(str(out_dir / f"{stem}_model_moi_gamma2.jpg"), res_gamma)

    # ----------------------------------------------------
    # 3. KIỂM THỬ PIPELINE GỘP THÔNG MINH (DetectionService)
    # ----------------------------------------------------
    print(f"\n--- [3] KIỂM THỬ PIPELINE GỘP THÔNG MINH (Dual-Model Fusion) ---")
    from src.services.detection_service import DetectionService
    service = DetectionService()
    fused_res = service.detect_frame(orig_img, frame_index=0, timestamp_sec=0.0)

    fused_boxes_data = [
        {
            "class_id": d.class_id,
            "class_name": d.class_name,
            "confidence": d.confidence,
            "bbox": d.bbox,
            "source": d.model_source
        }
        for d in fused_res.detections
    ]
    print(f"[*] Pipeline Gộp phát hiện: {len(fused_boxes_data)} đối tượng")
    print(f"[*] Trạng thái cảnh báo: {fused_res.alert_state} - {fused_res.alert_message}")
    for d in fused_boxes_data:
        print(f"    - Lớp: {d['class_name']} | Conf: {d['confidence']:.2%} | Nguồn: {d['source']} | Box: {d['bbox']}")

    res_fused = draw_boxes(orig_img, fused_boxes_data, "PIPELINE_GOP")
    out_fused = out_dir / f"{stem}_pipeline_gop.jpg"
    cv2.imwrite(str(out_fused), res_fused)
    print(f"[+] Đã lưu ảnh pipeline gộp: {out_fused.resolve()}")

    print("\n" + "=" * 70)
    print(f"[V] HOÀN TẤT KIỂM THỬ. Tất cả ảnh kết quả nằm trong thư mục: {out_dir.resolve()}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử phát hiện cháy/khói độc lập trên ảnh")
    parser.add_argument(
        "image_path",
        nargs="?",
        default="images/images.jpg",
        help="Đường dẫn file ảnh kiểm thử (mặc định: images/images.jpg)"
    )
    args = parser.parse_args()
    test_image(args.image_path)
