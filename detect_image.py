"""Phát hiện lửa và khói trên ảnh sử dụng pipeline AI Gộp (best.pt + fire_yolo.pt)."""

import argparse
import sys
from pathlib import Path
import cv2

# Thêm đường dẫn project
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo in ký tự Unicode/Emoji mượt mà trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    CV_COLORS,
    DEFAULT_CONF_THRESHOLD,
    MODEL_CU_PATH,
    MODEL_MOI_PATH,
    TEN_KHOI,
    TEN_LUA,
)
from src.services.detection_service import DetectionService


def detect_image(image_path: str, conf_thresh: float = DEFAULT_CONF_THRESHOLD) -> None:
    path = Path(image_path)
    if not path.exists():
        print(f"[!] Không tìm thấy ảnh: {path.resolve()}")
        sys.exit(1)

    print("=" * 65)
    print("🔥 AI FIRE & SMOKE DETECTION - PIPELINE GỘP DUAL-MODEL")
    print(f"   Ảnh kiểm thử: {path.name}")
    print(f"   Model Cũ: {MODEL_CU_PATH.name} | Model Mới: {MODEL_MOI_PATH.name}")
    print(f"   Ngưỡng confidence: {conf_thresh:.1%}")
    print("=" * 65)

    img = cv2.imread(str(path))
    if img is None:
        print(f"[!] Không thể đọc ảnh bằng OpenCV: {path}")
        sys.exit(1)

    # Khởi tạo DetectionService
    service = DetectionService(conf_threshold=conf_thresh)
    result = service.detect_frame(img, frame_index=0, timestamp_sec=0.0)

    print(f"\n[*] Thời gian xử lý AI: {result.inference_time_ms:.1f} ms")
    print(f"[*] Số đối tượng phát hiện: {len(result.detections)}")

    annotated = img.copy()
    for det in result.detections:
        cname_lower = det.class_name.lower()
        x1, y1, x2, y2 = det.bbox

        source_tag = {"gop": "GỘP", "moi": "Mới", "cu": "Cũ"}.get(det.model_source, det.model_source)
        if cname_lower in TEN_LUA:
            color = CV_COLORS["fire"]
            label_prefix = f"LUA [{source_tag}]"
        elif cname_lower in TEN_KHOI:
            color = CV_COLORS["smoke"]
            label_prefix = f"KHOI [{source_tag}]"
        else:
            color = CV_COLORS["default"]
            label_prefix = f"{det.class_name.upper()} [{source_tag}]"

        label = f"{label_prefix}: {det.confidence:.1%}"
        print(f"    - {label} | Box: ({x1}, {y1}, {x2}, y2={y2})")

        # Vẽ bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
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

    # Kết luận
    print("\n" + "=" * 35 + " KẾT QUẢ PHÂN TÍCH " + "=" * 35)
    if result.has_fire and result.has_smoke:
        print(f"🚨 CẢNH BÁO: PHÁT HIỆN CẢ LỬA ({result.max_fire_conf:.1%}) VÀ KHÓI ({result.max_smoke_conf:.1%})!")
    elif result.has_fire:
        print(f"🔥 CẢNH BÁO: PHÁT HIỆN LỬA! Độ tin cậy cao nhất: {result.max_fire_conf:.1%}")
    elif result.has_smoke:
        print(f"🌫️ CẢNH BÁO: PHÁT HIỆN KHÓI! Độ tin cậy cao nhất: {result.max_smoke_conf:.1%}")
    else:
        print("✅ KHÔNG PHÁT HIỆN NGUY CƠ CHÁY / KHÓI.")
    print("=" * 89)

    # Lưu ảnh kết quả
    out_dir = Path("runs/fire_smoke_image")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"result_{path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n[+] Đã lưu ảnh kết quả nhận diện tại: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nhận diện cháy và khói từ ảnh với Dual-Model AI")
    parser.add_argument("image_path", nargs="?", default="images/images.jpg", help="Đường dẫn file ảnh")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Ngưỡng confidence (0.0 - 1.0)")
    args = parser.parse_args()

    detect_image(args.image_path, conf_thresh=args.conf)
