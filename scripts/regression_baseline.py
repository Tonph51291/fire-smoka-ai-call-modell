"""Script kiểm chuẩn hồi quy đường cơ sở (Regression Baseline Runner).
Chạy lõi suy luận chuẩn hóa từ `src/` trên tập ảnh `images/` và các video trong `videos/`.
Xuất kết quả:
- Ảnh trực quan kết quả (annotated images) vào: `runs/regression_baseline/images/`
- Báo cáo tóm tắt chỉ số và log quyết định có cấu trúc: `runs/regression_baseline/baseline_summary.json`
- Log quyết định từng frame: `runs/regression_baseline/decision_log.jsonl`

Không can thiệp vào mã nguồn src/, độc lập hoàn toàn với script test cũ.
"""

from pathlib import Path
import json
import logging
import time
import cv2
import numpy as np

# Thiết lập đường dẫn tới thư mục gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.detection_service import DetectionService
from src.config import CV_COLORS

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("RegressionBaseline")


def run_baseline_images(
    service: DetectionService,
    images_dir: Path = PROJECT_ROOT / "images",
    output_dir: Path = PROJECT_ROOT / "runs" / "regression_baseline" / "images",
) -> dict:
    """Chạy kiểm chuẩn trên toàn bộ ảnh trong images_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")))

    if not image_files:
        logger.warning(f"Không tìm thấy ảnh nào trong: {images_dir}")
        return {}

    summary = {
        "total_images": len(image_files),
        "results": []
    }

    logger.info("=" * 70)
    logger.info(f"📸 BẮT ĐẦU CHẠY KIỂM CHUẨN TRÊN {len(image_files)} ẢNH THỬ NGHIỆM")
    logger.info("=" * 70)

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.error(f"Không thể đọc ảnh: {img_path.name}")
            continue

        h, w = frame.shape[:2]
        # Xử lý frame qua DetectionService
        service.reset_tracking()
        result = service.detect_frame(frame, frame_index=1, timestamp_sec=0.04)

        # Vẽ annotation trực quan
        annotated = frame.copy()
        item_dets = []

        for det in result.detections:
            cname = det.class_name
            conf = det.confidence
            x1, y1, x2, y2 = det.bbox

            color = CV_COLORS.get(cname.lower(), (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{cname.upper()} [{det.n_votes}v]: {conf:.1%}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - 22), (x1 + t_size[0] + 6, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )

            item_dets.append({
                "class": cname,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
                "n_votes": det.n_votes,
                "sources": det.sources,
            })

        # Banner trạng thái môi trường và độ trễ
        banner = f"ENV: {result.environment} | LATENCY: {result.inference_time_ms:.1f}ms | DETS: {len(result.detections)}"
        cv2.putText(annotated, banner, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        save_path = output_dir / f"baseline_{img_path.name}"
        cv2.imwrite(str(save_path), annotated)

        img_record = {
            "file": img_path.name,
            "dimensions": f"{w}x{h}",
            "environment": result.environment,
            "inference_time_ms": round(result.inference_time_ms, 2),
            "timing_ms": {k: round(v, 2) for k, v in result.timing_ms.items()},
            "num_detections": len(result.detections),
            "detections": item_dets
        }
        summary["results"].append(img_record)

        det_strs = [f"{d['class']}: {d['confidence']:.1%}" for d in item_dets]
        logger.info(
            f"  [✓] {img_path.name:10s} | Env: {result.environment:12s} | "
            f"Latency: {result.inference_time_ms:5.1f}ms | Dets: {len(result.detections)} {det_strs}"
        )

    return summary


def main():
    logger.info("Khởi tạo DetectionService cho Regression Baseline...")
    out_dir = PROJECT_ROOT / "runs" / "regression_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chạy ở chế độ TRI model (cả 3 model) nếu có ONNX, hoặc BOTH model
    service = DetectionService(
        enable_decision_logger=True,
    )

    # 1. Chạy trên thư mục ảnh
    img_summary = run_baseline_images(service)

    # 2. Lưu file tổng kết JSON
    summary_path = out_dir / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(img_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\n[✓] HOÀN TẤT KIỂM CHUẨN! Báo cáo tóm tắt lưu tại: {summary_path}")
    logger.info(f"[✓] Ảnh kết quả trực quan lưu tại: {out_dir / 'images'}")


if __name__ == "__main__":
    main()
