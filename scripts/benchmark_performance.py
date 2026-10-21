"""Script đo kiểm chuẩn hiệu năng toàn diện (Performance Benchmark Runner).
So sánh thông lượng (FPS) và độ trễ (latency breakdown) giữa:
1. Baseline: Chạy Full Ensemble cả 3 model mọi frame.
2. Stage 3: Adaptive Model Scheduling (ONNX Pre-screening + Dynamic Stride).

Kết quả lưu tại: runs/benchmark_performance.json
"""

import sys
import time
import json
import logging
from pathlib import Path
import cv2
import numpy as np

# Thêm thư mục gốc vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.detection_service import DetectionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")


def benchmark_mode(service: DetectionService, frames: list, num_repeats: int = 20) -> dict:
    """Đo đạc độ trễ chi tiết qua nhiều lần lặp."""
    latencies = []
    timing_breakdown = {
        "env_classify": [],
        "best_onnx": [],
        "fire_yolo_pt": [],
        "best_pt": [],
        "wbf_fusion": [],
        "threat_analyzer": [],
        "total_inference": []
    }

    # Warmup 2 frame
    for f in frames[:2]:
        service.detect_frame(f, frame_index=1, timestamp_sec=0.04)

    # Đo kiểm chính thức
    frame_idx = 2
    for rep in range(num_repeats):
        for f in frames:
            frame_idx += 1
            ts = frame_idx * 0.04
            t_start = time.perf_counter()
            res = service.detect_frame(f, frame_index=frame_idx, timestamp_sec=ts)
            t_end = time.perf_counter()

            total_lat = (t_end - t_start) * 1000.0
            latencies.append(total_lat)

            for k in timing_breakdown.keys():
                timing_breakdown[k].append(res.timing_ms.get(k, 0.0))

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    breakdown_mean = {k: round(float(np.mean(v)), 2) for k, v in timing_breakdown.items()}

    return {
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps": round(fps, 2),
        "timing_breakdown_ms": breakdown_mean
    }


def main():
    logger.info("=" * 70)
    logger.info("🚀 KHỞI ĐỘNG BỘ ĐO KIỂM HIỆU NĂNG GIAI ĐOẠN 3 (PERFORMANCE BENCHMARK)")
    logger.info("=" * 70)

    # Đọc ảnh từ thư mục images/ để làm tập benchmark thực tế
    images_dir = PROJECT_ROOT / "images"
    image_paths = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    frames = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)

    if not frames:
        logger.warning("Không tìm thấy ảnh trong images/, tạo 3 frame giả lập.")
        frames = [np.full((480, 640, 3), 120, dtype=np.uint8) for _ in range(3)]

    logger.info(f"Đang sử dụng {len(frames)} khung hình cho bài kiểm tra lặp lại.")

    # 1. Chạy Baseline: Full Ensemble
    logger.info("\n[1/2] Đang đo Baseline: Full Ensemble (chạy tất cả model trên mọi frame)...")
    service_baseline = DetectionService(
        enable_decision_logger=False,
        enable_adaptive_cascade=False
    )
    res_baseline = benchmark_mode(service_baseline, frames, num_repeats=5)

    # 2. Chạy Stage 3: Adaptive Model Scheduling
    logger.info("\n[2/2] Đang đo Stage 3: Adaptive Scheduling (ONNX Pre-screening + Dynamic Stride)...")
    service_adaptive = DetectionService(
        enable_decision_logger=False,
        enable_adaptive_cascade=True
    )
    res_adaptive = benchmark_mode(service_adaptive, frames, num_repeats=5)

    speedup = res_adaptive["fps"] / max(0.1, res_baseline["fps"])
    latency_reduction = (res_baseline["avg_latency_ms"] - res_adaptive["avg_latency_ms"]) / max(0.1, res_baseline["avg_latency_ms"])

    summary = {
        "baseline_full_ensemble": res_baseline,
        "stage_3_adaptive": res_adaptive,
        "speedup_factor": round(speedup, 2),
        "latency_reduction_percent": f"{latency_reduction * 100:.1f}%"
    }

    # In bảng so sánh trực quan
    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ SO SÁNH HIỆU NĂNG GIAI ĐOẠN 3")
    print("=" * 70)
    print(f"{'Chỉ số':<30} | {'Baseline (Full)':<18} | {'Stage 3 (Adaptive)':<18}")
    print("-" * 70)
    print(f"{'Thông lượng (FPS)':<30} | {res_baseline['fps']:<18.1f} | {res_adaptive['fps']:<18.1f}")
    print(f"{'Độ trễ trung bình (ms)':<30} | {res_baseline['avg_latency_ms']:<18.1f} | {res_adaptive['avg_latency_ms']:<18.1f}")
    print(f"{'Độ trễ P95 (ms)':<30} | {res_baseline['p95_latency_ms']:<18.1f} | {res_adaptive['p95_latency_ms']:<18.1f}")
    print(f"{'Thời gian ONNX 320 (ms)':<30} | {res_baseline['timing_breakdown_ms']['best_onnx']:<18.1f} | {res_adaptive['timing_breakdown_ms']['best_onnx']:<18.1f}")
    print(f"{'Thời gian fire_yolo (ms)':<30} | {res_baseline['timing_breakdown_ms']['fire_yolo_pt']:<18.1f} | {res_adaptive['timing_breakdown_ms']['fire_yolo_pt']:<18.1f}")
    print(f"{'Thời gian best.pt (ms)':<30} | {res_baseline['timing_breakdown_ms']['best_pt']:<18.1f} | {res_adaptive['timing_breakdown_ms']['best_pt']:<18.1f}")
    print("=" * 70)
    print(f"⚡ TỐC ĐỘ TĂNG GẤP: {speedup:.2f}x (Giảm {latency_reduction * 100:.1f}% thời gian xử lý)")
    print("=" * 70 + "\n")

    out_file = PROJECT_ROOT / "runs" / "benchmark_performance.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Đã lưu kết quả đo kiểm vào: {out_file}")


if __name__ == "__main__":
    main()
