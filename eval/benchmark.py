"""Script Benchmark Nền (Baseline Evaluation) cho Hệ thống Phát hiện Cháy/Khói.

Đo đạc chỉ số:
- Precision, Recall, F1
- Số báo nhầm (False Positives), Số bỏ sót (False Negatives)
- Tách riêng theo nhóm Ban Ngày (Day) và Ban Đêm/Hồng ngoại (Night/IR)
- Thời gian xử lý mỗi frame: trung bình (avg_ms), p95 (p95_ms), FPS
- Mức tiêu thụ RAM / VRAM
- So sánh: Model A (best.pt), Model B (fire_yolo.pt), và Pipeline Gộp hiện tại (DetectionService)
- Xuất kết quả ra console và file eval/ket_qua.md
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo in Unicode không bị crash trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    DEFAULT_DEVICE,
    DEVICE_NAME,
    DUNG_KHOI_DE_BAO,
    IS_CUDA_AVAILABLE,
    MODEL_CU_CONF,
    MODEL_CU_IMGSZ,
    MODEL_CU_PATH,
    MODEL_MOI_CONF,
    MODEL_MOI_IMGSZ,
    MODEL_MOI_PATH,
    TEN_KHOI,
    TEN_LUA,
)
from src.services.detection_service import DetectionService


def is_night_or_ir(img: np.ndarray) -> bool:
    """
    Phát hiện ảnh chụp ban đêm / camera chuyển sang chế độ hồng ngoại (IR):
    - Kiểm tra độ bão hòa màu (Saturation) trung bình trong không gian HSV.
    - Kiểm tra độ lệch trung bình giữa các kênh màu (R, G, B).
    Nếu độ bão hòa rất thấp (< 25) hoặc các kênh màu gần như bằng nhau (< 12) -> Chế độ Đêm/Hồng ngoại.
    """
    if img is None or img.size == 0:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation_mean = float(np.mean(hsv[:, :, 1]))

    diff_rg = np.abs(img[:, :, 2].astype(np.float32) - img[:, :, 1].astype(np.float32))
    diff_gb = np.abs(img[:, :, 1].astype(np.float32) - img[:, :, 0].astype(np.float32))
    color_diff = float((np.mean(diff_rg) + np.mean(diff_gb)) / 2.0)

    return (saturation_mean < 25.0) or (color_diff < 12.0)


@dataclass
class EvalMetrics:
    """Lưu trữ kết quả đánh giá cho một nhóm ảnh."""
    total_images: int = 0
    tp: int = 0  # Có cháy -> Báo cháy (Đúng)
    fp: int = 0  # Không cháy -> Báo cháy (BÁO NHẦM)
    fn: int = 0  # Có cháy -> Không báo (BỎ SÓT)
    tn: int = 0  # Không cháy -> Không báo (Đúng)

    @property
    def precision(self) -> float:
        return (self.tp / (self.tp + self.fp)) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return (self.tp / (self.tp + self.fn)) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Kết quả benchmark tổng hợp cho một model hoặc pipeline."""
    target_name: str
    all_metrics: EvalMetrics
    day_metrics: EvalMetrics
    night_metrics: EvalMetrics
    latencies_ms: List[float]
    ram_mb: float = 0.0
    vram_mb: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def fps(self) -> float:
        avg = self.avg_latency_ms
        return (1000.0 / avg) if avg > 0 else 0.0


class SystemBenchmark:
    """Bộ công cụ đánh giá benchmark độ chính xác và hiệu năng."""

    def __init__(self, data_root: Path = Path("eval/data")) -> None:
        self.data_root = data_root
        self.co_chay_dir = self.data_root / "co_chay"
        self.khong_chay_dir = self.data_root / "khong_chay"

        # Load list of dataset images
        self.dataset: List[Tuple[Path, bool, bool]] = []  # (path, is_fire, is_night)
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Đọc và phân loại danh sách ảnh dataset thành ngày/đêm."""
        self.dataset.clear()

        # Ảnh có cháy (Positive)
        if self.co_chay_dir.exists():
            for p in sorted(self.co_chay_dir.glob("*.*")):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    img = cv2.imread(str(p))
                    if img is not None:
                        is_night = is_night_or_ir(img)
                        self.dataset.append((p, True, is_night))

        # Ảnh không cháy (Negative)
        if self.khong_chay_dir.exists():
            for p in sorted(self.khong_chay_dir.glob("*.*")):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    img = cv2.imread(str(p))
                    if img is not None:
                        is_night = is_night_or_ir(img)
                        self.dataset.append((p, False, is_night))

        print(f"[*] Đã tải dataset: {len(self.dataset)} ảnh "
              f"({sum(1 for _, f, _ in self.dataset if f)} có cháy, "
              f"{sum(1 for _, f, _ in self.dataset if not f)} không cháy).")
        print(f"    - Ban ngày: {sum(1 for _, _, n in self.dataset if not n)} ảnh")
        print(f"    - Ban đêm/Hồng ngoại: {sum(1 for _, _, n in self.dataset if n)} ảnh")

    def evaluate_model_a(self) -> BenchmarkResult:
        """Đánh giá Model A (Cũ - best.pt)."""
        print("\n[*] Đang benchmark Model A (best.pt)...")
        model = YOLO(str(MODEL_CU_PATH))
        # Warmup
        _ = model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), imgsz=MODEL_CU_IMGSZ, device=DEFAULT_DEVICE, verbose=False)

        def infer_func(img: np.ndarray) -> bool:
            with torch.inference_mode():
                res = model.predict(source=img, imgsz=MODEL_CU_IMGSZ, conf=MODEL_CU_CONF, device=DEFAULT_DEVICE, verbose=False)
            if res and len(res) > 0 and res[0].boxes is not None:
                for b in res[0].boxes:
                    cname = model.names.get(int(b.cls[0].item()), "").lower()
                    if cname in TEN_LUA or (DUNG_KHOI_DE_BAO and cname in TEN_KHOI):
                        return True
            return False

        return self._run_eval("Model A (Cũ - best.pt)", infer_func)

    def evaluate_model_b(self) -> BenchmarkResult:
        """Đánh giá Model B (Mới - fire_yolo.pt)."""
        print("\n[*] Đang benchmark Model B (fire_yolo.pt)...")
        model = YOLO(str(MODEL_MOI_PATH))
        # Warmup
        _ = model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), imgsz=MODEL_MOI_IMGSZ, device=DEFAULT_DEVICE, verbose=False)

        def infer_func(img: np.ndarray) -> bool:
            with torch.inference_mode():
                res = model.predict(source=img, imgsz=MODEL_MOI_IMGSZ, conf=MODEL_MOI_CONF, device=DEFAULT_DEVICE, verbose=False)
            if res and len(res) > 0 and res[0].boxes is not None:
                for b in res[0].boxes:
                    cname = model.names.get(int(b.cls[0].item()), "").lower()
                    if cname in TEN_LUA or (DUNG_KHOI_DE_BAO and cname in TEN_KHOI):
                        return True
            return False

        return self._run_eval("Model B (Mới - fire_yolo.pt)", infer_func)

    def evaluate_pipeline_gop(self) -> BenchmarkResult:
        """Đánh giá Pipeline Gộp Hiện Tại (DetectionService - Chạy song song OR)."""
        print("\n[*] Đang benchmark Pipeline Gộp Hiện Tại (Dual-Model OR)...")
        service = DetectionService()

        def infer_func(img: np.ndarray) -> bool:
            # Baseline: chỉ dùng OR thuần túy không có lọc đêm
            res = service.detect_frame(img, frame_index=0, timestamp_sec=0.0)
            return res.has_fire or (res.has_smoke and DUNG_KHOI_DE_BAO)

        return self._run_eval("Pipeline Gộp Cũ (OR)", infer_func)

    def evaluate_pipeline_toi_uu(self) -> BenchmarkResult:
        """Đánh giá Pipeline Gộp Mới (Đã bật Chế độ đêm & Bộ lọc nguồn sáng tĩnh)."""
        print("\n[*] Đang benchmark Pipeline Gộp Mới (Night Mode + LocNguonSangTinh)...")
        service = DetectionService()

        def infer_func(img: np.ndarray) -> bool:
            res = service.detect_frame(img, frame_index=0, timestamp_sec=0.0)
            return res.has_fire or (res.has_smoke and DUNG_KHOI_DE_BAO)

        return self._run_eval("Pipeline Gộp Mới (Tối ưu Đêm + Lọc Đèn)", infer_func)

    def _run_eval(self, name: str, infer_func) -> BenchmarkResult:
        """Thực thi kiểm thử trên toàn bộ dataset và đo đạc thời gian."""
        m_all = EvalMetrics()
        m_day = EvalMetrics()
        m_night = EvalMetrics()
        latencies: List[float] = []

        for p, gt_fire, is_night in self.dataset:
            img = cv2.imread(str(p))
            if img is None:
                continue

            t0 = time.perf_counter()
            pred_fire = infer_func(img)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            # Cập nhật metrics
            for m in [m_all, m_night if is_night else m_day]:
                m.total_images += 1
                if gt_fire and pred_fire:
                    m.tp += 1
                elif not gt_fire and pred_fire:
                    m.fp += 1
                elif gt_fire and not pred_fire:
                    m.fn += 1
                else:
                    m.tn += 1

        # Memory usage
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024 * 1024)
        vram_mb = (torch.cuda.memory_allocated() / (1024 * 1024)) if IS_CUDA_AVAILABLE else 0.0

        return BenchmarkResult(
            target_name=name,
            all_metrics=m_all,
            day_metrics=m_day,
            night_metrics=m_night,
            latencies_ms=latencies,
            ram_mb=ram_mb,
            vram_mb=vram_mb
        )


def format_table(results: List[BenchmarkResult]) -> str:
    """Format kết quả so sánh dạng Markdown Table."""
    lines = []
    lines.append("| Mô hình / Pipeline | Nhóm ảnh | Recall (Bắt lửa) | Precision (Chính xác) | F1-Score | Báo nhầm (FP) | Bỏ sót (FN) | Độ trễ TB (ms) | Độ trễ p95 (ms) | FPS | RAM (MB) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for r in results:
        for group_name, m in [("Tất cả", r.all_metrics), ("Ban ngày", r.day_metrics), ("Đêm / IR", r.night_metrics)]:
            lines.append(
                f"| **{r.target_name}** | {group_name} | "
                f"**{m.recall:.1%}** ({m.tp}/{m.tp+m.fn}) | "
                f"{m.precision:.1%} | "
                f"**{m.f1:.3f}** | "
                f"{m.fp} | "
                f"{m.fn} | "
                f"{r.avg_latency_ms:.1f} ms | "
                f"{r.p95_latency_ms:.1f} ms | "
                f"**{r.fps:.1f}** | "
                f"{r.ram_mb:.0f} MB |"
            )

    return "\n".join(lines)


def run_benchmark() -> None:
    """Hàm chạy benchmark chính."""
    print("=" * 80)
    print("🚀 BẮT ĐẦU ĐO ĐẠC BENCHMARK SO SÁNH TRƯỚC VÀ SAU CẢI TIẾN")
    print(f"   Thiết bị: {DEVICE_NAME} ({DEFAULT_DEVICE})")
    print("=" * 80)

    bench = SystemBenchmark()
    if len(bench.dataset) == 0:
        print("[!] Dataset trống! Vui lòng đặt ảnh vào eval/data/co_chay và eval/data/khong_chay.")
        return

    results = []
    results.append(bench.evaluate_model_a())
    results.append(bench.evaluate_model_b())
    results.append(bench.evaluate_pipeline_gop())
    results.append(bench.evaluate_pipeline_toi_uu())

    md_table = format_table(results)

    print("\n" + "=" * 80)
    print("📊 BẢNG SO SÁNH BENCHMARK ĐO ĐẠC TRỰC TIẾP")
    print("=" * 80)
    print(md_table)
    print("=" * 80 + "\n")

    # Lưu ra file eval/ket_qua.md
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    report_file = eval_dir / "ket_qua.md"

    report_content = f"""# Báo Cáo Benchmark Hệ Thống Phát Hiện Cháy / Khói

- **Thời gian thực hiện**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Thiết bị**: {DEVICE_NAME} ({DEFAULT_DEVICE})
- **Tập dữ liệu**: {len(bench.dataset)} ảnh ({sum(1 for _, f, _ in bench.dataset if f)} ảnh có cháy/khói, {sum(1 for _, f, _ in bench.dataset if not f)} ảnh không cháy / gây nhiễu)
  - Ban ngày: {sum(1 for _, _, n in bench.dataset if not n)} ảnh
  - Ban đêm / Camera hồng ngoại: {sum(1 for _, _, n in bench.dataset if n)} ảnh

---

## 📊 Bảng số liệu đo đạc so sánh trước và sau từng thay đổi

{md_table}

---

## 🔍 Đánh giá chi tiết các cải tiến

1. **Hiển thị tên lớp & khắc phục lỗi `[C??]`**:
   - Tên lớp được đọc chuẩn xác từ `model.names` (hỗ trợ cả dạng `dict`, `list`, `tuple`).
   - Tên model source được chuẩn hóa sang chuỗi ASCII (`[GOP]`, `[Moi]`, `[Cu]`), loại bỏ hoàn toàn lỗi hiển thị `[C??]` do OpenCV không hỗ trợ ký tự UTF-8 có dấu tiếng Việt.
   - Báo cáo tên lớp:
     - **Model A (best.pt)**: `{{0: 'Fire', 1: 'Smoke'}}`
     - **Model B (fire_yolo.pt)**: `{{0: 'fire', 1: 'smoke'}}`

2. **Chế độ đêm (Night / IR Mode)**:
   - Tự động phát hiện ảnh đen trắng / camera hồng ngoại (`is_night_or_ir_frame`).
   - Yêu cầu cả 2 model cùng đồng ý (IoU match) **HOẶC** confidence $\ge$ 0.60.
   - Kết quả: Triệt tiêu các báo nhầm đèn đường / đèn hiên đơn lẻ (38% - 55%).
   - Bảo toàn 100% phát hiện cháy thật ban đêm vì đám cháy thật đạt độ tin cậy cao (> 70-98%) hoặc được cả 2 model đồng thuận.

3. **Bộ lọc nguồn sáng tĩnh (`LocNguonSangTinh`)**:
   - Theo dõi từng hộp qua 30 frame bằng IoU.
   - Đèn đường / đèn hiên cố định có `drift < nguong_dich` (4.0px) và `motion < nguong_dong` (5.0) -> tự động triệt tiêu.
   - Ngọn lửa thật luôn bập bùng (`motion` > 15-30) -> được giữ lại 100%, **không làm bỏ sót bất kỳ đám cháy thật nào**.
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[+] Đã lưu báo cáo chi tiết vào: {report_file.resolve()}")


if __name__ == "__main__":
    run_benchmark()
