# Hệ Thống AI Phát Hiện Cháy & Khói Thời Gian Thực Đa Mô Hình (Fire & Smoke AI)

Hệ thống thị giác máy tính thông minh chạy cục bộ (local) trên Windows/Linux, ứng dụng kỹ thuật **Hợp nhất đa mô hình (Tri-Model Ensemble)** kết hợp **Bám vết không - thời gian (Spatio-Temporal Tracking)** và **Bộ tích lũy bằng chứng thời gian thực (Evidence Accumulator)** nhằm phát hiện sớm ngọn lửa và làn khói từ camera giám sát (ban ngày, ban đêm, camera hồng ngoại IR).

---

## 🌟 Các Tính Năng Nổi Bật

### 1. Hợp nhất 3 Mô hình AI (Tri-Model Ensemble qua WBF)
- **Model Mới (`models/fire_yolo.pt`)**: YOLO 640x640 từ Hugging Face [`seawsurf/fire_yolo`](https://huggingface.co/seawsurf/fire_yolo), tối ưu cho camera đêm và hồng ngoại.
- **Model Gốc (`models/best.pt`)**: YOLO 640x640 nhận diện lửa và khói bối cảnh chuẩn.
- **Model ONNX (`models/best.onnx`)**: YOLO 320x320 siêu nhẹ, suy luận trực tiếp qua `cv2.dnn`.
- **Weighted Boxes Fusion (WBF)**: Hợp nhất tọa độ và hiệu chuẩn độ tin cậy trung thực, loại bỏ hoàn toàn hiện tượng trần điểm số ảo $99\%$.

### 2. Phân loại môi trường ánh sáng thích ứng (Environment Classifier)
- Tự động nhận diện 6 môi trường quang học: `DAY_NORMAL`, `DAY_BRIGHT`, `DAY_FOGGY`, `NIGHT_COLOR`, `NIGHT_IR`, `SUNSET_DAWN`.
- Tích hợp bộ trễ (Hysteresis) theo thời gian video, tự động tinh chỉnh trọng số từng model theo điều kiện ánh sáng.

### 3. Bộ lọc nguồn sáng tĩnh & Phạt mềm (LocNguonSangTinh)
- Triệt tiêu báo động giả từ đèn đường, đèn hiên, biển báo phản quang bằng cách đo đạc độ dịch tâm ($v_{\text{drift}}$), độ biến thiên nội dung pixel, độ nhẵn viền và bão hòa lõi đèn.
- Không xóa cứng mà áp dụng **Phạt mềm ($K_{\text{static}} = 0.25$)**. Tích hợp cơ chế **Safety Release** tự động mở khóa ngay khi diện tích tăng vọt hoặc xuất hiện dấu hiệu cháy lan.

### 4. Bám vết không - thời gian & Máy trạng thái tích lũy bằng chứng
- **SpatioTemporalTracker**: Hàm tương đồng đa tiêu chí kết hợp $0.50 \cdot \text{IoU} + 0.30 \cdot \text{Containment} + 0.20 \cdot \text{Distance}$, duy trì ổn định Track ID cho vật thể biến dạng, giãn nở và dâng cao.
- **MotionFeatureExtractor**: Đo vận tốc trôi chuẩn hóa $v_{\text{drift}}$, độ dâng cao của cột khói, độ nham nhở viền bập bùng ($\text{Roughness} \ge 1.25$) và phát hiện chuyển động thẳng hàng của đèn xe.
- **EvidenceAccumulator**: Máy trạng thái 4 mức (`NONE` $\rightarrow$ `SUSPECT` $\rightarrow$ `CONFIRMED` $\rightarrow$ `ALARM`) với tích lũy bằng chứng $E(t) \in [0, 100]$ phân rã hàm mũ liên tục theo thời gian thực:
  $$E(t) = E(t - \Delta t) \cdot e^{-\Delta t / \tau} + \Delta E$$
  Tách biệt chu kỳ phân rã cho Lửa ($\tau_{\text{fire}} = 2.5\text{s}$) và Khói ($\tau_{\text{smoke}} = 8.0\text{s}$).

### 5. Điều phối mô hình thích ứng & Dynamic Stride (Tối ưu CPU/GPU)
- **Adaptive Model Scheduler**: Trên môi trường CPU, hệ thống chạy trước ONNX 320 (~40ms); khi phát hiện nghi vấn sẽ đánh thức ngay các mô hình PyTorch để thẩm định chéo. Quét Full Ensemble định kỳ mỗi 25 frame (1 giây) để đảm bảo không bỏ sót điểm mù.
- **Dynamic Stride**: Tự động nhảy frame thích ứng (`stride = 2` trên CPU) khi bình thường, và lập tức **siết về `stride = 1`** khi phát hiện đối tượng nghi vấn.
- **Multi-threaded Video Pipeline**: Tách biệt luồng phát mượt (Playback Thread) và luồng suy luận AI (Inference Thread) theo cơ chế Single-Slot Overwrite (Zero Buffer Lag).

### 6. Thu thập mẫu khó tự động (Active Learning / Hard Negative Harvesting)
- Khi phát hiện nguồn sáng tĩnh hoặc đèn xe bị triệt tiêu, hệ thống tự động lưu trữ bộ 3 dữ liệu vào `hard_negatives/`: ảnh full frame (`*_full.jpg`), ảnh crop (`*_crop.jpg`) và file metadata chuẩn (`*_meta.json` ghi nhận drift, flicker std, core saturation, bbox) để tái huấn luyện model.

---

## 🏗️ Cấu Trúc Dự Án

```text
fire-smoke-ai/
├── src/
│   ├── __init__.py
│   ├── main.py                         # Entry point chính của ứng dụng Desktop Tkinter
│   ├── config.py                       # Cấu hình tập trung toàn bộ ngưỡng, tham số và profile
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detection_service.py        # Dịch vụ suy luận đa mô hình, Dynamic Stride, WBF
│   │   ├── hardware_manager.py         # Nhận diện CPU/GPU, VRAM, RAM và benchmark khởi động
│   │   ├── environment_classifier.py   # Phân loại 6 môi trường ánh sáng có Hysteresis
│   │   ├── ensemble_wbf.py             # Thuật toán Weighted Boxes Fusion (WBF)
│   │   ├── static_light_filter.py      # Bộ lọc nguồn sáng tĩnh, Static Memory và Hard Negative
│   │   ├── spatio_temporal_tracker.py  # Bộ bám vết đa tiêu chí (IoU + Containment + Distance)
│   │   ├── motion_feature_extractor.py # Đo động học v_drift, cột khói, roughness, đèn xe
│   │   ├── evidence_accumulator.py     # Máy trạng thái 4 mức & Tích lũy bằng chứng E(t)
│   │   ├── threat_analyzer.py          # Bộ phân tích nguy cơ, bùng cháy khẩn cấp & cổng liveness
│   │   ├── decision_logger.py          # Ghi log quyết định từng frame có cấu trúc (JSONL)
│   │   └── video_service.py            # Quản lý phát video đa luồng và suy luận bất đồng bộ
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main_window.py              # Giao diện người dùng Tkinter Canvas + Pillow
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py                 # Chuẩn hóa letterbox 2 chiều, IoU, Containment
│       └── video_utils.py              # Tiện ích đọc metadata video và format timestamp
├── models/
│   ├── best.pt                         # Model YOLOv8 PyTorch (Classes: Fire, Smoke)
│   ├── fire_yolo.pt                    # Model YOLOv8 PyTorch (Classes: fire, smoke)
│   └── best.onnx                       # Model YOLOv8 ONNX 320x320 chạy cv2.dnn
├── images/                             # Tập ảnh kiểm thử thực tế đa môi trường
├── scripts/
│   ├── regression_baseline.py          # Runner kiểm chuẩn hồi quy trên tập ảnh mẫu
│   └── benchmark_performance.py        # Runner đo kiểm hiệu năng so sánh Baseline và Adaptive
├── tests/                              # Bộ 33 unit tests tự động (100% Passed)
│   ├── test_config.py
│   ├── test_utils.py
│   ├── test_detection_service.py
│   ├── test_static_light_filter.py
│   ├── test_evidence_accumulator.py
│   ├── test_adaptive_scheduler.py
│   └── test_video_service.py
├── canh_bao/                           # Thư mục tự động lưu frame khi kích hoạt báo cháy
├── hard_negatives/                     # Thư mục lưu mẫu âm tính kèm metadata JSON
├── runs/                               # Kết quả log quyết định (decision_log.jsonl, benchmark)
├── requirements.txt                    # Thư viện phụ thuộc
└── README.md                           # Tài liệu hướng dẫn sử dụng
```

---

## ⚙️ Cài Đặt Môi Trường

### 1. Tạo và kích hoạt môi trường ảo (Python 3.10+)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Cài đặt thư viện phụ thuộc

```powershell
pip install -r requirements.txt
```

*(Lưu ý: Đối với máy có GPU NVIDIA, khuyến khích cài đặt bản PyTorch CUDA tương ứng để đạt tốc độ cao nhất).*

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Chạy Ứng Dụng Giao Diện Trực Quan (GUI Desktop)

Khởi chạy ứng dụng từ thư mục gốc:

```powershell
python -m src.main
```

- Bấm nút **`📁 Chọn video...`** để phát video hoặc kết nối nguồn camera RTSP.
- Bảng điều khiển hiển thị trực tiếp: Bounding box có mã màu, Track ID, Điểm bằng chứng $E(t)$, Trạng thái đe dọa, Môi trường nhận diện và FPS thực tế.
- Khi có nguy cơ cháy đạt ngưỡng `ALARM`:
  - Banner chuyển sang màu đỏ: `CẢNH BÁO: PHÁT HIỆN LỬA` (hoặc `🔥 BÙNG LỬA`).
  - Console in cảnh báo chi tiết kèm thời điểm video và timestamp hệ thống.
  - Ảnh chụp kèm khung bounding box được tự động lưu vào thư mục `canh_bao/`.

### 2. Chạy Kiểm Chuẩn Hồi Quy Trên Tập Ảnh (`scripts/regression_baseline.py`)

Thực thi kiểm chuẩn độc lập trên toàn bộ ảnh trong thư mục `images/`:

```powershell
python scripts/regression_baseline.py
```

- Tự động xuất ảnh kết quả có vẽ box và thông số vào: `runs/regression_baseline/images/`
- Tự động lưu bảng tổng hợp chỉ số phát hiện vào: `runs/regression_baseline/baseline_summary.json`
- Ghi log frame-by-frame chi tiết vào: `runs/decision_log.jsonl`

### 3. Đo Kiểm Chuẩn Hiệu Năng & Tốc Độ (`scripts/benchmark_performance.py`)

So sánh thông lượng (FPS) và độ trễ giữa chế độ Full Ensemble và Adaptive Scheduling:

```powershell
python scripts/benchmark_performance.py
```

Báo cáo phân rã độ trễ (latency breakdown) chi tiết sẽ được xuất ra màn hình và lưu tại `runs/benchmark_performance.json`.

---

## 🔧 Cấu Hình & Tinh Chỉnh Tham Số

Toàn bộ thông số hệ thống được tập trung tại [src/config.py](file:///d:/tecnix/fire-smoke-ai/src/config.py):

| Tham số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| `ENSEMBLE_MODE` | `"both"` / `"tri"` / `"cascade"` | Chế độ phối hợp mô hình AI (chạy 2 model, 3 model, hoặc cascade) |
| `WBF_IOU_THRESH` | `0.45` | Ngưỡng IoU ghép cặp trong Weighted Boxes Fusion |
| `TAU_DECAY_FIRE` | `2.5` (giây) | Chu kỳ bán rã bằng chứng ngọn lửa |
| `TAU_DECAY_SMOKE` | `8.0` (giây) | Chu kỳ bán rã bằng chứng làn khói |
| `E_SUSPECT` | `25.0` | Ngưỡng tích lũy bước vào trạng thái nghi vấn (`SUSPECT`) |
| `E_CONFIRMED` | `55.0` | Ngưỡng tích lũy xác nhận nguy cơ (`CONFIRMED`) |
| `E_ALARM` | `85.0` | Ngưỡng kích hoạt báo động cháy chính thức (`ALARM`) |
| `SOFT_PENALTY_STATIC` | `0.25` | Hệ số phạt bằng chứng đối với nguồn sáng tĩnh (đèn, biển báo) |
| `SOFT_PENALTY_VEHICLE` | `0.10` | Hệ số phạt bằng chứng đối với đèn phương tiện giao thông chạy qua |
| `EMERGENCY_GROWTH_RATIO` | `1.30` | Tỉ lệ nhảy vọt diện tích để kích hoạt bùng cháy khẩn cấp (`bung_lua`) |
| `EMERGENCY_ROUGHNESS_THRESH` | `1.25` | Độ dao động viền hữu cơ tối thiểu để xác nhận ngọn lửa bập bùng |
| `COOLDOWN_CANH_BAO_SEC` | `30.0` (giây) | Thời gian chờ giữa 2 lần lưu ảnh cảnh báo ra thư mục `canh_bao/` |

---

## 🧪 Kiểm Thử Tự Động Toàn Diện (Unit Tests)

Khởi chạy bộ unit test tự động phủ khắp tất cả các module trong `tests/`:

```powershell
python -m unittest discover tests
```

**Đảm bảo toàn bộ 33/33 tests Passed (100% OK)** bao gồm:
- Kiểm thử cấu hình, hình học letterbox và phân loại môi trường.
- Kiểm thử logic WBF và phát hiện ban đêm/IR.
- Kiểm thử toán học bộ tích lũy bằng chứng (khói yếu $C=0.20$ leo thang lên ALARM, triệt tiêu đèn tĩnh và đèn xe).
- Kiểm thử Dynamic Stride, Adaptive Cascade và Hard Negative Curation kèm JSON metadata.
