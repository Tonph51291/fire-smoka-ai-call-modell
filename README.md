# Ứng Dụng AI Phát Hiện Cháy & Khói Chạy Song Song 2 Model (Fire & Smoke AI)

Ứng dụng Desktop chạy cục bộ (local) trên Windows sử dụng Python, Tkinter, OpenCV và thư viện Ultralytics YOLO để phát hiện và cảnh báo Lửa (Fire) và Khói (Smoke) theo thời gian thực từ video/camera.

Hệ thống hỗ trợ **chạy song song 2 mô hình AI (Dual-Model Ensemble)** kết hợp logic OR, tối ưu hóa phát hiện lửa cả ban ngày lẫn ban đêm (chế độ hồng ngoại / ảnh đen trắng của camera giám sát).

---

## 📌 Tính năng nổi bật

1. **Chạy song song 2 Model AI (Dual-Model OR Logic)**:
   - **Model Hiện Có (`models/best.pt`)**: Nhận diện lửa và khói môi trường chuẩn.
   - **Model Bổ Sung (`models/fire_yolo.pt`)**: Tải từ Hugging Face [`seawsurf/fire_yolo`](https://huggingface.co/seawsurf/fire_yolo), tối ưu cho nhận diện lửa sáng trong camera đêm / hồng ngoại.
   - Kết hợp kiểu **OR**: Báo động cháy nếu một trong hai model phát hiện lửa.
2. **Bộ lọc thời gian giảm báo nhầm (Temporal Smoothing Filter)**:
   - Sử dụng `collections.deque(maxlen=10)`, chỉ kích hoạt báo động khi có **ít nhất 5 trên 10 lần** xử lý gần nhất phát hiện lửa.
3. **Phát cảnh báo & Cooldown**:
   - Khi có cháy thật được xác nhận: In `CẢNH BÁO CHÁY` ra console, tự động lưu ảnh chụp frame vào thư mục `canh_bao/`.
   - Có thời gian chờ (cooldown 30s) chống spam báo động liên tục.
4. **Phát video mượt mà trên giao diện**:
   - Hiển thị frame trực tiếp trên canvas Tkinter (sử dụng Pillow), không phụ thuộc vào `cv2.imshow()`.
   - Khung giao diện cố định 100%, không bị nhảy layout khi phát các video kích thước khác nhau.
5. **Đa luồng độc lập (3 Threads)**:
   - **Playback Thread**: Đồng bộ theo đồng hồ thực `time.monotonic()`, bỏ frame qua `cap.grab()` nếu bị chậm.
   - **Inference Thread**: Chạy AI nền bất đồng bộ với `FRAME_STRIDE = 3` (xử lý 1 trên 3 frame), không chặn video.
   - **UI Main Thread**: Render mượt mà, bounding box hiển thị trực tiếp trên canvas.
6. **Bộ script kiểm thử độc lập `test_anh.py`**:
   - Chạy riêng từng model trên ảnh tĩnh.
   - Hỗ trợ 4 chế độ test camera đêm: Ảnh gốc, Ngưỡng nhạy conf=0.05, Grayscale 3 kênh (hồng ngoại), Tăng tương phản Gamma. Kết quả lưu vào `runs_test/`.

---

## 🏗️ Cấu trúc dự án

```text
fire-smoke-ai/
├── src/
│   ├── __init__.py
│   ├── main.py                     # Entry point chính của ứng dụng
│   ├── config.py                   # CẤU HÌNH TẬP TRUNG TOÀN BỘ HẰNG SỐ VÀ NGƯỠNG
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detection_service.py    # Quản lý 2 model YOLO, OR logic, temporal filter, alert
│   │   └── video_service.py        # Quản lý phát video, đa luồng playback và inference
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main_window.py          # Giao diện người dùng Tkinter Canvas + Pillow
│   └── utils/
│       ├── __init__.py
│       └── video_utils.py          # Tiện ích format timestamp, letterbox, scale bbox
├── models/
│   ├── best.pt                     # Model Cũ (Classes: Fire, Smoke)
│   └── fire_yolo.pt                # Model Mới từ Hugging Face (Classes: fire, smoke)
├── canh_bao/                       # Thư mục tự động lưu ảnh khi kích hoạt báo cháy
├── runs_test/                      # Thư mục lưu ảnh kết quả khi chạy test_anh.py
├── test_anh.py                     # Script kiểm thử độc lập trên ảnh tĩnh
├── download_model.py               # Script tải model tự động từ Hugging Face Hub
├── tests/                          # Bộ 19 unit tests tự động
│   ├── test_config.py
│   ├── test_utils.py
│   ├── test_detection_service.py
│   └── test_video_service.py
├── videos/                         # Thư mục chứa video mẫu
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── .gitignore                      # Loại trừ file rác, virtualenv và cache
└── README.md                       # Tài liệu hướng dẫn sử dụng
```

---

## ⚙️ Cài đặt môi trường

```powershell
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện
pip install -r requirements.txt
```

*(Lưu ý: Yêu cầu `ultralytics >= 8.3.0` và `huggingface_hub >= 0.20.0`).*

---

## 🚀 Hướng dẫn chạy chương trình

### 1. Chạy Ứng Dụng Demo Giao Diện (Camera / Video)

Từ thư mục gốc dự án:

```powershell
python -m src.main
```

- Bấm nút **`📁 Chọn video...`** để chọn video camera từ máy tính.
- Video sẽ phát với tốc độ thực tế mượt mà, khung giao diện cố định.
- Khi một trong 2 model phát hiện lửa đủ 5/10 lần gần nhất:
  - Banner chuyển sang màu đỏ: `CẢNH BÁO: PHÁT HIỆN LỬA`.
  - Console in dòng thông báo `🔥 CẢNH BÁO CHÁY!`.
  - Ảnh chụp frame hiện tại được tự động lưu vào `canh_bao/`.

### 2. Kiểm Thử Độc Lập Trên Ảnh Tĩnh (`test_anh.py`)

Chạy script trên ảnh camera bất kỳ (ví dụ ảnh camera đêm bãi xe):

```powershell
# Chạy với ảnh mặc định:
python test_anh.py

# Hoặc truyền đường dẫn ảnh của bạn:
python test_anh.py path/to/your/night_camera_image.jpg
```

Script sẽ:
1. Chạy Model Cũ (`best.pt`) ở `conf=0.25`.
2. Chạy Model Mới (`fire_yolo.pt`) ở 4 chế độ:
   - [Cách 1] Ảnh gốc ở `conf=0.15`.
   - [Cách 2] Hạ ngưỡng nhạy `conf=0.05`.
   - [Cách 3] Chuyển ảnh sang Grayscale 3 kênh (chế độ hồng ngoại).
   - [Cách 4] Tăng tương phản Gamma (`gamma=2.0`).
3. Lưu toàn bộ ảnh có vẽ bounding box vào thư mục `runs_test/`.

---

## 🔧 Nơi cấu hình và điều chỉnh ngưỡng

Toàn bộ thông số được tập trung tại file [src/config.py](file:///d:/tecnix/fire-smoke-ai/src/config.py):

| Tham số | Giá trị mặc định | Giải thích ý nghĩa |
| :--- | :--- | :--- |
| `MODEL_MOI_CONF` | `0.15` | Ngưỡng confidence cho model mới `fire_yolo.pt` |
| `MODEL_CU_CONF` | `0.25` | Ngưỡng confidence cho model cũ `best.pt` |
| `TEN_LUA` | `{"fire", "flame", "lua", "chay"}` | Danh sách tên nhãn đồng nghĩa của lửa |
| `TEN_KHOI` | `{"smoke", "khoi"}` | Danh sách tên nhãn đồng nghĩa của khói |
| `DUNG_KHOI_DE_BAO` | `False` | Chỉ lửa kích hoạt báo động. Khói chỉ ghi log trừ khi đổi thành `True` |
| `CHI_GOI_MODEL_CU_KHI_MODEL_MOI_KHONG_THAY` | `False` | Chạy OR song song. Nếu đặt `True`: chỉ gọi model cũ khi model mới không thấy lửa (tiết kiệm CPU) |
| `DEQUE_WINDOW_SIZE` | `10` | Kích thước cửa sổ trượt bộ lọc thời gian (10 lần gần nhất) |
| `MIN_FIRE_DETECTIONS_TO_ALERT` | `5` | Số lần tối thiểu cần thấy lửa trong 10 lần gần nhất để báo động |
| `FRAME_STRIDE` | `3` | Tần suất xử lý AI: chỉ xử lý 1 trên N frame (mặc định 1/3) |
| `COOLDOWN_CANH_BAO_SEC` | `30.0` | Thời gian chờ giữa 2 lần lưu ảnh cảnh báo (tránh spam) |

---

## 🧪 Chạy kiểm thử tự động (Unit Tests)

```powershell
python -m unittest discover tests
```
*Đảm bảo 19/19 tests Passed.*
