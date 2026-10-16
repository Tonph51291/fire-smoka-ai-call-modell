# Ứng Dụng AI Phát Hiện Cháy & Khói Từ Video Camera (Fire & Smoke AI)

Ứng dụng Desktop chạy cục bộ (local) trên Windows sử dụng Python, Tkinter, OpenCV và mô hình Ultralytics YOLO để phát hiện và cảnh báo Lửa (Fire) và Khói (Smoke) từ video camera theo thời gian thực.

---

## 📌 Tính năng nổi bật

1. **Chọn video camera từ máy tính**: Hỗ trợ các định dạng phổ biến như `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`.
2. **Phát video mượt mà trên giao diện**: Hiển thị frame trực tiếp trên canvas Tkinter (sử dụng Pillow), không phụ thuộc vào cửa sổ `cv2.imshow()`.
3. **Phân tích AI theo thời gian thực**: Sử dụng model YOLO weights (`best.pt`) nhận diện các đối tượng `Fire` và `Smoke`.
4. **Vẽ Bounding Box trực quan**: Bao quanh vùng lửa/khói, hiển thị nhãn và độ tin cậy (confidence %) rõ nét.
5. **Khu vực Cảnh Báo Trực Tiếp (Live Alert Banner)**:
   - 🟢 **Trạng thái 1**: `CHƯA PHÁT HIỆN LỬA HOẶC KHÓI`
   - 🔴 **Trạng thái 2**: `CẢNH BÁO: PHÁT HIỆN LỬA` kèm độ tin cậy và mốc thời gian.
   - 🟠 **Trạng thái 3**: `CẢNH BÁO: PHÁT HIỆN KHÓI` kèm độ tin cậy và mốc thời gian.
   - 🚨 **Trạng thái 4**: `CẢNH BÁO: PHÁT HIỆN LỬA VÀ KHÓI` kèm độ tin cậy chi tiết của cả hai.
6. **Bộ điều khiển video đầy đủ**: Nút **Phát (Play)**, **Tạm dừng (Pause)**, **Tiếp tục (Resume)**, **Dừng (Stop)** và thanh trượt tiến trình (Timeline).
7. **Điều chỉnh độ nhạy (Confidence Threshold)**: Thanh trượt linh hoạt từ 10% đến 90% (mặc định 25%) để kiểm thử các tình huống nhạy cảm hoặc chống báo động giả.
8. **Đa luồng chống treo giao diện**: Quá trình đọc frame và AI inference chạy trên luồng nền (`threading.Thread`), kết hợp hàng đợi giới hạn (`queue.Queue`) giúp giao diện luôn mượt mà.

---

## 🏗️ Cấu trúc dự án

```text
fire-smoke-ai/
├── src/
│   ├── __init__.py
│   ├── main.py                     # Entry point chính của ứng dụng
│   ├── config.py                   # Cấu hình đường dẫn, hằng số, bảng màu và trạng thái cảnh báo
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detection_service.py    # Quản lý model YOLO, phân tích frame và vẽ bounding box
│   │   └── video_service.py        # Quản lý phát video, luồng worker và đồng bộ FPS
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main_window.py          # Giao diện người dùng Tkinter + Pillow
│   └── utils/
│       ├── __init__.py
│       └── video_utils.py          # Tiện ích định dạng thời gian, metadata, resize frame
├── models/
│   └── best.pt                     # Trọng số model YOLO (Classes: Fire, Smoke)
├── tests/                          # Bộ kiểm thử tự động (Unit tests)
│   ├── __init__.py
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

## ⚙️ Yêu cầu môi trường

- **Hệ điều hành**: Windows 10 hoặc Windows 11.
- **Python**: Phiên bản 3.9 trở lên (đã kiểm thử ổn định trên Python 3.10 / 3.11).
- **Phần cứng**: Chạy tốt trên CPU; nếu có GPU NVIDIA (CUDA), tốc độ nhận diện sẽ nhanh hơn.

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### 1. Tạo và kích hoạt Virtual Environment (Khuyến nghị)

Mở PowerShell tại thư mục gốc của dự án (`d:\tecnix\fire-smoke-ai`):

```powershell
# Tạo môi trường ảo .venv
python -m venv .venv

# Kích hoạt môi trường ảo trên Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

*(Nếu gặp lỗi PowerShell ExecutionPolicy, chạy: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi thử lại).*

### 2. Cài đặt các thư viện cần thiết

```powershell
pip install -r requirements.txt
```

*(Lưu ý: Nếu sử dụng model YOLO11 mới, yêu cầu `ultralytics >= 8.3.0`).*

### 3. Vị trí đặt file trọng số (Model Weights)

File model mặc định phải được đặt tại:
```text
models/best.pt
```
*(Nếu bạn dùng script [download_model.py](file:///d:/tecnix/fire-smoke-ai/download_model.py), model sẽ được tự động tải về thư mục này).*

### 4. Khởi chạy ứng dụng

Từ thư mục gốc của dự án, chạy lệnh:

```powershell
python -m src.main
```

---

## 🖥️ Hướng dẫn sử dụng giao diện

1. **Chọn video**: Bấm nút **`📁 Chọn video...`** ở thanh trên cùng và chọn một file video (ví dụ: `videos/sample_demo.mp4` hoặc video camera của bạn).
2. **Phát / Tạm dừng**:
   - Nhấn **`▶ Phát`** để bắt đầu phân tích video. Nút sẽ tự chuyển sang **`⏸ Tạm dừng`**.
   - Nhấn lại để tạm dừng hoặc tiếp tục.
3. **Dừng video**: Nhấn **`⏹ Dừng`** để dừng hẳn và đưa video về vị trí bắt đầu (giây 00:00).
4. **Tua video**: Kéo thanh trượt timeline bên dưới để chuyển nhanh đến đoạn video mong muốn.
5. **Điều chỉnh độ nhạy (Confidence)**:
   - Kéo thanh trượt **Ngưỡng tin cậy** ở góc dưới (từ 10% đến 90%).
   - Tăng ngưỡng để giảm thiểu cảnh báo giả.
   - Giảm ngưỡng nếu muốn phát hiện cả những tia lửa hoặc khói mờ ở xa.
6. **Theo dõi Cảnh Báo**: Banner cảnh báo ở đầu màn hình sẽ ngay lập tức đổi màu (Xanh lá -> Đỏ / Cam / Đỏ thẫm) kèm thông tin chi tiết khi AI phát hiện lửa hoặc khói.

---

## 🧪 Chạy kiểm thử tự động (Unit Tests)

Dự án đi kèm bộ test đầy đủ kiểm tra các module độc lập:

```powershell
python -m unittest discover tests
```

---

## ❓ Các lỗi thường gặp và cách xử lý

1. **Lỗi `AttributeError: Can't get attribute 'C3k2'` hoặc `C3k`**:
   - *Nguyên nhân*: File `best.pt` được huấn luyện bằng kiến trúc YOLO11 nhưng máy đang cài bản `ultralytics` cũ (< 8.3.0).
   - *Cách khắc phục*: Nâng cấp ultralytics bằng lệnh `pip install --upgrade ultralytics`.

2. **Lỗi `Không tìm thấy model weights tại models/best.pt`**:
   - *Cách khắc phục*: Đảm bảo bạn đã đặt file trọng số `best.pt` vào thư mục `models/` hoặc chạy `python download_model.py`.

3. **Lỗi không mở được file video**:
   - *Nguyên nhân*: Video bị hỏng hoặc thiếu bộ giải mã codec trên Windows.
   - *Cách khắc phục*: Thử mở video bằng file `.mp4` chuẩn H.264 hoặc dùng video mẫu được tạo sẵn tại `videos/sample_demo.mp4`.
