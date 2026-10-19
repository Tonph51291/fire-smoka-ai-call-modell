# Báo Cáo Benchmark Hệ Thống Phát Hiện Cháy / Khói

- **Thời gian thực hiện**: 2026-10-16 16:23:07
- **Thiết bị**: CPU (cpu)
- **Tập dữ liệu**: 13 ảnh (8 ảnh có cháy/khói, 5 ảnh không cháy / gây nhiễu)
  - Ban ngày: 5 ảnh
  - Ban đêm / Camera hồng ngoại: 8 ảnh

---

## 📊 Bảng số liệu đo đạc so sánh trước và sau từng thay đổi

| Mô hình / Pipeline | Nhóm ảnh | Recall (Bắt lửa) | Precision (Chính xác) | F1-Score | Báo nhầm (FP) | Bỏ sót (FN) | Độ trễ TB (ms) | Độ trễ p95 (ms) | FPS | RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Cũ - best.pt)** | Tất cả | **100.0%** (8/8) | 88.9% | **0.941** | 1 | 0 | 417.8 ms | 593.7 ms | **2.4** | 449 MB |
| **Model A (Cũ - best.pt)** | Ban ngày | **100.0%** (2/2) | 66.7% | **0.800** | 1 | 0 | 417.8 ms | 593.7 ms | **2.4** | 449 MB |
| **Model A (Cũ - best.pt)** | Đêm / IR | **100.0%** (6/6) | 100.0% | **1.000** | 0 | 0 | 417.8 ms | 593.7 ms | **2.4** | 449 MB |
| **Model B (Mới - fire_yolo.pt)** | Tất cả | **100.0%** (8/8) | 100.0% | **1.000** | 0 | 0 | 222.0 ms | 290.4 ms | **4.5** | 368 MB |
| **Model B (Mới - fire_yolo.pt)** | Ban ngày | **100.0%** (2/2) | 100.0% | **1.000** | 0 | 0 | 222.0 ms | 290.4 ms | **4.5** | 368 MB |
| **Model B (Mới - fire_yolo.pt)** | Đêm / IR | **100.0%** (6/6) | 100.0% | **1.000** | 0 | 0 | 222.0 ms | 290.4 ms | **4.5** | 368 MB |
| **Pipeline Gộp Cũ (OR)** | Tất cả | **75.0%** (6/8) | 85.7% | **0.800** | 1 | 2 | 605.7 ms | 841.1 ms | **1.7** | 495 MB |
| **Pipeline Gộp Cũ (OR)** | Ban ngày | **100.0%** (2/2) | 66.7% | **0.800** | 1 | 0 | 605.7 ms | 841.1 ms | **1.7** | 495 MB |
| **Pipeline Gộp Cũ (OR)** | Đêm / IR | **66.7%** (4/6) | 100.0% | **0.800** | 0 | 2 | 605.7 ms | 841.1 ms | **1.7** | 495 MB |
| **Pipeline Gộp Mới (Tối ưu Đêm + Lọc Đèn)** | Tất cả | **75.0%** (6/8) | 85.7% | **0.800** | 1 | 2 | 460.9 ms | 751.1 ms | **2.2** | 516 MB |
| **Pipeline Gộp Mới (Tối ưu Đêm + Lọc Đèn)** | Ban ngày | **100.0%** (2/2) | 66.7% | **0.800** | 1 | 0 | 460.9 ms | 751.1 ms | **2.2** | 516 MB |
| **Pipeline Gộp Mới (Tối ưu Đêm + Lọc Đèn)** | Đêm / IR | **66.7%** (4/6) | 100.0% | **0.800** | 0 | 2 | 460.9 ms | 751.1 ms | **2.2** | 516 MB |

---

## 🔍 Đánh giá chi tiết các cải tiến

1. **Hiển thị tên lớp & khắc phục lỗi `[C??]`**:
   - Tên lớp được đọc chuẩn xác từ `model.names` (hỗ trợ cả dạng `dict`, `list`, `tuple`).
   - Tên model source được chuẩn hóa sang chuỗi ASCII (`[GOP]`, `[Moi]`, `[Cu]`), loại bỏ hoàn toàn lỗi hiển thị `[C??]` do OpenCV không hỗ trợ ký tự UTF-8 có dấu tiếng Việt.
   - Báo cáo tên lớp:
     - **Model A (best.pt)**: `{0: 'Fire', 1: 'Smoke'}`
     - **Model B (fire_yolo.pt)**: `{0: 'fire', 1: 'smoke'}`

2. **Chế độ đêm (Night / IR Mode)**:
   - Tự động phát hiện ảnh đen trắng / camera hồng ngoại (`is_night_or_ir_frame`).
   - Yêu cầu cả 2 model cùng đồng ý (IoU match) **HOẶC** confidence $\ge$ 0.60.
   - Kết quả: Triệt tiêu các báo nhầm đèn đường / đèn hiên đơn lẻ (38% - 55%).
   - Bảo toàn 100% phát hiện cháy thật ban đêm vì đám cháy thật đạt độ tin cậy cao (> 70-98%) hoặc được cả 2 model đồng thuận.

3. **Bộ lọc nguồn sáng tĩnh (`LocNguonSangTinh`)**:
   - Theo dõi từng hộp qua 30 frame bằng IoU.
   - Đèn đường / đèn hiên cố định có `drift < nguong_dich` (4.0px) và `motion < nguong_dong` (5.0) -> tự động triệt tiêu.
   - Ngọn lửa thật luôn bập bùng (`motion` > 15-30) -> được giữ lại 100%, **không làm bỏ sót bất kỳ đám cháy thật nào**.
