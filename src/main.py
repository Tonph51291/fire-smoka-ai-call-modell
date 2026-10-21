"""Điểm khởi chạy chính của ứng dụng AI Phát Hiện Lửa & Khói (Desktop Tkinter GUI).
Khởi tạo giao diện, kết nối VideoService và DetectionService, xử lý sự kiện đóng ứng dụng.
"""

import logging
import sys
import tkinter as tk
from tkinter import messagebox

from src.config import DEFAULT_CONF_THRESHOLD, DEFAULT_MODEL_PATH
from src.services.detection_service import DetectionService
from src.services.video_service import VideoService
from src.ui.main_window import MainWindow

# Configure clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main")


def main() -> None:
    """Initialize core services and launch the GUI."""
    logger.info("Khởi động ứng dụng AI Fire & Smoke Detection...")

    # Initialize Tkinter root window
    root = tk.Tk()

    # Initialize AI Detection Service
    detection_service = None
    try:
        detection_service = DetectionService(
            model_path=DEFAULT_MODEL_PATH,
            conf_threshold=DEFAULT_CONF_THRESHOLD
            
        )
        logger.info(f"Đã nạp model AI thành công từ: {DEFAULT_MODEL_PATH}")
    except FileNotFoundError as fnf_err:
        logger.warning(f"Cảnh báo thiếu model: {fnf_err}")
        messagebox.showwarning(
            "Cảnh báo thiếu Model",
            f"Không tìm thấy file trọng số tại:\n{DEFAULT_MODEL_PATH}\n\n"
            "Ứng dụng vẫn sẽ mở để bạn xem video, nhưng tính năng nhận diện AI sẽ tạm tắt.\n"
            "Vui lòng đặt file weights 'best.pt' vào thư mục 'models/' để kích hoạt AI."
        )
    except Exception as e:
        logger.error(f"Lỗi khi nạp model AI: {e}")
        messagebox.showerror(
            "Lỗi nạp Model",
            f"Không thể nạp model AI:\n{e}\n\n"
            "Vui lòng kiểm tra phiên bản ultralytics và tính toàn vẹn của file trọng số."
        )

    # Initialize Video Service with attached Detection Service
    video_service = VideoService(detection_service=detection_service)

    # Create Main Application Window
    app = MainWindow(
        root=root,
        detection_service=detection_service,
        video_service=video_service
    )

    # Start Tkinter main event loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu dừng từ bàn phím (Ctrl+C).")
        app.on_close()
    except Exception as e:
        logger.critical(f"Lỗi không xác định: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
