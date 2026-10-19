"""Phát hiện lửa và khói theo thời gian thực từ Webcam / RTSP sử dụng pipeline AI Gộp (best.pt + fire_yolo.pt)."""

import argparse
import sys
import time
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
    DEFAULT_CAMERA_SOURCE,
    DEFAULT_CONF_THRESHOLD,
    FRAME_STRIDE,
    MODEL_CU_PATH,
    MODEL_MOI_PATH,
    STATE_BOTH,
    STATE_FIRE,
    STATE_SMOKE,
    TEN_KHOI,
    TEN_LUA,
)
from src.services.detection_service import DetectionService


def run_camera(camera_source=DEFAULT_CAMERA_SOURCE, conf_thresh: float = DEFAULT_CONF_THRESHOLD) -> None:
    print("=" * 65)
    print("🎥 AI FIRE & SMOKE DETECTION - REAL-TIME CAMERA")
    print(f"   Nguồn Camera: {camera_source}")
    print(f"   Model Cũ: {MODEL_CU_PATH.name} | Model Mới: {MODEL_MOI_PATH.name}")
    print(f"   Ngưỡng confidence: {conf_thresh:.1%}")
    print("=" * 65)

    service = DetectionService(conf_threshold=conf_thresh)

    # Chuyển đổi camera source nếu là chuỗi số
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)

    camera = cv2.VideoCapture(camera_source)
    if not camera.isOpened():
        print(f"[!] Không mở được camera tại nguồn: {camera_source}")
        sys.exit(1)

    print("\n[+] Camera đã kết nối. Nhấn 'q' để thoát.\n")

    frame_idx = 0
    fps_timer = time.time()
    fps_counter = 0
    current_fps = 0.0

    last_analysis = None

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("[!] Mất tín hiệu hình ảnh từ camera.")
                break

            frame_idx += 1
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                current_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # Chạy AI theo FRAME_STRIDE
            if frame_idx % FRAME_STRIDE == 0 or last_analysis is None:
                last_analysis = service.detect_frame(
                    frame=frame,
                    frame_index=frame_idx,
                    timestamp_sec=frame_idx / 25.0
                )

            display_frame = frame.copy()

            # Vẽ bounding boxes
            if last_analysis and last_analysis.detections:
                for det in last_analysis.detections:
                    cname_lower = det.class_name.lower()
                    x1, y1, x2, y2 = det.bbox

                    source_tag = {"gop": "GỘP", "moi": "Mới", "cu": "Cũ"}.get(det.model_source, det.model_source)
                    if cname_lower in TEN_LUA:
                        color = CV_COLORS["fire"]
                        label = f"LUA [{source_tag}]: {det.confidence:.1%}"
                    elif cname_lower in TEN_KHOI:
                        color = CV_COLORS["smoke"]
                        label = f"KHOI [{source_tag}]: {det.confidence:.1%}"
                    else:
                        color = CV_COLORS["default"]
                        label = f"{det.class_name.upper()} [{source_tag}]: {det.confidence:.1%}"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(
                        display_frame,
                        label,
                        (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

            # Vẽ thanh trạng thái cảnh báo phía trên
            state = last_analysis.alert_state if last_analysis else "NORMAL"
            if state == STATE_BOTH:
                msg = f"CANH BAO: PHAT HIEN LUA VA KHOI! (FPS: {current_fps:.1f})"
                bar_color = (0, 0, 200)
            elif state == STATE_FIRE:
                msg = f"CANH BAO: PHAT HIEN LUA! (FPS: {current_fps:.1f})"
                bar_color = (0, 0, 255)
            elif state == STATE_SMOKE:
                msg = f"CANH BAO: PHAT HIEN KHOI! (FPS: {current_fps:.1f})"
                bar_color = (0, 140, 255)
            else:
                msg = f"He thong an toan - Khong co lua/khoi (FPS: {current_fps:.1f})"
                bar_color = (30, 180, 50)

            # Vẽ banner cảnh báo trên đầu khung hình
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 45), bar_color, -1)
            cv2.putText(
                display_frame,
                msg,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("AI Fire & Smoke Detection (Dual-Model)", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\n[+] Đã đóng camera an toàn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera giám sát phát hiện cháy/khói thời gian thực")
    parser.add_argument("--source", default=str(DEFAULT_CAMERA_SOURCE), help="Chỉ số webcam (0, 1) hoặc đường dẫn RTSP")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Ngưỡng confidence (0.0 - 1.0)")
    args = parser.parse_args()

    run_camera(camera_source=args.source, conf_thresh=args.conf)
