
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.25

# Kiểm tra model
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

# Nạp model AI
print("Đang tải model AI...")
model = YOLO(MODEL_PATH)

# Mở webcam mặc định
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Không mở được camera. Hãy kiểm tra quyền camera "
        "hoặc thử đổi VideoCapture(0) thành VideoCapture(1)."
    )

print("Camera đã mở. Nhấn Q để thoát.")

try:
    while True:
        # Đọc một khung hình từ camera
        success, frame = camera.read()

        if not success:
            print("Không đọc được hình ảnh từ camera.")
            break

        # Đưa khung hình vào model AI
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        detected_fire = False
        detected_smoke = False

        # Vẽ khung nhận diện lên hình ảnh
        result = results[0]
        display_frame = result.plot()

        # Kiểm tra các đối tượng được phát hiện
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = str(result.names[class_id]).lower()

            if label == "fire":
                detected_fire = True
            elif label == "smoke":
                detected_smoke = True

            print(
                f"\rPhát hiện: {label} | "
                f"Độ tin cậy: {confidence:.2%}    ",
                end=""
            )

        # Hiển thị cảnh báo trên màn hình
        if detected_fire and detected_smoke:
            message = "CANH BAO: PHAT HIEN LUA VA KHOI!"
            color = (0, 0, 255)
        elif detected_fire:
            message = "CANH BAO: PHAT HIEN LUA!"
            color = (0, 0, 255)
        elif detected_smoke:
            message = "CANH BAO: PHAT HIEN KHOI!"
            color = (0, 165, 255)
        else:
            message = "Chua phat hien lua hoac khoi"
            color = (0, 200, 0)

        cv2.putText(
            display_frame,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

        # Hiển thị webcam
        cv2.imshow("AI Fire and Smoke Detection", display_frame)

        # Nhấn Q để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("\nĐã đóng camera.")
