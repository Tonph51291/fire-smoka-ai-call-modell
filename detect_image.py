
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
IMAGE_PATH = "images/images.jpg"

CONF_THRESHOLD = 0.25

# Kiểm tra file
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Không tìm thấy ảnh: {IMAGE_PATH}")

# Nạp model AI
model = YOLO(MODEL_PATH)

# Nhận diện ảnh
results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=CONF_THRESHOLD,
    save=True,
    project="runs",
    name="fire_smoke_image"
)

# Phân loại kết quả
detected_fire = False
detected_smoke = False

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = result.names[class_id].lower()

        print(f"Phát hiện: {label} | Độ tin cậy: {confidence:.2%}")

        if label == "fire":
            detected_fire = True
        elif label == "smoke":
            detected_smoke = True

# Kết luận cuối cùng
print("\n========== KẾT QUẢ PHÂN TÍCH ==========")

if detected_fire and detected_smoke:
    print("⚠️ CẢNH BÁO: PHÁT HIỆN CẢ LỬA VÀ KHÓI!")
elif detected_fire:
    print("🔥 CẢNH BÁO: PHÁT HIỆN LỬA!")
elif detected_smoke:
    print("🌫️ CẢNH BÁO: PHÁT HIỆN KHÓI!")
else:
    print("✅ KHÔNG PHÁT HIỆN LỬA HOẶC KHÓI.")

print("\nẢnh kết quả nằm trong thư mục: runs/fire_smoke_image/")
