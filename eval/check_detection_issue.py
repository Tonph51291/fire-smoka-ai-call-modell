from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

model_cu = YOLO('models/best.pt')
model_moi = YOLO('models/fire_yolo.pt')

all_images = list(Path('images').glob('*.jpg')) + list(Path('canh_bao').glob('*.jpg'))
print(f'Testing {len(all_images)} images...')

for img_p in all_images:
    img = cv2.imread(str(img_p))
    if img is None:
        continue
    print(f'\n--- {img_p.name} ({img.shape[1]}x{img.shape[0]}) ---')

    with torch.inference_mode():
        r_cu_416 = model_cu.predict(source=img, imgsz=416, conf=0.15, verbose=False)
        r_cu_640 = model_cu.predict(source=img, imgsz=640, conf=0.15, verbose=False)
        r_cu_640_05 = model_cu.predict(source=img, imgsz=640, conf=0.05, verbose=False)

        r_moi_416 = model_moi.predict(source=img, imgsz=416, conf=0.15, verbose=False)
        r_moi_640 = model_moi.predict(source=img, imgsz=640, conf=0.15, verbose=False)
        r_moi_640_05 = model_moi.predict(source=img, imgsz=640, conf=0.05, verbose=False)

    print(f"Model Cu  [416, conf=0.15]: {[model_cu.names[int(b.cls[0])] + f':{float(b.conf[0]):.2f}' for b in r_cu_416[0].boxes]}")
    print(f"Model Cu  [640, conf=0.15]: {[model_cu.names[int(b.cls[0])] + f':{float(b.conf[0]):.2f}' for b in r_cu_640[0].boxes]}")
    print(f"Model Cu  [640, conf=0.05]: {[model_cu.names[int(b.cls[0])] + f':{float(b.conf[0]):.2f}' for b in r_cu_640_05[0].boxes]}")

    print(f"Model Moi [416, conf=0.15]: {[model_moi.names[int(b.cls[0])] + f':{float(b.conf[0]):.2f}' for b in r_moi_416[0].boxes]}")
    print(f"Model Moi [640, conf=0.15]: {[model_moi.names[int(b.cls[0])] + f':{float(b.conf[0]):.2f}' for b in r_moi_640[0].boxes]}")
    print(f"Model Moi [640, conf=0.05]: {[model_moi.names[int(b.cls[0])] + f':{float(b.conf[0]):.2f}' for b in r_moi_640_05[0].boxes]}")
