import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
from src.config import DEFAULT_MODEL_PATH, DEFAULT_CONF_THRESHOLD
from src.services.detection_service import DetectionService

detection_service = DetectionService(
    model_path=DEFAULT_MODEL_PATH,
    conf_threshold=DEFAULT_CONF_THRESHOLD
)

cap = cv2.VideoCapture('videos/1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

print('--- Model Inferences (Detector & DetectionService) 28s - 50s ---')
for frame_idx in range(int(28 * fps), int(51 * fps), int(fps)):
    sec = frame_idx / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Test raw detector inside service
    raw_p = detection_service.model_detector.predict_primary(frame, conf=0.08)
    raw_l = detection_service.model_detector.predict_lowlight(frame, conf=0.08)
    raw_o = detection_service.model_detector.predict_onnx(frame, conf=0.08)
    raw_dets = raw_p + raw_l + raw_o
    raw_str = [f"{d.model_name}:{d.class_name}:{d.confidence:.2f} ({int(d.bbox[0])},{int(d.bbox[1])},{int(d.bbox[2])},{int(d.bbox[3])})" for d in raw_dets]
    
    # 2. Test full detection service
    res = detection_service.detect_frame(frame, timestamp_sec=sec, frame_index=frame_idx)
    det_str = [f"ID{d.track_id}:{d.class_name} E={d.evidence_score:.1f} state={d.track_state} conf={d.confidence:.2f} ({int(d.bbox[0])},{int(d.bbox[1])},{int(d.bbox[2])},{int(d.bbox[3])})" for d in res.detections]
    
    print(f"Sec {sec:.1f}s | alert={res.alert_state} (fire={res.has_fire}) | Raw ({len(raw_dets)}): {raw_str} | Dets ({len(res.detections)}): {det_str}")


