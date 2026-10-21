import cv2, os, sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
from src.config import DEFAULT_MODEL_PATH, DEFAULT_CONF_THRESHOLD
from src.services.detection_service import DetectionService

svc = DetectionService(model_path=DEFAULT_MODEL_PATH, conf_threshold=DEFAULT_CONF_THRESHOLD)
cap = cv2.VideoCapture('videos/4.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

cap.set(cv2.CAP_PROP_POS_FRAMES, int(4.0 * fps))
for f_idx in range(int(4.0 * fps), int(8.0 * fps)):
    ret, frame = cap.read()
    if not ret: break
    sec = f_idx / fps
    res = svc.detect_frame(frame, timestamp_sec=sec, frame_index=f_idx)
    if res.detections or res.alert_state == 'FIRE':
        det_strs = [f"{d.class_name}:{d.confidence:.2f}:{[int(x) for x in d.bbox]}:votes={d.n_votes}:state={d.track_state}" for d in res.detections]
        print(f"{sec:.2f}s (f={f_idx}): alert={res.alert_state} conf={res.is_alert_confirmed} dets={det_strs}")
cap.release()
