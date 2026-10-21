import cv2
import sys
import logging
from pathlib import Path

# Setup path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.services.detection_service import DetectionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    service = DetectionService()
    video_path = str(ROOT / "videos" / "1.mp4")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    frame_idx = 0
    print("=== PROBING 1.mp4 AROUND 50s - 105s ===")
    
    # Target windows around user's alerts: 55-60s, 70-75s, 84-88s, 97-101s
    target_times = [(55.0, 60.0), (70.0, 75.0), (84.0, 88.0), (97.0, 101.0)]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        ts = frame_idx / fps
        
        # Check if in any target window
        in_window = any(start <= ts <= end for start, end in target_times)
        
        # We must process every frame so state/trackers are updated accurately
        # But we print details when in target windows or when alert triggers
        last_alert_before = service.last_alert_time
        res = service.detect_frame(frame, frame_idx, ts)
        
        # Did an alert trigger?
        alert_fired = (service.last_alert_time != last_alert_before)
        
        if alert_fired or (in_window and frame_idx % 5 == 0):
            print(f"\n[FRAME {frame_idx} | {ts:.2f}s] AlertFired={alert_fired} has_fire={res.has_fire} num_dets={len(res.detections)}")
            for d in res.detections:
                ts_stat = getattr(d, 'threat_status', 'none')
                track_st = getattr(d, 'track_state', 'none')
                is_static = getattr(d, 'is_static_suspect', False)
                print(f"   -> cls={d.class_name} conf={d.confidence:.3f} bbox={[int(x) for x in d.bbox]} "
                      f"threat_st={ts_stat} track_st={track_st} is_static={is_static} "
                      f"flare_votes={getattr(d, 'flare_votes', 0)} model={d.model_source}")
                      
        frame_idx += 1
        if ts > 102.0:
            break
            
    cap.release()

if __name__ == "__main__":
    main()
