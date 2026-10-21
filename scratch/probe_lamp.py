import cv2
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.services.detection_service import DetectionService

logging.basicConfig(level=logging.WARNING)

def main():
    service = DetectionService()
    video_path = str(ROOT / "videos" / "1.mp4")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    frame_idx = 0
    # Let's fast forward to 50s
    start_sec = 50.0
    end_sec = 100.0
    
    print(f"Running probe from {start_sec}s to {end_sec}s...", flush=True)
    
    stride = 2
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        ts = frame_idx / fps
        
        if frame_idx % stride == 0:
            last_alert = service.last_alert_time
            res = service.detect_frame(frame=frame, timestamp_sec=ts, frame_index=frame_idx)
            alert_fired = (service.last_alert_time != last_alert)
            
            target_check = any(abs(ts - t) < 0.25 for t in [57.93, 72.07, 86.07, 99.23])
            
            if alert_fired or target_check:
                print(f"\n>>> [FRAME {frame_idx:04d} | {ts:6.2f}s] AlertFired={alert_fired} | has_fire={res.has_fire} | max_conf={res.max_fire_conf:.1%} | ratio={res.fire_ratio_in_window}", flush=True)
                for i, d in enumerate(res.detections):
                    print(f"    det[{i}]: cls={d.class_name} conf={d.confidence:.3f} bbox={[int(x) for x in d.bbox]} "
                          f"threat_st={getattr(d, 'threat_status', 'none')} track_id={getattr(d, 'track_id', -1)} "
                          f"track_st={getattr(d, 'track_state', 'none')} e_score={getattr(d, 'evidence_score', 0):.1f} "
                          f"static_suspect={getattr(d, 'is_static_suspect', False)}", flush=True)
                          
                if service.threat_analyzer and service.threat_analyzer.static_filter:
                    mem = service.threat_analyzer.static_filter.static_memory
                    if mem:
                        print(f"    StaticMemory: {[f'id={k}, bbox={v.bbox}' for k, v in mem.items()]}", flush=True)
                        
                if service.threat_analyzer:
                    lft = service.threat_analyzer.light_flare_trackers
                    unlocked_lft = {k: v for k, v in lft.items() if v.is_unlocked}
                    if unlocked_lft:
                        for fid, ft in unlocked_lft.items():
                            print(f"      UNLOCKED FlareTracker[{fid}]: bbox={ft.bbox} reason={ft.unlock_reason}", flush=True)
                        
        frame_idx += 1
        if ts > end_sec:
            break
            
    cap.release()
    print("Done probe.", flush=True)

if __name__ == "__main__":
    main()
