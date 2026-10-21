import cv2
import numpy as np

def extract_timeline_frames():
    cap = cv2.VideoCapture("videos/1.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Lưu frame tại các mốc giây: 28, 30, 32, 35, 40, 45, 50, 55, 60
    for sec in [28, 30, 32, 35, 40, 45, 50, 55, 60]:
        f_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            out_name = f"frame_{sec}s.jpg"
            cv2.imwrite(out_name, frame)
            print(f"Saved {out_name} (frame {f_idx})")
            
    cap.release()

if __name__ == "__main__":
    extract_timeline_frames()
