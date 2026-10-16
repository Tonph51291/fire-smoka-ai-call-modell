"""Video processing service with decoupled playback and inference threads."""

import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.services.detection_service import DetectionService, FrameAnalysisResult
from src.utils.video_utils import format_timestamp, get_video_metadata

logger = logging.getLogger(__name__)


class PlaybackState(str, Enum):
    """Video player operational states."""
    STOPPED = "STOPPED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"


class VideoService:
    """
    Manages dual-thread architecture:
    1. Playback Thread: Decodes video frames synchronized to monotonic clock.
    2. Inference Thread: Asynchronously runs YOLO detection on the newest available frame.
    """

    def __init__(self, detection_service: Optional[DetectionService] = None) -> None:
        self.detection_service = detection_service
        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.state: PlaybackState = PlaybackState.STOPPED

        # Video metadata
        self.metadata: Dict = {}
        self.fps: float = 25.0
        self.total_frames: int = 0
        self.duration_sec: float = 0.0

        # Position tracking
        self.current_frame_idx: int = 0
        self.current_timestamp_sec: float = 0.0

        # Monotonic clock tracking
        self._play_start_mono: float = 0.0
        self._play_start_video_sec: float = 0.0

        # Threading controls
        self._playback_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._new_frame_event = threading.Event()
        self._seek_lock = threading.Lock()

        # Slot 1: Latest Frame (Single-slot overwrite, zero buffer lag)
        self._frame_lock = threading.Lock()
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_frame_idx: int = -1
        self._latest_frame_ts: float = 0.0

        # Slot 2: Latest Inference Result (Single-slot overwrite)
        self._inference_lock = threading.Lock()
        self._latest_inference: Optional[FrameAnalysisResult] = None
        self._last_inferred_idx: int = -1

        # Real-time metrics
        self.display_fps: float = 0.0
        self.inference_fps: float = 0.0
        self._fps_counter = 0
        self._fps_timer = time.monotonic()
        self._inf_counter = 0
        self._inf_timer = time.monotonic()

        # Callbacks
        self.on_state_change: Optional[Callable[[PlaybackState], None]] = None
        self.on_video_ended: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def open_video(self, video_path: Union[str, Path]) -> bool:
        """Open video file, extract metadata, and prepare player."""
        self.stop()

        path = Path(video_path)
        if not path.exists():
            err = f"File video không tồn tại: {path}"
            logger.error(err)
            if self.on_error:
                self.on_error(err)
            return False

        try:
            self.metadata = get_video_metadata(path)
            self.video_path = path
            self.fps = float(self.metadata.get("fps", 25.0))
            self.total_frames = int(self.metadata.get("total_frames", 0))
            self.duration_sec = float(self.metadata.get("duration_sec", 0.0))
            self.current_frame_idx = 0
            self.current_timestamp_sec = 0.0

            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise RuntimeError(f"Không thể mở video stream từ {self.video_path.name}")

            logger.info(
                f"Đã mở video: {self.video_path.name} "
                f"({self.metadata.get('width')}x{self.metadata.get('height')}, "
                f"{self.fps:.1f} FPS, {self.metadata.get('duration_str')})"
            )
            return True

        except Exception as e:
            err = f"Lỗi khi đọc video {path.name}: {e}"
            logger.error(err)
            if self.on_error:
                self.on_error(err)
            self._cleanup_cap()
            return False

    def play(self) -> None:
        """Start or resume playback."""
        if not self.video_path or not self.cap:
            return

        if self.state == PlaybackState.PLAYING:
            return

        if self.state == PlaybackState.PAUSED:
            self._play_start_mono = time.monotonic()
            self._play_start_video_sec = self.current_timestamp_sec
            self._pause_event.clear()
            self._set_state(PlaybackState.PLAYING)
            return

        # Starting from STOPPED
        self._stop_event.clear()
        self._pause_event.clear()
        self._play_start_mono = time.monotonic()
        self._play_start_video_sec = self.current_timestamp_sec
        self._set_state(PlaybackState.PLAYING)

        # 1. Start Playback thread
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

        # 2. Start Inference thread (if detection service available)
        if self.detection_service:
            self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
            self._inference_thread.start()

    def pause(self) -> None:
        """Pause playback."""
        if self.state == PlaybackState.PLAYING:
            self._pause_event.set()
            self._set_state(PlaybackState.PAUSED)

    def resume(self) -> None:
        """Resume playback."""
        self.play()

    def stop(self) -> None:
        """Stop playback and reset to frame 0."""
        self._stop_event.set()
        self._pause_event.clear()
        self._new_frame_event.set()  # Wake up inference thread

        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=0.8)
        self._playback_thread = None

        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=0.8)
        self._inference_thread = None

        with self._seek_lock:
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.current_timestamp_sec = 0.0

        with self._frame_lock:
            self._latest_raw_frame = None
            self._latest_frame_idx = -1
            self._latest_frame_ts = 0.0

        with self._inference_lock:
            self._latest_inference = None
            self._last_inferred_idx = -1

        self._set_state(PlaybackState.STOPPED)

    def seek_frame(self, target_frame_idx: int) -> None:
        """Seek video to target frame and decode preview."""
        with self._seek_lock:
            if self.cap and self.cap.isOpened() and self.total_frames > 0:
                target_frame = max(0, min(self.total_frames - 1, target_frame_idx))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self.current_frame_idx = target_frame
                self.current_timestamp_sec = target_frame / self.fps if self.fps > 0 else 0.0

                # Read preview frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._frame_lock:
                        self._latest_raw_frame = frame
                        self._latest_frame_idx = self.current_frame_idx
                        self._latest_frame_ts = self.current_timestamp_sec

                # Reset clock base
                self._play_start_mono = time.monotonic()
                self._play_start_video_sec = self.current_timestamp_sec

    def get_latest_data(self) -> Tuple[Optional[np.ndarray], int, float, Optional[FrameAnalysisResult]]:
        """Non-blocking retrieval of latest frame and newest AI results for UI rendering."""
        with self._frame_lock:
            frame = self._latest_raw_frame
            idx = self._latest_frame_idx
            ts = self._latest_frame_ts

        with self._inference_lock:
            inf = self._latest_inference

        # Update display FPS counter
        self._fps_counter += 1
        now = time.monotonic()
        if now - self._fps_timer >= 1.0:
            self.display_fps = self._fps_counter / (now - self._fps_timer)
            self._fps_counter = 0
            self._fps_timer = now

        return frame, idx, ts, inf

    def close(self) -> None:
        """Cleanly terminate all threads and release video device."""
        self.stop()
        self._cleanup_cap()

    def _cleanup_cap(self) -> None:
        """Safely release VideoCapture."""
        with self._seek_lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.warning(f"Lỗi khi release VideoCapture: {e}")
                self.cap = None

    def _set_state(self, new_state: PlaybackState) -> None:
        """Update state and notify listener."""
        self.state = new_state
        if self.on_state_change:
            try:
                self.on_state_change(new_state)
            except Exception as e:
                logger.error(f"Lỗi trong on_state_change callback: {e}")

    # ==========================================
    # THREAD 1: PLAYBACK WORKER (Real-time Clock Sync)
    # ==========================================
    def _playback_worker(self) -> None:
        """Reads frames synchronized strictly to real-time monotonic clock."""
        logger.info("Luồng phát video (Playback Thread) đã khởi chạy.")
        frame_interval = 1.0 / self.fps if self.fps > 0 else 0.04

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.02)
                continue

            # Calculate where in the video we should be right now
            now = time.monotonic()
            target_video_sec = self._play_start_video_sec + (now - self._play_start_mono)
            target_frame_no = int(target_video_sec * self.fps)

            # If video has ended
            if self.total_frames > 0 and target_frame_no >= self.total_frames:
                logger.info("Video đã phát đến khung hình cuối.")
                self._set_state(PlaybackState.STOPPED)
                if self.on_video_ended:
                    try:
                        self.on_video_ended()
                    except Exception as e:
                        logger.error(f"Lỗi callback on_video_ended: {e}")
                break

            with self._seek_lock:
                if not self.cap or not self.cap.isOpened():
                    break

                # Frame skipping using cap.grab() if behind schedule by >= 2 frames
                current_cap_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                frames_behind = target_frame_no - current_cap_pos

                if frames_behind > 1:
                    # Skip up to frames_behind - 1 frames quickly without full decoding
                    for _ in range(min(frames_behind - 1, 10)):
                        if not self.cap.grab():
                            break

                # Read and decode the current target frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    self.current_timestamp_sec = (
                        self.current_frame_idx / self.fps if self.fps > 0 else 0.0
                    )

            if not ret or frame is None:
                logger.info("Không đọc được frame tiếp theo, video kết thúc.")
                self._set_state(PlaybackState.STOPPED)
                if self.on_video_ended:
                    try:
                        self.on_video_ended()
                    except Exception as e:
                        logger.error(f"Lỗi callback on_video_ended: {e}")
                break

            # Store frame into shared slot (single-slot overwrite)
            with self._frame_lock:
                self._latest_raw_frame = frame
                self._latest_frame_idx = self.current_frame_idx
                self._latest_frame_ts = self.current_timestamp_sec

            # Signal inference thread that a new frame is ready
            self._new_frame_event.set()

            # Synchronize sleep to real-time clock
            expected_next_mono = self._play_start_mono + (
                (self.current_frame_idx + 1) / self.fps - self._play_start_video_sec
            )
            sleep_needed = expected_next_mono - time.monotonic()
            if sleep_needed > 0.001:
                time.sleep(sleep_needed)

        logger.info("Luồng phát video (Playback Thread) đã kết thúc.")

    # ==========================================
    # THREAD 2: INFERENCE WORKER (Async YOLO Detection)
    # ==========================================
    def _inference_worker(self) -> None:
        """Asynchronously pulls latest available frame and runs YOLO inference."""
        logger.info("Luồng AI (Inference Thread) đã khởi chạy.")

        while not self._stop_event.is_set():
            # Wait until signaled by playback thread or timeout
            signaled = self._new_frame_event.wait(timeout=0.05)
            self._new_frame_event.clear()

            if self._stop_event.is_set():
                break

            if self._pause_event.is_set():
                time.sleep(0.02)
                continue

            # Grab the absolute latest frame available
            with self._frame_lock:
                frame_to_process = self._latest_raw_frame
                idx_to_process = self._latest_frame_idx
                ts_to_process = self._latest_frame_ts

            if frame_to_process is None or idx_to_process == self._last_inferred_idx:
                time.sleep(0.01)
                continue

            # Run detection
            try:
                result = self.detection_service.detect_frame(
                    frame=frame_to_process,
                    frame_index=idx_to_process,
                    timestamp_sec=ts_to_process
                )

                # Store newest result
                with self._inference_lock:
                    self._latest_inference = result
                    self._last_inferred_idx = idx_to_process

                # Update inference FPS metrics
                self._inf_counter += 1
                now = time.monotonic()
                if now - self._inf_timer >= 1.0:
                    self.inference_fps = self._inf_counter / (now - self._inf_timer)
                    self._inf_counter = 0
                    self._inf_timer = now

            except Exception as e:
                logger.error(f"Lỗi trong inference worker: {e}", exc_info=True)
                time.sleep(0.02)

        logger.info("Luồng AI (Inference Thread) đã kết thúc.")
