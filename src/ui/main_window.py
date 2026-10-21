"""Giao diện người dùng Tkinter cho ứng dụng AI Phát Hiện Lửa & Khói.
Hiển thị khung hình letterbox, bounding box cảnh báo, thanh điều khiển video,
thanh trượt độ nhạy và bảng chỉ số phân tích trực tiếp.
"""

import logging
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional
import numpy as np
from PIL import ImageTk

from src.config import (
    ALERT_MESSAGES,
    ALERT_SMOOTHING_FRAMES,
    BOX_PERSISTENCE_SEC,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_DISPLAY_HEIGHT,
    DEFAULT_DISPLAY_WIDTH,
    DEVICE_NAME,
    STATE_BOTH,
    STATE_FIRE,
    STATE_NORMAL,
    STATE_SMOKE,
    SUPPORTED_VIDEO_EXTENSIONS,
    UI_COLORS,
    UI_POLL_INTERVAL_MS,
)
from src.services.detection_service import DetectionService, FrameAnalysisResult
from src.services.video_service import PlaybackState, VideoService
from src.utils.video_utils import (
    bgr_to_pil,
    format_timestamp,
    resize_and_letterbox,
)

logger = logging.getLogger(__name__)


class MainWindow:
    """Tkinter-based GUI for Fire & Smoke AI Camera/Video Demo with fixed layout and Canvas."""

    def __init__(
        self,
        root: tk.Tk,
        detection_service: Optional[DetectionService] = None,
        video_service: Optional[VideoService] = None
    ) -> None:
        self.root = root
        self.detection_service = detection_service
        self.video_service = video_service or VideoService(detection_service=self.detection_service)

        # Connect video service callbacks
        self.video_service.on_state_change = self._on_playback_state_changed
        self.video_service.on_video_ended = self._on_video_ended
        self.video_service.on_error = self._on_video_error

        # UI state tracking
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None
        self.is_seeking = False
        self.poll_timer_id: Optional[str] = None
        self._last_rendered_frame_idx: int = -2

        # Canvas cached dimensions
        self._cached_canvas_w: int = DEFAULT_DISPLAY_WIDTH
        self._cached_canvas_h: int = DEFAULT_DISPLAY_HEIGHT

        # Alert smoothing and hold logic
        self.consecutive_alert_count: int = 0
        self.current_confirmed_state: str = STATE_NORMAL
        self.last_confirmed_alert_time: float = 0.0
        self.alert_hold_sec: float = 0.8

        self._setup_window()
        self._build_ui()
        self._start_render_loop()

    def _setup_window(self) -> None:
        """Configure root window title, geometry, and styling."""
        self.root.title(f"AI Fire & Smoke Detection Demo [{DEVICE_NAME}]")
        self.root.geometry("1100x820")
        self.root.minsize(960, 720)
        self.root.configure(bg=UI_COLORS["bg_dark"])

        # Prevent root auto-resizing
        self.root.pack_propagate(False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=UI_COLORS["bg_dark"], foreground=UI_COLORS["text_primary"])
        style.configure("TLabel", background=UI_COLORS["bg_dark"], foreground=UI_COLORS["text_primary"])
        style.configure("TFrame", background=UI_COLORS["bg_dark"])

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        """Construct the entire fixed widget hierarchy."""
        main_container = tk.Frame(self.root, bg=UI_COLORS["bg_dark"], padx=14, pady=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Top Bar (Fixed height 48px)
        self._build_top_bar(main_container)

        # 2. Alert Banner (Fixed height 48px, never jumps layout)
        self._build_alert_banner(main_container)

        # 3. Canvas Viewport (Expands smoothly without altering parent)
        self._build_video_viewport(main_container)

        # 4. Timeline Bar (Fixed height 32px)
        self._build_timeline(main_container)

        # 5. Controls & Status Bar (Fixed height 52px)
        self._build_controls_bar(main_container)

    def _build_top_bar(self, parent: tk.Widget) -> None:
        """Top section with 'Chọn video' button and file metadata."""
        top_container = tk.Frame(parent, bg=UI_COLORS["card_bg"], height=48, padx=12, pady=6)
        top_container.pack(fill=tk.X, pady=(0, 6))
        top_container.pack_propagate(False)

        btn_select = tk.Button(
            top_container,
            text="📁 Chọn video...",
            command=self.choose_video_file,
            font=("Segoe UI", 10, "bold"),
            bg=UI_COLORS["accent_primary"],
            fg="#FFFFFF",
            activebackground=UI_COLORS["accent_hover"],
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=14,
            pady=3,
            cursor="hand2"
        )
        btn_select.pack(side=tk.LEFT)

        self.lbl_filename = tk.Label(
            top_container,
            text="Chưa chọn video nào (Hỗ trợ MP4, AVI, MOV, MKV...)",
            font=("Segoe UI", 10),
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["card_bg"],
            padx=12,
            anchor="w"
        )
        self.lbl_scene_status = tk.Label(
            top_container,
            text="⚙️ Tự động thích ứng ngoại cảnh: Sẵn sàng",
            font=("Segoe UI", 9, "bold"),
            fg="#63B3ED",
            bg=UI_COLORS["card_bg"],
            anchor="e"
        )
        self.lbl_scene_status.pack(side=tk.RIGHT, padx=(0, 12))

        self.lbl_video_meta = tk.Label(
            top_container,
            text="",
            font=("Segoe UI", 9),
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["card_bg"],
            anchor="e"
        )
        self.lbl_video_meta.pack(side=tk.RIGHT)

    def _build_alert_banner(self, parent: tk.Widget) -> None:
        """Prominent fixed-height alert card."""
        self.banner_frame = tk.Frame(
            parent,
            bg=UI_COLORS[STATE_NORMAL],
            height=48,
            padx=14,
            pady=6,
            relief=tk.RAISED,
            bd=1
        )
        self.banner_frame.pack(fill=tk.X, pady=(0, 6))
        self.banner_frame.pack_propagate(False)

        self.lbl_alert_icon = tk.Label(
            self.banner_frame,
            text="✅",
            font=("Segoe UI", 16),
            bg=UI_COLORS[STATE_NORMAL],
            fg="#FFFFFF"
        )
        self.lbl_alert_icon.pack(side=tk.LEFT, padx=(4, 10))

        self.lbl_alert_title = tk.Label(
            self.banner_frame,
            text=ALERT_MESSAGES[STATE_NORMAL],
            font=("Segoe UI", 12, "bold"),
            bg=UI_COLORS[STATE_NORMAL],
            fg="#FFFFFF",
            anchor="w"
        )
        self.lbl_alert_title.pack(side=tk.LEFT, fill=tk.X)

        self.lbl_alert_details = tk.Label(
            self.banner_frame,
            text="",
            font=("Segoe UI", 10),
            bg=UI_COLORS[STATE_NORMAL],
            fg="#FFFFFF",
            anchor="e"
        )
        self.lbl_alert_details.pack(side=tk.RIGHT, padx=6)

    def _build_video_viewport(self, parent: tk.Widget) -> None:
        """Display area using fixed tk.Canvas with centered letterbox image."""
        self.viewport_container = tk.Frame(
            parent,
            bg="#000000",
            relief=tk.SUNKEN,
            bd=1
        )
        self.viewport_container.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.canvas = tk.Canvas(
            self.viewport_container,
            bg="#000000",
            highlightthickness=0,
            cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initial placeholder text on canvas
        self.canvas_text_id = self.canvas.create_text(
            DEFAULT_DISPLAY_WIDTH // 2,
            DEFAULT_DISPLAY_HEIGHT // 2,
            text="Chọn file video và bấm 'Phát' để bắt đầu phân tích AI",
            font=("Segoe UI", 13),
            fill=UI_COLORS["text_secondary"]
        )

        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _build_timeline(self, parent: tk.Widget) -> None:
        """Video timeline progress bar with fixed height."""
        timeline_container = tk.Frame(parent, bg=UI_COLORS["bg_dark"], height=32)
        timeline_container.pack(fill=tk.X, pady=(0, 6))
        timeline_container.pack_propagate(False)

        self.lbl_time_current = tk.Label(
            timeline_container,
            text="00:00",
            font=("Consolas", 10, "bold"),
            fg=UI_COLORS["text_primary"],
            bg=UI_COLORS["bg_dark"],
            width=6
        )
        self.lbl_time_current.pack(side=tk.LEFT)

        self.timeline_slider = ttk.Scale(
            timeline_container,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self._on_timeline_slider_drag
        )
        self.timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.timeline_slider.bind("<ButtonPress-1>", lambda e: setattr(self, "is_seeking", True))
        self.timeline_slider.bind("<ButtonRelease-1>", self._on_timeline_slider_release)

        self.lbl_time_total = tk.Label(
            timeline_container,
            text="00:00",
            font=("Consolas", 10),
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["bg_dark"],
            width=6
        )
        self.lbl_time_total.pack(side=tk.RIGHT)

    def _build_controls_bar(self, parent: tk.Widget) -> None:
        """Bottom toolbar with Play/Pause/Stop, Confidence Slider, and FPS indicators."""
        ctrl_container = tk.Frame(parent, bg=UI_COLORS["card_bg"], height=52, padx=12, pady=6)
        ctrl_container.pack(fill=tk.X)
        ctrl_container.pack_propagate(False)

        # Buttons
        self.btn_play_pause = tk.Button(
            ctrl_container,
            text="▶ Phát",
            command=self.toggle_play_pause,
            font=("Segoe UI", 10, "bold"),
            bg="#2B6CB0",
            fg="#FFFFFF",
            relief=tk.FLAT,
            padx=14,
            pady=3,
            cursor="hand2"
        )
        self.btn_play_pause.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = tk.Button(
            ctrl_container,
            text="⏹ Dừng",
            command=self.stop_video,
            font=("Segoe UI", 10),
            bg="#4A5568",
            fg="#FFFFFF",
            relief=tk.FLAT,
            padx=12,
            pady=3,
            cursor="hand2"
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 16))

        # Confidence Slider
        lbl_conf = tk.Label(
            ctrl_container,
            text="Độ tin cậy:",
            font=("Segoe UI", 9),
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["card_bg"]
        )
        lbl_conf.pack(side=tk.LEFT, padx=(0, 6))

        self.conf_slider = ttk.Scale(
            ctrl_container,
            from_=0.10,
            to=0.90,
            value=DEFAULT_CONF_THRESHOLD,
            orient=tk.HORIZONTAL,
            length=120,
            command=self._on_conf_slider_change
        )
        self.conf_slider.pack(side=tk.LEFT, padx=(0, 6))

        self.lbl_conf_val = tk.Label(
            ctrl_container,
            text=f"{int(DEFAULT_CONF_THRESHOLD * 100)}%",
            font=("Consolas", 10, "bold"),
            fg=UI_COLORS["text_primary"],
            bg=UI_COLORS["card_bg"],
            width=5
        )
        self.lbl_conf_val.pack(side=tk.LEFT, padx=(0, 14))

        # Real-time Performance Indicators in Status Bar
        self.lbl_status = tk.Label(
            ctrl_container,
            text=f"FPS Video: 0.0 | FPS AI: 0.0 | Thiết bị: {DEVICE_NAME}",
            font=("Consolas", 9),
            fg=UI_COLORS["text_secondary"],
            bg=UI_COLORS["card_bg"],
            anchor="e"
        )
        self.lbl_status.pack(side=tk.RIGHT)

    def _on_canvas_configure(self, event) -> None:
        """Handle canvas resizing smoothly without feedback loop."""
        if event.width > 50 and event.height > 50:
            self._cached_canvas_w = event.width
            self._cached_canvas_h = event.height
            if self.canvas_text_id is not None:
                self.canvas.coords(self.canvas_text_id, event.width // 2, event.height // 2)

    def _start_render_loop(self) -> None:
        """Periodically poll latest frame and AI result."""
        self._render_tick()

    def _render_tick(self) -> None:
        """Main rendering tick on Tkinter event loop."""
        try:
            frame, frame_idx, ts, inf = self.video_service.get_latest_data()

            # Render frame if new frame available or window resized
            if frame is not None and frame_idx != self._last_rendered_frame_idx:
                self._last_rendered_frame_idx = frame_idx
                self._draw_frame(frame, frame_idx, ts, inf)

            # Update timeline slider
            if not self.is_seeking and self.video_service.total_frames > 0 and frame_idx >= 0:
                progress = (frame_idx / self.video_service.total_frames) * 100.0
                self.timeline_slider.set(progress)
                self.lbl_time_current.configure(text=format_timestamp(ts))

            # Update real-time status metrics
            disp_fps = self.video_service.display_fps
            inf_fps = self.video_service.inference_fps
            inf_ms = inf.inference_time_ms if inf else 0.0
            self.lbl_status.configure(
                text=f"FPS Video: {disp_fps:4.1f} | FPS AI: {inf_fps:4.1f} ({inf_ms:3.0f}ms) | {DEVICE_NAME}"
            )
            if inf and getattr(inf, "scene_status_text", ""):
                self.lbl_scene_status.configure(text=f"⚙️ {inf.scene_status_text}")

        except Exception as e:
            logger.error(f"Lỗi trong render tick: {e}", exc_info=True)

        self.poll_timer_id = self.root.after(UI_POLL_INTERVAL_MS, self._render_tick)

    def _draw_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        inf: Optional[FrameAnalysisResult]
    ) -> None:
        """Letterbox resize frame, apply bounding boxes, and update canvas."""
        canvas_w = self._cached_canvas_w
        canvas_h = self._cached_canvas_h

        # 1. Letterbox resize to exact canvas dimensions
        canvas_img, scale, pad_x, pad_y = resize_and_letterbox(frame, canvas_w, canvas_h)

        # 2. Check if latest AI detection is fresh (within BOX_PERSISTENCE_SEC)
        now = time.monotonic()
        if inf is not None and self.detection_service:
            age = now - inf.updated_monotonic
            if age <= BOX_PERSISTENCE_SEC and len(inf.detections) > 0:
                # Fast direct drawing onto letterbox canvas
                canvas_img = self.detection_service.draw_annotations_on_letterbox(
                    canvas=canvas_img,
                    detections=inf.detections,
                    scale=scale,
                    pad_x=pad_x,
                    pad_y=pad_y
                )

        # 3. Convert to PhotoImage
        pil_img = bgr_to_pil(canvas_img)
        self.current_photo = ImageTk.PhotoImage(image=pil_img)

        # 4. Update canvas image item
        cx = canvas_w // 2
        cy = canvas_h // 2
        if self.canvas_image_id is None:
            if self.canvas_text_id is not None:
                self.canvas.delete(self.canvas_text_id)
                self.canvas_text_id = None
            self.canvas_image_id = self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self.current_photo)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.current_photo)
            self.canvas.coords(self.canvas_image_id, cx, cy)

        # 5. Update Alert Banner based on AI result
        self._update_alert_banner(inf, timestamp_sec)

    def _update_alert_banner(self, inf: Optional[FrameAnalysisResult], current_ts: float) -> None:
        """Smooth alert transitions and update banner UI."""
        now = time.monotonic()

        raw_state = inf.alert_state if inf else STATE_NORMAL
        is_alerting = raw_state in (STATE_FIRE, STATE_SMOKE, STATE_BOTH)

        if is_alerting:
            self.consecutive_alert_count += 1
            if self.consecutive_alert_count >= ALERT_SMOOTHING_FRAMES:
                self.current_confirmed_state = raw_state
                self.last_confirmed_alert_time = now
        else:
            self.consecutive_alert_count = 0
            if now - self.last_confirmed_alert_time > self.alert_hold_sec:
                self.current_confirmed_state = STATE_NORMAL

        state = self.current_confirmed_state
        time_str = format_timestamp(current_ts)
        bg_color = UI_COLORS.get(state, UI_COLORS[STATE_NORMAL])

        if state == STATE_BOTH and inf:
            icon = "🚨"
            title = f"{ALERT_MESSAGES[STATE_BOTH]} ({time_str})"
            details = f"Lửa: {inf.max_fire_conf:.1%} | Khói: {inf.max_smoke_conf:.1%}"
        elif state == STATE_FIRE and inf:
            icon = "🔥"
            title = f"{ALERT_MESSAGES[STATE_FIRE]} ({time_str})"
            details = f"Độ tin cậy: {inf.max_fire_conf:.1%}"
        elif state == STATE_SMOKE and inf:
            icon = "🌫️"
            title = f"{ALERT_MESSAGES[STATE_SMOKE]} ({time_str})"
            details = f"Độ tin cậy: {inf.max_smoke_conf:.1%}"
        else:
            icon = "✅"
            title = ALERT_MESSAGES[STATE_NORMAL]
            details = f"Thời gian: {time_str}"

        self.banner_frame.configure(bg=bg_color)
        self.lbl_alert_icon.configure(text=icon, bg=bg_color)
        self.lbl_alert_title.configure(text=title, bg=bg_color)
        self.lbl_alert_details.configure(text=details, bg=bg_color)

    def choose_video_file(self) -> None:
        """Open file dialog to choose a video file."""
        filetypes = [
            ("Video Files", " ".join(f"*{ext}" for ext in SUPPORTED_VIDEO_EXTENSIONS)),
            ("All Files", "*.*")
        ]
        chosen = filedialog.askopenfilename(
            title="Chọn file video camera",
            filetypes=filetypes
        )
        if not chosen:
            return

        success = self.video_service.open_video(chosen)
        if success:
            meta = self.video_service.metadata
            filename = meta.get("filename", Path(chosen).name)
            self.lbl_filename.configure(
                text=f"Đã chọn: {filename}",
                fg=UI_COLORS["text_primary"]
            )
            self.lbl_video_meta.configure(
                text=f"{meta.get('width')}x{meta.get('height')} | {meta.get('fps', 0):.1f} FPS | {meta.get('duration_str')}"
            )
            self.lbl_time_total.configure(text=meta.get("duration_str", "00:00"))
            self.lbl_time_current.configure(text="00:00")
            self.timeline_slider.set(0)
            self.btn_play_pause.configure(text="▶ Phát", bg="#2B6CB0")
            self._last_rendered_frame_idx = -2

            # Reset banner
            self.banner_frame.configure(bg=UI_COLORS[STATE_NORMAL])
            self.lbl_alert_icon.configure(text="✅", bg=UI_COLORS[STATE_NORMAL])
            self.lbl_alert_title.configure(text=ALERT_MESSAGES[STATE_NORMAL], bg=UI_COLORS[STATE_NORMAL])
            self.lbl_alert_details.configure(text="", bg=UI_COLORS[STATE_NORMAL])

    def toggle_play_pause(self) -> None:
        """Toggle Play/Pause state."""
        if not self.video_service.video_path:
            messagebox.showinfo("Thông báo", "Vui lòng chọn một file video trước khi phát.")
            return

        if self.video_service.state == PlaybackState.PLAYING:
            self.video_service.pause()
        else:
            self.video_service.play()

    def stop_video(self) -> None:
        """Stop playback and reset state."""
        self.video_service.stop()
        self.timeline_slider.set(0)
        self.lbl_time_current.configure(text="00:00")
        self.btn_play_pause.configure(text="▶ Phát", bg="#2B6CB0")
        self._last_rendered_frame_idx = -2

        # Reset banner
        self.banner_frame.configure(bg=UI_COLORS[STATE_NORMAL])
        self.lbl_alert_icon.configure(text="✅", bg=UI_COLORS[STATE_NORMAL])
        self.lbl_alert_title.configure(text=ALERT_MESSAGES[STATE_NORMAL], bg=UI_COLORS[STATE_NORMAL])
        self.lbl_alert_details.configure(text="", bg=UI_COLORS[STATE_NORMAL])

    def _on_playback_state_changed(self, new_state: PlaybackState) -> None:
        """Update buttons when playback state changes."""
        def _update():
            if new_state == PlaybackState.PLAYING:
                self.btn_play_pause.configure(text="⏸ Tạm dừng", bg="#DD6B20")
            elif new_state == PlaybackState.PAUSED:
                self.btn_play_pause.configure(text="▶ Tiếp tục", bg="#2B6CB0")
            elif new_state == PlaybackState.STOPPED:
                self.btn_play_pause.configure(text="▶ Phát", bg="#2B6CB0")

        self.root.after(0, _update)

    def _on_video_ended(self) -> None:
        """Notify user when video reaches end."""
        def _notify():
            self.btn_play_pause.configure(text="▶ Phát lại", bg="#2B6CB0")
            messagebox.showinfo("Thông báo", "Video đã phát hết.")
        self.root.after(0, _notify)

    def _on_video_error(self, err_msg: str) -> None:
        """Show video error."""
        def _show():
            messagebox.showerror("Lỗi Video", err_msg)
        self.root.after(0, _show)

    def _on_timeline_slider_drag(self, val: str) -> None:
        """Update timestamp preview while dragging."""
        if self.video_service.duration_sec > 0:
            pct = float(val) / 100.0
            sec = pct * self.video_service.duration_sec
            self.lbl_time_current.configure(text=format_timestamp(sec))

    def _on_timeline_slider_release(self, event) -> None:
        """Seek video position on slider release."""
        if self.video_service.total_frames > 0:
            pct = float(self.timeline_slider.get()) / 100.0
            target_frame = int(pct * self.video_service.total_frames)
            self.video_service.seek_frame(target_frame)
        self.is_seeking = False

    def _on_conf_slider_change(self, val: str) -> None:
        """Update confidence threshold on slider adjustment."""
        c = float(val)
        self.lbl_conf_val.configure(text=f"{int(c * 100)}%")
        if self.detection_service:
            self.detection_service.set_conf_threshold(c)

    def on_close(self) -> None:
        """Safely terminate all threads and destroy window."""
        logger.info("Đang đóng ứng dụng...")
        if self.poll_timer_id:
            try:
                self.root.after_cancel(self.poll_timer_id)
            except Exception:
                pass
        self.video_service.close()
        self.root.destroy()
