"""Hệ thống ghi log quyết định suy luận có cấu trúc (Structured Decision Logger).
Ghi lại chi tiết từng frame:
- Môi trường phát hiện (Environment).
- Thời gian thực thi (latency ms) từng bước (đọc, tiền xử lý, từng model, gộp WBF).
- Toàn bộ ứng viên phát hiện thô từ các model.
- Kết quả gộp, mức độ đồng thuận, lý do giữ lại hoặc triệt tiêu.
Lưu dưới định dạng JSON Lines (.jsonl) phục vụ kiểm chuẩn và phân tích thống kê.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
import threading

logger = logging.getLogger("FireSmokeAI.DecisionLogger")


class DecisionLogger:
    """Bộ ghi log quyết định dạng JSONL, luồng an toàn."""

    def __init__(self, log_path: Optional[Path] = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.log_path = log_path
        self._lock = threading.Lock()
        self._file_handle = None

        if self.enabled and self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.log_path, "a", encoding="utf-8")
            logger.info(f"[DecisionLogger] Đã mở file log quyết định tại: {self.log_path}")

    def log_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        environment: str,
        timing_ms: Dict[str, float],
        raw_candidates: list,
        fused_objects: list,
        suppressed_objects: Optional[list] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ghi nhận một sự kiện phân tích frame."""
        if not self.enabled or self._file_handle is None:
            return

        record = {
            "ts": round(timestamp_sec, 3),
            "frame": frame_index,
            "env": environment,
            "timing_ms": {k: round(v, 2) for k, v in timing_ms.items()},
            "raw_candidates": raw_candidates,
            "fused_objects": fused_objects,
            "suppressed_objects": suppressed_objects or [],
        }
        if extra:
            record["extra"] = extra

        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            try:
                self._file_handle.write(line + "\n")
                self._file_handle.flush()
            except Exception as e:
                logger.error(f"[DecisionLogger] Lỗi khi ghi log frame: {e}")

    def close(self) -> None:
        """Đóng file log an toàn."""
        with self._lock:
            if self._file_handle is not None and not self._file_handle.closed:
                self._file_handle.flush()
                self._file_handle.close()
                self._file_handle = None

    def __del__(self) -> None:
        self.close()
