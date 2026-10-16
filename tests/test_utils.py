"""Unit tests for video utilities."""

import unittest
import numpy as np
from src.utils.video_utils import (
    bgr_to_pil,
    format_timestamp,
    resize_and_letterbox,
    scale_bbox,
)


class TestVideoUtils(unittest.TestCase):
    """Test utility functions."""

    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0), "00:00")
        self.assertEqual(format_timestamp(59), "00:59")
        self.assertEqual(format_timestamp(60), "01:00")
        self.assertEqual(format_timestamp(75), "01:15")
        self.assertEqual(format_timestamp(3665), "01:01:05")

    def test_format_timestamp_millis(self):
        res = format_timestamp(12.345, include_millis=True)
        self.assertTrue(res.startswith("00:12."))

    def test_resize_and_letterbox(self):
        # 1920x1080 frame fitted into 960x540 canvas
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas, scale, pad_x, pad_y = resize_and_letterbox(dummy_frame, 960, 540)
        self.assertEqual(canvas.shape[1], 960)
        self.assertEqual(canvas.shape[0], 540)
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 0)

    def test_scale_bbox(self):
        # Original bbox (100, 200, 300, 400) with scale=0.5, pad_x=20, pad_y=10
        orig_box = (100, 200, 300, 400)
        scaled = scale_bbox(orig_box, scale=0.5, pad_x=20, pad_y=10)
        self.assertEqual(scaled, (70, 110, 170, 210))

    def test_bgr_to_pil(self):
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        pil_img = bgr_to_pil(dummy_frame)
        self.assertEqual(pil_img.size, (100, 100))


if __name__ == "__main__":
    unittest.main()
