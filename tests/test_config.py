"""Unit tests for configuration module."""

import unittest
from pathlib import Path
from src.config import (
    ALERT_MESSAGES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    STATE_BOTH,
    STATE_FIRE,
    STATE_NORMAL,
    STATE_SMOKE,
    SUPPORTED_VIDEO_EXTENSIONS,
    UI_COLORS,
)


class TestConfig(unittest.TestCase):
    """Test configuration paths and constants."""

    def test_project_root_exists(self):
        self.assertTrue(PROJECT_ROOT.exists())
        self.assertTrue(PROJECT_ROOT.is_dir())

    def test_default_model_path_location(self):
        expected = PROJECT_ROOT / "models" / "best.pt"
        self.assertEqual(DEFAULT_MODEL_PATH, expected)

    def test_alert_states_and_messages(self):
        for state in [STATE_NORMAL, STATE_FIRE, STATE_SMOKE, STATE_BOTH]:
            self.assertIn(state, ALERT_MESSAGES)
            self.assertTrue(len(ALERT_MESSAGES[state]) > 0)
            self.assertIn(state, UI_COLORS)

    def test_default_thresholds(self):
        self.assertGreater(DEFAULT_CONF_THRESHOLD, 0.0)
        self.assertLess(DEFAULT_CONF_THRESHOLD, 1.0)
        self.assertIn(DEFAULT_IMGSZ, [320, 416, 480, 640])

    def test_supported_video_extensions(self):
        self.assertIn(".mp4", SUPPORTED_VIDEO_EXTENSIONS)
        self.assertIn(".avi", SUPPORTED_VIDEO_EXTENSIONS)
        self.assertIn(".mov", SUPPORTED_VIDEO_EXTENSIONS)
        self.assertIn(".mkv", SUPPORTED_VIDEO_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()
