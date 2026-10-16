"""Unit tests for VideoService."""

import time
import unittest
from pathlib import Path

from src.services.video_service import PlaybackState, VideoService


class TestVideoService(unittest.TestCase):
    """Test VideoService states, video opening, and thread safety."""

    def setUp(self):
        self.video_service = VideoService()
        self.sample_video = Path("videos/sample_demo.mp4")

    def tearDown(self):
        self.video_service.close()

    def test_open_invalid_file(self):
        error_messages = []
        self.video_service.on_error = lambda msg: error_messages.append(msg)

        success = self.video_service.open_video("videos/non_existent.mp4")
        self.assertFalse(success)
        self.assertEqual(self.video_service.state, PlaybackState.STOPPED)
        self.assertTrue(len(error_messages) > 0)

    def test_open_valid_video(self):
        if not self.sample_video.exists():
            self.skipTest("Sample video does not exist.")

        success = self.video_service.open_video(self.sample_video)
        self.assertTrue(success)
        self.assertEqual(self.video_service.state, PlaybackState.STOPPED)
        self.assertGreater(self.video_service.fps, 0)
        self.assertGreater(self.video_service.total_frames, 0)
        self.assertGreater(self.video_service.duration_sec, 0)

    def test_playback_lifecycle(self):
        if not self.sample_video.exists():
            self.skipTest("Sample video does not exist.")

        state_history = []
        self.video_service.on_state_change = lambda s: state_history.append(s)

        # Open video
        self.assertTrue(self.video_service.open_video(self.sample_video))

        # Play
        self.video_service.play()
        self.assertEqual(self.video_service.state, PlaybackState.PLAYING)
        self.assertIn(PlaybackState.PLAYING, state_history)

        # Check get_latest_data after short sleep
        time.sleep(0.3)
        frame, idx, ts, inf = self.video_service.get_latest_data()
        self.assertIsNotNone(frame)
        self.assertGreaterEqual(idx, 0)

        # Pause
        self.video_service.pause()
        self.assertEqual(self.video_service.state, PlaybackState.PAUSED)
        self.assertIn(PlaybackState.PAUSED, state_history)

        # Resume
        self.video_service.resume()
        self.assertEqual(self.video_service.state, PlaybackState.PLAYING)

        # Stop
        self.video_service.stop()
        self.assertEqual(self.video_service.state, PlaybackState.STOPPED)

    def test_seek_frame(self):
        if not self.sample_video.exists():
            self.skipTest("Sample video does not exist.")

        self.video_service.open_video(self.sample_video)
        self.video_service.seek_frame(10)
        self.assertEqual(self.video_service.current_frame_idx, 10)


if __name__ == "__main__":
    unittest.main()
