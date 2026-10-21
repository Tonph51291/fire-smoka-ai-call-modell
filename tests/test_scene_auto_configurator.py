"""Unit tests for SceneAutoConfigurator."""

import unittest
import numpy as np

from src.services.scene_auto_configurator import SceneAutoConfigurator, DynamicSceneProfile
from src.core.entities import EnvironmentType


class TestSceneAutoConfigurator(unittest.TestCase):
    """Kiểm tra hoạt động của bộ tự động cấu hình ngoại cảnh."""

    def setUp(self):
        self.configurator = SceneAutoConfigurator(warmup_frames=10, env_hysteresis_sec=0.1)

    def test_daylight_standard_frame(self):
        # Tạo frame ban ngày chuẩn (ánh sáng trung bình, có màu)
        day_frame = np.full((120, 160, 3), 140, dtype=np.uint8)
        day_frame[:, :, 0] = 160  # Blue
        day_frame[:, :, 1] = 140  # Green
        day_frame[:, :, 2] = 120  # Red

        profile = self.configurator.update(day_frame, timestamp_sec=0.04)
        self.assertIsInstance(profile, DynamicSceneProfile)
        self.assertEqual(profile.environment, EnvironmentType.DAY_NORMAL)
        self.assertAlmostEqual(profile.confidence_fire, 0.25, places=2)
        self.assertAlmostEqual(profile.confidence_smoke, 0.30, places=2)

    def test_night_ir_frame_adaptation(self):
        # Tạo frame hồng ngoại đêm (đen trắng, saturation = 0, color_diff = 0, độ sáng thấp)
        ir_frame = np.full((120, 160, 3), 50, dtype=np.uint8)

        profile = self.configurator.update(ir_frame, timestamp_sec=0.04)
        self.assertEqual(profile.environment, EnvironmentType.NIGHT_IR)
        # Trong môi trường IR, ngưỡng khói phải tự nâng lên 0.42 để tránh bụi sáng
        self.assertGreaterEqual(profile.confidence_smoke, 0.40)
        # Model mới fire_yolo.pt phải được ưu tiên trọng số cao nhất (1.4)
        self.assertGreater(profile.model_weights["fire_yolo.pt"], profile.model_weights["best.pt"])

    def test_glare_sun_frame_adaptation(self):
        # Tạo frame ngược sáng / chói chang (vùng highlight trắng xóa >= 250 chiếm tỷ lệ lớn)
        glare_frame = np.full((120, 160, 3), 200, dtype=np.uint8)
        glare_frame[10:90, 10:140] = 255  # Vùng lóa mặt trời rộng

        profile = self.configurator.update(glare_frame, timestamp_sec=0.04)
        self.assertEqual(profile.environment, EnvironmentType.GLARE_SUN)
        # Ngưỡng lửa phải tự động nâng lên (>= 0.30) để chống lóa bóng mặt trời
        self.assertGreaterEqual(profile.confidence_fire, 0.30)

    def test_reset(self):
        frame = np.full((120, 160, 3), 50, dtype=np.uint8)
        self.configurator.update(frame, timestamp_sec=0.04)
        self.configurator.reset()
        self.assertEqual(self.configurator.frame_count, 0)
        self.assertEqual(len(self.configurator.noise_samples), 0)


if __name__ == "__main__":
    unittest.main()
