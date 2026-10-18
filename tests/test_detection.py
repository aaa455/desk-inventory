"""验证上传边界、方向校正与缩放坐标，不需要下载模型。"""

import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings
from app.main import app
from app.services.detector import Detector, detector
from app.services.images import InvalidImage, prepare_image


def make_image(size=(120, 80), color="white", format="PNG", **kwargs):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format=format, **kwargs)
    return output.getvalue()


class ImageTests(unittest.TestCase):
    def test_phone_orientation_is_corrected_before_resize(self):
        exif = Image.Exif()
        exif[274] = 6
        prepared = prepare_image(make_image((120, 80), format="JPEG", exif=exif))
        self.assertEqual((prepared.width, prepared.height), (80, 120))

    def test_transparency_uses_white_background(self):
        output = io.BytesIO()
        Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(output, format="PNG")
        self.assertTrue(np.all(prepare_image(output.getvalue()).bgr == 255))

    def test_corrupt_and_unsupported_images_are_rejected(self):
        for data in [b"", b"invalid", make_image(format="BMP"), make_image()[:40]]:
            with self.subTest(data=data[:10]), self.assertRaises(InvalidImage):
                prepare_image(data)

    def test_pixel_limit_checked_before_decode(self):
        with patch.object(settings, "max_image_pixels", 100), self.assertRaises(InvalidImage):
            prepare_image(make_image())

    def test_resized_boxes_return_to_original_coordinates(self):
        service = Detector()
        # 人工构造模型输出，只验证坐标与聚合逻辑，真实模型另做样本验证。
        result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=Mock(), conf=Mock(), cls=Mock()), names={41: "cup"},
        )
        result.boxes.xyxy.cpu.return_value.tolist.return_value = [[10, 5, 50, 25], [60, 10, 100, 30]]
        result.boxes.conf.cpu.return_value.tolist.return_value = [0.9, 0.8]
        result.boxes.cls.cpu.return_value.tolist.return_value = [41, 41]
        service._model = Mock()
        service._model.predict.return_value = [result]
        service._device = "cpu"
        with patch.object(settings, "max_image_side", 100):
            response = service.analyze(make_image((400, 200)), 0.25, 0.45)
        self.assertEqual(response.detections[0].bbox, (40, 20, 200, 100))
        self.assertEqual(response.counts, {"杯子": 2})
        self.assertEqual(response.total, 2)
        self.assertEqual((response.image.processed_width, response.image.processed_height), (100, 50))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_documentation(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        self.assertIn("/api/detect", self.client.get("/openapi.json").json()["paths"])

    def test_invalid_upload_is_400(self):
        for data in [b"", b"not an image"]:
            with self.subTest(data=data):
                response = self.client.post("/api/detect", files={"file": ("photo.jpg", data, "image/jpeg")})
                self.assertEqual(response.status_code, 400)

    def test_oversized_upload_is_413(self):
        with patch.object(settings, "max_upload_bytes", 50):
            response = self.client.post("/api/detect", files={"file": ("photo.png", make_image())})
        self.assertEqual(response.status_code, 413)

    def test_missing_file_and_invalid_threshold_are_422(self):
        self.assertEqual(self.client.post("/api/detect").status_code, 422)
        for query in ["confidence=0", "confidence=2", "iou=-1", "iou=nan"]:
            with self.subTest(query=query):
                response = self.client.post("/api/detect?" + query, files={"file": ("a.png", make_image())})
                self.assertEqual(response.status_code, 422)

    def test_missing_model_is_503_without_download(self):
        with patch.object(detector, "_model", None), patch.object(settings, "yolo_model_path", "models/不存在.pt"):
            response = self.client.post("/api/detect", files={"file": ("a.png", make_image())})
        self.assertEqual(response.status_code, 503)
        self.assertIn("未找到", response.json()["detail"])

    def test_inference_failure_does_not_expose_internal_details(self):
        with patch.object(detector, "analyze", side_effect=RuntimeError("内部路径或设备错误")):
            with self.assertLogs("app.api.detection", level="ERROR"):
                response = self.client.post("/api/detect", files={"file": ("a.png", make_image())})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("内部路径", response.text)


if __name__ == "__main__":
    unittest.main()
