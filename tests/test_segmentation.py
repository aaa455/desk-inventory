"""验证掩膜对应关系、清点规则与手动提示坐标，真实模型另做样本验证。"""

import base64
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings
from app.main import app
from app.schemas.segmentation import SegmentationPrompt
from app.services.detector import detector
from app.services.images import InvalidImage, prepare_image
from app.services.masks import mask_info
from app.services.segmenter import Segmenter, segmenter


def image_bytes(size=(120, 80)):
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


class MaskTests(unittest.TestCase):
    def test_binary_png_preserves_holes_and_area(self):
        mask = np.ones((10, 20), dtype=bool)
        mask[3:7, 8:12] = False
        info = mask_info(mask)
        decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(info.png.split(",", 1)[1]))))
        np.testing.assert_array_equal(decoded, mask.astype(np.uint8) * 255)
        self.assertEqual(info.area_pixels, 184)
        self.assertEqual(info.area_ratio, 0.92)

    def test_sam_prompt_indices_survive_filtered_output(self):
        service = Segmenter()
        service._model = Mock()
        service._device = "cpu"
        result = SimpleNamespace(masks=SimpleNamespace(data=Mock()), boxes=SimpleNamespace(cls=Mock(), conf=Mock()))
        result.masks.data.cpu.return_value.numpy.return_value = np.ones((2, 80, 120), dtype=bool)
        result.boxes.cls.cpu.return_value.tolist.return_value = [2, 0]
        result.boxes.conf.cpu.return_value.tolist.return_value = [0.8, 0.9]
        service._model.predict.return_value = [result]
        masks, quality = service._predict(prepare_image(image_bytes()), 3, bboxes=[[0, 0, 10, 10]] * 3)
        self.assertIsNone(masks[1])
        self.assertEqual(quality, [0.9, None, 0.8])

    def test_bad_mask_shape_and_duplicate_ids_are_not_silently_used(self):
        for indices, shape in [([0, 0], (2, 80, 120)), ([0], (1, 20, 30))]:
            with self.subTest(indices=indices):
                service = Segmenter()
                service._model = Mock()
                service._device = "cpu"
                result = SimpleNamespace(masks=SimpleNamespace(data=Mock()), boxes=SimpleNamespace(cls=Mock(), conf=Mock()))
                result.masks.data.cpu.return_value.numpy.return_value = np.ones(shape, dtype=bool)
                result.boxes.cls.cpu.return_value.tolist.return_value = indices
                result.boxes.conf.cpu.return_value.tolist.return_value = [0.9] * len(indices)
                service._model.predict.return_value = [result]
                with self.assertRaises(RuntimeError):
                    service._predict(prepare_image(image_bytes()), 2)


class InventoryTests(unittest.TestCase):
    def test_counting_exclusion_empty_mask_and_instance_ids(self):
        service = Segmenter()
        prediction = ([[0, 0, 40, 40], [0, 0, 120, 80], [50, 0, 100, 60]],
                      [0.9, 0.8, 0.7], [41, 60, 41], {41: "cup", 60: "dining table"}, 10)
        masks = [np.ones((80, 120), bool), np.ones((80, 120), bool), np.zeros((80, 120), bool)]
        with patch.object(detector, "predict", return_value=prediction), patch.object(detector, "_device", "cpu"), \
             patch.object(service, "_predict", return_value=(masks, [0.9, 0.8, 0.7])):
            result = service.inventory(image_bytes(), 0.25, 0.45, False)
            all_result = service.inventory(image_bytes(), 0.25, 0.45, True)
        self.assertEqual(result.counts, {"杯子": 2})
        self.assertEqual(result.inventory[0].instance_ids, [1, 3])
        self.assertEqual(result.instances[2].mask_status, "空掩膜")
        self.assertFalse(result.instances[1].included)
        self.assertEqual((result.total, result.excluded_total, result.segmented_total), (2, 1, 2))
        self.assertEqual(all_result.counts, {"杯子": 2, "餐桌": 1})

    def test_no_detections_does_not_load_sam(self):
        service = Segmenter()
        with patch.object(detector, "predict", return_value=([], [], [], {}, 5)), \
             patch.object(detector, "_device", "cpu"), patch.object(service, "_load") as load:
            result = service.inventory(image_bytes(), 0.25, 0.45, False)
        load.assert_not_called()
        self.assertEqual(result.total, 0)
        self.assertEqual(result.instances, [])

    def test_overlap_remains_two_instances_without_summing_physical_area(self):
        # 互相覆盖的掩膜仍属于不同提示，不因交叠就删除一个实例。
        service = Segmenter()
        prediction = ([[0, 0, 40, 40]] * 2, [0.9, 0.8], [41, 44], {41: "cup", 44: "spoon"}, 5)
        mask = np.ones((80, 120), bool)
        with patch.object(detector, "predict", return_value=prediction), patch.object(detector, "_device", "cpu"), \
             patch.object(service, "_predict", return_value=([mask, mask], [0.9, 0.8])):
            result = service.inventory(image_bytes(), 0.25, 0.45, False)
        self.assertEqual(result.total, 2)
        self.assertEqual(result.segmented_total, 2)


class PromptTests(unittest.TestCase):
    def test_invalid_prompts(self):
        invalid = [
            {"kind": "box", "box": [10, 10, 5, 5]},
            {"kind": "box", "box": [0, 0, float("nan"), 10]},
            {"kind": "points", "points": [[10, 10]], "labels": [0]},
            {"kind": "points", "points": [[10, 10]], "labels": [1, 0]},
            {"kind": "points", "points": [[-1, 10]], "labels": [1]},
            {"kind": "box", "box": [0, 0, 10, 10], "unknown": 1},
        ]
        for prompt in invalid:
            with self.subTest(prompt=prompt), self.assertRaises(ValidationError):
                SegmentationPrompt.model_validate(prompt)

    def test_point_group_is_scaled_as_one_object(self):
        service = Segmenter()
        service._device = "cpu"
        prompt = SegmentationPrompt(kind="points", points=[(40, 20), (200, 100)], labels=[1, 0])
        with patch.object(settings, "max_image_side", 100), \
             patch.object(service, "_predict", return_value=([np.ones((50, 100), bool)], [0.9])) as predict:
            result = service.prompted(image_bytes((400, 200)), prompt)
        self.assertEqual(predict.call_args.kwargs, {"points": [[[10, 5], [50, 25]]], "labels": [[1, 0]]})
        self.assertEqual((result.mask.width, result.mask.height), (100, 50))

    def test_box_scaled_to_processed_image(self):
        service = Segmenter()
        service._device = "cpu"
        with patch.object(settings, "max_image_side", 100), \
             patch.object(service, "_predict", return_value=([np.ones((50, 100), bool)], [0.9])) as predict:
            service.prompted(image_bytes((400, 200)), SegmentationPrompt(kind="box", box=(0, 0, 400, 200)))
        self.assertEqual(predict.call_args.kwargs, {"bboxes": [[0, 0, 100, 50]]})

    def test_out_of_range_prompt_fails_before_model_load(self):
        service = Segmenter()
        for prompt in [SegmentationPrompt(kind="box", box=(0, 0, 121, 20)),
                       SegmentationPrompt(kind="points", points=[(120, 0)], labels=[1])]:
            with self.subTest(prompt=prompt), patch.object(service, "_predict") as predict:
                with self.assertRaises(InvalidImage):
                    service.prompted(image_bytes(), prompt)
                predict.assert_not_called()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_invalid_json_and_out_of_range_prompt(self):
        for prompt, status in [("invalid", 422), ('{"kind":"box","box":[0,0,999,10]}', 400)]:
            response = self.client.post("/api/segment/prompt", data={"prompt": prompt}, files={"file": ("a.png", image_bytes())})
            self.assertEqual(response.status_code, status)

    def test_missing_sam_model(self):
        with patch.object(segmenter, "_model", None), patch.object(settings, "sam_model_path", "models/不存在.pt"):
            response = self.client.post("/api/segment/prompt", data={"prompt": '{"kind":"box","box":[0,0,20,20]}'},
                                        files={"file": ("a.png", image_bytes())})
        self.assertEqual(response.status_code, 503)

    def test_upload_limits_and_invalid_content(self):
        with patch.object(settings, "max_upload_bytes", 50):
            self.assertEqual(self.client.post("/api/segment", files={"file": ("a", image_bytes())}).status_code, 413)
        self.assertEqual(self.client.post("/api/segment", files={"file": ("a", b"bad")}).status_code, 400)
        self.assertEqual(self.client.post("/api/segment?confidence=2", files={"file": ("a", image_bytes())}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
