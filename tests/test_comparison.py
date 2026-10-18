"""验证同图编排、错误保留、计数评估和离线导出；不调用真实视觉服务。"""

import asyncio
import base64
import csv
import hashlib
import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import settings
from app.main import app
from app.schemas.comparison import ComparisonResponse, CountEntry, EvaluationRequest
from app.schemas.detection import ImageInfo
from app.schemas.segmentation import InventoryItem, SegmentationResponse
from app.schemas.vision import VisionItem, VisionResponse
from app.services.comparison import analyze_comparison, evaluate_counts
from app.services.detector import ModelUnavailable
from app.services.reporting import (
    TruthDocument, count_csv, evaluate_document, export_comparison, plain_text, print_comparison,
)
from app.services.vision import VisionError
import compare_image


def sample():
    output = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(output, format="PNG")
    return output.getvalue()


def local_result(data, items=None):
    items = items if items is not None else [("cup", "杯子", 1)]
    return SegmentationResponse(
        image=ImageInfo(sha256=hashlib.sha256(data).hexdigest(), width=60, height=40, processed_width=60, processed_height=40),
        detection_model="测试检测", segmentation_model="测试分割", device="cpu",
        confidence_threshold=.25, iou_threshold=.45, elapsed_ms=10, detection_ms=5, segmentation_ms=5,
        instances=[], inventory=[InventoryItem(label_en=en, label=cn, count=n, instance_ids=[]) for en, cn, n in items],
        counts={cn: n for en, cn, n in items}, total=sum(n for en, cn, n in items), detected_total=0,
        excluded_total=0, excluded_labels=[], segmented_total=0,
        annotated_image="data:image/jpeg;base64," + base64.b64encode(sample()).decode(), notes=[],
    )


def remote_result(data, items=None):
    local = local_result(data, items if items is not None else [("cup", "杯子", 1), ("saucer", "碟子", 1)])
    return VisionResponse(
        image=local.image, model="测试视觉", returned_model="测试视觉", response_mode="json_object",
        max_output_tokens=4096, token_limit_field="max_completion_tokens", prompt_version="测试",
        prompt="固定中文提示", sent_image_sha256="b" * 64, elapsed_ms=20, inference_ms=19,
        inventory=[VisionItem(label=item.label, label_en=item.label_en, count=item.count, uncertainty="") for item in local.inventory],
        excluded_items=[], excluded_labels=[], counts=local.counts, total=local.total,
        uncertainties=[], usage=None, notes=[],
    )


def run_mock(data=None, local=None, remote=None):
    data = data if data is not None else sample()
    with patch("app.services.comparison.segmenter.inventory", return_value=local or local_result(data)):
        with patch("app.services.comparison.vision_service.analyze", new=AsyncMock(return_value=remote or remote_result(data))):
            return asyncio.run(analyze_comparison(data))


def count(label, n):
    return CountEntry(label_en=label, count=n)


class ComparisonTests(unittest.TestCase):
    def test_same_original_bytes_and_parallel_execution(self):
        data = sample()
        entered, remote_entered = threading.Event(), threading.Event()
        def traditional(received, confidence, iou, background):
            self.assertIs(received, data)
            self.assertEqual((confidence, iou, background), (.3, .5, True))
            entered.set()
            self.assertTrue(remote_entered.wait(2), "远程分支必须在本地分支完成前启动")
            return local_result(data)
        async def remote(received, background):
            self.assertIs(received, data)
            self.assertTrue(background)
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            remote_entered.set()
            return remote_result(data)
        with patch("app.services.comparison.segmenter.inventory", side_effect=traditional):
            with patch("app.services.comparison.vision_service.analyze", side_effect=remote):
                result = asyncio.run(analyze_comparison(data, .3, .5, True))
        self.assertTrue(result.comparable)
        rows = {row.label_en: row for row in result.differences}
        self.assertEqual(rows["saucer"].delta, 1)
        self.assertEqual(rows["cup"].delta, 0)

    def test_one_branch_failure_does_not_become_zero(self):
        data = sample()
        with patch("app.services.comparison.segmenter.inventory", return_value=local_result(data)):
            with patch("app.services.comparison.vision_service.analyze", side_effect=VisionError(504, "视觉超时")):
                result = asyncio.run(analyze_comparison(data))
        self.assertEqual(result.traditional.status, "success")
        self.assertEqual(result.vision.error.status_code, 504)
        self.assertIsNone(result.vision.result)
        self.assertFalse(result.comparable)
        self.assertEqual(result.differences, [])
        csv_rows = list(csv.reader(io.StringIO(count_csv(result))))
        self.assertEqual(csv_rows[-1][2:5], ["1", "", ""])

    def test_local_failure_keeps_vision_result(self):
        data = sample()
        with patch("app.services.comparison.segmenter.inventory", side_effect=ModelUnavailable("缺少权重")):
            with patch("app.services.comparison.vision_service.analyze", return_value=remote_result(data)):
                result = asyncio.run(analyze_comparison(data))
        self.assertEqual(result.traditional.status, "error")
        self.assertEqual(result.vision.result.total, 2)

    def test_two_failures_are_sanitized_and_saved(self):
        with patch("app.services.comparison.segmenter.inventory", side_effect=RuntimeError("secret-internal")):
            with patch("app.services.comparison.vision_service.analyze", side_effect=VisionError(429, "额度不足")):
                with self.assertLogs("app.services.comparison", level="ERROR") as logs:
                    result = asyncio.run(analyze_comparison(sample()))
        self.assertFalse(result.comparable)
        self.assertEqual(result.traditional.status, "error")
        self.assertNotIn("secret-internal", result.model_dump_json() + str(logs.output))

    def test_mismatched_image_is_rejected_as_branch_failure(self):
        data = sample()
        local = local_result(data)
        local.image.sha256 = "a" * 64
        result = run_mock(data, local=local)
        self.assertEqual(result.traditional.status, "error")
        self.assertFalse(result.comparable)

    def test_invalid_image_or_upload_does_not_start_either_model(self):
        with TestClient(app) as client:
            with patch("app.services.comparison.segmenter.inventory") as local:
                with patch("app.services.comparison.vision_service.analyze") as remote:
                    self.assertEqual(client.post("/api/compare", files={"file": ("a.png", b"bad")}).status_code, 400)
                    with patch.object(settings, "max_upload_bytes", 10):
                        self.assertEqual(client.post("/api/compare", files={"file": ("a.png", sample())}).status_code, 413)
                    self.assertEqual(client.post("/api/compare").status_code, 422)
                    local.assert_not_called()
                    remote.assert_not_called()

    def test_api_returns_same_comparison_schema(self):
        data = sample()
        with TestClient(app) as client, patch("app.services.comparison.segmenter.inventory", return_value=local_result(data)):
            with patch("app.services.comparison.vision_service.analyze", return_value=remote_result(data)):
                response = client.post("/api/compare", files={"file": ("a.png", data)})
        result = ComparisonResponse.model_validate(response.json())
        self.assertTrue(result.comparable)

    def test_empty_success_is_distinct_from_failure(self):
        result = run_mock(local=local_result(sample(), []), remote=remote_result(sample(), []))
        self.assertTrue(result.comparable)
        self.assertEqual(result.differences, [])
        self.assertEqual(result.traditional.result.total, 0)

    def test_saved_result_requires_consistent_branch_state_and_digest(self):
        result = run_mock().model_dump()
        for field, value in (("comparable", False), ("image_sha256", "0" * 64)):
            altered = dict(result, **{field: value})
            with self.assertRaises(ValidationError):
                ComparisonResponse.model_validate(altered)
        result["vision"]["result"] = None
        with self.assertRaises(ValidationError):
            ComparisonResponse.model_validate(result)


class EvaluationTests(unittest.TestCase):
    def test_equal_total_does_not_hide_wrong_category(self):
        request = EvaluationRequest(traditional=[count("cup", 2)], vision=[count("bowl", 2)],
                                    truth=[count("cup", 2), count("bowl", 0)], confirmed=True)
        response = evaluate_counts(request)
        self.assertEqual(response.traditional.category_absolute_error, 0)
        self.assertEqual(response.vision.total_absolute_error, 0)
        self.assertEqual(response.vision.category_absolute_error, 4)
        self.assertEqual((response.vision.missing_count, response.vision.extra_count), (2, 2))

    def test_both_missed_category_is_evaluated(self):
        request = EvaluationRequest(traditional=[], vision=None, truth=[count("pen", 2)], confirmed=True)
        result = evaluate_counts(request)
        self.assertEqual(result.traditional.missing_count, 2)
        self.assertIsNone(result.vision)
        empty = evaluate_counts(EvaluationRequest(traditional=[], truth=[], confirmed=True))
        self.assertIsNone(empty.traditional.category_mean_absolute_error)

    def test_no_fuzzy_synonym_merge(self):
        request = EvaluationRequest(traditional=[count("coffee cup", 1)], vision=[count("cup", 1)],
                                    truth=[count("coffee cup", 0), count("cup", 1)], confirmed=True)
        self.assertEqual(evaluate_counts(request).traditional.category_absolute_error, 2)

    def test_unconfirmed_incomplete_and_duplicate_truth_are_rejected(self):
        for payload in [
            {"traditional": [], "truth": [], "confirmed": False},
            {"truth": [], "confirmed": True},
            {"traditional": [{"label_en": "cup", "count": 1}], "truth": [], "confirmed": True},
            {"traditional": [], "truth": [{"label_en": "cup", "count": 1}, {"label_en": " CUP ", "count": 1}], "confirmed": True},
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                EvaluationRequest.model_validate(payload)
        for value in (True, -1, 501, "1", 1.5):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                CountEntry(label_en="cup", count=value)

    def test_evaluation_api_works_without_models(self):
        with TestClient(app) as client:
            response = client.post("/api/compare/evaluate", json={
                "traditional": [{"label_en": "cup", "count": 1}], "vision": None,
                "truth": [{"label_en": "cup", "count": 2}], "confirmed": True,
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["traditional"]["missing_count"], 1)


class ReportingTests(unittest.TestCase):
    def test_exports_use_blank_truth_not_model_counts(self):
        result = run_mock()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report"
            export_comparison(result, output)
            loaded = ComparisonResponse.model_validate_json((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded.image_sha256, result.image_sha256)
            template = json.loads((output / "人工清点模板.json").read_text(encoding="utf-8"))
            self.assertTrue(all(item["真实数量"] is None for item in template["类别"]))
            self.assertTrue((output / "segmentation.jpg").is_file())
            with self.assertRaises(ValidationError):
                TruthDocument.model_validate(template)

    def test_truth_digest_prevents_cross_image_review(self):
        result = run_mock()
        document = TruthDocument.model_validate({
            "图片摘要": "0" * 64, "说明": "人工填写",
            "类别": [{"英文类别": "cup", "真实数量": 1}, {"英文类别": "saucer", "真实数量": 1}],
        })
        with self.assertRaises(ValueError):
            evaluate_document(result, document, True)

    def test_csv_quotes_and_terminal_control_characters(self):
        result = run_mock(local=local_result(sample(), [("cup", '=1,"危险"', 1)]))
        rows = list(csv.reader(io.StringIO(count_csv(result))))
        self.assertTrue(rows[3][1].startswith("'="))
        self.assertEqual(plain_text("\x1b[31m杯子\n"), " [31m杯子 ")
        with redirect_stdout(io.StringIO()) as stream:
            print_comparison(result)
        self.assertIn("同图双方案", stream.getvalue())

    def test_cli_offline_review_cannot_call_models(self):
        result = run_mock()
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            export_comparison(result, parent / "source")
            template = json.loads((parent / "source" / "人工清点模板.json").read_text(encoding="utf-8"))
            for item in template["类别"]:
                item["真实数量"] = 1
            truth = parent / "truth.json"
            truth.write_text(json.dumps(template, ensure_ascii=False), encoding="utf-8-sig")
            arguments = ["compare_image.py", "--evaluate-result", str(parent / "source" / "comparison.json"),
                         "--truth", str(truth), "--confirm-truth", "--output", str(parent)]
            with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                with patch("compare_image.analyze_comparison", side_effect=AssertionError("离线评估不允许调用模型")):
                    self.assertEqual(compare_image.main(), 0)
            files = list(parent.glob("evaluation-*/evaluation.json"))
            self.assertEqual(len(files), 1)
            metrics = json.loads(files[0].read_text(encoding="utf-8"))["评估"]
            self.assertEqual(metrics["traditional"]["missing_count"], 1)
            self.assertEqual(metrics["vision"]["category_absolute_error"], 0)

            with patch.object(sys, "argv", ["compare_image.py", "a.png", "--confidence", "nan"]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    compare_image.main()
            self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
