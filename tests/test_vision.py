"""使用可检查的模拟上游验证协议边界，不调用付费服务，也不加载本地权重。"""

import asyncio
import base64
import hashlib
import io
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import Settings, settings
from app.main import app
from app.services.vision import MAX_RESPONSE_BYTES, VisionService


def picture(size=(120, 80)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def item(label="杯子", label_en="cup", count=1):
    return {"label": label, "label_en": label_en, "count": count, "uncertainty": ""}


def envelope(items=None, finish_reason="stop", content=None):
    return {
        "model": "服务返回模型标识",
        "choices": [{"finish_reason": finish_reason, "message": {
            "content": content if content is not None else json.dumps(
                {"items": items if items is not None else [item()], "uncertainties": []},
                ensure_ascii=False,
            ),
        }}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
    }


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key, value in {
            "vision_api_base_url": "https://vision.example.test/v1",
            "vision_api_key": "测试密钥-不应泄露",
            "vision_model": "测试视觉模型",
            "vision_response_mode": "json_object",
            "vision_token_limit_field": "max_completion_tokens",
            "vision_timeout_seconds": 2,
            "vision_max_output_tokens": 4096,
        }.items():
            # 鉴权头采用 ASCII，避免把编码异常误当作协议错误。
            if key == "vision_api_key":
                value = "test-secret-no-leak"
            self.stack.enter_context(patch.object(settings, key, value))
        self.client = self.stack.enter_context(TestClient(app))
        self.requests = []

    def install(self, handler):
        def record(request):
            self.requests.append(request)
            return handler(request)
        service = VisionService(httpx.MockTransport(record))
        self.stack.enter_context(patch("app.api.vision.vision_service", service))

    def upload(self, data=None, query=""):
        return self.client.post("/api/vision" + query, files={
            "file": ("sample.png", data if data is not None else picture(), "image/png"),
        })

    def test_independent_image_request_and_server_counts(self):
        self.install(lambda request: httpx.Response(200, json=envelope([
            item(count=2), item("桌子", "dining table"),
        ])))
        with patch("app.services.detector.detector.predict", side_effect=AssertionError("不应检测")) as detect:
            with patch("app.services.segmenter.segmenter.inventory", side_effect=AssertionError("不应分割")) as segment:
                with patch.object(settings, "max_image_side", 100):
                    response = self.upload(picture((400, 200)))
        self.assertEqual(response.status_code, 200, response.text)
        detect.assert_not_called()
        segment.assert_not_called()
        result = response.json()
        self.assertEqual(result["counts"], {"杯子": 2})
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["excluded_items"][0]["label_en"], "dining table")
        self.assertEqual(result["image"]["sha256"], hashlib.sha256(picture((400, 200))).hexdigest())
        self.assertEqual(result["max_output_tokens"], 4096)
        self.assertEqual(result["token_limit_field"], "max_completion_tokens")
        self.assertEqual(result["returned_model"], "服务返回模型标识")
        self.assertEqual(result["usage"]["total_tokens"], 168)
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://vision.example.test/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer test-secret-no-leak")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "测试视觉模型")
        self.assertEqual(body["max_completion_tokens"], 4096)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(len(body["messages"]), 2)
        parts = body["messages"][1]["content"]
        self.assertEqual([part["type"] for part in parts], ["text", "image_url"])
        png = base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1])
        self.assertEqual(Image.open(io.BytesIO(png)).size, (100, 50))
        self.assertEqual(result["sent_image_sha256"], hashlib.sha256(png).hexdigest())
        self.assertNotIn("test-secret-no-leak", response.text)

    def test_deepseek_image_request_uses_compatible_json_protocol(self):
        self.install(lambda request: httpx.Response(200, json=envelope([item(count=2)])))
        with patch.object(settings, "vision_api_base_url", "https://api.deepseek.com"):
            with patch.object(settings, "vision_model", "deepseek-flash"):
                with patch.object(settings, "vision_token_limit_field", "max_tokens"):
                    response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        image = body["messages"][1]["content"][1]["image_url"]
        self.assertTrue(image["url"].startswith("data:image/png;base64,"))
        self.assertEqual(image["detail"], "high")
        self.assertEqual(response.json()["counts"], {"杯子": 2})
        self.assertEqual(response.json()["model"], "deepseek-flash")
        self.assertEqual(response.json()["token_limit_field"], "max_tokens")

    def test_include_background_and_empty_image_inventory(self):
        self.install(lambda request: httpx.Response(200, json=envelope([item("桌子", "dining table")])))
        result = self.upload(query="?include_background=true").json()
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["excluded_items"], [])
        self.assertEqual(result["excluded_labels"], [])
        self.install(lambda request: httpx.Response(200, json=envelope([])))
        result = self.upload().json()
        self.assertEqual((result["total"], result["counts"]), (0, {}))

    def test_schema_and_text_modes_are_explicit(self):
        self.install(lambda request: httpx.Response(200, json=envelope()))
        with patch.object(settings, "vision_response_mode", "json_schema"):
            self.assertEqual(self.upload().status_code, 200)
        body = json.loads(self.requests[-1].content)
        schema = body["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(set(schema["schema"]["required"]), {"items", "uncertainties"})
        with patch.object(settings, "vision_response_mode", "text"):
            with patch.object(settings, "vision_token_limit_field", "max_tokens"):
                self.install(lambda request: httpx.Response(200, json=envelope(
                    content="```json\n" + json.dumps({"items": [], "uncertainties": []}) + "\n```",
                )))
                self.assertEqual(self.upload().status_code, 200)
        body = json.loads(self.requests[-1].content)
        self.assertNotIn("response_format", body)
        self.assertIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)

    def test_invalid_counts_extra_fields_and_duplicate_categories(self):
        invalid_items = [[item(count=value)] for value in (True, 0, -1, 1.5, "2", 501)]
        invalid_items += [[item(), item()], [dict(item(), total=20)], [item(" ", "cup")]]
        for items in invalid_items:
            with self.subTest(items=items):
                self.install(lambda request: httpx.Response(200, json=envelope(items)))
                response = self.upload()
                self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(len(self.requests), len(invalid_items))

    def test_invalid_json_or_envelope_never_produces_empty_success(self):
        values = [
            {}, {"choices": []}, {"choices": [{}, {}]},
            envelope(content="不是 JSON"),
            envelope(content='{"items":[],"items":[],"uncertainties":[]}'),
            envelope(content='{"items":[],"uncertainties":[],"total":0}'),
            envelope(content='{"items":[],"uncertainties":[""]}'),
        ]
        for value in values:
            with self.subTest(value=value):
                self.install(lambda request: httpx.Response(200, json=value))
                self.assertEqual(self.upload().status_code, 502)

    def test_truncation_refusal_and_unfinished_response(self):
        for finish, expected in (("length", 502), ("content_filter", 502), (None, 502)):
            self.install(lambda request: httpx.Response(200, json=envelope(finish_reason=finish)))
            self.assertEqual(self.upload().status_code, expected)
        value = envelope()
        value["choices"][0]["message"]["refusal"] = "上游敏感原文不应泄露"
        self.install(lambda request: httpx.Response(200, json=value))
        response = self.upload()
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("上游敏感", response.text)

    def test_upstream_errors_are_sanitized_and_not_retried(self):
        for code, expected in ((401, 502), (403, 502), (429, 429), (400, 502), (404, 502), (500, 502), (302, 502)):
            with self.subTest(code=code):
                self.install(lambda request: httpx.Response(
                    code, text="test-secret-no-leak 上游内部正文", headers={"location": "https://other.example.test"},
                ))
                before = len(self.requests)
                response = self.upload()
                self.assertEqual(response.status_code, expected)
                self.assertEqual(len(self.requests), before + 1)
                self.assertNotIn("test-secret-no-leak", response.text)
                self.assertNotIn("上游内部正文", response.text)

    def test_network_and_total_timeouts(self):
        for error, expected in ((httpx.ConnectError("密钥内部信息"), 502), (httpx.ReadTimeout("超时"), 504)):
            def fail(request):
                raise error
            self.install(fail)
            self.assertEqual(self.upload().status_code, expected)
        async def slow(request):
            await asyncio.sleep(0.1)
            return httpx.Response(200, json=envelope())
        self.stack.enter_context(patch("app.api.vision.vision_service", VisionService(httpx.MockTransport(slow))))
        with patch.object(settings, "vision_timeout_seconds", 0.01):
            self.assertEqual(self.upload().status_code, 504)

    def test_response_size_and_non_json_transport_body(self):
        for data in (b"x" * (MAX_RESPONSE_BYTES + 1), b"<html>gateway error</html>", b"[]"):
            self.install(lambda request: httpx.Response(200, content=data))
            self.assertEqual(self.upload().status_code, 502)

    def test_missing_config_and_bad_urls_do_not_call_service(self):
        self.install(lambda request: self.fail("缺少配置时不能联网"))
        for field in ("vision_api_key", "vision_model", "vision_api_base_url"):
            with patch.object(settings, field, ""):
                self.assertEqual(self.upload().status_code, 503)
        for url in ("file:///tmp/example", "https://name:secret@example.test/v1", "https://example.test/v1?key=secret", "http://example.test:wrong/v1"):
            with patch.object(settings, "vision_api_base_url", url):
                response = self.upload()
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("secret", response.text)
        self.assertEqual(self.requests, [])

    def test_upload_validation_happens_before_remote_call(self):
        self.install(lambda request: self.fail("错误图片不能联网"))
        self.assertEqual(self.upload(b"invalid").status_code, 400)
        with patch.object(settings, "max_upload_bytes", 30):
            self.assertEqual(self.upload().status_code, 413)
        self.assertEqual(self.client.post("/api/vision").status_code, 422)
        self.assertEqual(self.requests, [])

    def test_unknown_errors_do_not_log_secrets(self):
        with patch("app.api.vision.vision_service.analyze", side_effect=RuntimeError("test-secret-no-leak")):
            with self.assertLogs("app.api.vision", level="ERROR") as logs:
                response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("test-secret-no-leak", response.text + str(logs.output))

    def test_both_pipelines_share_normalized_exclusion_config(self):
        config = Settings(_env_file=None, inventory_excluded_labels=[" CUP ", "cup", "Dining  Table"])
        self.assertEqual(config.inventory_excluded_labels, ["cup", "dining table"])

    def test_usage_is_optional_and_category_names_are_normalized(self):
        value = envelope([item(" 杯子 ", " CUP ")])
        value["usage"] = {"prompt_tokens": True, "completion_tokens": -1, "total_tokens": "8"}
        self.install(lambda request: httpx.Response(200, json=value))
        result = self.upload().json()
        self.assertEqual(result["inventory"][0]["label_en"], "cup")
        self.assertEqual(result["counts"], {"杯子": 1})
        self.assertIsNone(result["usage"])


if __name__ == "__main__":
    unittest.main()
