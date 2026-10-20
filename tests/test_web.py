"""网页首页和前端入口的基本连通性测试。"""

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app


class WebPageTests(unittest.TestCase):
    def test_home_page_contains_upload_and_comparison_workflow(self):
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("/api/compare", response.text)
        self.assertIn("本地方案 · YOLO + SAM", response.text)
        self.assertIn("视觉大模型", response.text)
        self.assertIn("视觉大模型 − 本地方案", response.text)
        self.assertIn("图片会发送至已配置的视觉模型服务", response.text)
        self.assertIn("item.uncertainty", response.text)
        self.assertIn("不确定说明", response.text)
        self.assertIn("/api/compare/evaluate", response.text)
        self.assertIn("truth-confirm", response.text)
        self.assertIn("下载 JSON", response.text)
        self.assertIn("下载 CSV", response.text)
        self.assertIn("function currentTruthEntries()", response.text)


if __name__ == "__main__":
    unittest.main()
