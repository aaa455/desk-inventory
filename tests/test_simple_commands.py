"""验证简化命令复用已有结果，人工输入不抄模型答案且不触发分析。"""

import csv
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from test_comparison import run_mock, sample
import compare_image
from app.schemas.comparison import AnalysisError, TraditionalBranch, VisionBranch
from app.services.reporting import export_comparison


class SimpleCommandTests(unittest.TestCase):
    def invoke(self, arguments, answers=()):
        with patch.object(sys, "argv", ["compare_image.py", *arguments]):
            with patch("builtins.input", side_effect=answers):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    return compare_image.main()

    def test_recent_uses_analysis_only_and_reports_empty_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            with self.assertRaises(ValueError):
                compare_image.recent_result(parent)
            for name in ("comparison-20261018-120000-a", "comparison-20261018-130000-b",
                         "evaluation-20261018-140000-c"):
                folder = parent / name
                folder.mkdir()
                (folder / "comparison.json").write_text("{}", encoding="utf-8")
            (parent / "comparison-20261018-150000-incomplete").mkdir()
            self.assertEqual(compare_image.recent_result(parent).parent.name,
                             "comparison-20261018-130000-b")

    def test_same_second_results_use_write_time_instead_of_random_suffix(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            paths = []
            for suffix in ("zzzzzzzz", "aaaaaaaa"):
                folder = parent / ("comparison-20261018-120000-" + suffix)
                folder.mkdir()
                path = folder / "comparison.json"
                path.write_text("{}", encoding="utf-8")
                paths.append(path)
            os.utime(paths[0], ns=(1_000_000_000, 1_000_000_000))
            os.utime(paths[1], ns=(2_000_000_000, 2_000_000_000))
            self.assertEqual(compare_image.recent_result(parent), paths[1])

    def test_show_only_reads_and_does_not_open_a_window_by_default(self):
        result = run_mock()
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            export_comparison(result, parent / "comparison-20261018-120000-a")
            before = set(parent.rglob("*"))
            with patch("compare_image.analyze_comparison", side_effect=AssertionError("查看不允许分析")):
                with patch("os.startfile", create=True) as opener:
                    self.assertEqual(self.invoke(["--show", "--output", temp]), 0)
                    opener.assert_not_called()
            self.assertEqual(set(parent.rglob("*")), before)

    def test_review_saves_manual_counts_and_both_missed_category_without_analysis(self):
        result = run_mock()
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            export_comparison(result, parent / "comparison-20261018-120000-a")
            # cup 和 saucer 按英文排序提问，pen 是两侧都没有的类别。
            answers = ["2", "1", "pen", "笔", "3", "", "是"]
            with patch("compare_image.analyze_comparison", side_effect=AssertionError("核对不允许分析")):
                self.assertEqual(self.invoke(["--review", "--output", temp], answers), 0)
            evaluation_path = next(parent.glob("evaluation-*/evaluation.json"))
            truth = json.loads((evaluation_path.parent / "人工清点.json").read_text(encoding="utf-8"))
            self.assertEqual({item["英文类别"]: item["真实数量"] for item in truth["类别"]},
                             {"cup": 2, "saucer": 1, "pen": 3})
            metrics = json.loads(evaluation_path.read_text(encoding="utf-8"))["评估"]
            self.assertEqual(metrics["traditional"]["missing_count"], 5)
            self.assertEqual(metrics["vision"]["missing_count"], 4)
            raw = (evaluation_path.parent / "counts.csv").read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            self.assertNotIn(b"\r\r\n", raw)
            with (evaluation_path.parent / "counts.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertTrue(all(rows))
            self.assertEqual(len(rows), 6)

    def test_empty_and_invalid_counts_are_not_accepted_as_zero(self):
        answers = ["", "true", "-1", "1.5", "501", "9" * 5000, "2"]
        with patch("builtins.input", side_effect=answers), redirect_stdout(io.StringIO()):
            self.assertEqual(compare_image.ask_count("杯子"), 2)

    def test_duplicate_and_invalid_extra_categories_are_reprompted(self):
        result = run_mock()
        answers = ["1", "1", " CUP ", "笔", "pen", "笔", "1", "", "是"]
        with patch("builtins.input", side_effect=answers), redirect_stdout(io.StringIO()):
            truth = compare_image.collect_truth(result)
        self.assertEqual([item.label_en for item in truth.items], ["cup", "saucer", "pen"])

    def test_cancellation_and_missing_confirmation_leave_no_review_files(self):
        result = run_mock()
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            export_comparison(result, parent / "comparison-20261018-120000-a")
            for answers in (["q"], ["1", "1", "", "否"], []):
                with self.subTest(answers=answers):
                    # 空输入模拟终端输入结束，而不是伪造确认。
                    if not answers:
                        with patch.object(sys, "argv", ["compare_image.py", "--review", "--output", temp]):
                            with patch("builtins.input", side_effect=EOFError):
                                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                                    self.assertEqual(compare_image.main(), 1)
                    else:
                        self.assertEqual(self.invoke(["--review", "--output", temp], answers), 1)
                    self.assertEqual(list(parent.glob("evaluation-*")), [])

    def test_partial_failure_can_be_reviewed_but_two_failures_cannot(self):
        result = run_mock()
        result.vision = VisionBranch(status="error", error=AnalysisError(status_code=503, message="未配置"))
        result.comparable = False
        result.differences = []
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            folder = parent / "comparison-20261018-120000-a"
            export_comparison(result, folder)
            self.assertEqual(self.invoke(["--review", "--output", temp], ["1", "", "是"]), 0)
            evaluation = json.loads(next(parent.glob("evaluation-*/evaluation.json")).read_text(encoding="utf-8"))
            self.assertIsNone(evaluation["评估"]["vision"])
            result.traditional = TraditionalBranch(
                status="error", error=AnalysisError(status_code=503, message="缺少权重")
            )
            (folder / "comparison.json").write_text(result.model_dump_json(), encoding="utf-8")
            with patch("builtins.input", side_effect=AssertionError("双侧失败不应提问")):
                with self.assertRaises(ValueError):
                    compare_image.collect_truth(result)

    def test_show_or_review_without_history_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as temp:
            for option in ("--show", "--review"):
                with patch("compare_image.analyze_comparison", side_effect=AssertionError("不允许分析")):
                    self.assertEqual(self.invoke([option, "--output", temp]), 1)

    def test_analysis_still_calls_existing_pipeline_once(self):
        data = sample()
        result = run_mock(data)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "含空格的照片.png"
            source.write_bytes(data)
            with patch("compare_image.analyze_comparison", new=AsyncMock(return_value=result)) as analyzer:
                self.assertEqual(self.invoke([str(source), "--output", temp]), 0)
                analyzer.assert_awaited_once_with(data, .25, .45, False)
            self.assertEqual(len(list(Path(temp).glob("comparison-*/comparison.json"))), 1)

    def test_conflicting_modes_are_rejected_before_analysis(self):
        for arguments in (["--show", "--review"], ["a.png", "--show"], ["--open-image"],
                          ["--review", "--truth", "a.json"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as raised:
                self.invoke(arguments)
            self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
