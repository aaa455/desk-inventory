"""不启动网页或 HTTP 服务，直接运行双方案；也能只评估已保存的结果。"""

import argparse
import asyncio
import math
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pydantic import ValidationError

from app.core.config import settings
from app.schemas.comparison import ComparisonResponse
from app.services.comparison import analyze_comparison
from app.services.images import InvalidImage
from app.services.reporting import (
    TruthDocument, count_csv, evaluate_document, export_comparison, print_comparison, write_json,
)


class ChineseParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, "参数错误：" + message + "\n")


def bounded_read(path: Path, limit: int) -> bytes:
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError("文件为空或超出读取上限。")
    return data


def unique_output(parent: Path, prefix: str) -> Path:
    return parent / (prefix + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = ChineseParser(description="直接在终端比较同图双方案，无需启动后端；真实分析会发送图片给所配置的视觉服务")
    parser.add_argument("image", type=Path, nargs="?", help="本地静态照片")
    parser.add_argument("--evaluate-result", type=Path, help="仅评估已保存的 comparison.json，不调用模型")
    parser.add_argument("--truth", type=Path, help="已填写的 UTF-8 人工清点模板")
    parser.add_argument("--confirm-truth", action="store_true", help="确认已看原图并覆盖全部类别")
    parser.add_argument("--include-background", action="store_true", help="将人和家具也计入清单")
    parser.add_argument("--confidence", type=float, default=.25, help="检测置信度，默认 0.25")
    parser.add_argument("--iou", type=float, default=.45, help="重叠框去重阈值，默认 0.45")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="保存结果的父目录")
    args = parser.parse_args()
    if bool(args.image) == bool(args.evaluate_result):
        parser.error("请选择图片分析或 --evaluate-result 离线评估，两者只用一个。")
    if not all(math.isfinite(value) and .01 <= value <= 1 for value in (args.confidence, args.iou)):
        parser.error("检测阈值必须是 0.01 到 1 的有限数值。")
    if args.evaluate_result and (not args.truth or not args.confirm_truth):
        parser.error("离线评估需要 --truth 和 --confirm-truth。")
    if args.image and (args.truth or args.confirm_truth):
        parser.error("先运行图片分析，再用保存的结果和对应模板离线评估。")
    try:
        if args.evaluate_result:
            comparison = ComparisonResponse.model_validate_json(bounded_read(args.evaluate_result, 128 * 1024 * 1024).decode("utf-8-sig"))
            document = TruthDocument.model_validate_json(bounded_read(args.truth, 256 * 1024).decode("utf-8-sig"))
            evaluation = evaluate_document(comparison, document, args.confirm_truth)
            output = unique_output(args.output, "evaluation")
            output.mkdir(parents=True)
            write_json(output / "evaluation.json", {
                "图片摘要": comparison.image_sha256, "评估": evaluation.model_dump(),
                "人工类别说明": [entry.model_dump(by_alias=True) for entry in document.items],
            })
            (output / "counts.csv").write_text(count_csv(comparison, evaluation), encoding="utf-8-sig", newline="")
            print_comparison(comparison, evaluation)
            print("\n离线评估已保存：" + str(output))
            print("本次没有调用模型或发送图片。")
            return 0
        data = bounded_read(args.image, settings.max_upload_bytes)
        print("正在分析同一张照片：YOLO + SAM 与视觉大模型。")
        print("图片会发送到 .env 中配置的视觉服务，可能产生费用；不自动重试。")
        comparison = asyncio.run(analyze_comparison(data, args.confidence, args.iou, args.include_background))
        output = unique_output(args.output, "comparison")
        export_comparison(comparison, output)
        print_comparison(comparison)
        print("\n结果目录：" + str(output))
        print("已保存完整 JSON、计数 CSV、人工清点模板，以及本地方案成功时的叠加图与掩膜。")
        print("填写 人工清点模板.json 后，用 -结果 和 -人工清单 参数重新核对，无需再次分析图片。")
        return 0 if comparison.comparable else 2
    except (ValidationError, ValueError, InvalidImage) as exc:
        if isinstance(exc, ValidationError):
            print("输入文件结构无效：请检查模板的类别名、整数数量、图片摘要，以及是否存在重复或遗漏类别。", file=sys.stderr)
        else:
            print("处理失败：" + str(exc), file=sys.stderr)
        return 1
    except OSError:
        print("文件读取或保存失败，请检查路径、权限和磁盘空间。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
