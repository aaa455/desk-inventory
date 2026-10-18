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
    TruthDocument, TruthEntry, category_names, count_csv, evaluate_document, export_comparison,
    plain_text, print_comparison, write_json,
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


def recent_result(parent: Path) -> Path:
    candidates = sorted(parent.glob("comparison-*/comparison.json"),
                        key=lambda path: (path.parent.name.rsplit("-", 1)[0], path.stat().st_mtime_ns),
                        reverse=True)
    if not candidates:
        raise ValueError("尚无分析结果，请先输入照片运行清点；自定义输出目录需使用同一个 -Output。")
    return candidates[0]


def load_comparison(path: Path) -> ComparisonResponse:
    return ComparisonResponse.model_validate_json(
        bounded_read(path, 128 * 1024 * 1024).decode("utf-8-sig")
    )


def save_evaluation(comparison, document, parent: Path):
    evaluation = evaluate_document(comparison, document, True)
    output = unique_output(parent, "evaluation")
    output.mkdir(parents=True)
    write_json(output / "人工清点.json", document.model_dump(by_alias=True))
    write_json(output / "evaluation.json", {
        "图片摘要": comparison.image_sha256, "评估": evaluation.model_dump(),
        "人工类别说明": [entry.model_dump(by_alias=True) for entry in document.items],
    })
    (output / "counts.csv").write_text(
        count_csv(comparison, evaluation), encoding="utf-8-sig", newline=""
    )
    print_comparison(comparison, evaluation)
    print("\n离线评估已保存：" + str(output))
    print("本次没有调用模型或发送图片。")


def ask_count(label: str) -> int:
    while True:
        value = input(plain_text(label) + " 的真实数量（0—500，输入 q 取消）：").strip()
        if value.lower() == "q":
            raise ValueError("已取消人工核对，没有保存本次输入。")
        if len(value) <= 3 and value.isascii() and value.isdecimal() and 0 <= int(value) <= 500:
            return int(value)
        print("请输入 0 到 500 的整数；空白不等于零。")


def collect_truth(comparison) -> TruthDocument:
    if comparison.traditional.status != "success" and comparison.vision.status != "success":
        raise ValueError("两侧都未成功，没有可评估的清单。")
    print("请对照原图输入实际数量，不要照抄模型答案；没有的类别填 0。")
    names = category_names(comparison)
    entries = [
        TruthEntry(label_en=english, label=label, count=ask_count(f"{label} / {english}"))
        for english, label in names.items()
    ]
    while True:
        english = input("补充两侧都漏掉的英文类别（如 pen；直接回车表示补充完毕）：").strip()
        if not english:
            break
        try:
            item = TruthEntry(label_en=english, count=0)
            if item.label_en in {entry.label_en for entry in entries}:
                print("该英文类别已经填写，请勿重复。")
                continue
            if len(entries) >= 200:
                raise ValueError("人工清单最多支持 200 个类别。")
        except ValidationError:
            print("英文类别格式无效，请使用小写英文单词，如 pen 或 pencil case。")
            continue
        label = input("中文类别（可直接回车）：").strip()
        if len(label) > 80:
            print("中文类别最多 80 个字符，请重新补充该类别。")
            continue
        entries.append(TruthEntry(
            label_en=item.label_en, label=label, count=ask_count(label or item.label_en)
        ))
    document = TruthDocument.model_validate({
        "图片摘要": comparison.image_sha256,
        "说明": "通过终端逐项输入人工数量，已确认包含两方案均漏掉的类别。",
        "类别": entries,
    })
    print("\n请检查本次人工清单：")
    for entry in entries:
        print(f"{plain_text(entry.label)} / {plain_text(entry.label_en)}：{entry.count}")
    if input("已看原图并确认所有类别和数量正确？输入“是”保存并评估：").strip() != "是":
        raise ValueError("未确认人工清点，没有保存本次输入。")
    return document


def show_saved(comparison, path: Path, open_image: bool):
    print("读取已有结果：" + str(path))
    print_comparison(comparison)
    print("\n结果文件：")
    for item in sorted(path.parent.iterdir()):
        if item.is_file():
            print("  " + item.name)
    if open_image:
        image = path.parent / "segmentation.jpg"
        if not image.is_file():
            print("没有本地分割图，请检查本地分支状态。")
        else:
            import os
            if hasattr(os, "startfile"):
                os.startfile(str(image.resolve()))
            else:
                print("请用本地看图工具打开：" + str(image))
    print("本次只读取文件，没有调用模型或发送图片。")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = ChineseParser(description="直接在终端比较同图双方案，无需启动后端；真实分析会发送图片给所配置的视觉服务")
    parser.add_argument("image", type=Path, nargs="?", help="本地静态照片")
    parser.add_argument("--evaluate-result", type=Path, help="仅评估已保存的 comparison.json，不调用模型")
    parser.add_argument("--show", action="store_true", help="查看最近分析，不调用模型")
    parser.add_argument("--review", action="store_true", help="终端输入人工数量并离线评估最近分析")
    parser.add_argument("--open-image", action="store_true", help="查看时用系统看图工具打开分割图")
    parser.add_argument("--truth", type=Path, help="已填写的 UTF-8 人工清点模板")
    parser.add_argument("--confirm-truth", action="store_true", help="确认已看原图并覆盖全部类别")
    parser.add_argument("--include-background", action="store_true", help="将人和家具也计入清单")
    parser.add_argument("--confidence", type=float, default=.25, help="检测置信度，默认 0.25")
    parser.add_argument("--iou", type=float, default=.45, help="重叠框去重阈值，默认 0.45")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="保存结果的父目录")
    args = parser.parse_args()
    saved_mode = args.show or args.review
    if args.show and args.review:
        parser.error("查看和人工核对只能选择一个。")
    if args.open_image and not args.show:
        parser.error("--open-image 需要 --show。")
    if saved_mode and (args.image or args.truth or args.confirm_truth):
        parser.error("简化查看或核对不接受图片、人工清单文件和确认开关。")
    if not saved_mode and bool(args.image) == bool(args.evaluate_result):
        parser.error("请选择图片分析或 --evaluate-result 离线评估，两者只用一个。")
    if not all(math.isfinite(value) and .01 <= value <= 1 for value in (args.confidence, args.iou)):
        parser.error("检测阈值必须是 0.01 到 1 的有限数值。")
    if not saved_mode and args.evaluate_result and (not args.truth or not args.confirm_truth):
        parser.error("离线评估需要 --truth 和 --confirm-truth。")
    if args.image and (args.truth or args.confirm_truth):
        parser.error("先运行图片分析，再用保存的结果和对应模板离线评估。")
    try:
        if saved_mode:
            path = args.evaluate_result or recent_result(args.output)
            comparison = load_comparison(path)
            if args.show:
                show_saved(comparison, path, args.open_image)
            else:
                print("正在核对：" + str(path))
                print("图片摘要：" + comparison.image_sha256)
                document = collect_truth(comparison)
                save_evaluation(comparison, document, args.output)
            return 0
        if args.evaluate_result:
            comparison = load_comparison(args.evaluate_result)
            document = TruthDocument.model_validate_json(
                bounded_read(args.truth, 256 * 1024).decode("utf-8-sig")
            )
            save_evaluation(comparison, document, args.output)
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
        print("接下来运行 inventory.ps1 -Show 或 -Review；自定义输出目录时继续带 -Output。")
        return 0 if comparison.comparable else 2
    except (ValidationError, ValueError, InvalidImage) as exc:
        if isinstance(exc, ValidationError):
            print("输入文件结构无效：请检查模板的类别名、整数数量、图片摘要，以及是否存在重复或遗漏类别。", file=sys.stderr)
        else:
            print("处理失败：" + str(exc), file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n已取消操作；已导出的结果仍然保留。", file=sys.stderr)
        return 1
    except OSError:
        print("文件读取或保存失败，请检查路径、权限和磁盘空间。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
