"""中文终端展示、本地结果导出与人工清点文件，不发起模型请求。"""

import base64
import csv
import io
import json
import unicodedata
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.comparison import ComparisonResponse, CountEntry, EvaluationRequest
from app.services.comparison import evaluate_counts


class TruthEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
    label_en: str = Field(alias="英文类别")
    label: str = Field(default="", alias="中文类别", max_length=80)
    count: int = Field(alias="真实数量", ge=0, le=500)

    @field_validator("label_en")
    @classmethod
    def check_name(cls, value):
        return CountEntry(label_en=value, count=0).label_en


class TruthDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_sha256: str = Field(alias="图片摘要", pattern=r"^[a-f0-9]{64}$")
    description: str = Field(alias="说明", max_length=1000)
    items: list[TruthEntry] = Field(alias="类别", max_length=200)


def successful_inventory(comparison, name):
    branch = getattr(comparison, name)
    return branch.result.inventory if branch.status == "success" else None


def category_names(comparison):
    result = {}
    for name in ("traditional", "vision"):
        for item in successful_inventory(comparison, name) or []:
            result.setdefault(item.label_en, item.label)
    return dict(sorted(result.items()))


def evaluate_document(comparison: ComparisonResponse, document: TruthDocument, confirmed: bool):
    if document.image_sha256 != comparison.image_sha256:
        raise ValueError("人工清点文件的图片摘要与分析结果不同，请使用对应的模板。")
    values = {}
    for name in ("traditional", "vision"):
        inventory = successful_inventory(comparison, name)
        values[name] = [CountEntry(label_en=item.label_en, count=item.count) for item in inventory] if inventory is not None else None
    return evaluate_counts(EvaluationRequest(
        **values, truth=[CountEntry(label_en=item.label_en, count=item.count) for item in document.items],
        confirmed=confirmed,
    ))


def safe_csv_cell(value):
    text = str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        text = "'" + text
    return text


def count_csv(comparison, evaluation=None):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["图片摘要", comparison.image_sha256])
    writer.writerow(["方案状态", "本地：" + comparison.traditional.status, "大模型：" + comparison.vision.status])
    writer.writerow(["英文类别", "中文类别", "本地数量", "大模型数量", "大模型减本地", "人工清点"])
    names = category_names(comparison)
    truth = {entry.label_en: entry.count for entry in evaluation.truth} if evaluation else {}
    for label in truth:
        names.setdefault(label, label)
    local_items, remote_items = successful_inventory(comparison, "traditional"), successful_inventory(comparison, "vision")
    local = {item.label_en: item.count for item in local_items or []}
    remote = {item.label_en: item.count for item in remote_items or []}
    for label, name in sorted(names.items()):
        left = local.get(label, 0) if local_items is not None else ""
        right = remote.get(label, 0) if remote_items is not None else ""
        delta = right - left if comparison.comparable else ""
        writer.writerow([safe_csv_cell(value) for value in (label, name, left, right, delta, truth.get(label, ""))])
    return buffer.getvalue()


def plain_text(value):
    # 模型或类别名不能在终端注入控制序列。
    return "".join(char if not unicodedata.category(char).startswith("C") else " " for char in str(value))


def print_table(headers, rows):
    values = [[plain_text(value) for value in row] for row in [headers, *rows]]
    def width(text):
        return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)
    widths = [max(width(row[index]) for row in values) for index in range(len(headers))]
    for index, row in enumerate(values):
        print("  ".join(value + " " * (size - width(value)) for value, size in zip(row, widths)))
        if index == 0:
            print("  ".join("─" * size for size in widths))


def print_comparison(comparison, evaluation=None):
    print("\n桌面清点 · 同图双方案")
    print("图片摘要：" + comparison.image_sha256)
    for name, title in (("traditional", "本地视觉"), ("vision", "视觉大模型")):
        branch = getattr(comparison, name)
        if branch.status == "success":
            result = branch.result
            print(f"{title}：{result.total} 件，总耗时 {result.elapsed_ms / 1000:.2f} 秒")
            if name == "vision":
                print("请求模型：" + plain_text(result.model))
                uncertain = [item.label + "：" + item.uncertainty for item in result.inventory if item.uncertainty]
                uncertain += result.uncertainties
                if uncertain:
                    for note in uncertain:
                        print("  不确定说明：" + plain_text(note))
                else:
                    print("  模型未报告不确定之处，仍需人工核对。")
        else:
            print(f"{title}：失败（{branch.error.status_code}），" + plain_text(branch.error.message))
    print("两方案总墙钟耗时：" + f"{comparison.elapsed_ms / 1000:.2f} 秒")
    names = category_names(comparison)
    local = {item.label_en: item.count for item in successful_inventory(comparison, "traditional") or []}
    remote = {item.label_en: item.count for item in successful_inventory(comparison, "vision") or []}
    rows = []
    for label, name in names.items():
        lc = local.get(label, 0) if comparison.traditional.status == "success" else "未完成"
        rc = remote.get(label, 0) if comparison.vision.status == "success" else "未完成"
        delta = f"{rc - lc:+d}" if comparison.comparable else "—"
        rows.append([name, label, lc, rc, delta])
    if rows:
        print_table(["类别", "英文类别", "本地", "大模型", "大模型减本地"], rows)
    else:
        print("没有可展示的成功清单；空清单与分析失败请看上方状态。")
    print("按英文类别精确对齐，差异不是准确率。未检出不等于没有物体。")
    if evaluation:
        print("\n人工清点 · 计数误差")
        rows = []
        for name, title in (("traditional", "本地视觉"), ("vision", "视觉大模型")):
            metrics = getattr(evaluation, name)
            if metrics:
                rows.append([title, metrics.truth_total, metrics.category_absolute_error,
                             metrics.category_mean_absolute_error if metrics.category_mean_absolute_error is not None else "无类别",
                             metrics.missing_count, metrics.extra_count, metrics.total_absolute_error])
        print_table(["方案", "人工总数", "类别误差合计", "类别平均误差", "少计", "多计", "总数误差"], rows)
        print("仅评价这张图片的类别计数，不代表检测、分割或数据集总体准确率。")


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_comparison(comparison: ComparisonResponse, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", comparison.model_dump())
    (output / "counts.csv").write_text(count_csv(comparison), encoding="utf-8-sig")
    template = {
        "图片摘要": comparison.image_sha256,
        "说明": "看原图填写真实数量（0 到 500 的整数），不要保留 null。补充两方案都漏掉的类别；同义词先统一口径。空图可保留空类别数组。",
        "类别": [{"中文类别": name, "英文类别": label, "真实数量": None} for label, name in category_names(comparison).items()],
    }
    write_json(output / "人工清点模板.json", template)
    branch = comparison.traditional
    if branch.status == "success":
        result = branch.result
        (output / "segmentation.jpg").write_bytes(base64.b64decode(result.annotated_image.split(",", 1)[1], validate=True))
        for item in result.instances:
            if item.mask:
                (output / f"mask-{item.instance_id}.png").write_bytes(
                    base64.b64decode(item.mask.png.split(",", 1)[1], validate=True))
