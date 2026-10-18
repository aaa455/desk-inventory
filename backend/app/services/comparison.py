"""同图双方案编排与计数评估；不做模糊同义词合并。"""

import asyncio
import hashlib
import logging
from time import perf_counter

from starlette.concurrency import run_in_threadpool

from app.schemas.comparison import (
    AnalysisError, ComparisonResponse, CountMetrics, DifferenceRow, EvaluationRequest, EvaluationResponse,
    TraditionalBranch, VisionBranch,
)
from app.services.detector import ModelUnavailable
from app.services.images import InvalidImage, prepare_image
from app.services.segmenter import segmenter
from app.services.vision import VisionError, vision_service

logger = logging.getLogger(__name__)





def compare_inventories(traditional, vision) -> list[DifferenceRow]:
    left = {item.label_en: item for item in traditional}
    right = {item.label_en: item for item in vision}
    rows = []
    for label in sorted(left.keys() | right.keys()):
        local, remote = left.get(label), right.get(label)
        lc = local.count if local else 0
        rc = remote.count if remote else 0
        rows.append(DifferenceRow(
            label_en=label, traditional_label=local.label if local else None,
            vision_label=remote.label if remote else None,
            traditional_count=lc, vision_count=rc, delta=rc - lc,
        ))
    return rows


def count_metrics(entries, truth) -> CountMetrics:
    counts = {item.label_en: item.count for item in entries}
    target = {item.label_en: item.count for item in truth}
    labels = counts.keys() | target.keys()
    missing = sum(max(0, target.get(label, 0) - counts.get(label, 0)) for label in labels)
    extra = sum(max(0, counts.get(label, 0) - target.get(label, 0)) for label in labels)
    total = sum(counts.values())
    actual = sum(target.values())
    error = missing + extra
    return CountMetrics(
        predicted_total=total, truth_total=actual, total_absolute_error=abs(total - actual),
        category_absolute_error=error, category_mean_absolute_error=round(error / len(labels), 4) if labels else None,
        missing_count=missing, extra_count=extra, category_count=len(labels),
    )


def evaluate_counts(request: EvaluationRequest) -> EvaluationResponse:
    return EvaluationResponse(
        traditional=count_metrics(request.traditional, request.truth) if request.traditional is not None else None,
        vision=count_metrics(request.vision, request.truth) if request.vision is not None else None,
        truth=request.truth,
        notes=[
            "人工清点由填写者提供，未由模型输出自动生成；评估仅针对本图和当前类别定义。",
            "按英文类别精确匹配，不模糊合并同义词；请先统一类别口径。",
            "类别绝对误差为各类别计数差绝对值之和；总数相同仍可能存在类别错误。",
            "这些是计数误差，不是目标检测准确率、分割 IoU 或数据集总体准确率。",
            "没有类别时类别平均误差为空，避免把没有评估项说成零误差。",
        ],
    )


def branch_error(exc: Exception) -> AnalysisError:
    if isinstance(exc, VisionError):
        return AnalysisError(status_code=exc.status_code, message=str(exc))
    if isinstance(exc, ModelUnavailable):
        return AnalysisError(status_code=503, message=str(exc))
    if isinstance(exc, InvalidImage):
        return AnalysisError(status_code=400, message=str(exc))
    logger.error("对比分支出现未预期错误，类型：%s", type(exc).__name__)
    return AnalysisError(status_code=503, message="本方案分析暂时失败，请检查服务或模型环境。")


async def analyze_comparison(data: bytes, confidence: float = 0.25, iou: float = 0.45,
                             include_background: bool = False) -> ComparisonResponse:
    """命令行与 HTTP 共用的同图分析，不依赖服务监听端口。"""
    started = perf_counter()
    await run_in_threadpool(prepare_image, data)
    results = await asyncio.gather(
        run_in_threadpool(segmenter.inventory, data, confidence, iou, include_background),
        vision_service.analyze(data, include_background),
        return_exceptions=True,
    )
    digest = hashlib.sha256(data).hexdigest()
    branches = []
    for result, branch_type in zip(results, (TraditionalBranch, VisionBranch)):
        if isinstance(result, Exception):
            branches.append(branch_type(status="error", error=branch_error(result)))
        elif result.image.sha256 != digest:
            branches.append(branch_type(status="error", error=AnalysisError(
                status_code=503, message="方案返回的图片摘要与上传文件不一致，已停止使用此结果。",
            )))
        else:
            branches.append(branch_type(status="success", result=result))
    traditional, vision = branches
    comparable = traditional.status == vision.status == "success"
    differences = compare_inventories(
        traditional.result.inventory, vision.result.inventory,
    ) if comparable else []
    return ComparisonResponse(
        image_sha256=digest, include_background=include_background,
        elapsed_ms=round((perf_counter() - started) * 1000, 2),
        traditional=traditional, vision=vision, comparable=comparable, differences=differences,
        notes=[
            "两条流程收到同一份原始上传字节及相同背景排除选项，彼此不传递识别结果。",
            "只有两方案成功且图片摘要一致时才生成差异；失败不等于识别数量为零。",
            "按英文类别精确对齐，同义词或类别粒度差异需要人工核对。",
            "未填写人工清点时只展示方案差异，不报告准确率。",
            "本地耗时包含预处理、加载和绘图；远程耗时包含网络与服务处理，不能据此比较纯模型速度。",
        ],
    )
