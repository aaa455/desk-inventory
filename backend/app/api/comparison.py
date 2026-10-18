"""同图对比接口；命令行直接调用相同服务，无需启动 HTTP。"""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.uploads import read_upload
from app.schemas.comparison import ComparisonResponse, EvaluationRequest, EvaluationResponse
from app.services.comparison import analyze_comparison, evaluate_counts
from app.services.images import InvalidImage

router = APIRouter(tags=["同图对比"])


@router.post("/compare", response_model=ComparisonResponse, summary="一次上传并发分析同一张图片")
async def compare(
    file: Annotated[UploadFile, File(description="供本地模型和已配置视觉服务分析的图片")],
    confidence: Annotated[float, Query(ge=0.01, le=1)] = 0.25,
    iou: Annotated[float, Query(ge=0.01, le=1)] = 0.45,
    include_background: Annotated[bool, Query()] = False,
) -> ComparisonResponse:
    data = await read_upload(file)
    try:
        return await analyze_comparison(data, confidence, iou, include_background)
    except InvalidImage as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/compare/evaluate", response_model=EvaluationResponse, summary="根据人工清点计算各方案计数误差")
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    return evaluate_counts(request)
