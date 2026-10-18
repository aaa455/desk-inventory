import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.uploads import read_upload
from app.schemas.detection import DetectionResponse
from app.services.detector import ModelUnavailable, detector
from app.services.images import InvalidImage

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/detect", response_model=DetectionResponse, summary="检测桌面物品",
             description="上传 JPEG、PNG 或 WebP 静态图片，返回类别、原图边界框、计数与标注图。",
             tags=["传统视觉"])
async def detect(
    file: Annotated[UploadFile, File(description="需要分析的静态图片")],
    confidence: Annotated[float, Query(ge=0.01, le=1, description="最低检测置信度")] = 0.25,
    iou: Annotated[float, Query(ge=0.01, le=1, description="重叠框去重阈值")] = 0.45,
) -> DetectionResponse:
    data = await read_upload(file)
    try:
        return await run_in_threadpool(detector.analyze, bytes(data), confidence, iou)
    except InvalidImage as exc:
        raise HTTPException(400, str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("图片检测失败")
        raise HTTPException(503, "检测暂时失败，请检查模型运行环境或稍后重试。") from exc
