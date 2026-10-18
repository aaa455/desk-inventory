import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.api.uploads import read_upload
from app.schemas.segmentation import PromptResponse, SegmentationPrompt, SegmentationResponse
from app.services.detector import ModelUnavailable
from app.services.images import InvalidImage
from app.services.segmenter import segmenter

router = APIRouter(tags=["实例分割"])
logger = logging.getLogger(__name__)


async def run_segmentation(function, *args):
    try:
        return await run_in_threadpool(function, *args)
    except InvalidImage as exc:
        raise HTTPException(400, str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("图片分割失败")
        raise HTTPException(503, "分割暂时失败，请检查模型环境或稍后重试。") from exc


@router.post("/segment", response_model=SegmentationResponse, summary="分割并生成物品清单")
async def segment(
    file: Annotated[UploadFile, File(description="桌面照片")],
    confidence: Annotated[float, Query(ge=0.01, le=1)] = 0.25,
    iou: Annotated[float, Query(ge=0.01, le=1)] = 0.45,
    include_background: Annotated[bool, Query(description="将默认排除的人和家具也计入清单")] = False,
) -> SegmentationResponse:
    return await run_segmentation(segmenter.inventory, await read_upload(file), confidence, iou, include_background)


@router.post("/segment/prompt", response_model=PromptResponse, summary="按点或框提示分割指定区域")
async def prompted(
    file: Annotated[UploadFile, File(description="需要手动分割的照片")],
    prompt: Annotated[str, Form(description='JSON，例如 {"kind":"box","box":[10,20,100,120]}')],
) -> PromptResponse:
    data = await read_upload(file)
    try:
        parsed = SegmentationPrompt.model_validate_json(prompt)
    except ValidationError as exc:
        raise HTTPException(422, "提示格式无效：填写有效的框或点 JSON，坐标应为有限数值。") from exc
    return await run_segmentation(segmenter.prompted, data, parsed)
