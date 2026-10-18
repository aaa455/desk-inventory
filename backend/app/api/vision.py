"""视觉模型分析接口；仅使用后端配置的服务与密钥。"""

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.uploads import read_upload
from app.schemas.vision import VisionResponse
from app.services.images import InvalidImage
from app.services.vision import VisionError, vision_service

router = APIRouter(tags=["视觉大模型"])
logger = logging.getLogger(__name__)


@router.post("/vision", response_model=VisionResponse, summary="独立调用视觉大模型生成清单")
async def analyze_vision(
    file: Annotated[UploadFile, File(description="会发送给所配置视觉服务的桌面照片")],
    include_background: Annotated[bool, Query(description="将默认排除的人和家具也计入清单")] = False,
) -> VisionResponse:
    data = await read_upload(file)
    try:
        return await vision_service.analyze(data, include_background)
    except InvalidImage as exc:
        raise HTTPException(400, str(exc)) from exc
    except VisionError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except Exception as exc:
        # 不记录原始异常与上游正文，避免日志泄露鉴权信息或图片编码。
        logger.error("视觉分析出现未预期错误，类型：%s", type(exc).__name__)
        raise HTTPException(503, "视觉分析暂时失败，请检查服务配置或稍后重试。") from exc
