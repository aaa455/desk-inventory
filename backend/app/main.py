from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.detection import router as detection_router
from app.api.segmentation import router as segmentation_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="对同一张桌面图片进行传统视觉分析与多模态模型分析，并比较物品清单。",
    version="0.3.0",
)
app.include_router(health_router, prefix="/api")
app.include_router(detection_router, prefix="/api")
app.include_router(segmentation_router, prefix="/api")
