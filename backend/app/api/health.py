from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="检查服务状态", tags=["服务状态"])
def health_check() -> dict[str, str]:
    """只检查后端服务是否正常响应，不代表模型已就绪。"""
    return {"status": "ok", "service": settings.app_name}
