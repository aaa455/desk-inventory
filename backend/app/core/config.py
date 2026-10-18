from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """集中读取项目根目录的配置，系统环境变量优先于配置文件。"""

    app_name: str = "桌面物品清点"
    app_env: str = "development"
    yolo_model_path: str = "models/yolo11n.pt"
    yolo_device: str = "auto"
    sam_model_path: str = "models/sam2.1_t.pt"
    inventory_excluded_labels: list[str] = ["person", "dining table", "chair", "couch", "bed"]
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=20_000_000, gt=0)
    max_image_side: int = Field(default=1600, ge=64, le=4096)
    vision_api_base_url: str = "https://api.openai.com/v1"
    vision_api_key: str = Field(default="", repr=False)
    vision_model: str = ""
    vision_timeout_seconds: float = Field(default=120, ge=1, le=600)
    vision_max_output_tokens: int = Field(default=4096, ge=256, le=32768)
    vision_response_mode: Literal["json_schema", "json_object", "text"] = "json_object"
    vision_token_limit_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"

    @field_validator("inventory_excluded_labels")
    @classmethod
    def normalize_excluded_labels(cls, labels):
        """两条流程统一使用小写且去重的英文类别配置。"""
        normalized = [" ".join(label.lower().split()) for label in labels]
        if any(not label for label in normalized):
            raise ValueError("排除类别不能为空。")
        return list(dict.fromkeys(normalized))

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
