from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """集中读取项目根目录的配置，系统环境变量优先于配置文件。"""

    app_name: str = "桌面物品清点"
    app_env: str = "development"
    yolo_model_path: str = "models/yolo11n.pt"
    vision_api_base_url: str = "https://api.openai.com/v1"
    vision_api_key: str = ""
    vision_model: str = ""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
