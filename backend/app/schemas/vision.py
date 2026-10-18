"""视觉大模型的受约束输出，以及后端计算的物品清单。"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.detection import ImageInfo


class VisionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label: str = Field(min_length=1, max_length=80, description="简短中文类别名")
    label_en: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9 '/-]*$", description="统一的英文类别名")
    count: int = Field(ge=1, le=500, description="该类别可见物体的整数数量")
    uncertainty: str = Field(max_length=300, description="遮挡或难辨认之处，无则为空字符串")

    @field_validator("label", "uncertainty", mode="before")
    @classmethod
    def trim_text(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("label_en", mode="before")
    @classmethod
    def normalize_label(cls, value):
        return " ".join(value.lower().split()) if isinstance(value, str) else value


class VisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[VisionItem] = Field(max_length=100)
    uncertainties: list[str] = Field(max_length=20, description="整图的不确定说明，没有则为空数组")

    @field_validator("uncertainties")
    @classmethod
    def check_notes(cls, values):
        if any(not value.strip() or len(value) > 300 for value in values):
            raise ValueError("不确定说明应为非空且不超过三百字的文本。")
        return [value.strip() for value in values]

    @model_validator(mode="after")
    def unique_categories(self):
        for field in ("label", "label_en"):
            labels = [getattr(item, field) for item in self.items]
            if len(labels) != len(set(labels)):
                raise ValueError("同一类别应合并为一条，避免重复清点。")
        return self


class ExcludedVisionItem(VisionItem):
    exclusion_reason: str


class TokenUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class VisionResponse(BaseModel):
    image: ImageInfo
    model: str = Field(description="请求的模型名")
    returned_model: str | None = Field(description="服务返回的模型标识，服务未提供时为空")
    response_mode: str
    max_output_tokens: int
    token_limit_field: str
    prompt_version: str
    prompt: str = Field(description="完整文字提示，不含图片编码")
    sent_image_sha256: str = Field(description="发送的处理后 PNG 文件摘要")
    elapsed_ms: float = Field(description="预处理、远程调用与校验总耗时，不含上传")
    inference_ms: float = Field(description="远程 HTTP 调用耗时，含网络传输")
    inventory: list[VisionItem]
    excluded_items: list[ExcludedVisionItem]
    excluded_labels: list[str]
    counts: dict[str, int]
    total: int
    uncertainties: list[str]
    usage: TokenUsage | None
    notes: list[str]
