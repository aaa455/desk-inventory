from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.detection import Detection, ImageInfo


class MaskInfo(BaseModel):
    width: int
    height: int
    area_pixels: int = Field(description="处理尺寸下掩膜的前景像素数，不代表实际物理面积")
    area_ratio: float = Field(description="掩膜面积占处理后整张图片的比例，实例间可能重叠")
    png: str = Field(description="处理尺寸的二值 PNG data URL，前景为 255，背景为 0，保留孔洞")


class SegmentedInstance(Detection):
    instance_id: int
    included: bool = Field(description="是否按当前类别排除规则纳入物品清单")
    exclusion_reason: str | None = None
    mask: MaskInfo | None = None
    mask_status: Literal["已生成", "空掩膜", "未返回"]
    mask_quality: float | None = Field(default=None, description="SAM 对掩膜质量的预测值，与类别置信度不同")


class InventoryItem(BaseModel):
    label: str
    label_en: str
    count: int
    instance_ids: list[int]


class SegmentationResponse(BaseModel):
    image: ImageInfo
    detection_model: str
    segmentation_model: str
    device: str
    confidence_threshold: float
    iou_threshold: float
    elapsed_ms: float
    detection_ms: float
    segmentation_ms: float
    instances: list[SegmentedInstance]
    inventory: list[InventoryItem]
    counts: dict[str, int]
    total: int = Field(description="纳入物品清单的检测实例数，不依赖掩膜是否成功")
    detected_total: int
    excluded_total: int
    excluded_labels: list[str]
    segmented_total: int
    annotated_image: str = Field(description="处理尺寸的彩色掩膜叠加图；编号对应 instance_id")
    notes: list[str]


class SegmentationPrompt(BaseModel):
    kind: Literal["box", "points"]
    box: tuple[float, float, float, float] | None = None
    points: list[tuple[float, float]] | None = Field(default=None, min_length=1, max_length=20)
    labels: list[Literal[0, 1]] | None = Field(default=None, min_length=1, max_length=20)

    model_config = {"allow_inf_nan": False, "extra": "forbid"}

    @model_validator(mode="after")
    def check_prompt(self):
        if self.kind == "box":
            if self.box is None or self.points is not None or self.labels is not None:
                raise ValueError("框提示只需填写 box。")
            x1, y1, x2, y2 = self.box
            if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
                raise ValueError("框坐标需要满足 0≤x1<x2、0≤y1<y2。")
        else:
            if self.box is not None or not self.points or not self.labels:
                raise ValueError("点提示需要 points 和 labels，不填写 box。")
            if len(self.points) != len(self.labels) or 1 not in self.labels:
                raise ValueError("点和标签数量应一致，且至少包含一个前景点。")
            if any(x < 0 or y < 0 for x, y in self.points):
                raise ValueError("点坐标不能为负数。")
        return self


class PromptResponse(BaseModel):
    image: ImageInfo
    segmentation_model: str
    device: str
    prompt: SegmentationPrompt
    mask: MaskInfo | None
    mask_status: Literal["已生成", "空掩膜", "未返回"]
    mask_quality: float | None
    elapsed_ms: float
    annotated_image: str
    notes: list[str]
