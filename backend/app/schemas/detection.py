from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    sha256: str = Field(description="上传文件的摘要，用于后续核对两种方案的输入")
    width: int = Field(description="方向校正后的原图宽度")
    height: int = Field(description="方向校正后的原图高度")
    processed_width: int
    processed_height: int


class Detection(BaseModel):
    class_id: int
    label: str = Field(description="中文类别")
    label_en: str = Field(description="模型原始类别")
    confidence: float
    bbox: tuple[float, float, float, float] = Field(description="原图像素坐标 x1、y1、x2、y2")


class DetectionResponse(BaseModel):
    image: ImageInfo
    model: str
    device: str
    confidence_threshold: float
    iou_threshold: float
    elapsed_ms: float = Field(description="图片预处理、模型加载、推理及绘图的总耗时，不含上传")
    inference_ms: float = Field(description="YOLO 推理调用耗时；GPU 首次调用可能包含预热")
    detections: list[Detection]
    counts: dict[str, int]
    total: int
    annotated_image: str = Field(description="缩放后的标注图，JPEG 格式的 data URL；框内编号对应 detections 顺序")
    notes: list[str]
