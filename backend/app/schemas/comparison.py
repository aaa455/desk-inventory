"""同图对比、独立失败与人工计数评估的数据结构。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.segmentation import SegmentationResponse
from app.schemas.vision import VisionResponse


class AnalysisError(BaseModel):
    status_code: int
    message: str


class BranchState(BaseModel):
    @model_validator(mode="after")
    def check_state(self):
        if self.status == "success":
            if self.result is None or self.error is not None:
                raise ValueError("成功分支必须包含结果且不能包含错误。")
        elif self.result is not None or self.error is None:
            raise ValueError("失败分支必须包含错误且不能包含结果。")
        return self


class TraditionalBranch(BranchState):
    status: Literal["success", "error"]
    result: SegmentationResponse | None = None
    error: AnalysisError | None = None


class VisionBranch(BranchState):
    status: Literal["success", "error"]
    result: VisionResponse | None = None
    error: AnalysisError | None = None


class DifferenceRow(BaseModel):
    label_en: str
    traditional_label: str | None
    vision_label: str | None
    traditional_count: int
    vision_count: int
    delta: int = Field(description="大模型数量减去传统方案数量")


class ComparisonResponse(BaseModel):
    image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    include_background: bool
    elapsed_ms: float = Field(description="两条分支并发执行的总墙钟耗时，不含上传")
    traditional: TraditionalBranch
    vision: VisionBranch
    comparable: bool
    differences: list[DifferenceRow]
    notes: list[str]

    @model_validator(mode="after")
    def check_consistency(self):
        both = self.traditional.status == self.vision.status == "success"
        if self.comparable != both:
            raise ValueError("可比较状态与两条分支状态不一致。")
        if not both and self.differences:
            raise ValueError("未完成双方案时不能生成差异。")
        for branch in (self.traditional, self.vision):
            if branch.status == "success":
                result = branch.result
                if result.image.sha256 != self.image_sha256:
                    raise ValueError("结果图片摘要不一致。")
                if result.total != sum(item.count for item in result.inventory):
                    raise ValueError("清单总数与明细不一致。")
        if both:
            left, right = self.traditional.result, self.vision.result
            if left.excluded_labels != right.excluded_labels:
                raise ValueError("两条流程的排除类别配置不一致。")
        return self


class CountEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    label_en: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9 '/-]*$")
    count: int = Field(ge=0, le=500)

    @field_validator("label_en", mode="before")
    @classmethod
    def normalize_name(cls, value):
        return " ".join(value.lower().split()) if isinstance(value, str) else value


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    traditional: list[CountEntry] | None = Field(default=None, max_length=100)
    vision: list[CountEntry] | None = Field(default=None, max_length=100)
    truth: list[CountEntry] = Field(max_length=200)
    confirmed: bool = Field(description="人工确认已覆盖全部类别，包括两方案都漏检的类别")

    @model_validator(mode="after")
    def validate_counts(self):
        if not self.confirmed:
            raise ValueError("请先人工确认清点范围完整。")
        if self.traditional is None and self.vision is None:
            raise ValueError("至少需要一个成功方案。")
        truth_names = {item.label_en for item in self.truth}
        predicted_names = set()
        for entries in (self.traditional, self.vision):
            if entries is not None:
                predicted_names.update(item.label_en for item in entries)
        if not predicted_names.issubset(truth_names):
            raise ValueError("人工清点需要覆盖两方案的全部类别，不存在的类别明确填写零。")
        for entries in (self.traditional, self.vision, self.truth):
            names = [entry.label_en for entry in entries or []]
            if len(names) != len(set(names)):
                raise ValueError("类别名不能重复。")
        return self


class CountMetrics(BaseModel):
    predicted_total: int
    truth_total: int
    total_absolute_error: int
    category_absolute_error: int = Field(description="各类别绝对计数误差之和")
    category_mean_absolute_error: float | None = Field(description="类别绝对误差均值；没有类别时为空")
    missing_count: int
    extra_count: int
    category_count: int


class EvaluationResponse(BaseModel):
    traditional: CountMetrics | None
    vision: CountMetrics | None
    truth: list[CountEntry]
    notes: list[str]
