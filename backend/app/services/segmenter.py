from pathlib import Path
from time import perf_counter

from app.core.config import PROJECT_ROOT, settings
from app.schemas.detection import ImageInfo
from app.schemas.segmentation import (
    InventoryItem, PromptResponse, SegmentationPrompt, SegmentationResponse, SegmentedInstance,
)
from app.services.detector import LABEL_BY_NAME, ModelUnavailable, detector
from app.services.images import InvalidImage, prepare_image
from app.services.masks import mask_info, overlay_masks
from app.services.runtime import configure_runtime, inference_lock


def image_info(image) -> ImageInfo:
    height, width = image.bgr.shape[:2]
    return ImageInfo(sha256=image.sha256, width=image.width, height=image.height,
                     processed_width=width, processed_height=height)


def summarize(instances: list[SegmentedInstance]) -> list[InventoryItem]:
    groups = {}
    for instance in instances:
        if not instance.included:
            continue
        if instance.label_en not in groups:
            groups[instance.label_en] = InventoryItem(
                label=instance.label, label_en=instance.label_en, count=0, instance_ids=[])
        item = groups[instance.label_en]
        item.count += 1
        item.instance_ids.append(instance.instance_id)
    return list(groups.values())


class Segmenter:
    def __init__(self) -> None:
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        path = Path(settings.sam_model_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise ModelUnavailable("未找到分割模型，请运行模型准备脚本的 --model sam 选项。")
        try:
            configure_runtime()
            import torch
            from ultralytics import SAM

            device = ("0" if torch.cuda.is_available() else "cpu") if settings.yolo_device == "auto" else settings.yolo_device
            model = SAM(str(path))
            if not model.is_sam2:
                raise ModelUnavailable("分割接口需要 SAM 2 系列权重。")
            self._model = model
            self._device = device
        except ModelUnavailable:
            raise
        except Exception as exc:
            raise ModelUnavailable("无法加载分割模型，请检查权重、依赖和设备配置。") from exc

    def _predict(self, image, count: int, **prompts):
        """用 SAM 返回的提示索引关联掩膜，不能用过滤后的列表位置冒充原索引。"""
        self._load()
        result = self._model.predict(image.bgr, device=self._device, imgsz=1024,
                                     conf=0.0, verbose=False, **prompts)[0]
        masks = [None] * count
        qualities = [None] * count
        if result.masks is None:
            return masks, qualities
        arrays = result.masks.data.cpu().numpy().astype(bool)
        indices = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        if len(arrays) != len(indices) or len(indices) != len(scores):
            raise RuntimeError("分割模型的掩膜与提示索引数量不一致。")
        seen = set()
        for array, index, score in zip(arrays, indices, scores):
            index = int(index)
            if index < 0 or index >= count or index in seen or array.shape != image.bgr.shape[:2]:
                raise RuntimeError("分割模型返回了无效的提示索引或掩膜尺寸。")
            seen.add(index)
            masks[index] = array
            qualities[index] = float(score)
        return masks, qualities

    def inventory(self, data: bytes, confidence: float, iou: float, include_background: bool):
        started = perf_counter()
        image = prepare_image(data)
        height, width = image.bgr.shape[:2]
        masks, qualities = [], []
        with inference_lock:
            boxes, scores, classes, names, detection_ms = detector.predict(image, confidence, iou, max_det=50)
            segmentation_started = perf_counter()
            if boxes:
                # 限制每批提示数，避免密集图片一次占用过多 GPU 显存。
                for offset in range(0, len(boxes), 8):
                    batch = boxes[offset:offset + 8]
                    batch_masks, batch_qualities = self._predict(image, len(batch), bboxes=batch)
                    masks.extend(batch_masks)
                    qualities.extend(batch_qualities)
            segmentation_ms = (perf_counter() - segmentation_started) * 1000
        instances = []
        excluded = [] if include_background else list(settings.inventory_excluded_labels)
        for index, (box, score, class_id, mask, quality) in enumerate(zip(boxes, scores, classes, masks, qualities), 1):
            label_en = names[int(class_id)]
            included = label_en not in excluded
            sx, sy = image.width / width, image.height / height
            original_box = tuple(round(max(0, min(v * (sx if i % 2 == 0 else sy),
                                                  image.width if i % 2 == 0 else image.height)), 2)
                                 for i, v in enumerate(box))
            info = mask_info(mask) if mask is not None else None
            instances.append(SegmentedInstance(
                instance_id=index, class_id=int(class_id), label=LABEL_BY_NAME.get(label_en, label_en),
                label_en=label_en, confidence=score, bbox=original_box, included=included,
                exclusion_reason=None if included else "按类别规则排除的人或大型家具，不计入物品清单。",
                mask=info, mask_quality=quality,
                mask_status="未返回" if info is None else ("已生成" if info.area_pixels else "空掩膜"),
            ))
        inventory = summarize(instances)
        annotated = overlay_masks(image.bgr, masks)
        return SegmentationResponse(
            image=image_info(image), detection_model=Path(settings.yolo_model_path).name,
            segmentation_model=Path(settings.sam_model_path).name, device=detector._device,
            confidence_threshold=confidence, iou_threshold=iou,
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
            detection_ms=round(detection_ms, 2), segmentation_ms=round(segmentation_ms, 2),
            instances=instances, inventory=inventory, counts={item.label: item.count for item in inventory},
            total=sum(item.count for item in inventory), detected_total=len(instances),
            excluded_total=sum(not item.included for item in instances), excluded_labels=excluded,
            segmented_total=sum(item.mask_status == "已生成" for item in instances), annotated_image=annotated,
            notes=["类别与实例数来自 YOLO，SAM 只按框分割，不增加识别类别或补回漏检。",
                   "默认仅按类别排除人和大型家具，不判断物体是否真正位于桌面上。",
                   "每图最多检测 50 个实例；掩膜失败的检测实例仍保留在清单中。",
                   "掩膜在处理尺寸上，边界框在方向校正后的原图坐标上。",
                   "掩膜可能重叠或混入背景；面积不是物体实际尺寸。"],
        )

    def prompted(self, data: bytes, prompt: SegmentationPrompt):
        started = perf_counter()
        image = prepare_image(data)
        height, width = image.bgr.shape[:2]
        sx, sy = width / image.width, height / image.height
        if prompt.kind == "box":
            x1, y1, x2, y2 = prompt.box
            if x2 > image.width or y2 > image.height:
                raise InvalidImage("提示框超出方向校正后的原图范围。")
            arguments = {"bboxes": [[x1 * sx, y1 * sy, x2 * sx, y2 * sy]]}
        else:
            if any(x >= image.width or y >= image.height for x, y in prompt.points):
                raise InvalidImage("提示点超出方向校正后的原图范围。")
            # 三维形状表示同一物体的一组前景和背景点，而非多个独立目标。
            arguments = {"points": [[[x * sx, y * sy] for x, y in prompt.points]], "labels": [prompt.labels]}
        with inference_lock:
            masks, qualities = self._predict(image, 1, **arguments)
        info = mask_info(masks[0]) if masks[0] is not None else None
        annotated = overlay_masks(image.bgr, masks)
        return PromptResponse(
            image=image_info(image), segmentation_model=Path(settings.sam_model_path).name,
            device=self._device, prompt=prompt, mask=info, mask_quality=qualities[0],
            mask_status="未返回" if info is None else ("已生成" if info.area_pixels else "空掩膜"),
            elapsed_ms=round((perf_counter() - started) * 1000, 2), annotated_image=annotated,
            notes=["提示坐标基于方向校正后的原图，掩膜为处理尺寸。",
                   "1 为前景点，0 为背景点；手动分割不自动命名物体，不计入自动清单。"],
        )


segmenter = Segmenter()
