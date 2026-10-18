import os
import threading
from collections import Counter
from pathlib import Path
from time import perf_counter

from app.core.config import PROJECT_ROOT, settings
from app.schemas.detection import Detection, DetectionResponse, ImageInfo
from app.services.images import annotate_image, prepare_image

# COCO 类别顺序与官方预训练 YOLO 模型一致。
LABELS_ZH = (
    "人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船", "交通灯",
    "消防栓", "停车标志", "停车计时器", "长椅", "鸟", "猫", "狗", "马", "羊", "牛",
    "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘",
    "滑雪板", "单板滑雪板", "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子",
    "高脚杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子",
    "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽", "床",
    "餐桌", "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱",
    "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷",
)

LABELS_EN = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
LABEL_BY_NAME = dict(zip(LABELS_EN, LABELS_ZH, strict=True))


class ModelUnavailable(RuntimeError):
    """模型尚未准备就绪。"""


class Detector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        path = Path(settings.yolo_model_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise ModelUnavailable("未找到检测模型，请先按照运行说明准备本地权重。")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache" / "ultralytics"))
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
        os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
        try:
            Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
            import torch
            from ultralytics import YOLO
            from ultralytics import settings as yolo_settings

            yolo_settings.update({
                "sync": False,
                "datasets_dir": str(PROJECT_ROOT / ".cache" / "datasets"),
                "weights_dir": str(PROJECT_ROOT / "models"),
                "runs_dir": str(PROJECT_ROOT / "outputs"),
            })
            device = ("0" if torch.cuda.is_available() else "cpu") if settings.yolo_device == "auto" else settings.yolo_device
            model = YOLO(str(path), task="detect")
            if model.task != "detect":
                raise ModelUnavailable("本接口需要目标检测权重，请检查 YOLO_MODEL_PATH。")
            self._device = device
            self._model = model
        except ModelUnavailable:
            raise
        except Exception as exc:
            raise ModelUnavailable("无法加载模型，请检查权重、依赖和设备配置。") from exc

    def analyze(self, data: bytes, confidence: float, iou: float) -> DetectionResponse:
        started = perf_counter()
        image = prepare_image(data)
        height, width = image.bgr.shape[:2]
        with self._lock:
            self._load()
            inference_started = perf_counter()
            result = self._model.predict(image.bgr, conf=confidence, iou=iou,
                                         device=self._device, imgsz=640, verbose=False)[0]
            inference_ms = (perf_counter() - inference_started) * 1000
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            names = result.names
        detections = []
        for box, score, class_id in zip(boxes, scores, classes):
            class_id = int(class_id)
            label_en = names[class_id]
            label = LABEL_BY_NAME.get(label_en, label_en)
            x1, y1, x2, y2 = box
            original_box = (
                max(0.0, min(image.width, x1 * image.width / width)),
                max(0.0, min(image.height, y1 * image.height / height)),
                max(0.0, min(image.width, x2 * image.width / width)),
                max(0.0, min(image.height, y2 * image.height / height)),
            )
            detections.append(Detection(class_id=class_id, label=label, label_en=label_en,
                                        confidence=score, bbox=tuple(round(v, 2) for v in original_box)))
        annotated = annotate_image(image.bgr, boxes, scores)
        return DetectionResponse(
            image=ImageInfo(sha256=image.sha256, width=image.width, height=image.height,
                            processed_width=width, processed_height=height),
            model=Path(settings.yolo_model_path).name, device=self._device,
            confidence_threshold=confidence, iou_threshold=iou,
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
            inference_ms=round(inference_ms, 2), detections=detections,
            counts=dict(Counter(item.label for item in detections)), total=len(detections),
            annotated_image=annotated,
            notes=["清点范围是整张图片，可能包含桌面以外的物体。",
                   "预训练模型仅支持其训练类别，空结果不代表图片中没有物品。",
                   "标注图为处理尺寸，返回的边界框是方向校正后的原图坐标。",
                   "首次请求包含模型加载与设备预热，不宜直接用作稳态速度比较。"],
        )


detector = Detector()
