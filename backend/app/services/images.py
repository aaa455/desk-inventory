import base64
import hashlib
import io
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings


class InvalidImage(ValueError):
    """上传内容无法作为受支持的静态图片使用。"""


@dataclass
class PreparedImage:
    bgr: np.ndarray
    width: int
    height: int
    sha256: str


def prepare_image(data: bytes) -> PreparedImage:
    if not data:
        raise InvalidImage("上传的图片为空。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("仅支持 JPEG、PNG、WebP 静态图片。")
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImage("请上传静态图片，不支持动画。")
                if image.width * image.height > settings.max_image_pixels:
                    raise InvalidImage(f"图片像素过多，上限为 {settings.max_image_pixels} 像素。")
                image.load()
                image = ImageOps.exif_transpose(image).convert("RGBA")
                background = Image.new("RGBA", image.size, "white")
                rgb = np.asarray(Image.alpha_composite(background, image).convert("RGB"))
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InvalidImage("图片内容无效、已损坏或尺寸过大。") from exc

    height, width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    scale = min(1.0, settings.max_image_side / max(width, height))
    if scale < 1:
        bgr = cv2.resize(bgr, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    return PreparedImage(bgr, width, height, hashlib.sha256(data).hexdigest())


def annotate_image(image: np.ndarray, boxes: list[tuple[float, float, float, float]],
                   scores: list[float]) -> str:
    canvas = image.copy()
    for index, (box, score) in enumerate(zip(boxes, scores), 1):
        x1, y1, x2, y2 = (round(value) for value in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 180, 50), 2)
        cv2.putText(canvas, f"#{index} {score:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 100, 30), 2)
    success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("生成标注图失败。")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
