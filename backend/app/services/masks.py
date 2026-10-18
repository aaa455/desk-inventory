import base64

import cv2
import numpy as np

from app.schemas.segmentation import MaskInfo


def mask_info(mask: np.ndarray) -> MaskInfo:
    binary = mask.astype(bool)
    success, encoded = cv2.imencode(".png", binary.astype(np.uint8) * 255)
    if not success:
        raise RuntimeError("生成掩膜图片失败。")
    height, width = binary.shape
    area = int(binary.sum())
    return MaskInfo(width=width, height=height, area_pixels=area,
                    area_ratio=round(area / (width * height), 6),
                    png="data:image/png;base64," + base64.b64encode(encoded).decode("ascii"))


def overlay_masks(image: np.ndarray, masks: list[np.ndarray | None]) -> str:
    canvas = image.copy()
    colors = [(40, 170, 250), (220, 100, 70), (80, 210, 100), (180, 70, 220)]
    # 大掩膜先绘制，尽量避免桌面等区域遮住小物体。
    for index in sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()) if masks[i] is not None else 0):
        mask = masks[index]
        if mask is None or not mask.any():
            continue
        color = colors[index % len(colors)]
        canvas[mask] = (canvas[mask].astype(float) * 0.55 + np.array(color) * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
    # 在填色完成后写编号，编号落在掩膜最内部的位置。
    for index, mask in enumerate(masks, 1):
        if mask is not None and mask.any():
            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
            y, x = np.unravel_index(distance.argmax(), distance.shape)
            text_width, text_height = cv2.getTextSize(f"#{index}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 3)[0]
            x = max(0, min(int(x), canvas.shape[1] - text_width - 2))
            y = max(text_height, min(int(y), canvas.shape[0] - 2))
            cv2.putText(canvas, f"#{index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(canvas, f"#{index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("生成分割叠加图失败。")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
