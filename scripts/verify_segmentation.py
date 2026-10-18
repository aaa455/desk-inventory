"""用用户指定的本地样本验证真实分割链路，需要已准备的权重和测试依赖。"""

import argparse
import base64
import io
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="验证真实 HTTP 分割、清单、提示与结果导出")
    parser.add_argument("image", type=Path, help="本地静态图片，建议使用有多个已知检测类别的图片")
    parser.add_argument("--cpu", action="store_true", help="额外验证 CPU 手动框分割")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "backend"))

    import httpx
    import numpy as np
    import uvicorn
    from PIL import Image, ImageOps
    from app.core.config import settings
    from app.main import app
    from app.schemas.segmentation import SegmentationPrompt
    from app.services.segmenter import Segmenter

    output_dir = root / "outputs" / "step3-verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = args.image.read_bytes()
    with Image.open(io.BytesIO(data)) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        width, height = source.size
        large = io.BytesIO()
        source.resize((2400, max(1, round(height * 2400 / width)))).save(large, format="PNG")
        tiled = Image.new("RGB", (width * 3, height * 2))
        for x in range(3):
            for y in range(2):
                tiled.paste(source, (width * x, height * y))
        tiles = io.BytesIO()
        tiled.save(tiles, format="PNG")

    def save_and_check(result, stem):
        image_info = result["image"]
        shape = (image_info["processed_height"], image_info["processed_width"])
        overlay = result.pop("annotated_image")
        (output_dir / f"{stem}.jpg").write_bytes(base64.b64decode(overlay.split(",", 1)[1]))
        items = result.get("instances", [{"mask": result.get("mask"), "instance_id": 1}])
        for item in items:
            if item["mask"] is None:
                continue
            mask = item["mask"]
            mask_bytes = base64.b64decode(mask.pop("png").split(",", 1)[1])
            array = np.asarray(Image.open(io.BytesIO(mask_bytes)))
            assert array.shape == shape
            assert set(np.unique(array)).issubset({0, 255})
            assert int(np.count_nonzero(array)) == mask["area_pixels"]
            name = f"{stem}-mask-{item['instance_id']}.png"
            (output_dir / name).write_bytes(mask_bytes)
            mask["png_file"] = name
        if "inventory" in result:
            assert result["total"] == sum(item["count"] for item in result["inventory"])
            assert result["detected_total"] == result["total"] + result["excluded_total"]
            assert result["detected_total"] == len(result["instances"])
            expected = {item["instance_id"] for item in result["instances"] if item["included"]}
            assert expected == {i for item in result["inventory"] for i in item["instance_ids"]}
        (output_dir / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    report = {}
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "本地验证服务未启动"
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
            def upload(endpoint, content=data, fields=None):
                response = client.post(endpoint, files={"file": ("sample.png", content)}, data=fields)
                assert response.status_code == 200, response.text
                return response.json()

            first = save_and_check(upload("/api/segment"), "auto")
            assert first["detected_total"] > 0, "样本未检出物体，请选择支持类别的照片"
            report["自动清单"] = {k: first[k] for k in ["counts", "total", "detected_total", "segmented_total", "elapsed_ms"]}
            all_result = save_and_check(upload("/api/segment?include_background=true"), "all")
            assert all_result["total"] == all_result["detected_total"]
            enlarged = save_and_check(upload("/api/segment", large.getvalue()), "large")
            assert enlarged["image"]["width"] == 2400
            assert enlarged["image"]["processed_width"] == settings.max_image_side
            report["大图"] = {"image": enlarged["image"], "counts": enlarged["counts"]}

            box = first["instances"][0]["bbox"]
            prompt = {"kind": "box", "box": box}
            prompted = save_and_check(upload("/api/segment/prompt", fields={"prompt": json.dumps(prompt)}), "box")
            assert prompted["mask_status"] == "已生成"
            report["框提示面积"] = prompted["mask"]["area_pixels"]
            # 前景点取自真实预测掩膜内部，背景点取其外部；它们只提示区域，不给出类别。
            mask = np.asarray(Image.open(output_dir / "box-mask-1.png")) > 0
            ys, xs = np.nonzero(mask)
            yb, xb = np.nonzero(~mask)
            points = [[float(xs[len(xs) // 2]) * width / mask.shape[1], float(ys[len(ys) // 2]) * height / mask.shape[0]]]
            labels = [1]
            if len(xb):
                points.append([float(xb[0]) * width / mask.shape[1], float(yb[0]) * height / mask.shape[0]])
                labels.append(0)
            point_prompt = {"kind": "points", "points": points, "labels": labels}
            point_result = save_and_check(upload("/api/segment/prompt", fields={"prompt": json.dumps(point_prompt)}), "points")
            assert point_result["mask_status"] == "已生成"
            report["点提示面积"] = point_result["mask"]["area_pixels"]
            prompt_path = output_dir / "prompt-input.json"
            prompt_path.write_text(json.dumps(point_prompt), encoding="utf-8")

            tile_result = save_and_check(upload("/api/segment", tiles.getvalue()), "tiles")
            report["拼图"] = {k: tile_result[k] for k in ["detected_total", "segmented_total", "counts"]}
            blank = io.BytesIO()
            Image.new("RGB", (640, 480), "white").save(blank, format="PNG")
            empty = upload("/api/segment", blank.getvalue())
            assert empty["total"] == 0 and empty["instances"] == []
            # 空白图之后重新处理原图，检查图像特征或提示是否串用。
            repeat = upload("/api/segment")
            assert repeat["counts"] == first["counts"]
            overlaps = []
            for original, repeated in zip(first["instances"], repeat["instances"], strict=True):
                assert original["instance_id"] == repeated["instance_id"]
                assert original["label_en"] == repeated["label_en"]
                if original["mask"] is None:
                    assert repeated["mask"] is None
                    continue
                first_mask = np.asarray(Image.open(output_dir / original["mask"]["png_file"])) > 0
                next_mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(repeated["mask"]["png"].split(",", 1)[1])))) > 0
                union = np.logical_or(first_mask, next_mask).sum()
                overlap = float(np.logical_and(first_mask, next_mask).sum() / union) if union else 1.0
                # GPU 阈值附近的浮点波动会改变少量边缘像素，不要求逐像素完全相等。
                assert overlap >= 0.99, f"重复请求掩膜变化过大：{overlap}"
                overlaps.append(overlap)
            report["重复掩膜重叠度"] = overlaps
            for options in [["--segment"], ["--prompt", str(prompt_path)]]:
                subprocess.run([sys.executable, "-B", str(root / "scripts/detect_image.py"), str(args.image.resolve()),
                                "--url", f"http://127.0.0.1:{port}", "--output", str(output_dir), *options], check=True)
            report["空白图、重复请求、命令行导出"] = "通过"
            if args.cpu:
                previous_device = settings.yolo_device
                try:
                    settings.yolo_device = "cpu"
                    cpu = Segmenter().prompted(data, SegmentationPrompt.model_validate(prompt))
                    assert cpu.device == "cpu" and cpu.mask_status == "已生成"
                    report["CPU 框提示面积"] = cpu.mask.area_pixels
                finally:
                    settings.yolo_device = previous_device
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
