"""向已启动的检测接口发送本地图片，将标注图与清单写入指定目录。"""

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="通过后端接口检测一张图片")
    parser.add_argument("image", type=Path, help="本地图片路径")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="本地后端地址")
    parser.add_argument("--confidence", type=float, default=0.25, help="最低置信度")
    parser.add_argument("--segment", action="store_true", help="同时分割并生成物品清单")
    parser.add_argument("--include-background", action="store_true", help="分割时将人和家具也计入清单")
    parser.add_argument("--prompt", type=Path, help="UTF-8 JSON 提示文件，用于手动点或框分割")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs",
                        help="结果保存目录")
    args = parser.parse_args()
    if args.prompt and (args.segment or args.include_background):
        parser.error("手动提示与自动清单选项不可同时使用。")
    if args.include_background and not args.segment:
        parser.error("--include-background 需要搭配 --segment。")
    if not args.image.is_file():
        parser.error("图片文件不存在。")
    if args.image.stat().st_size > 10 * 1024 * 1024:
        parser.error("本工具仅发送不超过 10 MiB 的文件。")
    boundary = uuid.uuid4().hex
    # 采用固定文件名，避免用户文件名影响 multipart 请求头。
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image\"\r\n"
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + args.image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    endpoint = f"/api/detect?confidence={args.confidence}"
    stem = "detection"
    if args.segment:
        endpoint = f"/api/segment?confidence={args.confidence}&include_background={str(args.include_background).lower()}"
        stem = "segmentation"
    if args.prompt:
        if not args.prompt.is_file():
            parser.error("提示文件不存在。")
        prompt_text = args.prompt.read_text(encoding="utf-8")
        try:
            json.loads(prompt_text)
        except ValueError:
            parser.error("提示文件不是有效 JSON。")
        body = body.removesuffix(f"\r\n--{boundary}--\r\n".encode())
        body += (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n"
                 + prompt_text + f"\r\n--{boundary}--\r\n").encode()
        endpoint, stem = "/api/segment/prompt", "prompt"
    request = urllib.request.Request(
        args.url.rstrip("/") + endpoint,
        data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"接口返回错误 {exc.code}：{exc.read().decode('utf-8', errors='replace')}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"无法连接后端，请先启动服务：{exc.reason}") from exc
    if stem != "detection":
        # 每次分割结果独立保存，防止上次的掩膜文件混入本次结果。
        args.output = args.output / f"{stem}-{uuid.uuid4().hex[:12]}"
    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.output / f"{stem}.jpg"
    image_path.write_bytes(base64.b64decode(result.pop("annotated_image").split(",", 1)[1]))
    if args.segment:
        for item in result["instances"]:
            if item["mask"]:
                mask_path = args.output / f"segmentation-mask-{item['instance_id']}.png"
                mask_path.write_bytes(base64.b64decode(item["mask"].pop("png").split(",", 1)[1]))
                item["mask"]["png_file"] = mask_path.name
    elif args.prompt and result["mask"]:
        mask_path = args.output / "prompt-mask.png"
        mask_path.write_bytes(base64.b64decode(result["mask"].pop("png").split(",", 1)[1]))
        result["mask"]["png_file"] = mask_path.name
    json_path = args.output / f"{stem}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.prompt:
        print(f"清单数量：{result['total']}；分类计数：{result['counts']}")
    else:
        print(f"手动分割状态：{result['mask_status']}")
    print(f"清单：{json_path}\n标注图：{image_path}")


if __name__ == "__main__":
    main()
