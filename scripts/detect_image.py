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
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs",
                        help="结果保存目录")
    args = parser.parse_args()
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
    request = urllib.request.Request(
        args.url.rstrip("/") + f"/api/detect?confidence={args.confidence}",
        data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"接口返回错误 {exc.code}：{exc.read().decode('utf-8', errors='replace')}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"无法连接后端，请先启动服务：{exc.reason}") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.output / "detection.jpg"
    image_path.write_bytes(base64.b64decode(result.pop("annotated_image").split(",", 1)[1]))
    json_path = args.output / "detection.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"检测数量：{result['total']}；分类计数：{result['counts']}")
    print(f"清单：{json_path}\n标注图：{image_path}")


if __name__ == "__main__":
    main()
