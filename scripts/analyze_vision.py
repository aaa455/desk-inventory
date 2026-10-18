"""将一张本地照片交给已配置的视觉模型接口，并保存中文清单。"""

import argparse
import json
import math
import sys
import uuid
from pathlib import Path

import httpx


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="调用本地后端的视觉大模型接口；照片将发送至配置的模型服务")
    parser.add_argument("image", type=Path, help="本地照片路径")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="本地后端地址")
    parser.add_argument("--include-background", action="store_true", help="也将人和家具计入清单")
    parser.add_argument("--timeout", type=float, default=150, help="等待本地后端的秒数")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs",
                        help="结果保存目录")
    args = parser.parse_args()
    if not args.image.is_file() or args.image.stat().st_size > 10 * 1024 * 1024:
        parser.error("请提供存在且不超过 10 MiB 的图片。")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("超时秒数必须大于零。")
    try:
        with args.image.open("rb") as file:
            response = httpx.post(
                args.url.rstrip("/") + "/api/vision",
                params={"include_background": str(args.include_background).lower()},
                files={"file": ("image", file, "application/octet-stream")}, timeout=args.timeout,
            )
        result = response.json()
        if response.status_code != 200:
            raise SystemExit(f"分析失败（{response.status_code}）：{result.get('detail', '后端返回异常。')}")
    except (httpx.RequestError, ValueError) as exc:
        raise SystemExit("连接后端失败或响应无效，请检查服务启动情况与超时配置。") from exc
    output = args.output / ("vision-" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=True)
    target = output / "vision.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"模型：{result['model']}；清单数量：{result['total']}；分类计数：{result['counts']}")
    print(f"总耗时：{result['elapsed_ms']} 毫秒；结果：{target}")


if __name__ == "__main__":
    main()
