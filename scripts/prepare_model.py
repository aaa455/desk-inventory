"""按需下载官方检测权重，校验摘要后保存，不重复下载已有有效权重。"""

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
SAM_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt"
SAM_SHA256 = "3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="检查并按需下载官方模型权重")
    parser.add_argument("--model", choices=["yolo", "sam"], default="yolo", help="检测或分割权重")
    args = parser.parse_args()
    url, expected, filename = (SAM_URL, SAM_SHA256, "sam2.1_t.pt") if args.model == "sam" else (URL, SHA256, "yolo11n.pt")
    target = Path(__file__).resolve().parents[1] / "models" / filename
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise SystemExit("已有权重的摘要不匹配，请核实来源；脚本不会覆盖现有文件。")
        print(f"复用已校验的模型：{target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=filename + "-", suffix=".part", dir=target.parent)
    temporary = Path(temporary)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit("下载内容的摘要不匹配，未安装权重。")
        temporary.replace(target)
        print(f"模型已下载并通过摘要检查：{target}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
