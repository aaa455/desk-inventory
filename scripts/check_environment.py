"""使用标准库盘点当前解释器及依赖，不安装包或下载模型。"""

import importlib.metadata
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    print(f"项目目录：{root}")
    print(f"解释器：{sys.executable}")
    print(f"Python 版本：{platform.python_version()}")
    print(f"项目虚拟环境：{'存在' if (root / '.venv').exists() else '未创建'}")

    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic_settings": "pydantic-settings",
        "torch": "torch",
        "torchvision": "torchvision",
        "cv2": "opencv-python",
        "ultralytics": "ultralytics",
        "transformers": "transformers",
        "PIL": "pillow",
        "openai": "openai",
    }
    for module, distribution in packages.items():
        if importlib.util.find_spec(module) is None:
            print(f"依赖 {distribution}：未发现")
            continue
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "已发现模块，未找到此发行包版本"
        print(f"依赖 {distribution}：{version}")

    weights = sorted(path.name for path in (root / "models").glob("*.pt"))
    print(f"项目模型文件：{'、'.join(weights) if weights else '未发现 .pt 权重'}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            print(f"显卡、显存、驱动：{result.stdout.strip()}")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"无法查询显卡：{exc}")
    else:
        print("未找到 nvidia-smi；后续可评估 CPU 推理。")

    print("以上结果只说明包和设备是否可发现，模型运行能力仍需实际推理验证。")


if __name__ == "__main__":
    main()
