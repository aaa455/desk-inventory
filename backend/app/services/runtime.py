import os
import threading
from pathlib import Path

from app.core.config import PROJECT_ROOT

# 两个模型共用设备，串行执行整条推理链；重入锁允许分割流程调用检测服务。
inference_lock = threading.RLock()


def configure_runtime() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache" / "ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    from ultralytics import settings as yolo_settings

    yolo_settings.update({
        "sync": False,
        "datasets_dir": str(PROJECT_ROOT / ".cache" / "datasets"),
        "weights_dir": str(PROJECT_ROOT / "models"),
        "runs_dir": str(PROJECT_ROOT / "outputs"),
    })
