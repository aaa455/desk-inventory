"""通过临时本地服务和命令行工具验证一次真实视觉模型调用，可能产生服务费用。"""

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="发送一张照片至配置的视觉模型，验证真实调用和导出，只调用一次")
    parser.add_argument("image", type=Path, help="允许发送给所配置模型服务的本地图片")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("图片文件不存在。")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "backend"))
    import uvicorn
    from app.core.config import settings
    from app.main import app
    from app.services.vision import VisionError, configured_endpoint

    try:
        configured_endpoint()
    except VisionError as exc:
        raise SystemExit(str(exc)) from exc
    output = root / "outputs" / ("step4-verification-" + str(time.time_ns()))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        if not server.started:
            raise SystemExit("临时本地服务启动失败。")
        command = [
            sys.executable, "-B", str(root / "scripts" / "analyze_vision.py"), str(args.image.resolve()),
            "--url", f"http://127.0.0.1:{port}", "--timeout", str(settings.vision_timeout_seconds + 20),
            "--output", str(output),
        ]
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode:
            raise SystemExit("真实调用未通过；未生成成功验证记录，请按上面的提示检查配置。")
        files = list(output.glob("vision-*/vision.json"))
        if len(files) != 1:
            raise SystemExit("导出文件数量异常。")
        result = json.loads(files[0].read_text(encoding="utf-8"))
        if result["image"]["sha256"] != hashlib.sha256(args.image.read_bytes()).hexdigest():
            raise SystemExit("图片摘要与输入不一致。")
        if result["total"] != sum(item["count"] for item in result["inventory"]):
            raise SystemExit("清单总数与明细不一致。")
        if result["counts"] != {item["label"]: item["count"] for item in result["inventory"]}:
            raise SystemExit("分类计数与明细不一致。")
        report = {
            "结论": "真实接口、图片摘要、后端汇总和命令行导出均通过；不代表识别准确率。",
            "请求模型": result["model"], "返回模型": result["returned_model"],
            "输入文件摘要": result["image"]["sha256"], "发送图片摘要": result["sent_image_sha256"],
            "分类计数": result["counts"], "总数": result["total"],
            "总耗时毫秒": result["elapsed_ms"], "接口耗时毫秒": result["inference_ms"],
            "令牌用量": result["usage"], "结果相对路径": str(files[0].relative_to(output)),
        }
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("真实验证通过，记录：" + str(output / "report.json"))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


if __name__ == "__main__":
    main()
