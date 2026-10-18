# 桌面物品清点

上传一张桌面照片，分别通过本地视觉模型与视觉大模型生成物品清单，并排比较识别类别、计数、耗时和使用体验。

## 当前进度

已完成第二步：图片上传与校验、方向校正和缩放、YOLO11n 检测、中文类别与计数、原图边界框和标注图。已经在本机 GPU 上跑通真实推理；分割、多模态对照与对比界面尚未实现。

## 项目方案

- 后端采用 FastAPI，统一管理图片输入、分析任务和结构化结果。
- 传统视觉方案先建立 YOLO 检测基线，再加入实例分割。具体分割组件在第三步依据识别效果与环境兼容性确定，评估 YOLO 实例分割或 SAM 2 框提示分割。
- 多模态方案通过兼容 OpenAI 接口的视觉模型，独立分析同一张图片，不使用传统模型结果作为提示。
- 界面并排展示原图、检测或分割结果、两份物品清单和分类计数。
- 使用人工清点的小型样本集核对两种方案，展示正确结果与失败案例。

YOLO 预训练模型只支持其训练类别，不能预先保证识别笔、橡皮等所有桌面物品。SAM 负责按提示分割，本身不会补全漏检类别。会记录支持类别，并根据实际样本决定是否增加开放词汇检测。

参考 [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) 的检测、提示式分割、可视化和结构化输出思路。业务代码独立实现，模型通过官方库或接口使用；第三方依赖与预训练权重的许可分别核对。

## 环境与依赖

建议使用 Python 3.12。安装前先运行标准库脚本，检查已有解释器、依赖、模型文件和显卡：

```powershell
python scripts/check_environment.py
```

本机已有 Python 3.12 可用于创建隔离环境，Python 3.15 暂不作为视觉模型环境。现有 `.venv` 已安装所需依赖，继续开发时直接复用即可。本机 RTX 5070 Ti Laptop GPU 已用 PyTorch 2.7.1 的 CUDA 12.8 构建验证实际运算与 YOLO 推理。

在项目根目录运行，使用本机已有的 Python 3.12 解释器创建环境：

```powershell
# 已配置 Python 启动器时使用。
py -3.12 -m venv .venv

# 当前机器也可直接复用以下已安装的解释器，无需重新下载 Python。
# & 'C:\Users\xingkui\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv

# 下面的 GPU 组合已在本机验证；其他机器请依据硬件选择官方兼容构建。
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache/pip torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache/pip -r requirements.txt
Copy-Item .env.example .env
```

只需选择一种创建环境的方式。项目运行环境、后续权重及输出文件都放在项目目录内。创建环境后可再用 `.\.venv\Scripts\python.exe scripts/check_environment.py` 检查隔离环境。

精确依赖保存在 `requirements.lock.txt`，它对应本机 Windows、Python 3.12、CUDA 12.8 环境。先执行上述 PyTorch 安装命令，再使用 `python -m pip install -r requirements.lock.txt` 复现；锁定文件已指明 PyTorch 官方额外索引。测试依赖单独放在 `requirements-dev.txt`。

## 准备检测模型

```powershell
.\.venv\Scripts\python.exe scripts/prepare_model.py
```

脚本先检查项目 `models/yolo11n.pt`，有效则复用；缺失时下载 Ultralytics 官方权重并校验 SHA256。权重约 5.6 MB，不进入 Git。它与 Ultralytics 库采用其官方的 AGPL-3.0 或企业许可规则，使用与分发时参考 [官方许可说明](https://www.ultralytics.com/license)。这是使用预训练模型，无需训练数据或自行训练。

## 启动后端

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

- 健康检查：<http://127.0.0.1:8000/api/health>
- 交互接口文档：<http://127.0.0.1:8000/docs>

`.env` 使用 UTF-8 编码，环境变量优先于配置文件。密钥只保存在本地，不进入 Git；第一步无需配置任何模型密钥。模板中的服务地址仅为示例，接入时需要确认服务支持图像输入以及所选模型。

## 检测图片

启动服务后，在接口文档中展开 `POST /api/detect`，点击尝试并上传图片。也可在另一个 PowerShell 窗口运行：

```powershell
.\.venv\Scripts\python.exe scripts/detect_image.py '你的桌面照片.jpg'
```

脚本向本地接口上传照片，生成 `outputs/detection.json` 和 `outputs/detection.jpg`；重复运行会更新这两个结果文件。默认使用 GPU（不可用时使用 CPU），可在 `.env` 设置 `YOLO_DEVICE=cpu`。当前服务器不保存上传原图或结果，脚本只在本地保存返回结果。

接口支持 JPEG、PNG、WebP 静态图片，默认最大 10 MiB、2000 万像素。手机方向校正后，最长边超过 1600 像素的图片按比例缩小；YOLO 内部再以 `imgsz=640` 保持比例补边。返回 `bbox` 为方向校正后的原图像素坐标 `x1,y1,x2,y2`；标注图为处理尺寸，图中编号从 1 开始，对应 `detections` 的列表顺序。`sha256` 标识原始上传文件，便于后续核对同图输入。

`confidence`（默认 0.25）过滤低置信度候选；`iou`（默认 0.45）控制重叠框去重。`counts` 为中文分类计数，`total` 包含整张图所有检出物体，可能包括桌子本身。它尚不是只统计桌面小物件的清单。未检出不表示没有物体，纸笔、橡皮等未训练类别不能可靠识别。

总耗时不含上传，包含图片预处理、首次模型加载、推理和绘图；单独返回 YOLO 推理调用耗时，首次调用还可能包含预热。当前请求串行推理，避免同一模型对象并发冲突，不用于高并发服务。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache/pip -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

回归测试不下载或加载真实权重；真实推理结果、样本来源及局限见 [第二步验证记录](docs/第二步验证.md)。健康检查仅代表后端响应，不表示模型已经加载。

## 目录结构

```text
backend/app/
  api/                  接口路由
  core/                 统一配置
  schemas/              接口返回数据结构
  services/             图片处理和模型推理
  main.py               后端入口
docs/
  开发计划.md           分步范围、验收要求和学习要点
  环境记录.md           环境盘点与复用结论
scripts/
  check_environment.py  无需额外安装依赖的环境检查
  prepare_model.py      检查并按需下载模型权重
  detect_image.py       上传图片并保存检测结果
tests/                  图片处理与接口回归测试
.env.example            配置模板
requirements.txt        当前阶段依赖范围
requirements.lock.txt   已验证环境的精确依赖版本
requirements-dev.txt    测试依赖
```
