# 桌面物品清点

上传一张桌面照片，分别通过本地视觉模型与视觉大模型生成物品清单，并排比较识别类别、计数、耗时和使用体验。

## 当前进度

已完成第四步：传统 YOLO + SAM 分割清单、手动点／框提示，以及视觉大模型独立清点、结构化校验、超时错误处理和 JSON 导出。两条流程均已跑通真实推理；同图并排界面与样本评估将在第五步实现。

## 项目方案

- 后端采用 FastAPI，统一管理图片输入、分析任务和结构化结果。
- 传统视觉方案使用 YOLO11n 检测，随后将检测框交给 SAM 2.1 tiny 分割。两者复用现有 Ultralytics、PyTorch 与 GPU 环境，不重新训练模型。
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
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

只需选择一种创建环境的方式。项目运行环境、后续权重及输出文件都放在项目目录内。创建环境后可再用 `.\.venv\Scripts\python.exe scripts/check_environment.py` 检查隔离环境。

精确依赖保存在 `requirements.lock.txt`，它对应本机 Windows、Python 3.12、CUDA 12.8 环境。先执行上述 PyTorch 安装命令，再使用 `python -m pip install -r requirements.lock.txt` 复现；锁定文件已指明 PyTorch 官方额外索引。测试依赖单独放在 `requirements-dev.txt`。

## 准备检测模型

```powershell
.\.venv\Scripts\python.exe scripts/prepare_model.py
.\.venv\Scripts\python.exe scripts/prepare_model.py --model sam
```

脚本先检查项目 `models/yolo11n.pt` 和 `models/sam2.1_t.pt`，有效则复用；缺失时下载 Ultralytics 官方发布的权重并校验 SHA256。分别约 5.6 MB 和 78.1 MB，不进入 Git。Ultralytics 库和使用方式涉及 AGPL-3.0 或企业许可规则，参考 [官方许可说明](https://www.ultralytics.com/license)；SAM 2 原始模型的 [Apache-2.0 许可](https://github.com/facebookresearch/sam2/blob/main/LICENSE) 与 Ultralytics 集成代码许可分别适用。这是使用预训练模型，无需训练数据或自行训练。

## 启动后端

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

- 健康检查：<http://127.0.0.1:8000/api/health>
- 交互接口文档：<http://127.0.0.1:8000/docs>

`.env` 使用 UTF-8 编码，环境变量优先于配置文件。密钥只保存在本地 .env，不进入 Git；不要将密钥填入会提交的 .env.example。传统视觉流程无需模型密钥，远程视觉模型按下面说明配置。已有 .env 时不要用模板覆盖它；修改配置后需要重启后端。

## 检测图片

启动服务后，在接口文档中展开 `POST /api/detect`，点击尝试并上传图片。也可在另一个 PowerShell 窗口运行：

```powershell
.\.venv\Scripts\python.exe scripts/detect_image.py '你的桌面照片.jpg'
```

脚本向本地接口上传照片，生成 `outputs/detection.json` 和 `outputs/detection.jpg`；重复运行会更新这两个结果文件。默认使用 GPU（不可用时使用 CPU），可在 `.env` 设置 `YOLO_DEVICE=cpu`。当前服务器不保存上传原图或结果，脚本只在本地保存返回结果。

接口支持 JPEG、PNG、WebP 静态图片，默认最大 10 MiB、2000 万像素。手机方向校正后，最长边超过 1600 像素的图片按比例缩小；YOLO 内部再以 `imgsz=640` 保持比例补边。返回 `bbox` 为方向校正后的原图像素坐标 `x1,y1,x2,y2`；标注图为处理尺寸，图中编号从 1 开始，对应 `detections` 的列表顺序。`sha256` 标识原始上传文件，便于后续核对同图输入。

`confidence`（默认 0.25）过滤低置信度候选；`iou`（默认 0.45）控制重叠框去重。`counts` 为中文分类计数，`total` 包含整张图所有检出物体，可能包括桌子本身。它尚不是只统计桌面小物件的清单。未检出不表示没有物体，纸笔、橡皮等未训练类别不能可靠识别。

总耗时不含上传，包含图片预处理、首次模型加载、推理和绘图；单独返回 YOLO 推理调用耗时，首次调用还可能包含预热。当前请求串行推理，避免同一模型对象并发冲突，不用于高并发服务。

## 分割与物品清单

启动服务后使用 `POST /api/segment`，或运行：

```powershell
.\.venv\Scripts\python.exe scripts/detect_image.py '你的桌面照片.jpg' --segment
# 也将人和家具计入清单。
.\.venv\Scripts\python.exe scripts/detect_image.py '你的桌面照片.jpg' --segment --include-background
```

分割先运行 YOLO，再把处理尺寸的检测框作为 SAM 提示；每图最多 50 个检测实例，按每批 8 个框执行分割。为复用环境并覆盖提示式分割学习要求，本步选择现有库支持的 SAM 2.1 tiny，而非另装参考项目或复制其代码。

默认清单按英文类别名排除 `person`、`dining table`、`chair`、`couch`、`bed`，可在 `.env` 中用 `INVENTORY_EXCLUDED_LABELS` 配置 JSON 数组。排除项仍在 `instances` 中，带 `included=false` 和原因。规则只是按类别排除，不会判断物体是否在桌面上。`include_background=true` 可将这些项也计入清单；原 `/api/detect` 行为不变。

`inventory` 提供类别、数量与实例编号；`counts`、`total` 仅统计纳入清单的项。`detected_total` 是全部检测实例数，`excluded_total` 是排除数，`segmented_total` 是成功得到非空掩膜的实例数。没有掩膜的检测项仍保留，`mask_status` 明确区分已生成、空掩膜与未返回。SAM 不负责类别识别，不增加清点数，也不会补回漏检。

每个实例的 `mask.png` 是二值 PNG data URL，前景 255、背景 0，保留内部孔洞，尺寸为 `processed_width × processed_height`；框仍为方向校正后的原图坐标。`area_pixels` 和 `area_ratio` 表示处理图上的区域大小，不是物体的物理面积；不同掩膜可重叠。叠加图包含所有有掩膜的实例，编号对应 `instance_id`。掩膜质量分数与 YOLO 类别置信度不同，不能直接比较。

命令行工具将每次分割保存到独立的 `outputs/segmentation-随机编号/`，包含中文 JSON 清单、叠加图和各实例 PNG，避免上次掩膜混入本次结果。JSON 用 `png_file` 指向导出的掩膜文件；HTTP 返回中则直接包含 PNG data URL。服务端仍不保存上传图片。

## 手动提示分割

`POST /api/segment/prompt` 接收图片和名为 `prompt` 的表单字段。提示使用方向校正后的原图坐标，在读取后按比例映射到处理图。框示例：

```json
{"kind":"box","box":[170,15,425,320]}
```

同一物体的多个提示点示例，`1` 为前景、`0` 为背景，至少有一个前景点：

```json
{"kind":"points","points":[[280,150],[50,350]],"labels":[1,0]}
```

保存为本地 UTF-8 `prompt.json` 后运行：

```powershell
.\.venv\Scripts\python.exe scripts/detect_image.py '你的桌面照片.jpg' --prompt prompt.json
```

工具将结果写入独立的 `outputs/prompt-随机编号/`。手动分割不命名物体，不计入自动物品清单。点可能指向物体的局部，需核对结果后调整框或前景／背景点。


## 视觉大模型独立清点

在本地 .env 填写 VISION_API_BASE_URL、VISION_API_KEY 和 VISION_MODEL。基础地址应包含服务要求的版本路径，例如 https://api.hangzhale.com/v1，后端会追加 /chat/completions；不要填写完整的补全接口路径。模型必须支持图像和 Chat Completions 协议。本机按用户配置使用 gpt-6.1-sol 已验证，使用其他服务时按其文档填写实际模型名。

图片会通过 HTTPS 发送到配置的视觉服务，可能产生费用。启动后端后，使用 POST /api/vision 上传同一张图片，或运行：

~~~powershell
.\.venv\Scripts\python.exe scripts/analyze_vision.py '你的桌面照片.jpg'
# 同时将人和家具计入清单。
.\.venv\Scripts\python.exe scripts/analyze_vision.py '你的桌面照片.jpg' --include-background
~~~

本流程只接收统一预处理后的图片与固定中文提示，不接收 YOLO / SAM 结果。它与传统方案返回同一原始文件 SHA256；发送的 PNG 另有摘要。清单返回 inventory、counts、total、excluded_items 和 uncertainties，总数由后端计算。默认复用同一份背景类别配置；不推断真实桌面范围。大模型可能识别出传统模型未支持的类别，也可能漏认或误认，结果需要人工核对。

工具每次保存至独立的 outputs/vision-随机编号/vision.json，包含文字提示及其版本、输入摘要、模型标识、输出模式与令牌上限、清单、耗时和服务提供的令牌用量。该接口不生成掩膜或检测图。模型返回的类别名仅做格式归一化，同义词差异留待人工检查，不自动模糊合并。

默认配置如下：

~~~dotenv
VISION_TIMEOUT_SECONDS=120
VISION_MAX_OUTPUT_TOKENS=4096
VISION_RESPONSE_MODE=json_object
VISION_TOKEN_LIMIT_FIELD=max_completion_tokens
~~~

json_object 只约束有效 JSON，后端仍严格校验字段、正整数数量和重复类别。服务明确支持严格结构时可使用 json_schema；仅支持文字响应时可显式改为 text。部分兼容服务只接受 max_tokens，请调整 VISION_TOKEN_LIMIT_FIELD。配置不匹配时返回错误，不会自动换模型、降级或重试。

缺少配置返回 503，超时返回 504，上游限流或额度不足返回 429，其他上游错误和无效清单返回 502，模型明确拒绝分析返回 422。失败不会伪装成空清单。默认 120 秒是整个远程调用时限，命令行默认等待 150 秒；提高服务时限时，相应设置工具 --timeout。后端不保存上传图片或原始上游响应，不向前端透传密钥和错误正文。

完整的协议、JSON 约束和 token、embedding、attention 等学习说明见 [视觉模型接入与学习记录](docs/视觉模型接入.md)；样例结果与局限见 [第四步验证记录](docs/第四步验证.md)。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache/pip -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

回归测试不下载或加载真实权重；真实推理结果、样本来源及局限见 [第二步验证记录](docs/第二步验证.md)。健康检查仅代表后端响应，不表示模型已经加载。

第三步的 24 项回归与真实分割记录见 [第三步验证记录](docs/第三步验证.md)。可复现端到端验证：

```powershell
.\.venv\Scripts\python.exe scripts/verify_segmentation.py '本地公开样本.png' --cpu
```

该脚本只临时启动本地服务，不联网下载图片；需要已准备的检测和分割权重，会将本地图像与派生验证图保存在 `outputs/step3-verification`。不指定 `--cpu` 时只验证当前配置设备；指定时额外运行一次 CPU 框分割。它验证数据链路和结果一致性，不代表准确率或跨设备像素一致。

第四步的 38 项回归测试全部通过。一次真实调用的端到端验证：

~~~powershell
.\.venv\Scripts\python.exe scripts/verify_vision.py '允许发送给模型服务的本地图片.png'
~~~

该脚本临时启动本地后端，通过命令行工具发起一次真实分析并检查图片摘要、明细、总数和导出，结束后关闭服务。它使用 .env 中的模型配置，会发送图片并可能产生费用，不自动重试。生成结果保存在独立的 outputs/step4-verification-* 目录，不进入 Git。模拟回归测试不联网，不代表真实模型准确率。

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
  verify_segmentation.py 真实分割链路验证
  analyze_vision.py     上传图片并保存视觉大模型清单
  verify_vision.py      一次真实视觉模型链路验证
tests/                  图片处理与接口回归测试
.env.example            配置模板
requirements.txt        当前阶段依赖范围
requirements.lock.txt   已验证环境的精确依赖版本
requirements-dev.txt    测试依赖
```
