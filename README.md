# 桌面物品清点

上传一张桌面照片，同时通过 YOLO11n + SAM 2.1 tiny 和 DeepSeek 视觉模型生成物品清单，在网页并排比较类别、数量与耗时，并支持人工核对和结果导出。保留 PowerShell 命令行入口。

## 功能概览

当前已完成网页双方案分析，并在 Ubuntu 云服务器上通过 Docker Compose 部署。日常使用可通过浏览器上传图片，也可在本机通过 PowerShell 直接调用共享分析服务。网页固定采用双方案并发分析，等待两侧结束后一起展示结果。

| 功能 | 实现与输出 |
| --- | --- |
| 图片预处理 | JPEG、PNG、WebP 校验，手机方向校正、透明背景合成、等比缩放及文件摘要 |
| 本地检测与分割 | YOLO11n 输出类别、置信度和边界框，SAM 2.1 tiny 按框生成实例掩膜 |
| 手动提示分割 | 支持框提示，以及同一物体的前景／背景点提示 |
| 视觉大模型清点 | 默认通过 DeepSeek 官方接口调用 deepseek-flash，独立观察图片并校验结构化清单 |
| 同图对比 | 并发执行两条方案，按英文类别展示数量、差值、耗时与不确定说明 |
| 网页核对与导出 | 填写人工数量、补充漏检类别，计算误差并下载 JSON、CSV；页面展示本地分割图 |
| 命令行文件导出 | 完整 JSON、中文 CSV、分割叠加图、二值掩膜和空白人工清点模板 |
| 人工离线核对 | 命令行校验图片摘要和人工数量，计算类别误差、总数误差、少计和多计 |
| 异常处理 | 单侧失败仍保留另一侧结果；失败不计为零，不自动重试付费调用 |

截至 2026 年 10 月 21 日，69 项回归测试通过，依赖一致性检查通过；已完成本机 GPU 推理、PowerShell 与离线评估验证，以及云端 CPU 检测分割和 DeepSeek 真实图像调用。CSV 导出保留 UTF-8 BOM；Windows 重复换行问题已有回归检查。

真实场景验证采用公开咖啡桌样例，不能据此推断复杂桌面的整体准确率。当前云端应用绑定服务器回环地址，通过 SSH 隧道访问；尚未配置 nginx、登录认证或直接公网入口。运行视频与个人材料单独准备，尚未确认完成最终交付，不属于程序运行依赖。

## 直接在 PowerShell 使用

完成下面的环境和权重准备，并在本地 .env 配置视觉服务后，直接运行：

~~~powershell
Set-Location 'D:\desk inventory'
.\scripts\inventory.ps1 -Image 'D:\照片\桌面.jpg'
~~~

无需打开网页或启动 Uvicorn。两条流程并发分析同一份图片，结果保存到独立的 outputs/comparison-* 目录。使用视觉服务会发送照片，可能产生费用。也可以不带参数运行 .\scripts\inventory.ps1，按提示输入照片路径。

命令名和参数使用英文，终端提示使用中文。分析后可用短命令查看和核对最近结果：

~~~powershell
.\scripts\inventory.ps1 -Show
.\scripts\inventory.ps1 -Review
~~~

-Show 重看已保存的分类表，-Review 在终端输入人工数量后直接计算误差，均不重新调用模型。需要查看分割图时使用 -Show -OpenImage。可以补充两侧都漏掉的类别，最后输入“是”保存。自定义输出目录时，三种操作需使用相同 -Output 参数。

输出包含终端分类对照、comparison.json、counts.csv、分割图、实例掩膜，以及初始数量全部为空的 人工清点模板.json。失败一侧显示“未完成”，不按零计数；成功一侧结果仍保留。

查看原图并填写模板后，只计算误差：

~~~powershell
.\scripts\inventory.ps1 -Result '结果目录\comparison.json' -Truth '结果目录\人工清点模板.json' -ConfirmTruth
~~~

离线核对不会调用模型或产生 API 费用。支持添加两边都漏掉的类别，计算类别绝对误差、总数误差、少计和多计；这些指标不是检测或分割准确率。

更多参数、中文模板格式、执行策略受限时的 Python 入口和适用场景见 [PowerShell 使用说明](docs/PowerShell使用说明.md)。真实结果和测试范围见 [第五步验证](docs/第五步验证.md)。

## 项目方案

- 后端采用 FastAPI，统一管理图片输入、分析任务和结构化结果。
- 传统视觉方案使用 YOLO11n 检测，随后将检测框交给 SAM 2.1 tiny 分割。两者复用现有 Ultralytics、PyTorch 与 GPU 环境，不重新训练模型。
- 多模态方案默认使用 DeepSeek 官方 OpenAI 兼容接口和 deepseek-flash，独立分析同一张图片，不使用传统模型结果作为提示。
- 前端使用 HTML、CSS 和原生 JavaScript，通过 FastAPI 提供页面；网页和 PowerShell 共用同图分析服务。
- 两条流程接收相同原始文件字节并核对 SHA256；只有两侧成功时才生成类别差异。HTTP 200 也可能包含单侧失败，需要查看各分支的 status。
- 网页支持人工核对与 JSON、CSV 下载；命令行另支持分割图、实例掩膜及人工模板文件导出。
- 支持计数误差评估；当前仅完成公开样例的开发验证，不宣称整体准确率。未实现数据库或历史记录功能。

YOLO 预训练模型只支持其训练类别，不能预先保证识别笔、橡皮等所有桌面物品。SAM 负责按提示分割，本身不会补全漏检类别。当前使用固定训练类别，未集成开放词汇检测。

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

## 准备本地模型

```powershell
.\.venv\Scripts\python.exe scripts/prepare_model.py
.\.venv\Scripts\python.exe scripts/prepare_model.py --model sam
```

脚本先检查项目 `models/yolo11n.pt` 和 `models/sam2.1_t.pt`，有效则复用；缺失时下载 Ultralytics 官方发布的权重并校验 SHA256。分别约 5.6 MB 和 78.1 MB，不进入 Git。Ultralytics 库和使用方式涉及 AGPL-3.0 或企业许可规则，参考 [官方许可说明](https://www.ultralytics.com/license)；SAM 2 原始模型的 [Apache-2.0 许可](https://github.com/facebookresearch/sam2/blob/main/LICENSE) 与 Ultralytics 集成代码许可分别适用。这是使用预训练模型，无需训练数据或自行训练。

## 网页与 HTTP 接口

先完成环境、权重准备，并按“视觉大模型独立清点”一节配置 `.env`。在本机 Windows PowerShell 的项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

- 网页首页：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- 健康检查：[http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- 交互接口文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

网页操作流程：

1. 选择 JPEG、PNG 或 WebP 桌面照片，点击“开始分析”。图片会发送给所配置的视觉服务，可能产生费用。
2. 等待本地方案和视觉大模型结束，查看各自完成状态、类别清单、差值、耗时及本地分割图。
3. 对照原图填写真实数量；两侧都漏掉的类别可用“补充类别”加入，不能仅以两侧结果一致作为正确依据。
4. 勾选已核对确认，点击“计算计数误差”。评估只使用本次结果，不重新调用模型。
5. 使用“下载 JSON”或“下载 CSV”保存结果。页面不提供历史记录，刷新后需要重新分析；需保留的结果请及时下载。

两侧请求并发执行，但页面会等待两侧全部结束；一侧失败时仍展示另一侧结果。网页不会自动重试远程调用。PowerShell 清点入口不要求启动此 HTTP 服务。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/` | 网页首页 |
| GET | `/api/health` | 应用存活检查，不代表模型推理成功 |
| POST | `/api/detect` | YOLO 检测 |
| POST | `/api/segment` | YOLO + SAM 自动清点 |
| POST | `/api/segment/prompt` | 点或框提示分割 |
| POST | `/api/vision` | 独立视觉大模型清点 |
| POST | `/api/compare` | 同图双方案并发分析 |
| POST | `/api/compare/evaluate` | 根据人工清点计算误差 |

`.env` 使用 UTF-8 编码，环境变量优先于配置文件。密钥只保存在本地 .env，不进入 Git；不要将密钥填入会提交的 .env.example。传统视觉流程无需模型密钥，远程视觉模型按下面说明配置。已有 .env 时不要用模板覆盖它；修改配置后需要重启后端。

## 检测图片

启动服务后，在接口文档中展开 `POST /api/detect`，点击尝试并上传图片。也可在另一个 PowerShell 窗口运行：

```powershell
.\.venv\Scripts\python.exe scripts/detect_image.py '你的桌面照片.jpg'
```

脚本向本地接口上传照片，生成 `outputs/detection.json` 和 `outputs/detection.jpg`；重复运行会更新这两个结果文件。默认使用 GPU（不可用时使用 CPU），可在 `.env` 设置 `YOLO_DEVICE=cpu`。当前服务器不保存上传原图或结果，脚本只在本地保存返回结果。

接口支持 JPEG、PNG、WebP 静态图片，默认最大 10 MiB、2000 万像素。手机方向校正后，最长边超过 1600 像素的图片按比例缩小；YOLO 内部再以 `imgsz=640` 保持比例补边。返回 `bbox` 为方向校正后的原图像素坐标 `x1,y1,x2,y2`；标注图为处理尺寸，图中编号从 1 开始，对应 `detections` 的列表顺序。`sha256` 标识原始上传文件，便于后续核对同图输入。

`confidence`（默认 0.25）过滤低置信度候选；`iou`（默认 0.45）控制重叠框去重。`counts` 为中文分类计数，`total` 包含整张图所有检出物体，可能包括桌子本身。它尚不是只统计桌面小物件的清单。未检出不表示没有物体，纸笔、橡皮等未训练类别不能可靠识别。

总耗时不含上传，包含图片预处理、首次模型加载、推理和绘图；单独返回 YOLO 推理调用耗时，首次调用还可能包含预热。本地推理使用共享锁，避免同一模型对象并发冲突；远程视觉分支可与本地推理并发，不用于高并发服务。

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

在运行应用的机器上，将 `.env.example` 复制为 `.env`（已有文件请直接修改，避免覆盖），填写 DeepSeek 开放平台密钥。当前默认使用 `https://api.deepseek.com` 和支持图片输入的 `deepseek-flash`，已在本机和云服务器完成真实调用。服务能力及模型名以 [DeepSeek 图像理解文档](https://api-docs.deepseek.com/zh-cn/guides/vision) 为准。

后端自动追加 `/chat/completions`，不要在基础地址中填写完整补全路径。切换其他兼容服务时，基础地址应包含该服务要求的版本路径，模型必须支持图像和 Chat Completions 协议；密钥需属于实际使用的服务商。

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
VISION_API_BASE_URL=https://api.deepseek.com
# 在本机或服务器的 .env 中填写，不要把真实密钥提交到仓库。
VISION_API_KEY=
VISION_MODEL=deepseek-flash
VISION_TIMEOUT_SECONDS=120
VISION_MAX_OUTPUT_TOKENS=4096
VISION_RESPONSE_MODE=json_object
VISION_TOKEN_LIMIT_FIELD=max_tokens
~~~

`json_object` 只约束有效 JSON，后端仍严格校验字段、正整数数量和重复类别。DeepSeek 当前使用 `max_tokens`。其他服务明确支持严格结构时可使用 `json_schema`；仅支持普通文本输出时可显式改为 `text`，仍要求模型能读取图片。按服务要求设置 `VISION_TOKEN_LIMIT_FIELD=max_tokens` 或 `max_completion_tokens`。配置不匹配时返回错误，不会自动换模型、降级或重试。

缺少配置返回 503，超时返回 504，上游限流或额度不足返回 429，其他上游错误和无效清单返回 502，模型明确拒绝分析返回 422。失败不会伪装成空清单。默认 120 秒是远程 HTTP 调用总时限。通过 HTTP 调用后端的 analyze_vision.py 默认等待 150 秒，可用 --timeout 调整；直接运行 compare_image.py 时由服务配置控制远程调用时限。后端不保存上传图片或原始上游响应，不向前端透传密钥和错误正文。

完整的协议、JSON 约束和 token、embedding、attention 等学习说明见 [视觉模型接入与学习记录](docs/视觉模型接入.md)；样例结果与局限见 [第四步验证记录](docs/第四步验证.md)。

## Linux + Docker 部署

已在 Ubuntu 22.04 amd64、2 vCPU、4 GB 内存的云服务器验证，容器运行 CPU 版 PyTorch。Dockerfile 负责安装系统库与 Python 依赖、复制后端并启动 Uvicorn；Compose 负责环境变量、模型和缓存挂载、端口、健康检查及重启策略。

部署需要已安装并运行 Docker Engine 与 Docker Compose。下面使用与已验证部署一致的 Compose 项目名 `deskinventory`。命令中的 `SERVER_IP` 和 SSH 用户 `admin` 应替换为自己的服务器信息，不要在公开材料中写入密码或私钥。

### 1. 本机 Windows PowerShell：构建并导出镜像

本机 Docker 需切换到 Linux 容器模式。先按前文准备好两份模型权重，在项目根目录执行：

```powershell
docker version
docker compose version
docker build --platform linux/amd64 -t deskinventory-desk-inventory:latest .
New-Item -ItemType Directory -Path '.tmp' -Force | Out-Null
docker save -o '.tmp/desk-inventory-image.tar' deskinventory-desk-inventory:latest
Get-FileHash -Algorithm SHA256 '.tmp/desk-inventory-image.tar'
```

构建失败时先处理错误，不要继续导出旧镜像。`.dockerignore` 排除 `.env`、模型、缓存、输出与文档；不要把密钥写入其他源码或放在构建目录中的配置备份里。镜像包含后端和网页，模型在服务器运行时单独挂载。

### 2. 本机 Windows PowerShell：上传部署文件

服务器无法拉取 Docker Hub 镜像时，可复用本机构建、导出、传输和服务器导入流程：

```powershell
ssh admin@SERVER_IP 'mkdir -p /home/admin/desk-inventory/models /home/admin/desk-inventory/outputs /home/admin/desk-inventory/.cache'
scp '.tmp/desk-inventory-image.tar' admin@SERVER_IP:/home/admin/desk-inventory-image.tar
scp 'compose.yaml' '.env.example' admin@SERVER_IP:/home/admin/desk-inventory/
scp 'models/yolo11n.pt' 'models/sam2.1_t.pt' admin@SERVER_IP:/home/admin/desk-inventory/models/
```

使用私钥时，可在 `ssh` 和 `scp` 中加入 `-i "$HOME\.ssh\id_ed25519"`，按实际路径调整。已有权重可直接复用，无需重复上传；不上传本机 `.env` 或私钥。

### 3. 云服务器 Ubuntu 终端：配置并启动

登录服务器后执行，先确认归档摘要与本机 `Get-FileHash` 输出一致：

```bash
sha256sum /home/admin/desk-inventory-image.tar
sudo docker load -i /home/admin/desk-inventory-image.tar
cd /home/admin/desk-inventory
if [ ! -f .env ]; then cp .env.example .env; fi
chmod 600 .env
nano .env
```

在 `.env` 中填写前文 DeepSeek 配置及自己的密钥。服务器与本机各自读取自己的配置文件，修改一端不会同步另一端。不要在终端日志、截图或 Git 中暴露密钥。

```bash
sudo docker compose -p deskinventory up -d --no-build --pull never
sudo docker compose -p deskinventory ps
sudo docker compose -p deskinventory logs --tail=60 desk-inventory
```

本流程显式使用已导入镜像，不在服务器重新构建或拉取。启动后健康状态可能先显示 `starting`，稍后应为 `healthy`；异常时检查日志。

默认映射为 `127.0.0.1:8000:8000`，服务只对服务器本机开放。模型目录只读挂载到 `/app/models`，输出与缓存分别挂载到 `/app/outputs`、`/app/.cache`。Compose 强制使用 CPU；网页请求本身不保存上传原图或推理结果，存在输出挂载不代表提供历史记录功能。

### 4. 本机 Windows PowerShell 与浏览器：通过 SSH 隧道验收

在本机另开一个 PowerShell 窗口，执行并保持窗口开启：

```powershell
ssh -N -L 8002:127.0.0.1:8000 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 admin@SERVER_IP
```

浏览器访问 [http://127.0.0.1:8002/](http://127.0.0.1:8002/) 使用网页，或访问 [http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs) 分别测试 `/api/segment` 和 `/api/vision`。页面上传同一张图后，应显示两侧完成状态、清单与本地分割图；再核对人工评估和 JSON、CSV 下载。健康检查通过仅说明应用存活，不能代替真实模型调用。

如连接远程视觉服务失败，可先在服务器执行 `curl -I --connect-timeout 10 --max-time 20 https://api.deepseek.com` 检查基础连通性。未携带密钥时返回 401 仍表明已收到 HTTP 响应，不代表已验证密钥或视觉推理；宿主机能访问也不保证容器内部能访问，应以容器中的真实调用为准。

### 5. 更新与日常维护（云服务器 Ubuntu 终端）

修改 `.env` 后需重新创建容器，单纯 `restart` 不会重新加载 Compose 的环境变量：

```bash
cd /home/admin/desk-inventory
cp -p .env "/home/admin/desk-inventory-env.$(date +%Y%m%d-%H%M%S).backup"
# 用 nano .env 修改配置，再重新创建容器。
sudo docker compose -p deskinventory up -d --no-build --pull never --force-recreate desk-inventory
```

配置备份保存在项目目录外，继续按密钥文件保管。更新业务代码或网页时，在本机构建新镜像，重新导出、上传、核对摘要并 `docker load`，再执行上面的重新创建命令。仅替换服务器目录里的源码不会更新镜像中的应用；只调整视觉服务配置时无需重建镜像，前提是旧镜像已支持对应请求格式。

```bash
sudo docker compose -p deskinventory ps
sudo docker compose -p deskinventory logs --tail=60 desk-inventory
sudo docker compose -p deskinventory stop desk-inventory
sudo docker compose -p deskinventory start desk-inventory
```

停止容器不会停止云服务器计费。关闭 SSH 隧道只会断开本机访问，不会停止服务。当前未配置 nginx、登录认证、TLS 公网入口、任务队列或高并发保障；正式公开访问需要另行设计访问控制及部署方案。

## 测试与验证

```powershell
.\.venv\Scripts\python.exe -m pip install --cache-dir .cache/pip -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

完整回归共 69 项，覆盖图片处理、检测与分割、提示坐标、视觉响应校验、DeepSeek 请求协议、同图并发、分支失败、网页入口、导出文件、人工模板和离线评估。导出测试直接读取落盘 CSV，检查中文 BOM、重复回车和空行。模拟测试不联网、不下载或加载真实权重，不代表真实识别准确率，也不能代替浏览器完整操作验收。

真实运行与样本来源分别见 [第二步验证记录](docs/第二步验证.md)、[第三步验证记录](docs/第三步验证.md)、[第四步验证记录](docs/第四步验证.md) 和 [第五步验证记录](docs/第五步验证.md)。这些记录保留当时的模型服务、交互方式和测试数量；当前配置和测试总数以本 README 为准。

截至 2026 年 10 月 21 日，最近云端阶段验收使用 860×574 的公开桌面图片，记录如下。两条分支分别调用时的耗时不是双方案并发总耗时，也不是稳定性能基准：

| 验证项 | 实际结果 |
| --- | --- |
| YOLO11n + SAM 2.1 tiny，CPU | 约 11.7 秒；4 个检测实例、4 个掩膜，默认纳入笔记本电脑 1、杯子 2，排除餐桌 |
| DeepSeek 独立图像调用 | 约 6.78 秒；请求及服务返回模型名均为 deepseek-flash；笔记本电脑 1、杯子 2、碟子 2 |
| 网页双方案流程 | 用户完成服务器首页操作并反馈运行正常；保留双方案并发，不提供仅本地模式选择 |

上述结果验证调用链及清单解析；服务返回的模型名是服务自报。漏检、同义词、遮挡与分割质量仍需人工检查。运行视频和个人材料的最终交付另行完成。

检测与分割的阶段验证记录见 [第三步验证记录](docs/第三步验证.md)。可复现端到端验证：

```powershell
.\.venv\Scripts\python.exe scripts/verify_segmentation.py '本地公开样本.png' --cpu
```

该脚本只临时启动本地服务，不联网下载图片；需要已准备的检测和分割权重，会将本地图像与派生验证图保存在 `outputs/step3-verification`。不指定 `--cpu` 时只验证当前配置设备；指定时额外运行一次 CPU 框分割。它验证数据链路和结果一致性，不代表准确率或跨设备像素一致。

视觉模型的阶段验证见 [第四步验证记录](docs/第四步验证.md)。如需复核真实接口，可运行以下端到端检查：

~~~powershell
.\.venv\Scripts\python.exe scripts/verify_vision.py '允许发送给模型服务的本地图片.png'
~~~

该脚本临时启动本地后端，通过命令行工具发起一次真实分析并检查图片摘要、明细、总数和导出，结束后关闭服务。它使用 .env 中的模型配置，会发送图片并可能产生费用，不自动重试。生成结果保存在独立的 outputs/step4-verification-* 目录，不进入 Git。模拟回归测试不联网，不代表真实模型准确率。

## 已知限制

- YOLO 使用预训练类别，不能可靠覆盖笔、橡皮等所有桌面小物件；SAM 不会补回漏检或增加类别。
- 默认仅按类别排除人和家具，没有检测真实桌面边界。背景类别排除可以关闭，但这不等于空间定位。
- 类别对比按小写英文名称精确匹配，同义词和类别粒度差异需要人工核对。
- 分割结果可能存在孔洞、边缘偏差和背景混入；像素面积不代表物体的物理面积。
- 本地自动分割每图最多检测 50 个实例，SAM 每批处理 8 个框。密集、遮挡和小目标场景尚缺少充分评估。
- 视觉服务依赖网络、权限与额度，可能漏认或误认；第三方返回的模型名及令牌用量属于服务自报。
- 真实样例数量有限；派生图片用于检查流程，不作为独立数据集，不报告整体准确率。
- 已完成云服务器 Docker 部署及实际推理，但仍面向单用户演示，模型推理使用共享锁，未进行高并发、长期稳定性或生产安全验收。
- 当前通过 SSH 隧道访问，无登录认证、数据库、历史记录或任务队列；刷新页面不会恢复旧结果。

## 文档导航

- [PowerShell 使用说明](docs/PowerShell使用说明.md)：英文参数、导出文件及人工核对。
- [视觉模型接入与学习记录](docs/视觉模型接入.md)：协议、结构约束和相关原理。
- [环境记录](docs/环境记录.md)：环境盘点、依赖复用和设备验证。
- [开发计划](docs/开发计划.md)：六步开发安排与技术选择的历史记录。

历史文档中的终端交互、旧视觉服务与阶段测试数量不代表当前版本。项目说明、个人感悟与录制指南另行准备，不作为应用运行依赖提交；视频由使用者自行录制。

## 目录结构

```text
backend/app/
  api/                  接口路由
  core/                 统一配置
  schemas/              接口返回数据结构
  services/             图片处理和模型推理
  templates/index.html  双方案网页、人工核对与结果下载
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
  compare_image.py      无服务的同图对比与离线核对
  inventory.ps1          PowerShell 入口（英文参数、中文提示）
tests/                  图片处理与接口回归测试
.env.example            配置模板
requirements.txt        运行依赖范围
requirements.lock.txt   已验证环境的精确依赖版本
requirements-dev.txt    测试依赖
Dockerfile              Linux CPU 镜像构建
compose.yaml            服务配置、挂载与健康检查
.dockerignore           排除密钥、模型、缓存和本地材料
```
