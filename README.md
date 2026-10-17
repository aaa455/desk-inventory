# 桌面物品清点

上传一张桌面照片，分别通过本地视觉模型与视觉大模型生成物品清单，并排比较识别类别、计数、耗时和使用体验。

## 当前进度

第一步：后端基础结构、健康检查接口、配置模板、环境检查脚本和开发计划。当前还没有接入识别模型或图片上传功能，健康检查仅代表后端正常响应。

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

本机已有 Python 3.12 可用于创建隔离环境，Python 3.15 暂不作为视觉模型环境。第一步只安装后端依赖；模型、PyTorch、OpenCV 和大模型 SDK 在对应步骤核对兼容性后再安装。

在项目根目录运行，使用本机已有的 Python 3.12 解释器创建环境：

```powershell
# 已配置 Python 启动器时使用。
py -3.12 -m venv .venv

# 当前机器也可直接复用以下已安装的解释器，无需重新下载 Python。
# & 'C:\Users\xingkui\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

只需选择一种创建环境的方式。项目运行环境、后续权重及输出文件都放在项目目录内。创建环境后可再用 `.\.venv\Scripts\python.exe scripts/check_environment.py` 检查隔离环境。

已验证环境的精确依赖版本保存在 `requirements.lock.txt`，复现当前版本时可使用 `python -m pip install -r requirements.lock.txt`。

## 启动后端

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

- 健康检查：<http://127.0.0.1:8000/api/health>
- 交互接口文档：<http://127.0.0.1:8000/docs>

`.env` 使用 UTF-8 编码，环境变量优先于配置文件。密钥只保存在本地，不进入 Git；第一步无需配置任何模型密钥。模板中的服务地址仅为示例，接入时需要确认服务支持图像输入以及所选模型。

## 目录结构

```text
backend/app/
  api/                  接口路由
  core/                 统一配置
  main.py               后端入口
docs/
  开发计划.md           分步范围、验收要求和学习要点
  环境记录.md           环境盘点与复用结论
scripts/
  check_environment.py  无需额外安装依赖的环境检查
.env.example            配置模板
requirements.txt        当前阶段依赖范围
requirements.lock.txt   已验证环境的精确依赖版本
```
