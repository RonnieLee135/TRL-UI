# TRL-UI

把 [TRL](https://github.com/huggingface/trl) 支持的强化学习算法与全部入参做成一个可以直接用的 Web 控制台：浏览算法、按四步填表、实时校验、生成 `trl <algo>` 命令，然后在这台机器上真实训练或模拟演示。

- **数据来自源码，不靠手抄**：21 个 trainer、5 个算法家族、四个参数平面（script / algorithm / model / runtime），参数名、类型、默认值、帮助文本都来自对 TRL 源码 dataclass 与 docstring 的解析。
- **零依赖启动**：后端只用 Python 标准库（含 `sqlite3`），前端是原生 JS + CSS，无 npm、无 CDN、无构建步骤——`python3 webui/server.py` 就能打开。
- **两种运行模式**：有 GPU 且依赖齐全时真实训练；否则用内置模拟器把界面、日志解析、曲线与产物走一遍，并在每处标明“模拟”。

## 功能

### 算法总览

- 一个 hero 概览：TRL 版本与 commit、算法数、稳定/实验比例、参数条目数、本机运行模式。
- 所有 trainer 共用的**八阶段管线**（环境与资源 → 模型 → 数据与提示 → 监督信号 → 采样 → 优势/偏好估计 → 损失与更新 → 记录与产物）。
- 按家族筛选（全部 / 在线强化学习 / 偏好对齐 / 蒸馏 / 奖励模型 / 监督微调）、按算法名/类名/参数名搜索。
- 每个算法点开抽屉看详情：训练流程映射、全部参数（带默认值与说明）、优势、代价、容易踩的坑、文档与论文链接，可复制命令或直接“用这个算法新建训练”。

### 新建训练（四步向导）

| 步骤 | 内容 |
| --- | --- |
| 1 选择算法 | 一个按家族分组的下拉框；选中后自动套用起步预设并进入下一步（算法说明在「算法总览」里看） |
| 2 数据与信号 | 数据集、奖励函数等输入（script 平面），带文档里的数据集样例 |
| 3 模型 | 模型 id、精度、LoRA、量化（model 平面） |
| 4 训练与资源 | 采样、优势、损失 + 并行、显存、日志（algorithm + runtime 平面），默认只显示关键参数 |

右侧栏贯穿全程：任务名、执行模式（真实训练 / 模拟）、实时校验结果、可复制的完整命令、`创建任务` / `创建并启动` / `套用起步预设`。

两个刻意的约定：

- **留空 = 不传**：没填的参数不会出现在命令里，TRL 升级后的新默认值会自动生效；要固定某个值就直接填上。
- **参数说明默认收起**，保持表单可扫读；在第 2–4 步或「环境」页用「显示参数说明」开关展开，偏好记在浏览器里。

### 任务

- 列表与详情：状态（`created` / `running` / `finished` / `failed` / `stopped`）、命令、实时日志（SSE 推送）、指标曲线、产物浏览与下载。
- 可启动、停止、删除任务；切走页面后日志与曲线仍由数据库保存。

### 环境

- 直接探测本机：核心依赖（trl / transformers / datasets / accelerate…）、torch 与加速器、TRL 源码路径、`trl` 命令、运行目录可写性，逐项给出 ok / warn / fail 与修复建议。
- 四项环境变量：`HF_HOME`、`HF_ENDPOINT`、`HF_TOKEN`、`CUDA_VISIBLE_DEVICES`。只注入真实训练进程（以及探测子进程）；`HF_TOKEN` 保存后不再回传浏览器，日志与产物里也只显示“已设置”。
- 顶栏右上角的模式标签由探测结果决定：依赖齐全且有 CUDA/NPU 设备时显示 `可实训 · N 设备`，否则显示 `演示模式 · 仅模拟`。

## 快速开始

### 前置条件

- Python 3.11+（与 TRL 的要求一致）。
- 启动界面本身**不需要**安装任何第三方包，也不需要事先准备 TRL 源码；仓库里已经带着生成好的算法目录 `catalog/catalog.json`。
- 真实训练另外需要一份 TRL 源码 checkout（默认 `../trl`）。不提供时界面照常可用，只会以模拟模式运行，并在「环境」页提示找不到 TRL 源码。

### 启动

```bash
cd TRL-UI
python3 webui/server.py                 # 默认 http://127.0.0.1:8770
python3 webui/server.py --port 8899 --trl /path/to/trl
```

Windows 上同理：`python webui\server.py`。

首屏打开 <http://127.0.0.1:8770/>。首次运行会自动创建 `webui/runs/` 与数据库 `webui/runs/studio.db`。

启动日志会打印实际端口（端口被占用时自动向后找）、TRL 版本与 commit、算法数量、产物目录、推荐运行模式。

### 第一次使用

1. 打开「环境」页确认识别到的依赖与设备。显示 `演示模式 · 仅模拟` 说明这台机器跑不了真训练，任务会以模拟模式提交。
2. 去「算法总览」挑一个算法，或在「新建训练」第 1 步直接下拉选择。
3. 按四步填参数（留空的项不会进入命令），右侧看校验与命令预览。
4. 点 `创建任务`，到「任务」页启动，看日志与曲线。产物落在 `webui/runs/<任务 id>/`。

### 真实训练需要什么

```bash
# 先按你的驱动装好 CUDA 版 torch，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 再装项目列出的训练依赖（torch/transformers/datasets/accelerate/peft…）
pip install -r requirements.txt
# 想与目录逐字节对齐时，安装被解析的那份 TRL：
pip install -e ../trl
```

装好回到「环境」页点「重新探测」，顶栏变成 `可实训 · N 设备` 后新建任务默认以真实训练提交。

## 运行模式

| | `real` 真实训练 | `simulate` 模拟演示 |
| --- | --- | --- |
| 前提 | trl + torch + GPU/NPU | 无 |
| 执行 | 用 `subprocess` 跑生成的命令，解析 TRL 日志 | 内置模拟器按算法家族生成日志与指标 |
| 产物 | 全部照常写盘 | `command.sh`、`params.json`、`README.md` 照常写盘 |
| 日志与曲线 | 真实 stdout 与指标 | 模拟数据，首行明确标注“不会真的训练” |
| 用途 | 真正训练 | 演示界面、校验命令、离线准备配置 |

模拟模式的日志在任务页有说明，任务记录里 `mode="simulate"`，产物 README 写明“模拟（未真实训练）”。**不要把模拟曲线当成训练结果。**

## 工作原理（简述）

1. `tools/extract_source_catalog.py` 用 `ast` 解析 TRL 源码里的 trainer 配置 dataclass（不 import TRL，不需要 torch），得到字段名、类型、默认值与 `metadata["help"]`，以及 docstring 里 `> Parameters that control …` 的分组标题。
2. `tools/build_catalog.py` 把解析结果与手工维护的 `catalog/algorithms.json`、`catalog/param_notes.json` 合并成 `catalog/catalog.json`——WebUI 的唯一数据源，并做严格校验（拼错的手工字段会让构建失败，而不是悄悄漏到界面上）。
3. WebUI 读取目录渲染表单；提交时 `builder` 负责把参数归一化、按 TRL 规则校验、拼出 `--name value` 形式的命令：
   - 有 CLI 子命令的算法 → `trl <algo> …`，并注入 `--accelerate_config` / `--num_processes`；
   - 实验性算法 → 生成 `train_<algo>.py`，用 `accelerate launch` 启动。
4. `runner` 启动任务并把 stdout/stderr 写进 SQLite，同时把日志里的 `{'loss': …}` 字典解析成指标；前端通过 SSE 实时读取。

## 目录结构

```
TRL-UI/
├── catalog/                 # 生成产物：WebUI 的唯一数据源
│   ├── catalog.json         #   合并后的完整目录
│   ├── source_catalog.json  #   从 TRL 源码解析出的字段
│   ├── algorithms.json      #   手工维护的算法元数据（家族、启动方式、数据契约、坑）
│   └── param_notes.json     #   手工补充的参数说明/阶段归属
├── docs/                    # 01 算法全景 / 02 参数总表（生成） / 03 训练流程
├── tools/                   # 目录与文档生成脚本
├── webui/
│   ├── server.py            # 入口：参数解析、端口选择、启动 HTTP 服务
│   ├── rlstudio/            # 标准库后端
│   │   ├── api.py           #   HTTP 路由、SSE、静态文件
│   │   ├── builder.py       #   校验 + 命令/脚本生成
│   │   ├── runner.py        #   真实/模拟执行、日志解析、产物写入
│   │   ├── store.py         #   SQLite：任务、日志、指标、设置
│   │   ├── doctor.py        #   环境探测（子进程跑 torch，避免拖垮服务）
│   │   ├── envvars.py       #   四项环境变量（白名单、校验、密钥不回显）
│   │   └── catalog.py       #   目录读取与查询
│   ├── static/              # 无构建前端：index.html / app.js / dom.js / styles.css / views/*
│   └── runs/                # 任务与产物（运行时生成，含 studio.db）
└── requirements.txt         # 训练后端依赖（启动界面不需要）
```

## 任务产物

每个任务在 `webui/runs/<任务 id>/` 下留下：

| 文件 | 内容 |
| --- | --- |
| `command.sh` | 可直接复制到任意 GPU 机器运行的完整命令 |
| `params.json` | 算法、执行模式、归一化后的参数、命令、并行配置 |
| `README.md` | 任务摘要、执行模式、训练器、关键参数表、注入的环境变量、复现步骤 |
| `train_<algo>.py` | 仅实验性算法：生成的训练脚本（`accelerate launch` 启动它） |

任务元数据、日志行与指标存在 `webui/runs/studio.db`（SQLite，WAL 模式），删掉这个库等于清空任务历史；已生成的产物目录不受影响。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8770` | 起始端口；被占用时自动向后找空闲端口 |
| `--no-port-shift` | 关闭 | 端口被占用时直接失败，而不是换端口 |
| `--trl` | `../trl` | TRL 源码目录（探测与命令生成用） |
| `--catalog` | `catalog/catalog.json` | 算法目录文件 |
| `--runs` | `webui/runs` | 任务与产物根目录 |
| `--db` | `webui/runs/studio.db` | SQLite 数据库路径 |

## HTTP API

前端使用的就是这些接口，脚本里也能直接调：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/catalog` | 算法目录摘要 + 运行模式 |
| GET | `/api/algorithms/<id>` | 单个算法的完整 schema、字段与起步预设 |
| GET | `/api/doctor?refresh` | 环境探测结果 |
| GET / POST | `/api/env-vars` | 读取/保存四项环境变量 |
| POST | `/api/validate` | 校验参数 |
| POST | `/api/preview` | 只生成命令与文件清单，不落盘 |
| GET / POST | `/api/jobs` | 任务列表 / 创建任务 |
| GET | `/api/jobs/<id>` | 任务详情、命令、产物清单 |
| POST | `/api/jobs/<id>/start`、`/stop`、`/delete` | 启动 / 停止 / 删除 |
| GET | `/api/jobs/<id>/logs?since=<id>` | 增量日志 |
| GET | `/api/jobs/<id>/metrics` | 指标序列 |
| GET | `/api/jobs/<id>/events?since=<id>` | SSE 实时日志 |
| GET | `/api/jobs/<id>/files/<name>` | 下载单个产物（单文件上限 512 KB） |

## 重新生成算法目录与文档

换了 TRL 版本或改了手工元数据后：

```bash
python3 tools/extract_source_catalog.py --trl ../trl   # 解析源码 -> source_catalog.json
python3 tools/build_catalog.py --trl ../trl            # 合并校验 -> catalog.json
python3 tools/gen_params_doc.py                        # 生成 docs/02-algorithm-params.md
```

`build_catalog.py --check` 只校验不写盘，适合放进 CI。当前目录由 TRL `1.16.0.dev0`（commit `8f39f8fc`）生成；`requirements.txt` 也按这个版本范围锁定。

## 常见问题

**顶栏显示“演示模式 · 仅模拟”是什么意思？**
不是开关，是探测结果：核心依赖没装齐、torch 只看到 CPU、或没有可用设备时，任务默认以模拟模式提交。命令与产物照常生成，但日志和曲线来自内置模拟器。把任务模式改成「真实训练」可以看到真实报错；要真训请在装好 CUDA 版 torch + trl 的 GPU 机器上启动。

**端口被占用？**
默认会自动向后找空闲端口，启动日志里打印实际地址；想固定端口就加 `--no-port-shift` 并显式换一个 `--port`。

**启动报“找不到算法目录”？**
确认 `catalog/catalog.json` 存在；没有就按上一节重新生成，或用 `--catalog` 指定路径。

**改了代码但页面没变？**
静态资源带 `Cache-Control: no-cache`，正常刷新即可；仍不对就强制刷新（Ctrl+F5）。改的是 `catalog/` 或 `webui/rlstudio/` 时记得重启服务。

**参数说明去哪了？**
默认收起。向导第 2–4 步底部、以及「环境」页右下角有「显示参数说明」开关，选择会记住。

**下载模型很慢 / 打不开 Hugging Face？**
在「环境」页设置 `HF_ENDPOINT`（如 `https://hf-mirror.com`）与 `HF_HOME`；gated 模型填 `HF_TOKEN`，令牌不会出现在界面回显、日志或产物里。

**任务删不掉？**
运行中的任务不允许删除，先停止再删。

## 相关文档

- `docs/01-trl-rl-algorithms.md` — TRL 强化学习算法全景、选型决策树、数据与奖励函数契约。
- `docs/02-algorithm-params.md` — 每个算法的参数总表（由 `catalog/catalog.json` 生成，勿手改）。
- `docs/03-training-workflow.md` — 训练流程与四个平面的对应关系。
