# Day 1 开发日志

**日期：** 2026-11-12

## 今日目标

基础环境搭建 + 本地 LLM 连通性验证（对应 30 天计划 D1）。

## 完成内容

- [x] 环境检查：Apple M5 Pro / macOS 27.2.1 / arm64 / 48GB 内存
- [x] 安装 uv 0.13.7（`~/.local/bin`）
- [x] 安装 Python 3.11.18（`uv python install 3.11`）
- [x] 下载本地模型 `qwen3:30b-a3b-instruct-2507-q4_K_M`（18 GB，Q4_K_M）
- [x] 创建别名 `qwen3:30b`
- [x] 初始化项目 `financial-risk-ai-platform`（uv init --package）
- [x] 安装基础依赖（pandas/numpy/scipy/scikit-learn/matplotlib/jupyterlab/ollama/ucimlrepo/pydantic/fastapi/uvicorn/streamlit）+ dev 依赖（ruff/pytest）
- [x] 创建目录结构（data/notebooks/src 各子包/scripts/tests/docs/reports）
- [x] Python 调用本地 Qwen3 成功
- [x] 下载 UCI Default of Credit Card Clients 数据集（30000 × 25）到 `data/raw/`
- [x] Git 首次提交

## 关键问题与解决方案

### 1. Python 调用本地 Ollama 返回 503（空响应）

**现象：** `ollama.chat()` / httpx POST `127.0.0.1:11434/api/chat` 返回 503 空响应，
但 `curl` 直连正常返回 200。

**根因：** 本机开启了 macOS **系统级代理**（127.0.0.1:1087 http / 1088 socks）。
httpx（以及 urllib）会读取 macOS 系统代理配置，把 localhost 请求也塞进代理，
代理对 127.0.0.1 回包返回 503。`curl` 不读系统代理所以正常。

**解决：** 封装 `src/financial_risk_ai_platform/models/llm.py`，用
`ollama.Client(host=..., trust_env=False)` 直连本地 Ollama。

**遗留建议：** 可在代理客户端把 `localhost` / `127.0.0.1` 加入绕过列表，从系统层面根治。

### 2. 模型加载导致瞬时 503

冷启动时 Ollama 需要加载 18GB 模型，早期调用可能撞上加载中的 503。
正式代码需对 503 做重试或预加载。

## 明日计划（D2）

下载 Home Credit 主表 `application_train.csv`，完成 EDA、缺失率、标签分布、变量分析。
