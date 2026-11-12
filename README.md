# 金融风控 AI 平台

面向金融行业的端到端 AI 解决方案作品集，围绕一个统一平台实现三个业务应用：

1. **智能信贷决策助手** — 传统风控模型（PD / SHAP / 规则引擎）+ Agent 生成风险分析报告
2. **金融监管知识助手** — 官方监管文件 RAG（BGE-M3 / Chroma / BM25 / Rerank）与证据溯源
3. **AI 模型风险管理 Copilot** — AUC / KS / PSI 监控与漂移诊断

## 技术栈

- Python 3.11 + uv
- 本地 LLM：Ollama + Qwen3-30B-A3B Instruct-2507（Q4_K_M）
- 风控建模：scikit-learn、LightGBM、SHAP
- RAG：BGE-M3、Chroma、BM25、Rerank
- Agent：LangGraph（Supervisor + 专业 Agent）
- 后端/前端：FastAPI + Streamlit

## 环境准备

```bash
uv sync
```

本地模型（Ollama）：

```bash
ollama run qwen3:30b
```

> 注意：本机开启了 macOS 系统级代理，代码中调用本地 Ollama 时已通过
> `trust_env=False` 直连，避免 localhost 请求被代理劫持（详见
> `src/financial_risk_ai_platform/models/llm.py`）。

## 目录结构

见 `docs/` 与 30 天执行计划。
