# Day 6 开发日志

**日期：** 2026-11-20

## 今日目标

模拟准入规则、额度规则、FastAPI 模型接口（风控模型服务 API，对应 30 天计划 D6）。

## 完成内容

- [x] 规则引擎 `src/financial_risk_ai_platform/rules/engine.py`（准入 + 额度）
- [x] 特征工程迁至 `src/.../models/features.py`（生产唯一实现）
- [x] 模型服务 `src/.../models/credit_model.py`（加载模型 + 预处理 + 预测 PD）
- [x] `scripts/save_model.py`（训练 + 保存模型到 data/models/，gitignore）
- [x] FastAPI 接口 `src/.../api/main.py`（`POST /predict` + `GET /health`）
- [x] 端到端测试通过（真实违约客户 → reject，真实未违约 → approve）

## 规则引擎设计（模拟策略）

**准入规则**（硬规则优先于模型）：
1. 年龄 <18 或缺失 → 拒绝
2. 年收入 <5 万 → 拒绝
3. 模型 PD ≥ 20% → 拒绝（高风险）
4. PD ∈ [5%, 20%) → 人工复核
5. PD < 5% → 通过

**额度规则**（min(申请额, 年收入 × PD 档位系数)）：
`PD<2% ×0.5`、`PD<5% ×0.3`、`PD<10% ×0.15`、`PD<20% ×0.05`，向下取整到千位。

**原则**：确定性计算（规则/模型）与生成式 AI 分工——准入和额度由 Python 计算，LLM 只负责解释和报告。

## 端到端测试结果

| 用例 | PD | 决策 | 额度 |
|---|---:|---|---:|
| 真实违约客户 | 39.3% | reject | — |
| 真实未违约客户 | 3.3% | approve | 8.1 万 |
| 年龄缺失+低收入 | — | reject（年龄） | — |
| 中风险 | 8.0% | manual_review | — |
| 高风险 | 25.0% | reject | — |

## 关键问题与解决方案

### 1. 数值列含 None 被推断成 object，LightGBM 拒绝

API 输入 dict 里 NaN→None 后，`OWN_CAR_AGE` 等数值列被 pandas 推断成 object dtype，LightGBM 报 "dtypes must be int, float or bool"。修复：预处理时把非类别列统一 `astype(float)`（None→NaN）。

### 2. 特征工程出现两份拷贝（待清理）

`add_derived_features` 现同时存在于 `scripts/train_baseline.py`（训练用）和 `src/.../models/features.py`（生产用），逻辑一致但需合并：后续把 train_baseline.py 改为从 src 导入，避免漂移。

## 明日计划（D7）

按 30 天计划进入第二阶段。待确认：监管知识库数据收集（RAG 线）或继续完善信贷决策助手。
