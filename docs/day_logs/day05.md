# Day 5 开发日志

**日期：** 2026-11-20

## 今日目标

SHAP 特征贡献解释、风险因素解释、原因码整理（模型解释模块，对应 30 天计划 D5）。

## 完成内容

- [x] 安装 `shap 0.51.0`
- [x] `scripts/explain_model.py`（TreeExplainer + 全局重要性 + 局部原因码）
- [x] `reports/model_explanation_report.md` 解释报告

## 核心结论

**全局 SHAP 重要性 top5**：EXT_SOURCE_MEAN（0.42，主导）> RATIO_CREDIT_GOODS（0.12）> RATIO_ANNUITY_CREDIT（0.12）> EXT_SOURCE_3（0.11）> CODE_GENDER（0.09）。

**关键发现**：
1. 外部评分 EXT_SOURCE_MEAN 是压倒性第一风险因素（3.5 倍于第二名）。
2. 偿债比率衍生特征稳居第 2、3 名，印证 D3 特征工程收益。
3. 原因码可落地：高风险客户（PD 0.693）主要由"外部评分极低（均值 0.074）+ 年金/贷款比高"驱动；低风险客户（PD 0.005）由"评分高 + 征信查询少"驱动。

## 关键问题与解决方案

### shap 0.51 API 变化

`shap.TreeExplainer(model).shap_values(X)` 对二分类 LightGBM **直接返回单数组 (n, n_features)**（违约类别 SHAP），不再是老版的 `[shap_0, shap_1]` 列表。老写法 `shap_values[1]` 会取错维度报错。修复：直接用返回值。

## 明日计划（D6）

模拟准入规则、额度规则、FastAPI 模型接口（风控模型服务 API，对应 30 天计划 D6）。
