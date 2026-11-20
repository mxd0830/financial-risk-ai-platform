# D5 模型解释报告：SHAP 特征贡献 + 风险因素 + 原因码

**日期：** 2026-11-20　**模型：** LightGBM（AUC 0.7691）
**方法：** SHAP TreeExplainer（采样 3000 测试客户）

---

## 一、全局 SHAP 重要性（top 20，mean |SHAP|）

| 排名 | 特征 | mean\|SHAP\| | 业务含义 |
|---:|---|---:|---|
| 1 | EXT_SOURCE_MEAN | 0.4223 | 三个外部评分均值（**主导，3.5 倍于第二名**） |
| 2 | RATIO_CREDIT_GOODS | 0.1206 | 贷款/商品价（衍生） |
| 3 | RATIO_ANNUITY_CREDIT | 0.1204 | 年金/贷款额（衍生） |
| 4 | EXT_SOURCE_3 | 0.1103 | 外部评分 3 |
| 5 | CODE_GENDER | 0.0860 | 性别 |
| 6 | ORGANIZATION_TYPE | 0.0787 | 职业组织 |
| 7 | EXT_SOURCE_MAX | 0.0730 | 最高外部评分 |
| 8 | OWN_CAR_AGE | 0.0701 | 车龄 |
| 9 | EXT_SOURCE_MIN | 0.0659 | 最低外部评分 |
| 10 | NAME_EDUCATION_TYPE | 0.0634 | 教育程度 |
| ... | AMT_ANNUITY、OCCUPATION_TYPE、DAYS_EMPLOYED、DAYS_BIRTH、FLAG_DOCUMENT_3 等 | | |

**与 gain 重要度的差异**：SHAP（平均边际贡献）把 `RATIO_CREDIT_GOODS`、`CODE_GENDER`、`OWN_CAR_AGE` 排得更靠前，gain（分裂收益）更看重 `ORGANIZATION_TYPE`、`EXT_SOURCE_3`。两种口径一致指向：**外部评分 + 偿债比率是核心风险因素**。

---

## 二、风险因素归类（业务可读）

| 风险因素组 | 代表特征 | 方向 |
|---|---|---|
| **外部信用评分** | EXT_SOURCE_MEAN/MIN/MAX/1/3 | 评分越低 → 风险越高 |
| **偿债负担** | RATIO_ANNUITY_CREDIT、RATIO_CREDIT_GOODS、AMT_ANNUITY | 负担比越高 → 风险越高 |
| **职业稳定性** | ORGANIZATION_TYPE、OCCUPATION_TYPE | 某些行业 → 风险偏高 |
| **人口/教育** | CODE_GENDER、NAME_EDUCATION_TYPE、DAYS_BIRTH（年龄） | 男性/低教育/年轻 → 风险偏高 |
| **资料完整性** | FLAG_DOCUMENT_3、DOC_COUNT | 证件/资料少 → 风险偏高 |

---

## 三、局部原因码（可解释性核心）

### 高风险客户（预测 PD = 0.693）

| 特征 | 特征值 | SHAP | 方向 |
|---|---:|---:|---|
| EXT_SOURCE_MEAN | 0.074 | **+1.20** | ↑ 推高风险 |
| RATIO_ANNUITY_CREDIT | 0.079 | +0.44 | ↑ |
| EXT_SOURCE_MIN | 0.014 | +0.40 | ↑ |
| EXT_SOURCE_3 | 0.135 | +0.38 | ↑ |
| EXT_SOURCE_MAX | 0.135 | +0.24 | ↑ |
| RATIO_CREDIT_GOODS | 1.158 | +0.14 | ↑ |
| REGION_RATING_CLIENT_W_CITY | 3 | +0.11 | ↑ |

**业务解读**：该客户被判高风险，几乎完全由**外部评分极低**（均值 0.074、最低 0.014，远低于正常）驱动——征信局综合评分差是主因，加上年金/贷款比偏高（还款负担重）。

### 低风险客户（预测 PD = 0.005）

| 特征 | 特征值 | SHAP | 方向 |
|---|---:|---:|---|
| EXT_SOURCE_MEAN | 0.731 | **-0.73** | ↓ 拉低风险 |
| ORGANIZATION_TYPE | Industry: type 9 | -0.40 | ↓ |
| DOC_COUNT | 2 | -0.19 | ↓ |
| OWN_CAR_AGE | 3 | -0.16 | ↓ |
| AMT_REQ_CREDIT_BUREAU_QRT | 1 | -0.15 | ↓ |
| EXT_SOURCE_3 | 0.677 | -0.14 | ↓ |

**业务解读**：外部评分高（均值 0.731）、征信查询少（QRT=1）、职业稳定 → 风险极低。

---

## 四、结论

1. **外部信用评分（EXT_SOURCE_MEAN）是压倒性的第一风险因素**（mean |SHAP| 0.42，是第二名的 3.5 倍），模型高度依赖征信局综合评分。
2. **衍生特征（偿债比率）确实有效**：RATIO_CREDIT_GOODS、RATIO_ANNUITY_CREDIT 稳居 SHAP 第 2、3 名，印证 D3 特征工程的收益。
3. **原因码可落地**：每个客户的预测都能拆解成"哪些特征推高/拉低了风险 + 幅度"，可直接用于业务解释（"为什么拒绝这位客户"）。
4. **局限**：SHAP 只解释"相关性"，不构成因果关系；单表单模型，未含多表时序因素。

**可复现**：`scripts/explain_model.py`。
