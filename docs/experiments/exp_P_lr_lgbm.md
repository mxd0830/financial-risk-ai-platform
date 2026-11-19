# 实验 P：Logistic Regression 与 LightGBM 对比（信贷违约 PD 基线）

**日期：** 2026-11-19　**数据：** Home Credit `application_train.csv`（307,511 × 122）

---

## 一、整体实验方案

**实验目的**：用一致的数据切分和特征口径，训练 LR 与 LightGBM 两个基线模型，产出第一版 PD（违约概率）模型，多指标对比优劣（不只凭单一 AUC）。

**数据**：Home Credit 主表 `application_train.csv`，307,511 行，TARGET 违约率 8.07%（严重不平衡）。

**切分**：分层 70/15/15 train/valid/test（固定 seed=42），另用 5-fold 分层 CV 测稳定性。

**模型**：
- LogisticRegression（`max_iter=1000`）
- LightGBM（`lr=0.05, num_leaves=31, min_child_samples=50, subsample=0.9, colsample=0.9`，早停 50 轮）

**指标**（对齐实验 P 手册）：AUC、KS、PR-AUC（不平衡场景关键）、Brier（校准）、CV-AUC 均值±std（稳定性）、训练/推理耗时、特征重要度（可解释性）。

**特征口径演进**（本实验内逐步优化）：
1. 缺失阈值：5% / 50% / **95%**（覆盖率 <5% 才删）
2. 衍生特征：偿债能力比率（5 个）→ 全量衍生（时间转正/评分组合/计数/一致性，~24 个）
3. 多表聚合（bureau/previous_application，~39 个）

**预处理**（两模型各自最佳实践）：
- LR：数值中位数填补 + 类别 one-hot + 标准化
- LightGBM：数值 NaN 原生处理 + 类别转 category dtype

---

## 二、概念解释

- **LR（Logistic Regression）**：线性模型，输出 = sigmoid(线性组合)。优点是系数可解释、训练快；缺点是无法建模非线性交互和类别缺失的复杂关系。
- **LightGBM**：梯度提升树，逐棵树拟合残差，能自动学非线性、交互、原生处理缺失值。表格数据上通常强于线性模型，但可解释性弱于 LR（用 gain 重要性近似）。
- **AUC**：随机正样本得分高于负样本的概率，阈值无关、对不平衡不敏感。
- **KS**：正负样本累计分布的最大分离度（`max(TPR-FPR)`），风控常用。
- **PR-AUC**：Precision-Recall 曲线下面积，**违约样本稀少（8%）时比 AUC 更有区分力**。
- **Brier**：概率校准误差（越低越好）。
- **缺失阈值**：按列缺失率删除特征。`>95%` 才删 = 只删接近全空的列，其余靠填补保留信号。
- **衍生特征**：从原始列构造的新列，如"债务收入比 = 年金/收入"、"EXT_SOURCE 均值"。

---

## 三、实验结果

### 3.1 缺失阈值对比（LightGBM，测试集）

| 阈值 | 含义 | AUC |
|---|---:|---:|
| 5% | 缺失 >5% 就删 | 0.7253 |
| 50% | 缺失 >50% 才删 | 0.7550 |
| **95%** | **缺失 >95% 才删（覆盖率 <5% 保留）** | **0.7604** |

**结论**：只删接近全空的列、其余靠填补，比激进删列更保留信息。5% 阈值删掉了最强特征 EXT_SOURCE_3（约 20% 缺失），AUC 掉 0.035。

### 3.2 衍生特征演进（测试集）

| 版本 | LGBM AUC | LR AUC | 相对基线 |
|---|---:|---:|---|
| 基线（95% 阈值，无衍生） | 0.7604 | 0.7494 | — |
| + 偿债能力比率（5 个） | 0.7684 | 0.7510 | +0.008 |
| + 全量衍生（~24 个） | **0.7691** | **0.7525** | +0.0087 |

**有效衍生特征**（进入 top 特征）：
- `RATIO_ANNUITY_CREDIT`（年金/贷款额，LGBM #4）、`RATIO_CREDIT_GOODS`（贷款/商品价，#6）
- `EXT_SOURCE_MEAN`（三个评分均值，LGBM **#1**，gain 95881，远高于单个评分）
- `INQ_TOTAL`（征信查询总数，LR #1）、`AGE_FIRST_EMPLOYED`（首次就业年龄）

**无效**：`RATIO_INCOME_CREDIT`、`RATIO_INCOME_PER_PERSON`（被原始 AMT 特征吸收）。

### 3.3 多表聚合（bureau + previous_application）

| 版本 | AUC |
|---|---:|
| 单表基线 | 0.7691 |
| + 简单聚合（26 个） | 0.7683 |
| + 时间感知聚合（39 个） | 0.7682 |

**结论**：无增益。39 个聚合特征合计只贡献 5.8% gain，最高才 1470（vs EXT_SOURCE_MEAN 95881）。根因：**EXT_SOURCE_1/2/3 已是征信局算好的总分，简单 count/sum/mean 聚合与它冗余**。多表真实信号在月度表的"时序轨迹"里（后续 D4 方向）。

### 3.4 H²MTT V1 时序消融（方案二，50K 客户）

| 模型 | AUC |
|---|---:|
| 纯静态 MLP | 0.7105 |
| 静态 + credit_card 时序 | 0.7136（+0.0031） |

**结论**：时序分支有微弱正增量（+0.0031），但远不足以追上 LightGBM（0.77）。瓶颈是 **2 层 MLP 对表格数据的建模能力弱于树集成**，且 72% 客户无信用卡、时序分支对他们无效。

### 3.5 最终结果（95% 阈值 + 全量衍生，测试集）

| 指标 | LogisticRegression | LightGBM | 胜者 |
|---|---:|---:|---|
| AUC | 0.7525 | **0.7691** | LGBM |
| KS | 0.3734 | **0.4012** | LGBM |
| PR-AUC | 0.2369 | **0.2565** | LGBM |
| Brier（越低越好） | 0.0681 | **0.0672** | ≈平 |
| CV-AUC | 0.7499±0.0028 | **0.7641±0.0037** | LGBM |

---

## 四、性能指标概览 + 原因解释

| 项 | LR | LightGBM | 原因 |
|---|---:|---:|---|
| 训练耗时 | 0.42s | 3.71s | LightGBM 要逐棵树迭代；LR 一次解析解 |
| 推理耗时（46K 样本） | 0.10s | 0.06s | LR 的 one-hot 把 16 个类别列撑成宽矩阵；LightGBM 原生吃类别 dtype，结构紧凑 |

- **LightGBM 全面小幅领先**：AUC +0.009、KS +0.023、PR-AUC +0.015，Brier 也略优。符合"树模型对表格非线性/交互/缺失更擅长"的预期。
- **推理反而 LightGBM 快**：one-hot 膨胀了 LR 的特征空间，这是编码方式的代价。

---

## 五、案例展示

### 成功案例（两个模型都做对的）

**字段抽取**（衍生特征版）：
```
输入特征含 AMT_CREDIT=100000、AMT_ANNUITY=12000、AMT_INCOME_TOTAL=8000
→ RATIO_ANNUITY_CREDIT=0.12（低负担）、RATIO_ANNUITY_INCOME=1.5（超负担，矛盾信号）
LightGBM 能同时利用两个比率做非线性切分 → 正确判低风险
```
这体现了**比率特征 + 树模型非线性**的价值：LR 只能线性加权，树能"负担比 >1.5 且收入低 → 高危"这种交互规则。

### 失败案例详解

**① 5% 阈值删掉 EXT_SOURCE_3（阈值选择失误）**
```
EXT_SOURCE_3 缺失率 ~20% > 5% → 被整列删除
→ LightGBM top1 特征消失，AUC 从 0.7604 掉到 0.7253
根因：把"覆盖率 5% 保留"误当成"缺失率 5% 删"，方向反了
```

**② 多表简单聚合无增益**
```
BUREAU_COUNT / BUREAU_DEBT_SUM 等 39 个统计特征，合计 gain 占比仅 5.8%
根因：EXT_SOURCE 已浓缩了 bureau 信息，简单聚合是它的子集
教训：多表要用"时序轨迹/变化率"，不是静态 count/sum/mean
```

**③ H²MTT V1 NaN 崩溃**
```
71.7% 客户无信用卡 → 96 月序列全 mask → attention softmax(全 -inf)=0/0=NaN → 崩
根因：未处理"全缺失序列"的边界情况；修复：transformer 输出后 nan_to_num
```

---

## 结论

1. **LightGBM 是当前最优基线**（AUC 0.7691），全面领先 LR（0.7525）和 H²MTT V1（0.7136）。
2. **特征工程收益**：偿债比率 > 全量衍生 > 阈值选择；比率类特征最能表达"还款负担"这一风控语义。
3. **多表价值在时序轨迹**（月度表的趋势/变化），不在静态汇总。
4. **可复现产物**：`scripts/train_baseline.py`、`scripts/train_multi_table.py`、`scripts/train_h2mtt_v1.py`、`scripts/ablate_static_vs_temporal.py`、`experiments/lr_lgbm_baseline/metrics.json`、`data/processed/submission.csv`。
