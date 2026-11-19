# Day 3 开发日志

**日期：** 2026-11-19

## 今日目标

Logistic Regression + LightGBM 基线模型，训练/验证划分，产出第一版 PD 模型（对应实验 P：LR 与 LightGBM 对比）。

## 完成内容

- [x] 安装 `lightgbm 4.7.0`（连带 `libomp` OpenMP 依赖）
- [x] 安装 PyTorch 2.15.0（为 H²MTT 方案二做准备）
- [x] LR + LightGBM 基线模型（`scripts/train_baseline.py`）
- [x] 缺失阈值对比：5% / 50% / 95%，**95% 最优**
- [x] 偿债能力比率衍生特征（5 个，+0.008 AUC）
- [x] 全量衍生特征（时间转正 / 评分组合 / 计数 / 一致性，共 ~24 个）
- [x] 下载 Home Credit 全部 9 张关联表（bureau、previous_application、bureau_balance、POS_CASH、credit_card、installments 等，~2.5GB）
- [x] 多表聚合尝试（bureau + previous_application，含时间窗口/速度特征）
- [x] 时间字段粒度检查（日度 vs 月度）
- [x] H²MTT 方案二 V1 + 时序消融实验
- [x] 生成 `application_test.csv` 提交文件（`data/processed/submission.csv`）

## 最终基线结果（95% 阈值 + 全量衍生，测试集）

| 指标 | LogisticRegression | LightGBM |
|---|---:|---:|
| AUC | 0.7525 | **0.7691** |
| KS | 0.3734 | **0.4012** |
| PR-AUC | 0.2369 | **0.2565** |
| Brier | 0.0681 | **0.0672** |
| CV-AUC | 0.7499±0.0028 | **0.7641±0.0037** |
| 训练耗时(s) | 0.42 | 3.71 |
| 推理耗时(s) | 0.10 | 0.06 |

## 关键问题与解决方案

### 1. LightGBM 导入报 libomp 缺失

`import lightgbm` 报 `Library not loaded: @rpath/libomp.dylib`。根因：LightGBM 原生库依赖 OpenMP。解决：`brew install libomp`。

### 2. 缺失阈值理解偏差（重要）

我最初写"缺失率 >50% 才删"；用户本意是"缺失率 >95% 才删（覆盖率 ≥5% 保留）"。中间 VS Code 里 Claude Code 误把"覆盖率 5%"当成"缺失率 5%"，导致模型 AUC 掉到 0.725。最终确认 **95% 阈值最优**（0.7604），因为它保留了 EXT_SOURCE_3 等高缺失但高价值的特征。

### 3. 多表简单聚合无增益

bureau/prev 聚合出 26 个特征（后加到 39 个含时间特征），AUC 反而平（0.768）。诊断：26 个聚合特征只贡献 5.8% gain，最高才 1470（vs EXT_SOURCE_MEAN 95881）。根因：**EXT_SOURCE_1/2/3 本身是征信局算好的总分，简单 count/sum/mean 聚合与它高度冗余**。

### 4. H²MTT V1 的 NaN 崩溃

两次跑崩在 eval 的 `roc_auc_score`，报 "Input contains NaN"。逐层排查：静态特征干净（0 NaN/inf）→ 真正根因是 **71.7% 客户完全没有信用卡记录，96 个月序列全 mask，attention 在"全 mask"行上 softmax(全 -inf) = 0/0 = NaN**。修复：transformer 输出后 `nan_to_num`。另：MPS 缺 `_nested_tensor_from_mask_left_aligned` 算子，需 `PYTORCH_ENABLE_MPS_FALLBACK=1` 回退 CPU。

### 5. MPS 提速

CPU 训练 10 epoch 需 ~15 分钟，MPS 只需 ~50 秒（快十几倍）。用户明确要求：**后续深度学习训练优先用 MPS**。

## 关键结论

1. **LightGBM 0.7691 是当前最佳基线**，显著优于 LR 和 H²MTT V1。
2. **特征工程收益排序**：偿债比率（+0.008）> 全量衍生（+0.001 对 LGBM、+0.003 对 LR）> 阈值选择（0.725→0.760）。
3. **多表简单聚合走不通**，信号在月度表的"时序轨迹"里，不在静态汇总里。
4. **H²MTT V1（AUC 0.7136）打不过 LightGBM**：时序分支消融只 +0.003，真正瓶颈是 MLP 结构对表格数据建模能力弱于树集成。
5. 训练/推理性能：LightGBM 推理反而更快（类别原生 + 结构紧凑），解码速度与采样参数无关。

## 明日计划（D4，待定）

- 多表月度时序特征（credit_card / POS_CASH / bureau_balance 的 MONTHS_BALANCE 时序聚合）
- LightGBM 超参调优（Optuna）
- 或 H²MTT V2（跨表交互 + 事件分支）
