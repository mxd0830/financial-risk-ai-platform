# Day 2 开发日志

**日期：** 2026-11-13

## 今日目标

下载 Home Credit 主表，完成 EDA（缺失率、标签分布、变量分析）。

## 完成内容

- [x] 下载 `application_train.csv`（307,511 行 × 122 列，158MB）
- [x] EDA：数据规模、缺失率、标签分布、关键变量与违约率关系
- [x] 补 D1 实验 A（推理参数对照）
- [x] 建 `docs/experiments/`、`docs/adr/` 目录
- [x] 修复 `import ollama` 崩溃（装 socksio）

## 关键问题与解决方案

### 1. Kaggle API 认证方式变了（花了不少时间）

**现象：** 用旧的 `username + key` basic auth 一直 401。

**根因：** Kaggle 已改用 API token：环境变量 `KAGGLE_API_TOKEN=KGAT_xxx`，
请求头 `Authorization: Bearer <token>`（`KGAT_` 是 token 的一部分，不是标签）。

**连锁坑：**
- 竞赛数据需先在网页接受规则，否则 403「You must accept this competition's rules」
- Kaggle 对 `python-requests` 默认 User-Agent 返回 404，必须带浏览器 UA
- 返回的是 `application_train.csv.zip`，需解压

token 存 `.env`（gitignore，不入库）。

### 2. `import ollama` 本身会崩

ollama 库 import 时创建模块级默认 client（trust_env=True），读系统 `all_proxy`
SOCKS 代理，缺 socksio 报 ImportError。装 `socksio` 解决（实际调用仍 trust_env=False）。

## EDA 核心结论

| 维度 | 发现 |
|---|---|
| 规模 | 307,511 × 122，内存 341MB |
| 标签不平衡 | 违约 8.07%，未违约 91.93% |
| 缺失 | 67/122 列有缺失，41 列缺失 >50%（房屋/建筑信息） |
| 最强单变量 | EXT_SOURCE_3：最低分档违约 16.4% → 最高分档 3.4% |
| 年龄 | <25 岁 12.3% → >55 岁 5.2%（单调） |
| 性别 | 男 10.1% vs 女 7.0% |
| 收入 | 仅最高档 Q5 明显下降（6.5%） |
| 异常值 | DAYS_EMPLOYED=365243（失业占位符）；AMT_INCOME_TOTAL 最高 1.17 亿 |

可复现脚本：`scripts/eda_train.py`

## 明日计划（D3）

Logistic Regression + LightGBM 基线模型，训练/验证划分，产出第一版 PD 模型。
（对应实验 P：LR 与 LightGBM 对比）
