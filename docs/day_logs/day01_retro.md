# Day 1 复盘

> 日期：2026-11-12

## 一、目标回顾

D1 定义为「基础环境 + 本地 LLM 连通性验证」，7 项验收全部闭环。产出超出最小清单，并提前搭好 D2 下载脚本。

## 二、完成情况

| 验收项 | 结果 |
|---|---|
| Ollama 跑 Qwen3 | ✅ 下载 18GB + 别名 `qwen3:30b`，实测返回正确 |
| Python 3.11 + uv | ✅ uv 0.13.7 / Python 3.11.18 |
| 项目目录 | ✅ `~/Projects/financial-risk-ai-platform`（src 布局 7 个子包） |
| 依赖安装 | ✅ 基础 13 项 + dev（ruff/pytest） |
| Python 调 Qwen3 | ✅ 打通（踩代理坑，见下） |
| UCI 数据 | ✅ 30000×25 → `data/raw/` |
| Git 首次提交 | ✅ `db7ea1f`，原始数据未入库 |

**超额完成**：`llm.py` 封装、`fetch_uci_credit.py` / `fetch_home_credit.py` 两个可复现脚本、开发日志、单测 + lint 全绿。

## 三、踩坑与根因

**坑 1：模型下载限速** — 53 MB/s 冲到 81% 后掉到 2 MB/s，重启两次（断点续传）才爬完。
教训：Ollama 拉大模型尾部会限速，重启能换 CDN 节点，且不丢进度。

**坑 2（核心）：Python 连本地 Ollama 报 503** — 排查链条：

1. 先是 `socksio` ImportError → 发现 `all_proxy` SOCKS 被 httpx 读到
2. 清空环境变量后仍 503 → 发现响应头有 `proxy-connection`，说明还在走代理
3. `httpx.get_environment_proxies()` 抓到 `http://127.0.0.1:1087` → 根因是 macOS **系统级代理**（V2rayU/xray），不是环境变量
4. 代理没设 localhost 绕过，把连 `127.0.0.1:11434` 的请求也塞进代理 → 远端 vless 节点连不到本地 → 空 503

**解决**：`ollama.Client(host=..., trust_env=False)` 直连。比改代理客户端更稳（代理开/关都不影响）。

**坑 3：`.gitignore` 误伤** — `data/` 规则把 `src/financial_risk_ai_platform/data/` 包也忽略了，导致包里的 `__init__.py` 没入库。改成 `/data/` 锚定根目录解决。

## 四、关键决策

- **选型**：确认 Instruct-2507 + Q4_K_M（MoE 激活 3B，M5 Pro 48GB 跑得动）
- **代理处理**：选代码层 `trust_env=False` 而非改系统代理，因为后者会随模式切换/重启失效
- **凭据策略**：Kaggle key 走 `~/.kaggle/kaggle.json` 或环境变量，绝不进 git

## 五、做得好的 / 可改进

**好**：问题排查到根因而非表面；把代理坑固化进 `llm.py` + 日志，避免重复踩；边做边留痕。

**可改进**：
1. `.gitignore` 应一开始就锚定根目录（`/data/`）
2. 代理这个「机器级」问题值得沉淀到全局记忆（已处理）

## 六、遗留（D2）

1. Kaggle 凭据 → 跑 `fetch_home_credit.py` 下 `application_train.csv`
2. `.pytest_cache/`、`.ruff_cache/` 补进 `.gitignore`（已处理）
3. （可选）push 到 GitHub 做异地备份
