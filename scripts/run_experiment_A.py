"""实验 A v2：推理参数对速度、稳定性、结构化输出与资源的影响。

对照技术手册 2.2 实验 A。相比 v1（只测 temperature / num_ctx），本版扩展为
7 组单变量对照，并补上 v1 报告自指出的方法缺陷：

- temperature ：重复 10 次，用「归一化编辑距离 + 完全一致占比」量化稳定性（v1 只重复 3 次、定性）
- num_ctx     ：固定 num_predict=256，消除「输出变长」对总耗时的混淆（v1 的局限 #1），并加测 Ollama 进程 RSS
- top_p / top_k：采样候选范围对多样性的影响（v1 未测）
- max_tokens (num_predict)：输出截断率、延迟、结论是否仍完整（v1 未测）
- seed        ：固定 seed 能否复现输出（v1 未测）
- 结构化 JSON ：字段完整率 / 数值一致率（v1 未测）

每组只改变一个变量，其余固定为默认（temperature=0.1, num_ctx=4096, num_predict=None）。
结果保存到 docs/experiments/exp_A_results.json，报告见 exp_A_inference_params.md。

注意：本机系统代理会劫持 localhost，get_client() 已用 trust_env=False 直连 Ollama。
"""
from __future__ import annotations

import json
import subprocess
from itertools import combinations
from pathlib import Path

from financial_risk_ai_platform.models.llm import get_client

MODEL = "qwen3:30b"
DOCS_EXP = Path(__file__).resolve().parents[1] / "docs" / "experiments"

# 固定任务集（3 个问题，全实验复用）
Q_DEF = "用一句话解释什么是信用风险。"
Q_DTI = "某客户月收入8000元，已有每月5000元负债，评估其偿债能力风险属于高、中还是低水平？"
Q_JSON = (
    "请从下面这段客户信息中提取风控字段，只输出一个 JSON 对象，不要输出任何解释或多余文字：\n"
    "客户张三，月收入 8000 元，现有每月负债 5000 元，近 6 个月逾期 2 次。\n"
    "JSON 字段要求：monthly_income（整数，单位元）、monthly_debt（整数，单位元）、"
    "overdue_6m（整数，次数）、risk_level（字符串，取值 low/medium/high 之一）。"
)
JSON_REQUIRED = {"monthly_income", "monthly_debt", "overdue_6m", "risk_level"}
JSON_EXPECTED = {"monthly_income": 8000, "monthly_debt": 5000, "overdue_6m": 2}
JSON_LEVELS = {"low", "medium", "high"}


def levenshtein(a: str, b: str) -> int:
    """经典编辑距离（插入/删除/替换），用于量化文本稳定性。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = curr
    return prev[lb]


def norm_edit(a: str, b: str) -> float:
    """归一化编辑距离（0=完全相同，1=完全不同）。"""
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def get_model_ps() -> dict:
    """读取 Ollama /api/ps 中已加载模型的显存与上下文（含权重 + KV cache）。

    注意：模型跑在 ollama serve fork 出的 llama-server 子进程里，直接读 serve
    进程 RSS 只能拿到 ~30MB 的空壳，因此这里走 /api/ps 的 size_vram 字段。
    """
    try:
        out = subprocess.run(
            ["curl", "-s", "--noproxy", "*", "http://127.0.0.1:11434/api/ps"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        models = json.loads(out).get("models", [])
        if models:
            m = models[0]
            return {
                "context_length": m.get("context_length"),
                "size_vram_mb": round(m.get("size_vram", 0) / 1024**2, 1),
            }
    except Exception:
        pass
    return {}


def call(
    client,
    question: str,
    group: str,
    *,
    temperature: float = 0.1,
    num_ctx: int = 4096,
    num_predict: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    repeat: int = 0,
) -> dict:
    """单次调用，返回带指标的记录。单个失败不抛异常，记录 error 字段后继续。"""
    options: dict = {"temperature": temperature, "num_ctx": num_ctx}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if seed is not None:
        options["seed"] = seed
    rec = {
        "group": group,
        "temperature": temperature,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "repeat": repeat,
    }
    try:
        resp = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            options=options,
            keep_alive="10m",
        )
        inference_s = (resp["total_duration"] - resp["load_duration"]) / 1e9
        eval_count = resp["eval_count"]
        rec.update(
            {
                "total_duration_s": round(resp["total_duration"] / 1e9, 3),
                "load_duration_s": round(resp["load_duration"] / 1e9, 3),
                "inference_s": round(inference_s, 3),
                "prompt_eval_count": resp["prompt_eval_count"],
                "prompt_eval_duration_s": round(resp["prompt_eval_duration"] / 1e9, 3),
                "eval_count": eval_count,
                "tokens_per_s": round(eval_count / inference_s, 1) if inference_s > 0 else None,
                "output_chars": len(resp["message"]["content"]),
                "output": resp["message"]["content"],
                "truncated": num_predict is not None and eval_count >= num_predict,
                "error": None,
            }
        )
    except Exception as exc:  # 单个失败不中断整轮实验
        rec.update({"error": f"{type(exc).__name__}: {exc}"})
    return rec


def try_parse_json(text: str) -> dict | None:
    """从可能带 markdown 代码块/多余文字的回复里解析 JSON 对象。"""
    cands = [text]
    # 去 ```json ... ``` 围栏
    if text.strip().startswith("```"):
        fence = text.strip().split("\n")
        inner = "\n".join(fence[1:-1]) if len(fence) >= 3 else text
        cands.append(inner)
    # 取第一个 { 到最后一个 }
    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi > lo:
        cands.append(text[lo : hi + 1])
    for c in cands:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def summarize(results: list[dict]) -> None:
    """按组打印汇总，供填写报告。"""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)

    def ok(rs):
        return [r for r in rs if r.get("error") is None]

    # 1) temperature / top_p / top_k / seed：稳定性与多样性
    for g in ("temperature", "top_p", "top_k", "seed"):
        rs = ok(groups.get(g, []))
        if not rs:
            continue
        print(f"\n=== {g} ===")
        if g in ("top_p", "top_k"):
            key = "top_p" if g == "top_p" else "top_k"
            for val in sorted({r[key] for r in rs}):
                sub = [r for r in rs if r[key] == val]
                outs = [r["output"] for r in sub]
                pairs = [norm_edit(a, b) for a, b in combinations(outs, 2)] if len(sub) > 1 else []
                mean_edit = round(sum(pairs) / len(pairs), 3) if pairs else 0.0
                uniq = len(set(outs))
                tps = [r["tokens_per_s"] for r in sub if r["tokens_per_s"]]
                print(
                    f"  {key}={val}: 平均编辑距离{mean_edit} 唯一输出{uniq}/{len(sub)} "
                    f"平均{round(sum(tps)/len(tps),1) if tps else 0}tok/s"
                )
        else:
            for val in sorted({r["temperature"] for r in rs}):
                sub = [r for r in rs if r["temperature"] == val]
                outs = [r["output"] for r in sub]
                pairs = [norm_edit(a, b) for a, b in combinations(outs, 2)] if len(sub) > 1 else []
                mean_edit = round(sum(pairs) / len(pairs), 3) if pairs else 0.0
                uniq = len(set(outs))
                identical = sum(1 for a, b in combinations(outs, 2) if a == b)
                n_pairs = len(pairs)
                tps = [r["tokens_per_s"] for r in sub if r["tokens_per_s"]]
                print(
                    f"  temp={val}: 平均编辑距离{mean_edit} 唯一输出{uniq}/{len(sub)} "
                    f"完全一致{identical}/{n_pairs} 平均{round(sum(tps)/len(tps),1) if tps else 0}tok/s"
                )
        # seed 组额外看固定 seed 是否可复现
        if g == "seed":
            for val in sorted({r["seed"] for r in rs}, key=lambda x: (x is None, x)):
                sub = [r for r in rs if r["seed"] == val]
                outs = [r["output"] for r in sub]
                label = "seed=42" if val is not None else "seed=None"
                identical = all(o == outs[0] for o in outs) if outs else False
                print(f"  {label}: 3 次输出完全一致 = {identical}")

    # 2) num_ctx：延迟 / 吞吐 / 内存
    rs = ok(groups.get("num_ctx", []))
    if rs:
        print("\n=== num_ctx（固定 num_predict=256）===")
        for r in sorted(rs, key=lambda x: x["num_ctx"]):
            print(
                f"  num_ctx={r['num_ctx']}: 推理{r['inference_s']}s 输出{r['eval_count']}tok "
                f"{r['tokens_per_s']}tok/s 截断={r.get('truncated')} "
                f"size_vram={r.get('size_vram_mb')}MB(context={r.get('context_length')})"
            )

    # 3) max_tokens：截断与结论完整性
    rs = ok(groups.get("max_tokens", []))
    if rs:
        print("\n=== max_tokens（num_predict）===")
        for r in sorted(rs, key=lambda x: x["num_predict"]):
            concl = ("高风险" in r["output"]) or ("高水平" in r["output"])
            print(
                f"  num_predict={r['num_predict']}: 推理{r['inference_s']}s 输出{r['eval_count']}tok "
                f"截断={r.get('truncated')} 结论完整={concl}"
            )

    # 4) JSON 结构化：可解析率 / 字段完整率 / 数值一致率
    rs = ok(groups.get("json_format", []))
    if rs:
        print("\n=== json_format（结构化抽取）===")
        for val in sorted({r["temperature"] for r in rs}):
            sub = [r for r in rs if r["temperature"] == val]
            parsed = [try_parse_json(r["output"]) for r in sub]
            n_parse = sum(1 for d in parsed if d is not None)
            n_fields = sum(1 for d in parsed if d is not None and JSON_REQUIRED.issubset(d))
            n_value = sum(
                1
                for d in parsed
                if d is not None
                and all(d.get(k) == v for k, v in JSON_EXPECTED.items())
                and d.get("risk_level") in JSON_LEVELS
            )
            print(
                f"  temp={val}: 可解析{n_parse}/{len(sub)} 字段完整{n_fields}/{len(sub)} "
                f"数值一致{n_value}/{len(sub)}"
            )


def main() -> None:
    client = get_client()
    results: list[dict] = []

    # 预热：加载模型，避免首个数据点混入冷启动
    call(client, Q_DEF, "warmup")

    # R1 温度实验：10 次重复量化稳定性
    for temp in (0.1, 0.5):
        for rep in range(10):
            results.append(call(client, Q_DEF, "temperature", temperature=temp, repeat=rep))

    # R2 上下文实验：固定 num_predict=256 消除输出长度混淆；每个 ctx 先预热再计时，
    # 并在模型仍处于该 ctx 时立即采样 /api/ps 记录显存（否则后续组会把它重载回 4096）
    for ctx in (4096, 8192, 16384):
        call(client, Q_DTI, "num_ctx_warmup", num_ctx=ctx, num_predict=256)
        rec = call(client, Q_DTI, "num_ctx", num_ctx=ctx, num_predict=256)
        rec.update(get_model_ps())
        results.append(rec)

    # R3 top_p 实验：固定 temp=0.5，观察采样候选范围对多样性的影响
    for top_p in (0.5, 0.9, 1.0):
        for rep in range(5):
            results.append(
                call(client, Q_DEF, "top_p", temperature=0.5, top_p=top_p, repeat=rep)
            )

    # R4 top_k 实验：同上，观察 top_k
    for top_k in (10, 40, 100):
        for rep in range(5):
            results.append(
                call(client, Q_DEF, "top_k", temperature=0.5, top_k=top_k, repeat=rep)
            )

    # R5 max_tokens 实验：num_predict=64/256/1024 的截断率与延迟
    for np_ in (64, 256, 1024):
        results.append(call(client, Q_DTI, "max_tokens", num_predict=np_))

    # R6 seed 实验：固定 seed=42 能否复现（temp=0.5 下采样才有意义）
    for seed in (42, None):
        for rep in range(3):
            results.append(call(client, Q_DEF, "seed", temperature=0.5, seed=seed, repeat=rep))

    # R7 结构化 JSON：不同温度下格式稳定性
    for temp in (0.1, 0.5):
        for rep in range(5):
            results.append(call(client, Q_JSON, "json_format", temperature=temp, repeat=rep))

    DOCS_EXP.mkdir(parents=True, exist_ok=True)
    out = DOCS_EXP / "exp_A_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"结果已保存: {out}\n")

    summarize(results)

    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\n⚠️ 有 {len(errors)} 次调用失败（已记录 error 字段）")


if __name__ == "__main__":
    main()
