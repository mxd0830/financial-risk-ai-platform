"""实验 A：推理参数（temperature / num_ctx）对速度、输出长度与稳定性的影响。

对应技术手册 2.2 实验 A。结果保存到 docs/experiments/exp_A_results.json，
报告见 docs/experiments/exp_A_inference_params.md。

设计：
- 温度实验：固定 num_ctx=4096，同一问题在 temperature=0.1 / 0.5 各重复 3 次，
  观察输出稳定性和生成耗时。
- 上下文实验：固定 temperature=0.1，同一问题在 num_ctx=4096/8192/16384 各测一次，
  观察推理耗时与输出 token 数。
- 每个 num_ctx 首次运行会触发 KV cache 重建（load_duration），故每个 ctx 先预热再计时。
"""
from __future__ import annotations

import json
from pathlib import Path

from financial_risk_ai_platform.models.llm import get_client

MODEL = "qwen3:30b"
DOCS_EXP = Path(__file__).resolve().parents[1] / "docs" / "experiments"

Q_STABILITY = "用一句话解释什么是信用风险。"
Q_RISK = "某客户月收入8000元，已有每月5000元负债，评估其偿债能力风险属于高、中还是低水平？"


def call(client, question: str, temperature: float, num_ctx: int) -> dict:
    resp = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        options={"temperature": temperature, "num_ctx": num_ctx},
        keep_alive="10m",
    )
    return {
        "temperature": temperature,
        "num_ctx": num_ctx,
        "total_duration_s": round(resp["total_duration"] / 1e9, 3),
        "load_duration_s": round(resp["load_duration"] / 1e9, 3),
        "inference_s": round((resp["total_duration"] - resp["load_duration"]) / 1e9, 3),
        "prompt_eval_count": resp["prompt_eval_count"],
        "eval_count": resp["eval_count"],
        "output_chars": len(resp["message"]["content"]),
        "output": resp["message"]["content"],
    }


def main() -> None:
    client = get_client()
    results: list[dict] = []

    # 预热：加载模型，避免首个数据点混入冷启动
    call(client, Q_STABILITY, 0.1, 4096)

    # 温度实验
    for temp in (0.1, 0.5):
        for rep in range(3):
            r = call(client, Q_STABILITY, temp, 4096)
            r["repeat"] = rep
            results.append(r)

    # 上下文实验：每个 num_ctx 先预热（触发 KV cache 重建），再计时
    for ctx in (4096, 8192, 16384):
        call(client, Q_RISK, 0.1, ctx)  # 预热
        r = call(client, Q_RISK, 0.1, ctx)
        r["repeat"] = "ctx"
        results.append(r)

    DOCS_EXP.mkdir(parents=True, exist_ok=True)
    out = DOCS_EXP / "exp_A_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"结果已保存: {out}\n")

    print("=== 温度实验（Q：一句话解释信用风险；num_ctx=4096）===")
    for r in results:
        if r.get("repeat") != "ctx":
            print(
                f"temp={r['temperature']} rep={r['repeat']}: "
                f"推理{r['inference_s']}s 输出{r['eval_count']}tok/{r['output_chars']}字 "
                f"→ {r['output']}"
            )

    print("\n=== 上下文实验（Q：偿债能力风险判断；temp=0.1）===")
    for r in results:
        if r.get("repeat") == "ctx":
            print(
                f"num_ctx={r['num_ctx']}: "
                f"推理{r['inference_s']}s 输出{r['eval_count']}tok "
                f"→ {r['output']}"
            )


if __name__ == "__main__":
    main()
