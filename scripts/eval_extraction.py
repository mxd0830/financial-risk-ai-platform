"""结构化抽取忠实度评测：在 Home Credit 构造的评测集上跑本地模型，比对 ground truth。

指标：JSON 可解析率、字段完整率、数值/字符串一致率（整体 + 分字段）。
用法：
  .venv/bin/python scripts/eval_extraction.py --limit 500   # 500 条验证
  .venv/bin/python scripts/eval_extraction.py               # 全量
结果保存到 data/processed/eval_extraction_results.json（含失败样本供分析）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_risk_ai_platform.models.llm import get_client

MODEL = "qwen3:30b"
EVAL_SET = "data/processed/eval_extraction.jsonl"
OUT = "data/processed/eval_extraction_results.json"

STR_FIELDS = {"gender", "education", "contract_type"}
INT_FIELDS = {"age", "children"}
REQUIRED = set(
    ["gender", "age", "education", "contract_type", "income_total",
     "loan_amount", "annuity", "goods_price", "children", "employment_years"]
)


def try_parse_json(text: str) -> dict | None:
    """从可能带 markdown 代码块/多余文字的回复里解析 JSON 对象。"""
    cands = [text]
    if text.strip().startswith("```"):
        fence = text.strip().split("\n")
        if len(fence) >= 3:
            cands.append("\n".join(fence[1:-1]))
    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi > lo:
        cands.append(text[lo : hi + 1])
    for c in cands:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def field_match(parsed: dict, gt: dict) -> tuple[bool, dict[str, bool]]:
    """逐字段比对。返回 (是否全对, 分字段结果)。"""
    per = {}
    all_ok = True
    for k, v in gt.items():
        if k not in parsed:
            per[k] = False
            all_ok = False
            continue
        p = parsed[k]
        if k in STR_FIELDS:
            ok = p == v
        elif k in INT_FIELDS:
            try:
                ok = float(p) == float(v)
            except (TypeError, ValueError):
                ok = False
        else:  # 数值浮点，容差 0.1% 或 0.01
            try:
                tol = max(0.01, abs(float(v)) * 0.001)
                ok = abs(float(p) - float(v)) <= tol
            except (TypeError, ValueError):
                ok = False
        per[k] = ok
        all_ok = all_ok and ok
    return all_ok, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（验证用）")
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    client = get_client()
    lines = Path(EVAL_SET).read_text(encoding="utf-8").strip().splitlines()
    if args.limit:
        lines = lines[: args.limit]

    n_parse = 0
    n_complete = 0
    n_all_correct = 0
    field_ok: dict[str, int] = {k: 0 for k in REQUIRED}
    failures: list[dict] = []

    for i, line in enumerate(lines, 1):
        rec = json.loads(line)
        resp = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": rec["prompt"]}],
            options={"temperature": args.temperature, "num_ctx": 4096},
            keep_alive="10m",
        )
        output = resp["message"]["content"]
        parsed = try_parse_json(output)

        if parsed is None:
            failures.append({"id": rec["id"], "output": output, "ground_truth": rec["ground_truth"], "parse_failed": True})
        else:
            n_parse += 1
            complete = REQUIRED.issubset(parsed)
            if complete:
                n_complete += 1
            all_ok, per = field_match(parsed, rec["ground_truth"])
            for k, ok in per.items():
                field_ok[k] += int(ok)
            if all_ok and complete:
                n_all_correct += 1
            if not (all_ok and complete):
                failures.append({"id": rec["id"], "output": output, "parsed": parsed,
                                 "ground_truth": rec["ground_truth"]})

        if i % 100 == 0:
            print(f"  进度 {i}/{len(lines)}（可解析 {n_parse}，全对 {n_all_correct}）", flush=True)

    total = len(lines)
    summary = {
        "model": MODEL,
        "temperature": args.temperature,
        "n": total,
        "parse_rate": round(n_parse / total, 4),
        "completeness_rate": round(n_complete / total, 4),
        "all_correct_rate": round(n_all_correct / total, 4),
        "per_field_accuracy": {k: round(v / total, 4) for k, v in field_ok.items()},
    }
    out = {"summary": summary, "failures": failures}
    Path(OUT).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n" + "=" * 56)
    print(f"结构化抽取忠实度（n={total}, temp={args.temperature}）")
    print("=" * 56)
    print(f"JSON 可解析率      : {summary['parse_rate']:.2%}")
    print(f"字段完整率         : {summary['completeness_rate']:.2%}")
    print(f"全字段全对率       : {summary['all_correct_rate']:.2%}")
    print("\n分字段一致率：")
    for k, v in summary["per_field_accuracy"].items():
        print(f"  {k:<20} {v:.2%}")
    print(f"\n失败样本数：{len(failures)}（已存 {OUT}）")


if __name__ == "__main__":
    main()
