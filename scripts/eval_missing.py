"""字段缺失变体评测：模型是否会编造档案里没有的字段。

指标：JSON 可解析率、在场字段完整率、在场字段一致率、省略字段编造率。
用法：
  .venv/bin/python scripts/eval_missing.py --limit 500   # 验证
  .venv/bin/python scripts/eval_missing.py               # 全量
结果保存到 data/processed/eval_missing_results.json。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_risk_ai_platform.models.llm import get_client

MODEL = "qwen3:30b"
EVAL_SET = "data/processed/eval_extraction_missing.jsonl"
OUT = "data/processed/eval_missing_results.json"

STR_FIELDS = {"gender", "education", "contract_type"}
INT_FIELDS = {"age", "children"}


def try_parse_json(text: str) -> dict | None:
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


def value_ok(p, v) -> bool:
    if v is None:  # 省略字段：正确 = 不编造（缺失或 null）
        return (p is None)
    if isinstance(v, str):
        return p == v
    try:
        if isinstance(v, int) and not isinstance(v, bool):
            return float(p) == float(v)
        tol = max(0.01, abs(float(v)) * 0.001)
        return abs(float(p) - float(v)) <= tol
    except (TypeError, ValueError):
        return False


def score(parsed: dict, gt: dict) -> dict:
    """返回单样本评分：在场字段逐字段正确性、省略字段编造数。"""
    present = {k: v for k, v in gt.items() if v is not None}
    omitted = [k for k, v in gt.items() if v is None]

    complete = present.keys() <= parsed.keys()
    per_field = {}
    for k, v in present.items():
        per_field[k] = (k in parsed) and value_ok(parsed.get(k), v)

    fabricated = [k for k in omitted if k in parsed and parsed[k] is not None]
    return {
        "complete": complete,
        "per_field": per_field,
        "present_ok": complete and all(per_field.values()),
        "fabricated": fabricated,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    client = get_client()
    lines = Path(EVAL_SET).read_text(encoding="utf-8").strip().splitlines()
    if args.limit:
        lines = lines[: args.limit]

    n_parse = 0
    n_complete = 0
    n_present_ok = 0
    n_all_ok = 0
    field_ok: dict[str, int] = {}
    field_total: dict[str, int] = {}
    n_omitted_total = 0
    n_fabricated_total = 0
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
            failures.append({"id": rec["id"], "output": output, "ground_truth": rec["ground_truth"],
                             "omitted": rec["omitted"], "parse_failed": True})
        else:
            n_parse += 1
            sc = score(parsed, rec["ground_truth"])
            if sc["complete"]:
                n_complete += 1
            for k, ok in sc["per_field"].items():
                field_ok[k] = field_ok.get(k, 0) + int(ok)
                field_total[k] = field_total.get(k, 0) + 1
            if sc["present_ok"]:
                n_present_ok += 1
            n_omitted_total += len(rec["omitted"])
            n_fabricated_total += len(sc["fabricated"])
            if sc["present_ok"] and not sc["fabricated"]:
                n_all_ok += 1
            if not (sc["present_ok"] and not sc["fabricated"]):
                failures.append({"id": rec["id"], "output": output, "parsed": parsed,
                                 "ground_truth": rec["ground_truth"], "omitted": rec["omitted"],
                                 "fabricated": sc["fabricated"]})

        if i % 100 == 0:
            print(f"  进度 {i}/{len(lines)}（可解析 {n_parse}，全对 {n_all_ok}，编造 {n_fabricated_total}）", flush=True)

    total = len(lines)
    summary = {
        "model": MODEL,
        "temperature": args.temperature,
        "n": total,
        "parse_rate": round(n_parse / total, 4),
        "present_completeness_rate": round(n_complete / total, 4),
        "present_ok_rate": round(n_present_ok / total, 4),
        "fabrication_rate": round(n_fabricated_total / n_omitted_total, 4) if n_omitted_total else None,
        "all_ok_rate": round(n_all_ok / total, 4),
        "per_field_accuracy": {k: round(v / field_total[k], 4) for k, v in field_ok.items()},
        "omitted_events": n_omitted_total,
        "fabricated_events": n_fabricated_total,
    }
    Path(OUT).write_text(json.dumps({"summary": summary, "failures": failures}, ensure_ascii=False, indent=2))

    print("\n" + "=" * 56)
    print(f"字段缺失变体（n={total}, temp={args.temperature}）")
    print("=" * 56)
    print(f"JSON 可解析率       : {summary['parse_rate']:.2%}")
    print(f"在场字段完整率      : {summary['present_completeness_rate']:.2%}")
    print(f"在场字段全对率      : {summary['present_ok_rate']:.2%}")
    print(f"省略字段编造率      : {summary['fabrication_rate']:.2%}（{n_fabricated_total}/{n_omitted_total}）")
    print(f"全对且不编造率      : {summary['all_ok_rate']:.2%}")
    print("\n在场字段一致率：")
    for k, v in summary["per_field_accuracy"].items():
        print(f"  {k:<20} {v:.2%}")
    print(f"\n失败样本数：{len(failures)}（已存 {OUT}）")


if __name__ == "__main__":
    main()
