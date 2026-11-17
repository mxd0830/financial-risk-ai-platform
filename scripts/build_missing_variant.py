"""构造「字段缺失」变体评测集：从基础评测集每条随机删 2 个字段。

用途：测模型是否会**编造**档案里没有的字段（理想行为是省略或输出 null）。
生成物写入 data/processed/eval_extraction_missing.jsonl（gitignore、可重生成）。
每行 JSON：{"id", "target", "prompt", "ground_truth", "omitted"}；
其中被省略字段在 ground_truth 里值为 null（表示"不应编造"）。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SRC = "data/processed/eval_extraction.jsonl"
OUT = "data/processed/eval_extraction_missing.jsonl"
SEED = 42
OMIT_N = 2

FIELDS = [
    "gender", "age", "education", "contract_type", "income_total",
    "loan_amount", "annuity", "goods_price", "children", "employment_years",
]
LABELS = {
    "gender": "性别", "age": "年龄", "education": "学历", "contract_type": "贷款类型",
    "income_total": "收入总额", "loan_amount": "贷款金额", "annuity": "月供",
    "goods_price": "商品价格", "children": "子女数", "employment_years": "已工作年限",
}
SUFFIX = {"employment_years": " 年"}

SCHEMA = (
    "JSON 字段要求：gender（字符串）、age（整数）、education（字符串）、contract_type（字符串）、"
    "income_total（数值）、loan_amount（数值）、annuity（数值）、goods_price（数值）、"
    "children（整数）、employment_years（数值）。"
)


def rebuild_prompt(gt: dict, omitted: list[str]) -> str:
    lines = [
        "以下是某贷款申请人的档案，请从中提取字段并只输出一个 JSON 对象（不要输出任何解释或多余文字）："
    ]
    for field in FIELDS:
        if field in omitted:
            continue
        lines.append(f"{LABELS[field]}：{gt[field]}{SUFFIX.get(field, '')}")
    lines.append(SCHEMA)
    return "\n".join(lines)


def main() -> None:
    rng = random.Random(SEED)
    src_lines = Path(SRC).read_text(encoding="utf-8").strip().splitlines()

    out_path = Path(OUT)
    with out_path.open("w", encoding="utf-8") as f:
        for line in src_lines:
            rec = json.loads(line)
            omitted = rng.sample(FIELDS, OMIT_N)
            gt = dict(rec["ground_truth"])
            for k in omitted:
                gt[k] = None  # null 表示"不应编造"
            f.write(json.dumps(
                {
                    "id": rec["id"],
                    "target": rec["target"],
                    "prompt": rebuild_prompt(gt, omitted),
                    "ground_truth": gt,
                    "omitted": omitted,
                },
                ensure_ascii=False,
            ) + "\n")
    print(f"字段缺失变体已生成: {out_path}（{len(src_lines)} 条，每条删 {OMIT_N} 字段）")

    with out_path.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    print("\n=== 第 1 条示例（省略了", first["omitted"], "）===")
    print(first["prompt"])
    print("ground_truth:", json.dumps(first["ground_truth"], ensure_ascii=False))


if __name__ == "__main__":
    main()
