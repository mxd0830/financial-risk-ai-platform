"""从 Home Credit application_train.csv 构造「结构化抽取」评测集。

每条样本 = 一个真实借款人的「客户档案」自然语言描述 + 结构化字段 ground truth。
用于实验 A 的结构化抽取忠实度评测（先 500 条验证 → 全量 5000 条）。

生成物写入 data/processed/eval_extraction.jsonl（data/ 已被 gitignore，可随时重生成）。
每行 JSON：{"id", "target", "prompt", "ground_truth"}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW = "data/raw/application_train.csv"
OUT = "data/processed/eval_extraction.jsonl"
N_SAMPLES = 5000
SEED = 42

EDU_MAP = {
    "Secondary / secondary special": "中等教育",
    "Higher education": "高等教育",
    "Incomplete higher": "高等教育肄业",
    "Lower secondary": "初中教育",
    "Academic degree": "研究生学历",
}
CONTRACT_MAP = {"Cash loans": "现金贷", "Revolving loans": "循环贷"}
GENDER_MAP = {"M": "男", "F": "女", "XNA": "未知"}


def fmt_num(v: float):
    """数值格式化：整数值去小数点，非整保留 1 位小数。"""
    v = float(v)
    return int(v) if v == int(v) else round(v, 1)


def build_profile(row: pd.Series) -> tuple[str, dict]:
    gender = GENDER_MAP[row["CODE_GENDER"]]
    age = int(round(-row["DAYS_BIRTH"] / 365.25))
    education = EDU_MAP[row["NAME_EDUCATION_TYPE"]]
    contract = CONTRACT_MAP[row["NAME_CONTRACT_TYPE"]]
    income = fmt_num(row["AMT_INCOME_TOTAL"])
    loan = fmt_num(row["AMT_CREDIT"])
    annuity = fmt_num(row["AMT_ANNUITY"])
    goods = fmt_num(row["AMT_GOODS_PRICE"])
    children = int(row["CNT_CHILDREN"])
    emp_years = round(-row["DAYS_EMPLOYED"] / 365.25, 1)

    gt = {
        "gender": gender,
        "age": age,
        "education": education,
        "contract_type": contract,
        "income_total": income,
        "loan_amount": loan,
        "annuity": annuity,
        "goods_price": goods,
        "children": children,
        "employment_years": emp_years,
    }

    prompt = (
        "以下是某贷款申请人的档案，请从中提取字段并只输出一个 JSON 对象（不要输出任何解释或多余文字）：\n"
        f"性别：{gender}\n"
        f"年龄：{age}\n"
        f"学历：{education}\n"
        f"贷款类型：{contract}\n"
        f"收入总额：{income}\n"
        f"贷款金额：{loan}\n"
        f"月供：{annuity}\n"
        f"商品价格：{goods}\n"
        f"子女数：{children}\n"
        f"已工作年限：{emp_years} 年\n"
        "JSON 字段要求：gender（字符串）、age（整数）、education（字符串）、contract_type（字符串）、"
        "income_total（数值）、loan_amount（数值）、annuity（数值）、goods_price（数值）、"
        "children（整数）、employment_years（数值）。"
    )
    return prompt, gt


def main() -> None:
    df = pd.read_csv(RAW)
    # 剔除关键字段缺失 + 失业占位符（DAYS_EMPLOYED=365243），保证 ground truth 无空值、可比对
    cols = [
        "CODE_GENDER", "DAYS_BIRTH", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "CNT_CHILDREN", "DAYS_EMPLOYED",
    ]
    df = df.dropna(subset=cols)
    df = df[df["DAYS_EMPLOYED"] != 365243]
    df = df[df["NAME_EDUCATION_TYPE"].isin(EDU_MAP)]
    df = df[df["NAME_CONTRACT_TYPE"].isin(CONTRACT_MAP)]

    sample = df.sample(n=N_SAMPLES, random_state=SEED)

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in sample.iterrows():
            prompt, gt = build_profile(row)
            f.write(json.dumps(
                {
                    "id": int(row["SK_ID_CURR"]),
                    "target": int(row["TARGET"]),
                    "prompt": prompt,
                    "ground_truth": gt,
                },
                ensure_ascii=False,
            ) + "\n")
    print(f"评测集已生成: {out_path}（{N_SAMPLES} 条，从 {len(df)} 条合格样本中抽取）")

    with out_path.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    print("\n=== 第 1 条示例 ===")
    print(first["prompt"])
    print("ground_truth:", json.dumps(first["ground_truth"], ensure_ascii=False))


if __name__ == "__main__":
    main()
