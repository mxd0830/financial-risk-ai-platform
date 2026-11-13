"""Home Credit application_train.csv 探索性数据分析（D2）。

输出：数据规模、缺失率、标签分布、关键变量与违约率的关系。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW = "data/raw/application_train.csv"


def main() -> None:
    df = pd.read_csv(RAW)
    target = df["TARGET"]

    print("=" * 60)
    print("1. 数据规模")
    print("=" * 60)
    print(f"形状: {df.shape}  (行 × 列)")
    print(f"内存占用: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print("\n" + "=" * 60)
    print("2. 标签分布（TARGET：0=未违约，1=违约）")
    print("=" * 60)
    counts = target.value_counts().sort_index()
    for label, cnt in counts.items():
        print(f"  TARGET={label}: {cnt:>8,}  ({cnt / len(df):.2%})")

    print("\n" + "=" * 60)
    print("3. 缺失率（前 20 列）")
    print("=" * 60)
    missing = df.isnull().mean().sort_values(ascending=False)
    print(f"  有缺失的列: {(missing > 0).sum()} / {len(missing)}")
    print(f"  缺失率 > 50% 的列: {(missing > 0.5).sum()}")
    for col, rate in missing.head(20).items():
        print(f"  {col:<35} {rate:>7.2%}")

    print("\n" + "=" * 60)
    print("4. 关键数值特征统计")
    print("=" * 60)
    key_cols = [
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_INCOME_TOTAL",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_CHILDREN",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]
    print(df[key_cols].describe().round(1).to_string())

    print("\n" + "=" * 60)
    print("5. 违约率 vs 关键变量")
    print("=" * 60)
    # 年龄（DAYS_BIRTH 为负，转成年）
    df["age"] = -df["DAYS_BIRTH"] / 365.25
    print("\n[违约率 vs 年龄分箱]")
    age_bins = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 100], labels=["<25", "25-35", "35-45", "45-55", ">55"])
    print((target.groupby(age_bins).mean() * 100).round(2).to_string(), "%")

    print("\n[违约率 vs 收入分箱]")
    inc_bins = pd.qcut(df["AMT_INCOME_TOTAL"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    print((target.groupby(inc_bins).mean() * 100).round(2).to_string(), "%")

    print("\n[违约率 vs 性别]")
    print((target.groupby(df["CODE_GENDER"]).mean() * 100).round(2).to_string(), "%")

    print("\n[违约率 vs 外部评分 EXT_SOURCE_3（缺失已剔除）]")
    es3 = df["EXT_SOURCE_3"]
    es3_bins = pd.qcut(es3, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    print((target[es3.notna()].groupby(es3_bins[es3.notna()]).mean() * 100).round(2).to_string(), "%")


if __name__ == "__main__":
    main()
