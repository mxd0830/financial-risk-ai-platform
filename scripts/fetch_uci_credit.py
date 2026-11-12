"""下载 UCI Default of Credit Card Clients 数据集到 data/raw/。

数据集说明：历史信用卡客户下一期违约预测数据，用于验证
数据处理、EDA 和信用卡存量客户违约预测流程（不等同于贷前贷款申请评分）。
来源：https://archive.ics.uci.edu/dataset/350/default+credit+card+clients
"""
from __future__ import annotations

from pathlib import Path

from ucimlrepo import fetch_ucirepo

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=350)
    df = dataset.data.original
    out = RAW / "uci_default_credit.csv"
    df.to_csv(out, index=False)
    print(f"已保存: {out}")
    print(f"形状: {df.shape}")


if __name__ == "__main__":
    main()
