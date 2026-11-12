"""下载 Home Credit Default Risk 竞赛主表 application_train.csv。

凭据优先读 ~/.kaggle/kaggle.json，其次读环境变量 KAGGLE_USERNAME / KAGGLE_KEY。
原始数据不提交到公开仓库（/data/ 已 gitignore）。

获取 API token：https://www.kaggle.com/settings → API → Create New Token
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
COMPETITION = "home-credit-default-risk"
FILENAME = "application_train.csv"


def load_credentials() -> tuple[str, str]:
    """读取 Kaggle 凭据（kaggle.json 或环境变量），不写入仓库。"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        creds = json.loads(kaggle_json.read_text())
        return creds["username"], creds["key"]
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        return username, key
    raise RuntimeError(
        "未找到 Kaggle 凭据。请把 kaggle.json 放到 ~/.kaggle/ 目录，"
        "或设置 KAGGLE_USERNAME / KAGGLE_KEY 环境变量。"
    )


def main() -> None:
    username, key = load_credentials()
    url = (
        f"https://www.kaggle.com/api/v1/competitions/data/download/"
        f"{COMPETITION}/{FILENAME}"
    )
    print(f"下载 {FILENAME} ...")
    resp = requests.get(url, auth=(username, key), timeout=600)
    resp.raise_for_status()
    out = RAW / FILENAME
    out.write_bytes(resp.content)
    print(f"已保存: {out} ({len(resp.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
