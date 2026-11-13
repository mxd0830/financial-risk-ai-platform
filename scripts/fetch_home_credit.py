"""下载 Home Credit Default Risk 竞赛主表 application_train.csv。

认证：读取 KAGGLE_API_TOKEN（Kaggle 新版 API token，形如 KGAT_xxx），
用 Authorization: Bearer <token> 请求。

前置条件：
1. 需先在 Kaggle 网页接受该竞赛规则（否则 403）：
   https://www.kaggle.com/competitions/home-credit-default-risk/rules
2. 必须带浏览器 User-Agent（Kaggle 对 requests 默认 UA 返回 404）。

注意：Kaggle 返回的是 application_train.csv.zip，需解压。
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv()

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
COMPETITION = "home-credit-default-risk"
FILENAME = "application_train.csv"

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def load_token() -> str:
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "未找到 KAGGLE_API_TOKEN。请在 .env 中设置，或 export KAGGLE_API_TOKEN=KGAT_xxx"
        )
    return token


def main() -> None:
    token = load_token()
    url = (
        f"https://www.kaggle.com/api/v1/competitions/data/download/"
        f"{COMPETITION}/{FILENAME}"
    )
    headers = {"Authorization": f"Bearer {token}", "User-Agent": BROWSER_UA}
    print(f"下载 {FILENAME} ...")
    resp = requests.get(url, headers=headers, timeout=600)
    if resp.status_code == 403:
        raise RuntimeError(
            "403：需先在 Kaggle 网页接受该竞赛规则。打开 "
            "https://www.kaggle.com/competitions/home-credit-default-risk/rules "
            "点击 I Understand and Accept。"
        )
    resp.raise_for_status()

    # Kaggle 返回的是 zip 包
    zip_path = RAW / f"{FILENAME}.zip"
    zip_path.write_bytes(resp.content)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(FILENAME, RAW)
    zip_path.unlink()
    out = RAW / FILENAME
    print(f"已保存: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
