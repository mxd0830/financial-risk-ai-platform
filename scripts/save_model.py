"""训练并保存 LightGBM 模型 + 预处理元数据，供 CreditModel 服务加载。"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "scripts")

from train_baseline import fit_lgb, load_and_prep, split

MODEL_DIR = Path("data/models")


def main() -> None:
    X, y, num_cols, cat_cols, drop_cols = load_and_prep()
    parts = split(X, y)
    model, _, _ = fit_lgb(parts, num_cols, cat_cols)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "lgbm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    meta = {"drop_cols": drop_cols, "cat_cols": cat_cols}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"模型已保存: {MODEL_DIR}/lgbm_model.pkl")
    print(f"元数据已保存: {MODEL_DIR}/meta.json（drop_cols {len(drop_cols)}，cat_cols {len(cat_cols)}）")


if __name__ == "__main__":
    main()
