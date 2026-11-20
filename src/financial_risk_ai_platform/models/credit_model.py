"""信贷违约模型服务：加载训练好的 LightGBM，预处理 + 预测 PD。

模型与元数据由 scripts/save_model.py 生成，存于 data/models/（gitignore）。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_derived_features

MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "models"


class CreditModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        with open(model_dir / "lgbm_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.feature_names = self.model.booster_.feature_name()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """原始特征 → 模型输入（与训练完全一致的处理）。"""
        out = df.copy()
        for drop in ("SK_ID_CURR", "TARGET"):
            if drop in out.columns:
                out = out.drop(columns=[drop])
        if "DAYS_EMPLOYED" in out.columns:
            out.loc[out["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
        for c in self.meta["drop_cols"]:          # 与训练一致：先删高缺失列
            if c in out.columns:
                out = out.drop(columns=[c])
        out = add_derived_features(out)           # 再衍生
        for c in self.meta["cat_cols"]:           # 再转类别 dtype
            if c in out.columns:
                out[c] = out[c].astype("category")
        num_cols = [c for c in out.columns if c not in self.meta["cat_cols"]]
        out[num_cols] = out[num_cols].astype(float)   # 其余列转 float（None→NaN）
        return out.reindex(columns=self.feature_names)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """返回违约概率 PD（0~1）。"""
        X = self.preprocess(df)
        return self.model.predict_proba(X)[:, 1]
