"""特征工程（衍生特征）：时间转正、偿债比率、评分组合/缺失、计数与一致性。

从 scripts/train_baseline.py 迁到 src，作为生产模型服务的唯一实现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """衍生特征：时间转正、偿债比率、评分组合/缺失、计数与一致性。"""
    out = X.copy()

    def ratio(num: str, den: str, name: str) -> None:
        out[name] = out[num] / out[den]
        out[name] = out[name].replace([np.inf, -np.inf], np.nan)

    # ── ① 时间转正（DAYS_* 是相对申请日的负天数）─────────────────
    out["AGE"] = -out["DAYS_BIRTH"] / 365.25
    out["YEARS_EMPLOYED"] = -out["DAYS_EMPLOYED"] / 365.25
    out["YEARS_REGISTRATION"] = -out["DAYS_REGISTRATION"] / 365.25
    out["YEARS_ID"] = -out["DAYS_ID_PUBLISH"] / 365.25
    out["DAYS_SINCE_PHONE_CHANGE"] = -out["DAYS_LAST_PHONE_CHANGE"]
    out["AGE_FIRST_EMPLOYED"] = out["AGE"] - out["YEARS_EMPLOYED"]

    # ── ② 偿债能力比率 ──────────────────────────────────────────
    ratio("AMT_ANNUITY", "AMT_INCOME_TOTAL", "RATIO_ANNUITY_INCOME")
    ratio("AMT_INCOME_TOTAL", "AMT_CREDIT", "RATIO_INCOME_CREDIT")
    ratio("AMT_ANNUITY", "AMT_CREDIT", "RATIO_ANNUITY_CREDIT")
    ratio("AMT_CREDIT", "AMT_GOODS_PRICE", "RATIO_CREDIT_GOODS")
    ratio("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "RATIO_INCOME_PER_PERSON")

    # ── ③ 外部评分组合 + 缺失指示 ───────────────────────────────
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)
    out["EXT_SOURCE_MIN"] = out[ext_cols].min(axis=1)
    out["EXT_SOURCE_MAX"] = out[ext_cols].max(axis=1)
    out["EXT_SOURCE_MISSING_COUNT"] = out[ext_cols].isna().sum(axis=1)
    for c in ext_cols:
        out[f"{c}_ISMISSING"] = out[c].isna().astype(int)

    # ── ④ 家庭 ─────────────────────────────────────────────────
    out["ADULTS"] = out["CNT_FAM_MEMBERS"] - out["CNT_CHILDREN"]
    ratio("CNT_CHILDREN", "CNT_FAM_MEMBERS", "CHILD_RATIO")

    # ── ⑤ 计数 / 一致性 ─────────────────────────────────────────
    out["DOC_COUNT"] = out[[c for c in out.columns if c.startswith("FLAG_DOCUMENT")]].sum(axis=1)
    contact_cols = ["FLAG_PHONE", "FLAG_EMAIL", "FLAG_MOBIL", "FLAG_EMP_PHONE",
                    "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE"]
    out["CONTACT_COUNT"] = out[contact_cols].sum(axis=1)
    inq_cols = [c for c in out.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    out["INQ_TOTAL"] = out[inq_cols].sum(axis=1)
    region_cols = ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
                   "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
                   "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]
    out["REGION_MISMATCH"] = out[region_cols].sum(axis=1)
    housing_cols = [c for c in out.columns if c.startswith(
        ("APARTMENTS", "BASEMENTAREA", "NONLIVING", "LIVING", "FLOORS",
         "YEARS_", "COMMONAREA", "ELEVATORS", "ENTRANCES", "LANDAREA"))]
    out["HOUSING_MISSING_COUNT"] = out[housing_cols].isna().sum(axis=1)

    return out
