"""D3+ 多表基线（第一步）：bureau + previous_application 聚合特征 + LightGBM。

先用两张最小的明细表（bureau 1.7M 行 / previous_application 1.67M 行）做尝试：
按 SK_ID_CURR 聚合后 join 主表，对比「单表基线」与「+多表聚合」的 AUC。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from train_baseline import add_derived_features, SEED, ks_score

RAW = "data/raw/"


def agg_bureau(b: pd.DataFrame, all_ids: np.ndarray) -> pd.DataFrame:
    """bureau.csv → 每客户一行（信用历史统计）。"""
    g = b.groupby("SK_ID_CURR")

    def s(col, how):
        return getattr(g[col], how)().reindex(all_ids)

    def cnt(col, val):
        return b.loc[b[col] == val].groupby("SK_ID_CURR").size().reindex(all_ids, fill_value=0)

    agg = pd.DataFrame({
        "BUREAU_COUNT": g.size().reindex(all_ids),
        "BUREAU_ACTIVE_COUNT": cnt("CREDIT_ACTIVE", "Active"),
        "BUREAU_CLOSED_COUNT": cnt("CREDIT_ACTIVE", "Closed"),
        "BUREAU_BAD_DEBT_COUNT": cnt("CREDIT_ACTIVE", "Bad debt"),
        "BUREAU_DEBT_SUM": s("AMT_CREDIT_SUM_DEBT", "sum"),
        "BUREAU_OVERDUE_SUM": s("AMT_CREDIT_SUM_OVERDUE", "sum"),
        "BUREAU_MAX_OVERDUE_DAYS": s("CREDIT_DAY_OVERDUE", "max"),
        "BUREAU_MEAN_OVERDUE_DAYS": s("CREDIT_DAY_OVERDUE", "mean"),
        "BUREAU_MAX_OVERDUE_AMT": s("AMT_CREDIT_MAX_OVERDUE", "max"),
        "BUREAU_CREDIT_SUM": s("AMT_CREDIT_SUM", "sum"),
        "BUREAU_LIMIT_SUM": s("AMT_CREDIT_SUM_LIMIT", "sum"),
        "BUREAU_PROLONG_SUM": s("CNT_CREDIT_PROLONG", "sum"),
        "BUREAU_CREDIT_HISTORY": -s("DAYS_CREDIT", "min"),
        "BUREAU_RECENT_UPDATE": s("DAYS_CREDIT_UPDATE", "max"),
    })
    agg["BUREAU_DEBT_RATIO"] = agg["BUREAU_DEBT_SUM"] / agg["BUREAU_CREDIT_SUM"].replace(0, np.nan)

    # ── 时间感知：最近 6 个月窗口 + 速度/比率 ─────────────────────
    rec = b[b["DAYS_CREDIT"] >= -180]
    overdue_any = b[b["CREDIT_DAY_OVERDUE"] > 0]
    agg["BUREAU_RECENT_6M_COUNT"] = rec.groupby("SK_ID_CURR").size().reindex(all_ids, fill_value=0)
    agg["BUREAU_RECENT_6M_OVERDUE_COUNT"] = (
        rec[rec["CREDIT_DAY_OVERDUE"] > 0].groupby("SK_ID_CURR").size().reindex(all_ids, fill_value=0))
    agg["BUREAU_RECENT_6M_BAD_DEBT_COUNT"] = (
        rec[rec["CREDIT_ACTIVE"] == "Bad debt"].groupby("SK_ID_CURR").size().reindex(all_ids, fill_value=0))
    agg["BUREAU_RECENT_6M_DEBT_SUM"] = rec.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].sum().reindex(all_ids)
    agg["BUREAU_RECENT_OVERDUE_GAP"] = overdue_any.groupby("SK_ID_CURR")["DAYS_CREDIT"].max().reindex(all_ids)
    agg["BUREAU_OVERDUE_RATE"] = (
        overdue_any.groupby("SK_ID_CURR").size().reindex(all_ids, fill_value=0) / agg["BUREAU_COUNT"])
    agg["BUREAU_ACTIVE_RATIO"] = agg["BUREAU_ACTIVE_COUNT"] / agg["BUREAU_COUNT"]
    agg["BUREAU_CREDIT_VELOCITY"] = agg["BUREAU_COUNT"] / agg["BUREAU_CREDIT_HISTORY"].replace(0, np.nan)
    return agg


def agg_prev(p: pd.DataFrame, all_ids: np.ndarray) -> pd.DataFrame:
    """previous_application.csv → 每客户一行（历史申请统计）。"""
    g = p.groupby("SK_ID_CURR")

    def s(col, how):
        return getattr(g[col], how)().reindex(all_ids)

    def cnt(col, val):
        return p.loc[p[col] == val].groupby("SK_ID_CURR").size().reindex(all_ids, fill_value=0)

    agg = pd.DataFrame({
        "PREV_COUNT": g.size().reindex(all_ids),
        "PREV_APPROVED_COUNT": cnt("NAME_CONTRACT_STATUS", "Approved"),
        "PREV_REFUSED_COUNT": cnt("NAME_CONTRACT_STATUS", "Refused"),
        "PREV_HC_REJECT_COUNT": cnt("CODE_REJECT_REASON", "HC"),
        "PREV_CREDIT_DIFF": s("AMT_APPLICATION", "mean") - s("AMT_CREDIT", "mean"),
        "PREV_APPLICATION_MEAN": s("AMT_APPLICATION", "mean"),
        "PREV_ANNUITY_MEAN": s("AMT_ANNUITY", "mean"),
        "PREV_DOWN_PAYMENT_MEAN": s("RATE_DOWN_PAYMENT", "mean"),
        "PREV_INTEREST_MEAN": s("RATE_INTEREST_PRIMARY", "mean"),
        "PREV_RECENT_DECISION": s("DAYS_DECISION", "max"),
        "PREV_CNT_PAYMENT_MEAN": s("CNT_PAYMENT", "mean"),
    })
    agg["PREV_APPROVED_RATE"] = agg["PREV_APPROVED_COUNT"] / agg["PREV_COUNT"]

    # ── 时间感知：最近 6 个月窗口 + 速度/变化率 ───────────────────
    rec = p[p["DAYS_DECISION"] >= -180]
    agg["PREV_RECENT_6M_COUNT"] = rec.groupby("SK_ID_CURR").size().reindex(all_ids, fill_value=0)
    agg["PREV_RECENT_6M_APPROVED_COUNT"] = (
        rec[rec["NAME_CONTRACT_STATUS"] == "Approved"].groupby("SK_ID_CURR").size().reindex(all_ids, fill_value=0))
    agg["PREV_RECENT_6M_REFUSED_COUNT"] = (
        rec[rec["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR").size().reindex(all_ids, fill_value=0))
    agg["PREV_RECENT_6M_APPROVED_RATE"] = agg["PREV_RECENT_6M_APPROVED_COUNT"] / agg["PREV_RECENT_6M_COUNT"]
    span = (s("DAYS_DECISION", "max") - s("DAYS_DECISION", "min")).replace(0, np.nan)
    agg["PREV_APPLICATION_VELOCITY"] = agg["PREV_COUNT"] / span
    agg["PREV_APPROVED_RATE_CHANGE"] = agg["PREV_RECENT_6M_APPROVED_RATE"] - agg["PREV_APPROVED_RATE"]
    return agg


def main() -> None:
    train = pd.read_csv(RAW + "application_train.csv")
    y = train["TARGET"].values.astype(int)
    all_ids = train["SK_ID_CURR"].values

    print("聚合 bureau.csv ...", flush=True)
    b_agg = agg_bureau(pd.read_csv(RAW + "bureau.csv"), all_ids)
    print("聚合 previous_application.csv ...", flush=True)
    p_agg = agg_prev(pd.read_csv(RAW + "previous_application.csv"), all_ids)

    X = train.drop(columns=["TARGET", "SK_ID_CURR"])
    X.loc[X["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    X = X.join(b_agg).join(p_agg)  # 索引即 SK_ID_CURR（train 原始顺序）

    X = add_derived_features(X)
    drop_cols = X.columns[X.isna().mean() > 0.95].tolist()
    X = X.drop(columns=drop_cols)
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    print(f"特征维度: {X.shape}（单表 145 → 多表 {X.shape[1]}）", flush=True)

    # 与单表基线相同的切分
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.15 / 0.85, random_state=SEED, stratify=y_tr
    )

    model = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
        random_state=SEED, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    proba = model.predict_proba(X_te)[:, 1]

    print("\n=== 结果（bureau + previous_application 聚合后）===")
    print(f"  test AUC = {roc_auc_score(y_te, proba):.4f}  （单表基线 0.7691）")
    print(f"  KS = {ks_score(y_te, proba):.4f}")
    print(f"  最优树数 = {model.best_iteration_}")

    imp = pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
    print("\nTop 20 特征（gain）:")
    print(imp.sort_values(ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
