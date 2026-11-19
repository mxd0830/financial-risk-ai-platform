"""D3 基线模型：Logistic Regression vs LightGBM（实验 P）。

一致的数据切分与特征口径，多指标对比两个基线 PD 模型。

特征口径（两模型一致）：
- 丢弃 TARGET / SK_ID_CURR / 缺失率 >95% 的列（覆盖率 <5% 才删）
- DAYS_EMPLOYED=365243（失业占位符）按缺失处理
- 预处理按模型各自最佳实践：LR 走「中位数填补 + one-hot + 标准化」；
  LightGBM 走「原生 NaN + 类别 dtype」（两者编码方式不同是各自模型的必要差异）
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW = "data/raw/application_train.csv"
OUT = Path("experiments/lr_lgbm_baseline")
SEED = 42


def ks_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS：正负样本累计分布的最大分离度。"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """衍生特征：时间转正、偿债比率、评分组合/缺失、计数与一致性。"""
    out = X.copy()

    def ratio(num: str, den: str, name: str) -> None:
        out[name] = out[num] / out[den]
        out[name] = out[name].replace([np.inf, -np.inf], np.nan)

    # ── ① 时间转正（DAYS_* 是相对申请日的负天数）─────────────────
    out["AGE"] = -out["DAYS_BIRTH"] / 365.25                    # 年龄
    out["YEARS_EMPLOYED"] = -out["DAYS_EMPLOYED"] / 365.25      # 工龄（365243 已置 NaN）
    out["YEARS_REGISTRATION"] = -out["DAYS_REGISTRATION"] / 365.25
    out["YEARS_ID"] = -out["DAYS_ID_PUBLISH"] / 365.25          # 证件已发年数
    out["DAYS_SINCE_PHONE_CHANGE"] = -out["DAYS_LAST_PHONE_CHANGE"]
    out["AGE_FIRST_EMPLOYED"] = out["AGE"] - out["YEARS_EMPLOYED"]  # 首次就业年龄

    # ── ② 偿债能力比率 ──────────────────────────────────────────
    ratio("AMT_ANNUITY", "AMT_INCOME_TOTAL", "RATIO_ANNUITY_INCOME")   # DSR
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


def load_and_prep() -> tuple[pd.DataFrame, np.ndarray, list[str], list[str], list[str]]:
    df = pd.read_csv(RAW)
    y = df["TARGET"].values.astype(int)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    # 失业占位符 → 缺失
    X.loc[X["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    # 丢弃缺失率 >95% 的列（覆盖率 <5% 才删）
    drop_cols = X.columns[X.isna().mean() > 0.95].tolist()
    X = X.drop(columns=drop_cols)
    # 衍生特征
    X = add_derived_features(X)
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, num_cols, cat_cols, drop_cols


def split(X: pd.DataFrame, y: np.ndarray) -> dict[str, pd.DataFrame | np.ndarray]:
    X_trva, X_te, y_trva, y_te = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trva, y_trva, test_size=0.15 / 0.85, random_state=SEED, stratify=y_trva
    )
    return dict(
        X_tr=X_tr, X_va=X_va, X_te=X_te,
        y_tr=y_tr, y_va=y_va, y_te=y_te,
        X_trva=X_trva, y_trva=y_trva,
    )


# ---------- LR ----------
def fit_lr(parts, num_cols, cat_cols):
    num_imp = SimpleImputer(strategy="median")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    Xtr = num_imp.fit_transform(parts["X_tr"][num_cols])
    Xtr = np.hstack([Xtr, ohe.fit_transform(parts["X_tr"][cat_cols])])
    Xtr = scaler.fit_transform(Xtr)
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    t0 = time.time()
    model.fit(Xtr, parts["y_tr"])
    train_s = time.time() - t0

    def transform(df: pd.DataFrame) -> np.ndarray:
        num = num_imp.transform(df[num_cols])
        num = np.hstack([num, ohe.transform(df[cat_cols])])
        return scaler.transform(num)

    return model, transform, train_s


# ---------- LightGBM ----------
def fit_lgb(parts, num_cols, cat_cols):
    def to_lgb(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cat_cols:
            out[c] = out[c].astype("category")
        return out

    Xtr = to_lgb(parts["X_tr"])
    Xva = to_lgb(parts["X_va"])
    model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(
        Xtr, parts["y_tr"],
        eval_set=[(Xva, parts["y_va"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    train_s = time.time() - t0
    return model, to_lgb, train_s


def metrics(y_true, y_score) -> dict:
    return dict(
        auc=round(roc_auc_score(y_true, y_score), 4),
        ks=round(ks_score(y_true, y_score), 4),
        pr_auc=round(average_precision_score(y_true, y_score), 4),
        brier=round(brier_score_loss(y_true, y_score), 4),
    )


def cv_auc(model_fn, parts, num_cols, cat_cols, n_splits=5) -> tuple[float, float]:
    """5-fold 分层 CV，返回 AUC 均值与标准差（稳定性）。"""
    X = parts["X_trva"]
    y = parts["y_trva"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        sub = dict(
            X_tr=X.iloc[tr_idx], X_va=X.iloc[va_idx],
            y_tr=y[tr_idx], y_va=y[va_idx],
        )
        model, transform, _ = model_fn(sub, num_cols, cat_cols)
        aucs.append(roc_auc_score(y[va_idx], model.predict_proba(transform(sub["X_va"]))[:, 1]))
    return round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, num_cols, cat_cols, drop_cols = load_and_prep()
    parts = split(X, y)

    print(f"数据: {X.shape[0]:,} × {X.shape[1]} 特征（数值 {len(num_cols)}，类别 {len(cat_cols)}）")
    print(f"切分: train {len(parts['y_tr']):,} / valid {len(parts['y_va']):,} / test {len(parts['y_te']):,}")
    print(f"测试集违约率: {parts['y_te'].mean():.2%}\n")

    result = {"meta": {"data": RAW, "seed": SEED, "dropped_gt95pct_missing": drop_cols}}

    # LR
    lr_model, lr_transform, lr_train_s = fit_lr(parts, num_cols, cat_cols)
    t0 = time.time()
    lr_prob = lr_model.predict_proba(lr_transform(parts["X_te"]))[:, 1]
    lr_infer_s = time.time() - t0
    lr_m = metrics(parts["y_te"], lr_prob)
    lr_cv = cv_auc(fit_lr, parts, num_cols, cat_cols)
    result["lr"] = dict(
        metrics=lr_m, cv_auc_mean=lr_cv[0], cv_auc_std=lr_cv[1],
        train_time_s=round(lr_train_s, 3), inference_time_s=round(lr_infer_s, 4),
    )

    # LGBM
    lgb_model, lgb_transform, lgb_train_s = fit_lgb(parts, num_cols, cat_cols)
    t0 = time.time()
    lgb_prob = lgb_model.predict_proba(lgb_transform(parts["X_te"]))[:, 1]
    lgb_infer_s = time.time() - t0
    lgb_m = metrics(parts["y_te"], lgb_prob)
    lgb_cv = cv_auc(fit_lgb, parts, num_cols, cat_cols)
    result["lgbm"] = dict(
        metrics=lgb_m, cv_auc_mean=lgb_cv[0], cv_auc_std=lgb_cv[1],
        train_time_s=round(lgb_train_s, 3), inference_time_s=round(lgb_infer_s, 4),
        best_iter=lgb_model.best_iteration_,
    )

    # 特征重要度（LR 取 |系数|，LGBM 取 gain）
    lr_coef = pd.Series(lr_model.coef_[0], index=lr_transform_feature_names(lr_model, parts, num_cols, cat_cols))
    lr_top = lr_coef.abs().sort_values(ascending=False).head(15)
    lgb_imp = pd.Series(lgb_model.booster_.feature_importance("gain"), index=lgb_model.booster_.feature_name())
    lgb_top = lgb_imp.sort_values(ascending=False).head(15)

    result["lr"]["top_features"] = {k: round(float(v), 5) for k, v in lr_top.items()}
    result["lgbm"]["top_features"] = {k: int(v) for k, v in lgb_top.items()}

    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # 打印对比表
    print("=" * 78)
    print("实验结果对比（测试集）")
    print("=" * 78)
    print(f"{'指标':<16}{'LogisticRegression':>22}{'LightGBM':>22}")
    for key, label in [("auc", "AUC"), ("ks", "KS"), ("pr_auc", "PR-AUC"), ("brier", "Brier")]:
        print(f"{label:<16}{lr_m[key]:>22}{lgb_m[key]:>22}")
    print(f"{'CV-AUC 均值±std':<16}{str(lr_cv[0])+'±'+str(lr_cv[1]):>22}{str(lgb_cv[0])+'±'+str(lgb_cv[1]):>22}")
    print(f"{'训练耗时(s)':<16}{lr_train_s:>22.3f}{lgb_train_s:>22.3f}")
    print(f"{'推理耗时(s)':<16}{lr_infer_s:>22.4f}{lgb_infer_s:>22.4f}")
    print(f"\nLR 顶级特征（|系数|）:\n{lr_top.to_string()}")
    print(f"\nLightGBM 顶级特征（gain）:\n{lgb_top.to_string()}")
    print(f"\n结果已存: {OUT / 'metrics.json'}")


def lr_transform_feature_names(model, parts, num_cols, cat_cols):
    """LR 的 one-hot 后特征名（用于系数对齐）。"""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(parts["X_tr"][cat_cols])
    return num_cols + list(ohe.get_feature_names_out(cat_cols))


if __name__ == "__main__":
    main()
