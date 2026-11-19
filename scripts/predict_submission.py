"""在 application_test.csv 上预测并生成提交文件（当前单表模型 LightGBM）。

复用 train_baseline 的衍生特征逻辑；训练集全量训练，先早停定最优树数再全量重训。
注意：application_test.csv 无 TARGET，本地无法算 AUC；真正的分数需提交 Kaggle。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from train_baseline import add_derived_features, SEED

RAW = "data/raw/"
OUT = "data/processed/submission.csv"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """与训练一致：失业占位符→NaN + 衍生特征。"""
    out = df.copy()
    out.loc[out["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    return add_derived_features(out)


def main() -> None:
    train = pd.read_csv(RAW + "application_train.csv")
    test = pd.read_csv(RAW + "application_test.csv")
    y = train["TARGET"].values.astype(int)
    test_id = test["SK_ID_CURR"].values

    X = prep(train.drop(columns=["TARGET", "SK_ID_CURR"]))
    X_test = prep(test.drop(columns=["SK_ID_CURR"]))

    # 用训练集计算的缺失率丢弃列（>95%），测试集同列丢弃
    drop_cols = X.columns[X.isna().mean() > 0.95].tolist()
    X = X.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)
    X_test = X_test[X.columns]  # 列对齐

    # 类别列统一 category dtype（train/test 用同一套 categories，防编码错位）
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    for c in cat_cols:
        cats = pd.concat([X[c], X_test[c]], axis=0).astype("category").cat.categories.tolist()
        X[c] = pd.Categorical(X[c], categories=cats)
        X_test[c] = pd.Categorical(X_test[c], categories=cats)

    print(f"训练集: {X.shape[0]:,} × {X.shape[1]}，测试集: {X_test.shape[0]:,} × {X_test.shape[1]}")
    assert X.columns.tolist() == X_test.columns.tolist(), "特征列不一致"

    params = dict(
        n_estimators=1500, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
        random_state=SEED, verbose=-1,
    )

    # 第 1 步：10% 留出早停，定最优树数
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    m1 = lgb.LGBMClassifier(**params)
    m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
           callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = m1.best_iteration_
    va_auc = roc_auc_score(y_va, m1.predict_proba(X_va)[:, 1])
    print(f"留出验证 AUC={va_auc:.4f}，最优树数={best_iter}")

    # 第 2 步：全量重训
    m2 = lgb.LGBMClassifier(**{**params, "n_estimators": best_iter})
    m2.fit(X, y)

    proba = m2.predict_proba(X_test)[:, 1]
    sub = pd.DataFrame({"SK_ID_CURR": test_id, "TARGET": proba})
    sub.to_csv(OUT, index=False)

    print(f"\n已生成提交文件: {OUT}（{len(sub):,} 行）")
    print(f"PD 分布: min={proba.min():.4f}  中位={np.median(proba):.4f}  "
          f"均值={proba.mean():.4f}  max={proba.max():.4f}")
    print(f"预测平均违约率 {proba.mean():.2%}（训练集真实 8.07%，两者应接近）")


if __name__ == "__main__":
    main()
