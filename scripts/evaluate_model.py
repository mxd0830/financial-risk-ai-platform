"""D4 模型评测：LightGBM 基线的指标汇总、混淆矩阵、风险分层、校准。

复用 train_baseline 的预处理与训练配置，产出评测报告数据。
"""
from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from train_baseline import fit_lgb, ks_score, load_and_prep, split


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """等频分箱的期望校准误差 ECE。"""
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-6
    bin_idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y_true)) * abs(y_prob[m].mean() - y_true[m].mean())
    return float(ece)


def main() -> None:
    X, y, num_cols, cat_cols, _ = load_and_prep()
    parts = split(X, y)
    model, to_lgb, _ = fit_lgb(parts, num_cols, cat_cols)
    y_test = parts["y_te"]
    y_prob = model.predict_proba(to_lgb(parts["X_te"]))[:, 1]

    print("=" * 70)
    print("一、指标汇总（测试集，46,127 样本，违约率 8.07%）")
    print("=" * 70)
    print(f"  AUC    = {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  KS     = {ks_score(y_test, y_prob):.4f}")
    print(f"  PR-AUC = {average_precision_score(y_test, y_prob):.4f}")
    print(f"  Brier  = {brier_score_loss(y_test, y_prob):.4f}")
    print(f"  ECE    = {ece(y_test, y_prob):.4f}")

    # 二、混淆矩阵 @ 0.5
    print("\n" + "=" * 70)
    print("二、混淆矩阵（阈值 0.5）")
    print("=" * 70)
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    print(f"  精确率(违约)= {p[1]:.4f}   召回率(违约)= {r[1]:.4f}   F1= {f1[1]:.4f}")
    print(f"  预测违约率 = {(y_pred==1).mean():.2%}  vs 真实 {y_test.mean():.2%}")

    # 不同阈值下的 tradeoff
    print("\n  不同阈值的业务 tradeoff：")
    print(f"  {'阈值':>8} {'通过率':>8} {'精确率':>8} {'召回率':>8} {'F1':>8}")
    for t in [0.05, 0.10, 0.20, 0.30, 0.50]:
        yp = (y_prob >= t).astype(int)
        pr, rc, f, _ = precision_recall_fscore_support(y_test, yp, zero_division=0)
        print(f"  {t:>8.2f} {1-(yp==1).mean():>8.1%} {pr[1]:>8.4f} {rc[1]:>8.4f} {f[1]:>8.4f}")

    # Youden 最优阈值
    fpr, tpr, thresh = roc_curve(y_test, y_prob)
    j = tpr - fpr
    best_t = thresh[np.argmax(j)]
    print(f"\n  Youden 最优阈值 = {best_t:.4f}（约登指数 max= {j.max():.4f}）")

    # 三、风险分层
    print("\n" + "=" * 70)
    print("三、风险分层（PD 分箱）")
    print("=" * 70)
    bins = [0, 0.02, 0.05, 0.10, 0.20, 1.0]
    labels = ["<2%", "2-5%", "5-10%", "10-20%", ">20%"]
    df = pd.DataFrame({"pd": y_prob, "y": y_test})
    df["band"] = pd.cut(df["pd"], bins=bins, labels=labels)
    strat = df.groupby("band", observed=True).agg(
        count=("y", "size"),
        pred_mean=("pd", "mean"),
        actual=("y", "mean"),
    )
    strat["占比"] = strat["count"] / len(df)
    strat["lift"] = strat["actual"] / y_test.mean()
    order = ["<2%", "2-5%", "5-10%", "10-20%", ">20%"]
    strat = strat.reindex(order)
    strat["违约占比"] = strat["count"] * strat["actual"] / y_test.sum()   # 该档违约数/总违约数
    strat["累计捕获率(≥该档)"] = strat["违约占比"][::-1].cumsum()[::-1]   # 从该档往上累计
    print(strat.round(4).to_string())
    print(f"\n  整体违约率 {y_test.mean():.2%}；lift=各档真实违约率/整体违约率（>1 表示高于平均风险）")

    # 四、校准曲线
    print("\n" + "=" * 70)
    print("四、校准曲线（10 等频分箱：预测均值 vs 真实违约率）")
    print("=" * 70)
    qbins = np.quantile(y_prob, np.linspace(0, 1, 11))
    qbins[0] -= 1e-6
    df["cal_bin"] = np.digitize(y_prob, qbins)
    cal = df.groupby("cal_bin").agg(
        pred_mean=("pd", "mean"),
        actual=("y", "mean"),
        n=("y", "size"),
    )
    for _, row in cal.iterrows():
        bar = "█" * int(row["pred_mean"] * 200)
        print(f"  pred={row['pred_mean']:.3f}  actual={row['actual']:.3f}  n={row['n']:>5,}  {bar}")
    print(f"\n  ECE = {ece(y_test, y_prob):.4f}（越小校准越好）")


if __name__ == "__main__":
    main()
