"""D5 模型解释：SHAP 全局重要性 + 局部风险因素（原因码）。

复用 train_baseline 训练 LightGBM，用 TreeExplainer 算 SHAP，
产出全局特征重要性 + 单个客户的风险原因码。
"""
from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import shap

from train_baseline import fit_lgb, load_and_prep, split


def fmt_val(v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    return str(v)


def main() -> None:
    X, y, num_cols, cat_cols, _ = load_and_prep()
    parts = split(X, y)
    model, to_lgb, _ = fit_lgb(parts, num_cols, cat_cols)
    X_test = to_lgb(parts["X_te"])
    y_prob = model.predict_proba(X_test)[:, 1]

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_test), 3000, replace=False)
    X_sample = X_test.iloc[sample_idx]
    sample_prob = y_prob[sample_idx]

    print("计算 SHAP（TreeExplainer，采样 3000）...")
    explainer = shap.TreeExplainer(model)
    shap_pos = explainer.shap_values(X_sample)  # shap 0.51 直接返回 (n, n_features) 的违约类别 SHAP
    feat_names = model.booster_.feature_name()
    shap_df = pd.DataFrame(shap_pos, columns=feat_names)

    print("\n" + "=" * 70)
    print("一、全局 SHAP 重要性（top 20，mean |SHAP|）")
    print("=" * 70)
    print(shap_df.abs().mean().sort_values(ascending=False).head(20).round(4).to_string())

    for label, i in [("高风险", int(np.argmax(sample_prob))), ("低风险", int(np.argmin(sample_prob)))]:
        row = shap_df.iloc[i]
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(10)
        print("\n" + "=" * 70)
        print(f"二、{label}客户原因码（预测 PD={sample_prob[i]:.3f}）top10")
        print("=" * 70)
        print(f"  {'特征':<30} {'特征值':>14} {'SHAP':>9}  方向")
        for feat, shv in top.items():
            direction = "↑ 推高风险" if shv > 0 else "↓ 拉低风险"
            print(f"  {feat:<30} {fmt_val(X_sample.iloc[i][feat]):>14} {shv:>+9.4f}  {direction}")


if __name__ == "__main__":
    main()
