"""H²MTT V1 最小试点：静态 + 月度(credit_card) + 时序 Transformer → 单 TARGET 二分类。

试点规模：50K 客户，验证「月度时序序列是否比纯静态特征更有预测力」这一核心假设。
（事件分支、跨表注意力、多窗口 PD 留到 V2/V3）
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS 缺失算子回退 CPU

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train_baseline import add_derived_features, SEED

RAW = "data/raw/"
N_CLIENTS = 50000
N_MONTHS = 36  # 历史窗口 36 个月（H²MTT 文档建议 36 起步）
D_MODEL = 128


def build_static(train: pd.DataFrame) -> np.ndarray:
    """主表静态特征（数值列，中位数填补 + 标准化）。"""
    X = train.drop(columns=["TARGET", "SK_ID_CURR"])
    X.loc[X["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    X = add_derived_features(X)
    X = X.drop(columns=X.columns[X.isna().mean() > 0.95])
    X = X.select_dtypes(include="number")
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X = sc.fit_transform(imp.fit_transform(X.values))
    return X.astype(np.float32)


def build_monthly_seq(cc: pd.DataFrame, ids: np.ndarray, n_months: int) -> tuple[np.ndarray, np.ndarray]:
    """credit_card_balance → 每客户 (n_months × 特征) 序列 + 掩码。"""
    id_set = set(ids)
    cc = cc[cc["SK_ID_CURR"].isin(id_set)]
    monthly = cc.groupby(["SK_ID_CURR", "MONTHS_BALANCE"]).agg(
        AMT_BALANCE=("AMT_BALANCE", "sum"),
        AMT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "sum"),
        AMT_PAYMENT=("AMT_PAYMENT_CURRENT", "sum"),
        SK_DPD=("SK_DPD", "max"),
        SK_DPD_DEF=("SK_DPD_DEF", "max"),
        N_CARDS=("SK_ID_PREV", "nunique"),
    ).reset_index()
    monthly["UTILIZATION"] = (monthly["AMT_BALANCE"] / monthly["AMT_LIMIT"].replace(0, np.nan)).astype(np.float32)
    feat_cols = ["AMT_BALANCE", "AMT_LIMIT", "AMT_PAYMENT", "SK_DPD", "SK_DPD_DEF", "N_CARDS", "UTILIZATION"]

    id_to_idx = {sk: i for i, sk in enumerate(ids)}
    seq = np.zeros((len(ids), n_months, len(feat_cols)), dtype=np.float32)
    mask = np.zeros((len(ids), n_months), dtype=np.float32)
    for row in monthly.itertuples(index=False):
        sk = row.SK_ID_CURR
        m = int(row.MONTHS_BALANCE)
        if sk not in id_to_idx or not (-n_months <= m < 0):
            continue
        idx = id_to_idx[sk]
        j = m + n_months
        seq[idx, j] = [getattr(row, c) for c in feat_cols]
        mask[idx, j] = 1.0
    seq = np.nan_to_num(seq, nan=0.0)

    # 显式相对时间：距预测时点的月数（Δt）+ 距上次观测的间隔（事件间隔）
    delta_t = (n_months - np.arange(n_months)).astype(np.float32)          # 96,95,...,1（月前）
    delta_t = np.broadcast_to(delta_t[None, :, None], (len(ids), n_months, 1))
    gap = np.zeros((len(ids), n_months, 1), dtype=np.float32)
    for i in range(len(ids)):
        last = -1
        for j in range(n_months):
            if mask[i, j] == 1:
                gap[i, j, 0] = float(j - last) if last >= 0 else 0.0
                last = j
    seq = np.concatenate([seq, delta_t, gap], axis=-1)
    # 标准化序列特征（用观测到的格子算 mean/std，防止大额数值导致数值溢出）
    for f in range(seq.shape[2]):
        obs = seq[:, :, f][mask == 1]
        if len(obs) > 0 and obs.std() > 1e-6:
            seq[:, :, f] = (seq[:, :, f] - obs.mean()) / (obs.std() + 1e-6)
    return seq, mask


class Time2Vec(nn.Module):
    """连续时间编码：第 0 维为线性项，其余为周期性项（sin）。"""
    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model) * 0.1)
        self.b = nn.Parameter(torch.randn(d_model) * 0.1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t * self.w + self.b                                  # (B, T, d)
        return torch.cat([x[..., :1], torch.sin(x[..., 1:])], dim=-1)


class H2MTT_V1(nn.Module):
    def __init__(self, static_dim: int, seq_dim: int, d_model: int = D_MODEL):
        super().__init__()
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.seq_proj = nn.Linear(seq_dim, d_model)
        self.t2v = Time2Vec(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dropout=0.1)
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, 1),
        )

    def forward(self, static: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor):
        B, T, _ = seq.shape
        h_static = self.static_mlp(static)                                # (B, d)
        # 相对时间：距预测时点的月数，归一化到 0~1，经 Time2Vec 编码
        t = torch.arange(T, device=seq.device).float() / T               # (T,)
        t = t.unsqueeze(0).unsqueeze(-1).expand(B, T, 1)                 # (B, T, 1)
        h_seq = self.seq_proj(seq) + self.t2v(t)                         # (B, T, d)
        pad = (mask == 0)                                                 # True=无数据
        h = self.temporal(h_seq, src_key_padding_mask=pad)                # (B, T, d)
        h = torch.nan_to_num(h, nan=0.0)   # 全 mask 行 attention 产生 NaN，置 0（无记录客户贡献 0）
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        h_pool = (h * mask.unsqueeze(-1)).sum(dim=1) / denom              # (B, d)
        return self.head(torch.cat([h_static, h_pool], dim=-1)).squeeze(-1)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    train = pd.read_csv(RAW + "application_train.csv")
    # 分层抽样 50K（train 在前，修掉之前的子集/测试集拆反 bug）
    train, _ = train_test_split(train, train_size=N_CLIENTS, random_state=SEED, stratify=train["TARGET"])
    train = train.reset_index(drop=True)
    ids = train["SK_ID_CURR"].values
    y = train["TARGET"].values.astype(np.float32)

    print("构建静态特征 ...")
    static = build_static(train)
    print("构建 credit_card 月度序列 ...")
    cc = pd.read_csv(RAW + "credit_card_balance.csv")
    seq, mask = build_monthly_seq(cc, ids, N_MONTHS)
    print(f"static={static.shape}  seq={seq.shape}  有月度数据占比={mask.sum()/mask.size:.2%}")

    # 训练/验证切分
    idx = np.arange(len(ids))
    tr, va = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

    static_t = torch.tensor(static, device=device)
    seq_t = torch.tensor(seq, device=device)
    mask_t = torch.tensor(mask, device=device)
    y_t = torch.tensor(y, device=device)

    model = H2MTT_V1(static.shape[1], seq.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - y.mean()) / y.mean()], device=device))

    B = 256
    n_epochs = 10
    model.train()
    for ep in range(n_epochs):
        perm = np.random.RandomState(ep).permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), B):
            b = perm[i:i + B]
            opt.zero_grad()
            out = model(static_t[b], seq_t[b], mask_t[b])
            loss = criterion(out, y_t[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 2 == 0:
            print(f"epoch {ep+1}/{n_epochs}  train_loss={tot/len(tr):.4f}", flush=True)

    # 验证
    model.eval()
    with torch.no_grad():
        va_pred = model(static_t[va], seq_t[va], mask_t[va]).cpu().numpy()
        va_auc = roc_auc_score(y[va], va_pred)
    print(f"\nV1 验证 AUC = {va_auc:.4f}  （LightGBM 单表基线约 0.77）", flush=True)


if __name__ == "__main__":
    main()
