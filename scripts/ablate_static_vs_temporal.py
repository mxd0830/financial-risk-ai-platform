"""消融：纯静态 MLP vs 静态+时序（H²MTT V1），定位时序分支是否有增量。

同一批 50K 数据、同一 129 静态特征、同一训练配置，只改「有没有时序分支」。
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train_baseline import SEED
from train_h2mtt_v1 import H2MTT_V1, N_MONTHS, build_monthly_seq, build_static

RAW = "data/raw/"
N_CLIENTS = 50000
D_MODEL = 128
N_EPOCHS = 10
BATCH = 256


class StaticOnly(nn.Module):
    """纯静态：只有主表 129 特征 → MLP → 头。"""

    def __init__(self, static_dim: int, d_model: int = D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(static_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, static: torch.Tensor):
        return self.head(self.mlp(static)).squeeze(-1)


def train_eval(model, inputs, y_t, tr, va, device, n_epochs=N_EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([(1 - y_t.mean()) / y_t.mean()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for ep in range(n_epochs):
        perm = np.random.RandomState(ep).permutation(tr)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(*[x[b] for x in inputs])
            loss = criterion(out, y_t[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(*[x[va] for x in inputs]).cpu().numpy()
        pred = np.nan_to_num(pred, nan=0.0)
    return roc_auc_score(y_t[va].cpu().numpy(), pred)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    train = pd.read_csv(RAW + "application_train.csv")
    train, _ = train_test_split(train, train_size=N_CLIENTS, random_state=SEED, stratify=train["TARGET"])
    train = train.reset_index(drop=True)
    ids = train["SK_ID_CURR"].values
    y = train["TARGET"].values.astype(np.float32)

    static = build_static(train)
    cc = pd.read_csv(RAW + "credit_card_balance.csv")
    seq, mask = build_monthly_seq(cc, ids, N_MONTHS)

    idx = np.arange(len(ids))
    tr, va = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

    static_t = torch.tensor(static, device=device)
    seq_t = torch.tensor(seq, device=device)
    mask_t = torch.tensor(mask, device=device)
    y_t = torch.tensor(y, device=device)

    # ① 纯静态
    m_static = StaticOnly(static.shape[1]).to(device)
    auc_static = train_eval(m_static, [static_t], y_t, tr, va, device)
    print(f"\n① 纯静态 MLP           AUC = {auc_static:.4f}")

    # ② 静态 + 时序（完整 V1）
    m_full = H2MTT_V1(static.shape[1], seq.shape[2]).to(device)
    auc_full = train_eval(m_full, [static_t, seq_t, mask_t], y_t, tr, va, device)
    print(f"② 静态 + 时序（V1）    AUC = {auc_full:.4f}")

    print(f"\n时序分支增量 = {auc_full - auc_static:+.4f}  "
          f"（正数=时序有用，负数=时序拖后腿）")


if __name__ == "__main__":
    main()
