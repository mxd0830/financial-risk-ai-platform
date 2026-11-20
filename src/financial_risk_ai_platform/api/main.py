"""风控模型服务 API：预测 PD + 准入决策 + 额度。

启动：uvicorn financial_risk_ai_platform.api.main:app --reload
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from financial_risk_ai_platform.models.credit_model import CreditModel
from financial_risk_ai_platform.rules.engine import decide

app = FastAPI(title="金融风控模型服务", version="0.1.0")
model = CreditModel()


class PredictRequest(BaseModel):
    features: dict[str, float | str | None]  # 原始客户特征（application_train 列）


def _num(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    f = req.features
    df = pd.DataFrame([f])
    pd_val = float(model.predict(df)[0])

    days_birth = _num(f, "DAYS_BIRTH")
    age = -days_birth / 365.25 if days_birth is not None else None
    income = _num(f, "AMT_INCOME_TOTAL")
    requested = _num(f, "AMT_CREDIT")

    return decide(pd_val, age=age, income=income, requested=requested)
