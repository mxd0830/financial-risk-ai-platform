"""风控规则引擎：模拟准入规则 + 额度规则。

确定性计算，与 LLM 分工：准入/额度由本模块计算，LLM 只负责解释和报告。
规则为「模拟策略」，用于演示，不代表真实生产策略。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 准入阈值（模拟策略，可调）
PD_REJECT_THRESHOLD = 0.20      # PD ≥ 20% 直接拒绝
PD_MANUAL_REVIEW = 0.05         # PD ∈ [5%, 20%) 转人工复核
MIN_AGE = 18                    # 最小年龄
MIN_INCOME = 50_000             # 最低年收入（Home Credit 口径，单位本币）

# 额度系数：按 PD 档位决定「额度/年收入」上限（模拟策略）
LIMIT_FACTORS = [
    (0.02, 0.50),   # PD < 2%    → 额度 ≤ 年收入 × 0.50
    (0.05, 0.30),   # PD < 5%    → ≤ × 0.30
    (0.10, 0.15),   # PD < 10%   → ≤ × 0.15
    (0.20, 0.05),   # PD < 20%   → ≤ × 0.05
]


@dataclass
class AdmissionResult:
    decision: str                        # approve / manual_review / reject
    reasons: list[str] = field(default_factory=list)


def admission_decision(pd: float, age: float | None, income: float | None) -> AdmissionResult:
    """准入决策：硬规则 → 模型 PD 阈值 → 人工复核。

    硬规则（缺关键信息或明显不合规）优先于模型结果。
    """
    if age is None or age < MIN_AGE:
        return AdmissionResult("reject", ["年龄未满 18 或缺失"])
    if income is None or income < MIN_INCOME:
        return AdmissionResult("reject", ["年收入低于最低门槛"])

    if pd >= PD_REJECT_THRESHOLD:
        return AdmissionResult("reject", [f"模型 PD={pd:.1%} ≥ {PD_REJECT_THRESHOLD:.0%}（高风险）"])
    if pd >= PD_MANUAL_REVIEW:
        return AdmissionResult("manual_review", [f"模型 PD={pd:.1%} 处于人工复核区间 [{PD_MANUAL_REVIEW:.0%}, {PD_REJECT_THRESHOLD:.0%})"])
    return AdmissionResult("approve", [])


def credit_limit(pd: float, income: float, requested: float) -> float:
    """额度：min(申请额, 年收入 × PD 档位系数)，并向下取整到千位。"""
    factor = LIMIT_FACTORS[-1][1]
    for threshold, f in LIMIT_FACTORS:
        if pd < threshold:
            factor = f
            break
    limit = min(requested, income * factor)
    return float(round(limit / 1000) * 1000)  # 取整到千


def decide(pd: float, *, age: float | None, income: float | None, requested: float | None) -> dict:
    """一站式决策：准入 + 额度。"""
    adm = admission_decision(pd, age, income)
    result = {
        "pd": round(float(pd), 4),
        "decision": adm.decision,
        "reasons": adm.reasons,
        "credit_limit": None,
    }
    if adm.decision == "approve" and requested is not None and income is not None:
        result["credit_limit"] = credit_limit(pd, income, requested)
    return result
