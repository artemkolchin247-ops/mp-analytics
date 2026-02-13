"""Утилиты общего назначения."""
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(numerator, denominator, default=np.nan):
    """Безопасное деление: если знаменатель 0 или NaN → default."""
    if isinstance(numerator, (pd.Series, np.ndarray)):
        result = pd.Series(np.where(
            (denominator == 0) | pd.isna(denominator) | pd.isna(numerator),
            default,
            numerator / denominator,
        ), index=getattr(numerator, "index", None))
        return result
    if pd.isna(denominator) or denominator == 0:
        return default
    if pd.isna(numerator):
        return default
    return numerator / denominator


def safe_pct_change(val_new, val_old):
    """Δ% = (new − old) / |old| * 100. Если old=0 → NaN."""
    delta_abs = val_new - val_old
    return safe_div(delta_abs, np.abs(val_old)) * 100


def fmt_number(val, decimals: int = 1, suffix: str = "") -> str:
    """Форматирование числа для UI."""
    if pd.isna(val):
        return "—"
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:,.{decimals}f}M{suffix}"
    if abs(val) >= 1_000:
        return f"{val / 1_000:,.{decimals}f}K{suffix}"
    return f"{val:,.{decimals}f}{suffix}"


def fmt_pct(val, decimals: int = 1) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:+.{decimals}f}%"


def weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    """Взвешенное среднее. Если сумма весов = 0 → NaN."""
    mask = values.notna() & weights.notna()
    v = values[mask]
    w = weights[mask]
    total_w = w.sum()
    if total_w == 0:
        return np.nan
    return (v * w).sum() / total_w


def error_tag(val, tag: str = "ошибка данных") -> str | float:
    """Возвращает пометку если val NaN."""
    if pd.isna(val):
        return tag
    return val
