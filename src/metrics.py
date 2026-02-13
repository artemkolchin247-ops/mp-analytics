"""Расчёт маржи до СПП, годовой доходности, дельт (п2 vs п1). п2 — текущий, п1 — прошлый."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import (
    CALC_ANNUAL_YIELD,
    CALC_MARGIN_RATE,
    METRIC_BASES,
    col_p,
)
from src.utils import safe_div, safe_pct_change


# ---------------------------------------------------------------------------
# Маржа до СПП и годовая доходность — построчно
# ---------------------------------------------------------------------------

def compute_margin_rate(margin: pd.Series | float, sales: pd.Series | float) -> pd.Series | float:
    """Маржа до СПП, % = (Маржа / Продажи) * 100."""
    return safe_div(margin, sales) * 100


def compute_annual_yield(roi: pd.Series | float, turnover_days: pd.Series | float) -> pd.Series | float:
    """Годовая доходность, % = Маржа до СПП, % / Оборот_дни * 365."""
    return safe_div(roi, turnover_days) * 365





def add_calculated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет Маржу до СПП, % и Годовую доходность для п1 и п2."""
    df = df.copy()
    for p in ("п1", "п2"):
        margin = col_p("Маржа, ₽", p)
        sales = col_p("Продажи до СПП, ₽", p)
        turn = col_p("Оборот продаж, дни", p)
        margin_rate_col = col_p(CALC_MARGIN_RATE, p)
        ay_col = col_p(CALC_ANNUAL_YIELD, p)

        if margin in df.columns and sales in df.columns:
            df[margin_rate_col] = compute_margin_rate(df[margin], df[sales])
        if turn in df.columns and margin_rate_col in df.columns:
            df[ay_col] = compute_annual_yield(df[margin_rate_col], df[turn])

    # Удаляем временные колонки, если были
    tmp_cols = [c for c in df.columns if c.startswith("_tmp_")]
    if tmp_cols:
        df.drop(columns=tmp_cols, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Дельты (п2 − п1): п2=текущий, п1=прошлый
# ---------------------------------------------------------------------------

ALL_DELTA_BASES = METRIC_BASES + [CALC_MARGIN_RATE, CALC_ANNUAL_YIELD]


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет Δабс и Δ% для всех метрик. Δ = п2 (текущий) − п1 (прошлый)."""
    df = df.copy()
    for base in ALL_DELTA_BASES:
        c1 = col_p(base, "п1")
        c2 = col_p(base, "п2")
        if c1 not in df.columns or c2 not in df.columns:
            continue
        df[f"{base} Δабс"] = df[c2] - df[c1]
        df[f"{base} Δ%"] = safe_pct_change(df[c2], df[c1])
    return df


def compute_deltas_series(row_p1: pd.Series, row_p2: pd.Series) -> pd.Series:
    """Дельты для одной пары агрегированных строк. Δ = п2 (текущий) − п1 (прошлый)."""
    delta_abs = row_p2 - row_p1
    delta_pct = safe_pct_change(row_p2, row_p1)
    return pd.concat([
        row_p1.rename(lambda x: f"{x} п1"),
        row_p2.rename(lambda x: f"{x} п2"),
        delta_abs.rename(lambda x: f"{x} Δабс"),
        delta_pct.rename(lambda x: f"{x} Δ%"),
    ])
