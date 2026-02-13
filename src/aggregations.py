"""Группировки по уровням: Артикул, Модель, Color code+Коллекция, Статус, Склейка."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.schema import (
    CALC_ANNUAL_YIELD,
    CALC_MARGIN_RATE,
    METRIC_BASES,
    SUM_METRICS,
    UNIT_METRICS,
    col_p,
)
from src.metrics import compute_margin_rate, compute_annual_yield
from src.utils import safe_div, weighted_avg


# ---------------------------------------------------------------------------
# Агрегация одного периода
# ---------------------------------------------------------------------------

def _agg_period(df: pd.DataFrame, group_cols: list[str], period: str) -> pd.DataFrame:
    """Агрегирует метрики одного периода по указанным группам."""
    agg_dict: dict[str, str | tuple] = {}

    # Суммируемые
    for base in SUM_METRICS:
        c = col_p(base, period)
        if c in df.columns:
            agg_dict[c] = "sum"

    # На единицу — weighted avg по Продажи, шт
    weight_col = col_p("Продажи, шт", period)
    unit_cols_present = [col_p(b, period) for b in UNIT_METRICS if col_p(b, period) in df.columns]

    result = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    # Weighted avg для unit-метрик
    if weight_col in df.columns and unit_cols_present:
        for uc in unit_cols_present:
            wa = df.groupby(group_cols, dropna=False).apply(
                lambda g: weighted_avg(g[uc], g[weight_col]),
                include_groups=False,
            ).reset_index(name=uc)
            result = result.merge(wa, on=group_cols, how="left", suffixes=("", "_wa"))
            # drop duplicate if exists
            dup = f"{uc}_wa"
            if dup in result.columns:
                result[uc] = result[dup]
                result.drop(columns=[dup], inplace=True)

    # --- Пересчёт derived rate metrics ---
    orders_qty = col_p("Заказы, шт", period)
    orders_rub = col_p("Заказы до СПП, ₽", period)
    sales_qty = col_p("Продажи, шт", period)
    sales_rub = col_p("Продажи до СПП, ₽", period)
    ad_int = col_p("Реклама внутр., ₽", period)
    ad_ext = col_p("Реклама внеш., ₽", period)
    margin = col_p("Маржа, ₽", period)
    cost = col_p("Себес-ть, ₽", period)

    # Ср.чек заказа до СПП
    c = col_p("Ср.чек заказа до СПП, ₽", period)
    if orders_rub in result.columns and orders_qty in result.columns:
        result[c] = safe_div(result[orders_rub], result[orders_qty])

    # Ср.чек продажи до СПП
    c = col_p("Ср.чек продажи до СПП, ₽", period)
    if sales_rub in result.columns and sales_qty in result.columns:
        result[c] = safe_div(result[sales_rub], result[sales_qty])

    # Ср.чек заказа после СПП — weighted avg
    c_src = col_p("Ср.чек заказа после СПП, ₽", period)
    if c_src in df.columns and orders_qty in df.columns:
        wa = df.groupby(group_cols, dropna=False).apply(
            lambda g: weighted_avg(g[c_src], g[col_p("Заказы, шт", period)]) if col_p("Заказы, шт", period) in g.columns else np.nan,
            include_groups=False,
        ).reset_index(name=c_src)
        result = result.merge(wa, on=group_cols, how="left", suffixes=("", "_tmp"))
        tmp = f"{c_src}_tmp"
        if tmp in result.columns:
            result[c_src] = result[tmp]
            result.drop(columns=[tmp], inplace=True)

    # Комиссия до СПП, % — weighted avg по sales_rub
    c_comm = col_p("Комиссия до СПП, %", period)
    if c_comm in df.columns and sales_rub in df.columns:
        wa = df.groupby(group_cols, dropna=False).apply(
            lambda g: weighted_avg(g[c_comm], g[sales_rub]) if sales_rub in g.columns else np.nan,
            include_groups=False,
        ).reset_index(name=c_comm)
        result = result.merge(wa, on=group_cols, how="left", suffixes=("", "_tmp"))
        tmp = f"{c_comm}_tmp"
        if tmp in result.columns:
            result[c_comm] = result[tmp]
            result.drop(columns=[tmp], inplace=True)

    # DRR
    total_ad = None
    if ad_int in result.columns and ad_ext in result.columns:
        total_ad = result[ad_int].fillna(0) + result[ad_ext].fillna(0)
    elif ad_int in result.columns:
        total_ad = result[ad_int].fillna(0)
    elif ad_ext in result.columns:
        total_ad = result[ad_ext].fillna(0)

    if total_ad is not None:
        c_drr_ord = col_p("ДРР от заказов (до СПП), %", period)
        if orders_rub in result.columns:
            result[c_drr_ord] = safe_div(total_ad, result[orders_rub]) * 100

        c_drr_sal = col_p("ДРР от продаж (до СПП), %", period)
        if sales_rub in result.columns:
            result[c_drr_sal] = safe_div(total_ad, result[sales_rub]) * 100

    # Маржа до СПП, %
    c_marg_pct = col_p("Маржа до СПП, %", period)
    if margin in result.columns and sales_rub in result.columns:
        result[c_marg_pct] = safe_div(result[margin], result[sales_rub]) * 100

    # Оборот продаж, дни — weighted avg по Продажи до СПП, ₽ (fallback Продажи, шт)
    c_turn_s = col_p("Оборот продаж, дни", period)
    if c_turn_s in df.columns:
        wa = df.groupby(group_cols, dropna=False).apply(
            lambda g: _weighted_turnover(g, c_turn_s, sales_rub, sales_qty),
            include_groups=False,
        ).reset_index(name=c_turn_s)
        result = result.merge(wa, on=group_cols, how="left", suffixes=("", "_tmp"))
        tmp = f"{c_turn_s}_tmp"
        if tmp in result.columns:
            result[c_turn_s] = result[tmp]
            result.drop(columns=[tmp], inplace=True)

    # Оборот заказов, дни — weighted avg по Заказы до СПП, ₽
    c_turn_o = col_p("Оборот заказов, дни", period)
    if c_turn_o in df.columns:
        wa = df.groupby(group_cols, dropna=False).apply(
            lambda g: _weighted_turnover(g, c_turn_o, orders_rub, orders_qty),
            include_groups=False,
        ).reset_index(name=c_turn_o)
        result = result.merge(wa, on=group_cols, how="left", suffixes=("", "_tmp"))
        tmp = f"{c_turn_o}_tmp"
        if tmp in result.columns:
            result[c_turn_o] = result[tmp]
            result.drop(columns=[tmp], inplace=True)

    # Маржа до СПП, % и годовая доходность
    margin_rate_col = col_p(CALC_MARGIN_RATE, period)
    ay_col = col_p(CALC_ANNUAL_YIELD, period)
    if margin in result.columns and sales_rub in result.columns:
        result[margin_rate_col] = compute_margin_rate(result[margin], result[sales_rub])
    else:
        result[margin_rate_col] = np.nan

    turn_col = col_p("Оборот продаж, дни", period)
    if margin_rate_col in result.columns and turn_col in result.columns:
        result[ay_col] = compute_annual_yield(result[margin_rate_col], result[turn_col])
    else:
        result[ay_col] = np.nan

    return result


def _weighted_turnover(g: pd.DataFrame, turn_col: str, primary_weight: str, fallback_weight: str) -> float:
    """Взвешенный оборот: сначала по primary_weight, потом fallback, потом NaN."""
    if turn_col not in g.columns:
        return np.nan
    if primary_weight in g.columns and g[primary_weight].sum() > 0:
        return weighted_avg(g[turn_col], g[primary_weight])
    if fallback_weight in g.columns and g[fallback_weight].sum() > 0:
        return weighted_avg(g[turn_col], g[fallback_weight])
    return np.nan


# ---------------------------------------------------------------------------
# Полная агрегация (п1 + п2 + дельты)
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Агрегация данных по group_cols для обоих периодов + дельты."""
    agg1 = _agg_period(df, group_cols, "п1")
    agg2 = _agg_period(df, group_cols, "п2")

    result = agg1.merge(agg2, on=group_cols, how="outer", suffixes=("", "__p2"))

    # Resolve duplicates: п2-колонки могут оказаться в обоих
    for c in result.columns:
        if c.endswith("__p2"):
            base_col = c[:-4]
            if base_col in result.columns:
                result[base_col] = result[base_col].fillna(result[c])
            else:
                result.rename(columns={c: base_col}, inplace=True)
            if c in result.columns:
                result.drop(columns=[c], inplace=True)

    # Дельты: единый источник — compute_deltas() из metrics.py
    # Δабс = п2 (текущий) − п1 (прошлый); Δ% = (п2−п1)/|п1|×100
    from src.metrics import compute_deltas as _cd
    result = _cd(result)

    return result


# ---------------------------------------------------------------------------
# Итоговая строка по всей площадке
# ---------------------------------------------------------------------------

def total_kpi(df: pd.DataFrame) -> pd.Series:
    """Одна строка KPI по всему DataFrame (без группировки)."""
    df_copy = df.copy()
    df_copy["__all__"] = "Итого"
    agg = aggregate(df_copy, ["__all__"])
    if agg.empty:
        return pd.Series(dtype=float)
    row = agg.iloc[0].drop("__all__")
    return row


# ---------------------------------------------------------------------------
# Удобные обёртки для стандартных группировок
# ---------------------------------------------------------------------------

def agg_by_article(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["Артикул"])


def agg_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегация по Модели. Фильтрует строки с пустой моделью."""
    if df.empty:
        return pd.DataFrame()
    # Фильтруем строки где Модель пустая/NaN
    df_filtered = df[df["Модель"].notna() & (df["Модель"].str.strip() != "")]
    return aggregate(df_filtered, ["Модель"])


def agg_by_color_collection(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["Color code", "Коллекция"])


def agg_by_status(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["Статус"])


def agg_by_glue(df: pd.DataFrame) -> pd.DataFrame:
    """Только для WB — по Склейке."""
    if "Склейка на WB" not in df.columns:
        return pd.DataFrame()
    return aggregate(df, ["Склейка на WB"])


def agg_by_glue_article(df: pd.DataFrame) -> pd.DataFrame:
    """Внутри склейки по артикулам (WB)."""
    if "Склейка на WB" not in df.columns:
        return pd.DataFrame()
    return aggregate(df, ["Склейка на WB", "Артикул"])
