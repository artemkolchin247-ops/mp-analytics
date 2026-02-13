"""Витрины, KPI и агрегации рекламы WB (из API)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.wb_api.adapters import ADS_PREFIX
from src.utils import safe_div, safe_pct_change


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------

def _ap(base: str, period: str) -> str:
    """Имя ads-колонки: A_{base} {period}."""
    return f"{ADS_PREFIX}{base} {period}"


def has_ads_cols(df: Optional[pd.DataFrame]) -> bool:
    """Проверяет, есть ли ads-колонки в DataFrame."""
    if df is None or df.empty:
        return False
    return any(c.startswith(ADS_PREFIX) for c in df.columns)


# ---------------------------------------------------------------------------
# A) WB Ads Summary — KPI (п1, п2, Δ)
# ---------------------------------------------------------------------------

def build_ads_kpi(df: pd.DataFrame) -> pd.DataFrame:
    """Итоговые KPI рекламы WB (п1, п2, Δабс, Δ%)."""
    display_metrics = [
        "views", "clicks", "spend", "orders", "atbs",
        "true_ctr", "cpc", "cpm", "cr_ads", "cart_rate_ads",
    ]
    labels = {
        "views": "Показы (ads)",
        "clicks": "Клики (ads)",
        "spend": "Затраты, ₽",
        "orders": "Заказы из рекламы",
        "atbs": "Корзина из рекламы",
        "true_ctr": "True CTR (ads), %",
        "cpc": "CPC, ₽",
        "cpm": "CPM, ₽",
        "cr_ads": "CR (ads), %",
        "cart_rate_ads": "CartRate (ads), %",
    }

    # Суммируем по всем артикулам для sum-метрик, пересчитываем rate
    sum_bases = ["views", "clicks", "spend", "orders", "atbs", "shks", "canceled", "sum_price"]

    rows = []
    for m in display_metrics:
        c1 = _ap(m, "п1")
        c2 = _ap(m, "п2")
        da = f"{ADS_PREFIX}{m} Δабс"
        dp = f"{ADS_PREFIX}{m} Δ%"

        if m in sum_bases:
            v1 = df[c1].sum() if c1 in df.columns else np.nan
            v2 = df[c2].sum() if c2 in df.columns else np.nan
        else:
            # Rate-метрики: пересчитываем из суммированных
            v1 = _compute_total_rate(df, m, "п1")
            v2 = _compute_total_rate(df, m, "п2")

        d_abs = v2 - v1 if not (np.isnan(v1) or np.isnan(v2)) else np.nan
        d_pct = safe_div(d_abs, abs(v1)) * 100 if not (np.isnan(d_abs) or v1 == 0) else np.nan

        rows.append({
            "Метрика": labels.get(m, m),
            "п1": v1,
            "п2": v2,
            "Δабс": d_abs,
            "Δ%": d_pct,
        })
    return pd.DataFrame(rows)


def _compute_total_rate(df: pd.DataFrame, metric: str, period: str) -> float:
    """Пересчитывает rate-метрику из суммированных абсолютных значений."""
    views = df[_ap("views", period)].sum() if _ap("views", period) in df.columns else 0
    clicks = df[_ap("clicks", period)].sum() if _ap("clicks", period) in df.columns else 0
    spend = df[_ap("spend", period)].sum() if _ap("spend", period) in df.columns else 0
    orders = df[_ap("orders", period)].sum() if _ap("orders", period) in df.columns else 0
    atbs = df[_ap("atbs", period)].sum() if _ap("atbs", period) in df.columns else 0

    if metric == "true_ctr":
        return safe_div(clicks, views) * 100 if views else np.nan
    if metric == "cpc":
        return safe_div(spend, clicks) if clicks else np.nan
    if metric == "cpm":
        return safe_div(spend, views) * 1000 if views else np.nan
    if metric == "cr_ads":
        return safe_div(orders, clicks) * 100 if clicks else np.nan
    if metric == "cart_rate_ads":
        return safe_div(atbs, clicks) * 100 if clicks else np.nan
    return np.nan


# ---------------------------------------------------------------------------
# B) Ads по артикулам (nmId → Артикул)
# ---------------------------------------------------------------------------

def build_ads_by_article(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Таблица эффективности рекламы по артикулам + флаги деградации."""
    if not has_ads_cols(df):
        return pd.DataFrame()

    cols = ["Артикул"]
    if "nmId" in df.columns:
        cols.append("nmId")

    ads_show = [
        _ap("spend", "п2"), f"{ADS_PREFIX}spend Δабс",
        _ap("views", "п2"), _ap("clicks", "п2"),
        _ap("orders", "п2"), f"{ADS_PREFIX}orders Δабс",
        _ap("true_ctr", "п2"), f"{ADS_PREFIX}true_ctr Δабс",
        _ap("cpc", "п2"), f"{ADS_PREFIX}cpc Δабс",
        _ap("cr_ads", "п2"), f"{ADS_PREFIX}cr_ads Δабс",
    ]
    for c in ads_show:
        if c in df.columns:
            cols.append(c)

    if len(cols) <= 1:
        return pd.DataFrame()

    result = df[[c for c in cols if c in df.columns]].copy()
    result = result[result.get(_ap("spend", "п2"), pd.Series(dtype=float)).notna()].copy()

    if result.empty:
        return pd.DataFrame()

    # Диагностические флаги
    flags = pd.Series("", index=result.index)

    spend_d = f"{ADS_PREFIX}spend Δабс"
    orders_d = f"{ADS_PREFIX}orders Δабс"
    ctr_d = f"{ADS_PREFIX}true_ctr Δабс"
    cpc_d = f"{ADS_PREFIX}cpc Δабс"
    cr_d = f"{ADS_PREFIX}cr_ads Δабс"

    if spend_d in result.columns and orders_d in result.columns:
        mask = (result[spend_d].fillna(0) > 0) & (result[orders_d].fillna(0) <= 0)
        flags = flags.where(~mask, flags + "🔴Spend↑Orders↓ ")

    if ctr_d in result.columns:
        mask = result[ctr_d].fillna(0) < -1
        flags = flags.where(~mask, flags + "🟠CTR↓ ")

    if cpc_d in result.columns and cr_d in result.columns:
        mask = (result[cpc_d].fillna(0) > 0) & (result[cr_d].fillna(0) < 0)
        flags = flags.where(~mask, flags + "🟡CPC↑CR↓ ")

    result["Флаги"] = flags.str.strip()

    sort_c = _ap("spend", "п2")
    if sort_c in result.columns:
        result = result.sort_values(sort_c, ascending=False, na_position="last")

    return result.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# C) Ads × Funnel × Price
# ---------------------------------------------------------------------------

def build_ads_funnel_price(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Таблица: реклама × воронка × цена для сопоставления."""
    if not has_ads_cols(df):
        return pd.DataFrame()

    from src.wb_funnel_metrics import _fp
    from src.wb_funnel_io import CALC_CONV_CART, CALC_CONV_ORDER

    cols = ["Артикул"]
    wanted = [
        _ap("spend", "п2"), _ap("true_ctr", "п2"), _ap("cr_ads", "п2"),
        _fp(CALC_CONV_CART, "п2"), _fp(CALC_CONV_ORDER, "п2"),
        "Ср.чек заказа после СПП, ₽ п2", "Ср.чек заказа после СПП, ₽ Δабс",
        _ap("orders", "п2"), "Заказы, шт п2",
    ]
    for c in wanted:
        if c in df.columns:
            cols.append(c)

    if len(cols) <= 1:
        return pd.DataFrame()

    result = df[[c for c in cols if c in df.columns]].copy()
    result = result[result.get(_ap("spend", "п2"), pd.Series(dtype=float)).notna()].copy()

    sort_c = _ap("spend", "п2")
    if sort_c in result.columns:
        result = result.sort_values(sort_c, ascending=False, na_position="last")

    return result.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Агрегации ads по уровням
# ---------------------------------------------------------------------------

_ADS_SUM_BASES = ["views", "clicks", "spend", "orders", "atbs", "shks", "canceled", "sum_price"]
_ADS_RATE_BASES = ["true_ctr", "cpc", "cpm", "cr_ads", "cart_rate_ads"]


def aggregate_ads(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Агрегирует ads-метрики по group_cols. Суммы суммируются, rate пересчитываются."""
    if df.empty or not has_ads_cols(df):
        return pd.DataFrame()

    agg_dict: dict[str, str] = {}
    for period in ("п1", "п2"):
        for base in _ADS_SUM_BASES:
            c = _ap(base, period)
            if c in df.columns:
                agg_dict[c] = "sum"

    if not agg_dict:
        return pd.DataFrame()

    valid_groups = [c for c in group_cols if c in df.columns]
    if not valid_groups:
        return pd.DataFrame()

    result = df.groupby(valid_groups, dropna=False).agg(agg_dict).reset_index()

    # Пересчёт rate из агрегированных сумм
    for period in ("п1", "п2"):
        views = _ap("views", period)
        clicks = _ap("clicks", period)
        spend = _ap("spend", period)
        orders = _ap("orders", period)
        atbs = _ap("atbs", period)

        if views in result.columns and clicks in result.columns:
            result[_ap("true_ctr", period)] = safe_div(result[clicks], result[views]) * 100
        if spend in result.columns and clicks in result.columns:
            result[_ap("cpc", period)] = safe_div(result[spend], result[clicks])
            result[_ap("cpm", period)] = safe_div(result[spend], result[views]) * 1000
        if orders in result.columns and clicks in result.columns:
            result[_ap("cr_ads", period)] = safe_div(result[orders], result[clicks]) * 100
        if atbs in result.columns and clicks in result.columns:
            result[_ap("cart_rate_ads", period)] = safe_div(result[atbs], result[clicks]) * 100

    # Дельты
    for base in _ADS_SUM_BASES + _ADS_RATE_BASES:
        c1 = _ap(base, "п1")
        c2 = _ap(base, "п2")
        if c1 in result.columns and c2 in result.columns:
            result[f"{ADS_PREFIX}{base} Δабс"] = result[c2] - result[c1]
            result[f"{ADS_PREFIX}{base} Δ%"] = safe_pct_change(result[c2], result[c1])

    return result


def ads_agg_by_model(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_ads(df, ["Модель"])


def ads_agg_by_status(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_ads(df, ["Статус"])


def ads_agg_by_color_collection(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_ads(df, ["Color code", "Коллекция"])


def ads_agg_by_glue(df: pd.DataFrame) -> pd.DataFrame:
    if "Склейка на WB" not in df.columns:
        return pd.DataFrame()
    return aggregate_ads(df, ["Склейка на WB"])
