"""Адаптеры: приведение API DataFrame к форматам существующего пайплайна."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.wb_api.normalize import normalize_vendor_code, normalize_nm_id
from src.wb_funnel_io import (
    CALC_CTR,
    CALC_CONV_CART,
    CALC_CONV_ORDER,
    CALC_PURCHASE_RATE,
    FUNNEL_ID_COL,
    normalize_article_key,
)
from src.wb_funnel_metrics import _fp
from src.utils import safe_div, safe_pct_change


# ---------------------------------------------------------------------------
# Funnel API → формат, совместимый с существующим join_funnel_to_economics
# ---------------------------------------------------------------------------

def api_funnel_to_excel_format(df_api: pd.DataFrame) -> pd.DataFrame:
    """Конвертирует DataFrame из funnel.py (API) в формат, ожидаемый wb_funnel_io.

    Существующий пайплайн ожидает колонки:
    - "Артикул продавца" (= vendorCode)
    - "Показы", "Переходы в карточку", "Положили в корзину", "Заказали, шт",
      "Выкупили, шт", "Отменили, шт", + суммовые
    """
    if df_api.empty:
        return pd.DataFrame()

    result = pd.DataFrame()
    result[FUNNEL_ID_COL] = df_api["vendorCode"]
    result["Артикул WB"] = df_api["nmId"]

    # Маппинг API → Excel-колонки (для selected period — п2, текущий)
    _map_period(df_api, result, prefix="sel")
    return result


def api_funnel_to_excel_format_pair(
    df_api: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет API funnel (с sel_* и past_*) на два DF: п1 (прошлый) и п2 (текущий).

    Каждый DF имеет формат, совместимый с load_funnel_excel output.
    sel (selected) → п2 (текущий), past → п1 (прошлый).
    """
    if df_api.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_p1 = pd.DataFrame()
    df_p1[FUNNEL_ID_COL] = df_api["vendorCode"]
    df_p1["Артикул WB"] = df_api["nmId"]
    _map_period(df_api, df_p1, prefix="past")

    df_p2 = pd.DataFrame()
    df_p2[FUNNEL_ID_COL] = df_api["vendorCode"]
    df_p2["Артикул WB"] = df_api["nmId"]
    _map_period(df_api, df_p2, prefix="sel")

    return df_p1, df_p2


def _map_period(src: pd.DataFrame, dst: pd.DataFrame, prefix: str) -> None:
    """Маппит колонки одного периода из API-формата в Excel-формат."""
    mapping = {
        f"{prefix}_openCount": "Показы",
        f"{prefix}_cartCount": "Положили в корзину",
        f"{prefix}_orderCount": "Заказали, шт",
        f"{prefix}_orderSum": "Заказали на сумму, ₽",
        f"{prefix}_buyoutCount": "Выкупили, шт",
        f"{prefix}_buyoutSum": "Выкупили на сумму, ₽",
        f"{prefix}_cancelCount": "Отменили, шт",
        f"{prefix}_cancelSum": "Отменили на сумму, ₽",
        f"{prefix}_addToWishlist": "Добавили в отложенные",
    }
    for api_col, excel_col in mapping.items():
        if api_col in src.columns:
            dst[excel_col] = src[api_col].values
        else:
            dst[excel_col] = np.nan

    # "Переходы в карточку" — в API это openCount * addToCartPct / 100 ?
    # Нет: openCount = показы (просмотры карточки). cartCount = корзина.
    # Но в Excel-воронке "Переходы в карточку" — это клики по карточке в поиске.
    # В API v3 openCount = число открытий карточки (= переходы в карточку).
    # Поэтому Показы = openCount (переходы/просмотры карточки).
    # Но в Excel "Показы" — это показы в каталоге, а "Переходы в карточку" — открытия.
    # В API v3 НЕТ отдельного поля "показы в каталоге" — openCount = переходы в карточку.
    # Ставим: "Показы" и "Переходы в карточку" = openCount (best effort).
    dst["Переходы в карточку"] = dst["Показы"].values


# ---------------------------------------------------------------------------
# Ads API → формат для wb_ads_metrics
# ---------------------------------------------------------------------------

ADS_PREFIX = "A_"


def ads_df_with_prefix(df_ads: pd.DataFrame, period: str) -> pd.DataFrame:
    """Добавляет суффикс периода к ads-колонкам: A_{metric} {period}.

    Колонки на входе: nmId, name, views, clicks, spend, orders, atbs, shks,
                      canceled, sum_price, true_ctr, cpc, cpm, cr_ads, cart_rate_ads
    """
    if df_ads.empty:
        return pd.DataFrame()

    result = pd.DataFrame()
    result["nmId"] = df_ads["nmId"]
    if "name" in df_ads.columns:
        result["ads_name"] = df_ads["name"]

    ads_cols = [
        "views", "clicks", "spend", "orders", "atbs", "shks",
        "canceled", "sum_price", "true_ctr", "cpc", "cpm", "cr_ads", "cart_rate_ads",
    ]
    for c in ads_cols:
        if c in df_ads.columns:
            result[f"{ADS_PREFIX}{c} {period}"] = df_ads[c].values

    return result


def merge_ads_periods(df_p1: pd.DataFrame, df_p2: pd.DataFrame) -> pd.DataFrame:
    """Объединяет ads п1 и п2 по nmId + вычисляет дельты."""
    if df_p1.empty and df_p2.empty:
        return pd.DataFrame()
    if df_p1.empty:
        return df_p2
    if df_p2.empty:
        return df_p1

    df = df_p1.merge(df_p2, on="nmId", how="outer", suffixes=("", "_drop"))
    # Сохраняем ads_name из п1
    if "ads_name_drop" in df.columns:
        df["ads_name"] = df["ads_name"].fillna(df["ads_name_drop"])
        df.drop(columns=["ads_name_drop"], inplace=True, errors="ignore")

    # Дельты
    sum_metrics = ["views", "clicks", "spend", "orders", "atbs", "shks", "canceled", "sum_price"]
    rate_metrics = ["true_ctr", "cpc", "cpm", "cr_ads", "cart_rate_ads"]

    for m in sum_metrics + rate_metrics:
        c1 = f"{ADS_PREFIX}{m} п1"
        c2 = f"{ADS_PREFIX}{m} п2"
        if c1 in df.columns and c2 in df.columns:
            df[f"{ADS_PREFIX}{m} Δабс"] = df[c2] - df[c1]
            df[f"{ADS_PREFIX}{m} Δ%"] = safe_pct_change(df[c2], df[c1])

    return df


# ---------------------------------------------------------------------------
# Join ads → funnel → economics (по nmId → vendorCode → Артикул)
# ---------------------------------------------------------------------------

def join_ads_to_economics(
    df_econ: pd.DataFrame,
    df_ads_merged: pd.DataFrame,
    df_funnel_api: Optional[pd.DataFrame] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Присоединяет ads к экономике через nmId → vendorCode → Артикул.

    Steps:
    1. nmId → vendorCode через funnel_api (если есть)
    2. vendorCode → Артикул через normalize
    3. Overrides: nmId → Артикул напрямую

    Returns:
        (df_merged, coverage_info)
    """
    if df_ads_merged.empty or df_econ.empty:
        return df_econ, {}

    overrides = overrides or {}
    coverage = {
        "ads_nm_total": len(df_ads_merged),
        "matched_via_funnel": 0,
        "matched_via_override": 0,
        "unmatched": 0,
        "has_ads": True,
    }

    # Нормализованный ключ экономики
    econ = df_econ.copy()
    econ["_art_key"] = econ["Артикул"].apply(normalize_article_key)

    # Шаг 1: nmId → vendorCode через funnel
    nm_to_vc: Dict[str, str] = {}
    if df_funnel_api is not None and not df_funnel_api.empty:
        for _, row in df_funnel_api.iterrows():
            nm = normalize_nm_id(row.get("nmId"))
            vc = str(row.get("vendorCode", "")).strip()
            if nm and vc:
                nm_to_vc[nm] = vc

    # Шаг 2: маппинг nmId → _art_key
    ads = df_ads_merged.copy()
    ads["_nm_str"] = ads["nmId"].apply(normalize_nm_id)

    def _resolve(row):
        nm = row["_nm_str"]
        # Override
        if nm in overrides:
            return normalize_article_key(overrides[nm]), "override"
        # Via funnel vendorCode
        vc = nm_to_vc.get(nm)
        if vc:
            return normalize_vendor_code(vc), "funnel"
        return "", "unmatched"

    resolved = ads.apply(_resolve, axis=1, result_type="expand")
    ads["_art_key"] = resolved[0]
    ads["_match_type"] = resolved[1]

    coverage["matched_via_funnel"] = int((ads["_match_type"] == "funnel").sum())
    coverage["matched_via_override"] = int((ads["_match_type"] == "override").sum())
    coverage["unmatched"] = int((ads["_match_type"] == "unmatched").sum())

    # Вычисляем % spend matched
    total_spend_col = f"{ADS_PREFIX}spend п2"
    if total_spend_col in ads.columns:
        total_spend = ads[total_spend_col].sum()
        matched_spend = ads.loc[ads["_art_key"] != "", total_spend_col].sum()
        coverage["spend_total"] = float(total_spend)
        coverage["spend_matched"] = float(matched_spend)
        coverage["spend_matched_pct"] = round(matched_spend / total_spend * 100, 1) if total_spend > 0 else 0

    # Join
    ads_for_join = ads[ads["_art_key"] != ""].copy()
    if not ads_for_join.empty:
        join_cols = [c for c in ads_for_join.columns if c not in ("_nm_str", "_match_type")]
        econ = econ.merge(
            ads_for_join[join_cols],
            on="_art_key",
            how="left",
        )

    return econ, coverage


def get_unmatched_ads(
    df_ads_merged: pd.DataFrame,
    df_funnel_api: Optional[pd.DataFrame] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Возвращает DataFrame непривязанных nmId с суммой spend."""
    if df_ads_merged.empty:
        return pd.DataFrame()

    overrides = overrides or {}
    nm_to_vc: Dict[str, str] = {}
    if df_funnel_api is not None and not df_funnel_api.empty:
        for _, row in df_funnel_api.iterrows():
            nm = normalize_nm_id(row.get("nmId"))
            vc = str(row.get("vendorCode", "")).strip()
            if nm and vc:
                nm_to_vc[nm] = vc

    rows = []
    for _, row in df_ads_merged.iterrows():
        nm = normalize_nm_id(row.get("nmId"))
        if nm in overrides:
            continue
        if nm in nm_to_vc:
            continue
        r = {"nmId": row.get("nmId"), "ads_name": row.get("ads_name", "")}
        spend_col = f"{ADS_PREFIX}spend п2"
        if spend_col in df_ads_merged.columns:
            r["spend_p2"] = row.get(spend_col, 0)
        rows.append(r)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("spend_p2", ascending=False, na_position="last").reset_index(drop=True)
