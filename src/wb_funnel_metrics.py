"""Join воронки WB с экономикой, coverage, агрегации воронки на уровнях."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.wb_funnel_io import (
    CALC_CONV_CART,
    CALC_CONV_ORDER,
    CALC_CTR,
    CALC_PURCHASE_RATE,
    FUNNEL_ID_COL,
    FUNNEL_SUM_COLS,
    aggregate_funnel_by_article,
    normalize_article_key,
)
from src.schema import col_p
from src.utils import safe_div, safe_pct_change


# ---------------------------------------------------------------------------
# Колонки воронки с суффиксом периода
# ---------------------------------------------------------------------------

FUNNEL_DISPLAY_METRICS: list[str] = [
    "Показы",
    "Переходы в карточку",
    "Положили в корзину",
    "Заказали, шт",
    "Выкупили, шт",
    "Отменили, шт",
    CALC_CTR,
    CALC_CONV_CART,
    CALC_CONV_ORDER,
    CALC_PURCHASE_RATE,
]

FUNNEL_SUM_BASES: list[str] = [
    "Показы",
    "Переходы в карточку",
    "Положили в корзину",
    "Добавили в отложенные",
    "Заказали, шт",
    "Заказали ВБ клуб, шт",
    "Выкупили, шт",
    "Выкупили ВБ клуб, шт",
    "Отменили, шт",
    "Отменили ВБ клуб, шт",
    "Заказали на сумму, ₽",
    "Заказали на сумму ВБ клуб, ₽",
    "Выкупили на сумму, ₽",
    "Выкупили на сумму ВБ клуб, ₽",
    "Отменили на сумму, ₽",
    "Отменили на сумму ВБ клуб, ₽",
]

FUNNEL_RATE_BASES: list[str] = [
    CALC_CTR,
    CALC_CONV_CART,
    CALC_CONV_ORDER,
    CALC_PURCHASE_RATE,
]


def _fp(base: str, period: str) -> str:
    """Имя колонки воронки с суффиксом периода."""
    return f"F_{base} {period}"


# ---------------------------------------------------------------------------
# Join воронки с экономикой WB
# ---------------------------------------------------------------------------

def join_funnel_to_economics(
    df_econ: pd.DataFrame,
    funnel_p1: Optional[pd.DataFrame],
    funnel_p2: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Присоединяет агрегированную воронку (п1 и/или п2) к экономическим данным WB.

    Returns:
        (df_merged, coverage_info)
    """
    df = df_econ.copy()
    df["_art_key"] = df["Артикул"].apply(normalize_article_key)

    coverage: Dict[str, any] = {
        "econ_total": len(df),
        "funnel_p1_total": 0,
        "funnel_p2_total": 0,
        "matched_p1": 0,
        "matched_p2": 0,
    }

    _period_key = {"п1": "p1", "п2": "p2"}
    for period, funnel_raw in [("п1", funnel_p1), ("п2", funnel_p2)]:
        pk = _period_key[period]
        if funnel_raw is None or funnel_raw.empty:
            continue

        funnel_agg = aggregate_funnel_by_article(funnel_raw)
        if funnel_agg.empty:
            continue

        coverage[f"funnel_{pk}_total"] = len(funnel_agg)

        # Переименовываем колонки с суффиксом периода
        rename_map: dict[str, str] = {}
        for c in funnel_agg.columns:
            if c in ("_art_key", FUNNEL_ID_COL):
                continue
            rename_map[c] = _fp(c, period)
        funnel_renamed = funnel_agg.rename(columns=rename_map)

        # Join
        new_funnel_cols = list(rename_map.values())
        df = df.merge(
            funnel_renamed.drop(columns=[FUNNEL_ID_COL], errors="ignore"),
            on="_art_key",
            how="left",
        )
        # Подсчёт совпавших: хотя бы одна funnel-колонка не NaN
        check_cols = [c for c in new_funnel_cols if c in df.columns]
        if check_cols:
            matched = df[check_cols].notna().any(axis=1).sum()
        else:
            matched = 0
        coverage[f"matched_{pk}"] = int(matched)

    # Дельты воронки (п2 − п1): п2=текущий, п1=прошлый
    df = _compute_funnel_deltas(df)

    return df, coverage


def _compute_funnel_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет Δабс и Δ% для метрик воронки. Δ = п2 (текущий) − п1 (прошлый)."""
    df = df.copy()
    all_bases = FUNNEL_SUM_BASES + FUNNEL_RATE_BASES
    for base in all_bases:
        c1 = _fp(base, "п1")
        c2 = _fp(base, "п2")
        if c1 in df.columns and c2 in df.columns:
            df[f"F_{base} Δабс"] = df[c2] - df[c1]
            df[f"F_{base} Δ%"] = safe_pct_change(df[c2], df[c1])
    return df


# ---------------------------------------------------------------------------
# Агрегация воронки на произвольном уровне
# ---------------------------------------------------------------------------

def aggregate_funnel(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Агрегирует метрики воронки по group_cols.

    - Суммы: суммируются.
    - Конверсии: пересчитываются из агрегированных сумм.
    """
    if df.empty:
        return pd.DataFrame()

    agg_dict: dict[str, str] = {}
    for period in ("п1", "п2"):
        for base in FUNNEL_SUM_BASES:
            c = _fp(base, period)
            if c in df.columns:
                agg_dict[c] = "sum"

    if not agg_dict:
        return pd.DataFrame()

    valid_groups = [c for c in group_cols if c in df.columns]
    if not valid_groups:
        return pd.DataFrame()

    result = df.groupby(valid_groups, dropna=False).agg(agg_dict).reset_index()

    # Пересчёт конверсий
    for period in ("п1", "п2"):
        shows = _fp("Показы", period)
        clicks = _fp("Переходы в карточку", period)
        cart = _fp("Положили в корзину", period)
        ordered = _fp("Заказали, шт", period)
        bought = _fp("Выкупили, шт", period)

        if shows in result.columns and clicks in result.columns:
            result[_fp(CALC_CTR, period)] = safe_div(result[clicks], result[shows]) * 100

        if clicks in result.columns and cart in result.columns:
            result[_fp(CALC_CONV_CART, period)] = safe_div(result[cart], result[clicks]) * 100

        if clicks in result.columns and ordered in result.columns:
            result[_fp(CALC_CONV_ORDER, period)] = safe_div(result[ordered], result[clicks]) * 100

        if ordered in result.columns and bought in result.columns:
            result[_fp(CALC_PURCHASE_RATE, period)] = safe_div(result[bought], result[ordered]) * 100

    # Дельты
    result = _compute_funnel_deltas(result)

    return result


# ---------------------------------------------------------------------------
# Удобные обёртки
# ---------------------------------------------------------------------------

def funnel_agg_by_article(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_funnel(df, ["Артикул"])


def funnel_agg_by_model(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_funnel(df, ["Модель"])


def funnel_agg_by_color_collection(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_funnel(df, ["Color code", "Коллекция"])


def funnel_agg_by_status(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_funnel(df, ["Статус"])


def funnel_agg_by_glue(df: pd.DataFrame) -> pd.DataFrame:
    if "Склейка на WB" not in df.columns:
        return pd.DataFrame()
    return aggregate_funnel(df, ["Склейка на WB"])


# ---------------------------------------------------------------------------
# Витрина A: Итог воронки WB
# ---------------------------------------------------------------------------

def build_funnel_kpi(df: pd.DataFrame) -> pd.DataFrame:
    """Итоговые KPI воронки WB (п1, п2, Δабс, Δ%)."""
    # Список метрик для отображения в краткой воронке.
    # Ключи — внутренняя база (используется для поиска колонок F_{base} п1/п2),
    # значение — подпись в колонке "Метрика".
    display_metrics: list[tuple[str, str]] = [
        ("Показы", "Показы"),
        ("Переходы в карточку", "Переходы в карточку"),
        ("Положили в корзину", "Положили в корзину"),
        ("Заказали, шт", "Заказали, шт"),
        # Убраны: "Выкупили, шт", "Отменили, шт", CTR, PurchaseRate
        (CALC_CONV_CART, "Конверсия в корзину, %"),
        (CALC_CONV_ORDER, "Конверсия в заказ, %"),
    ]

    # Нужна «итоговая» строка
    df_copy = df.copy()
    df_copy["__all__"] = "Итого"
    agg = aggregate_funnel(df_copy, ["__all__"])
    if agg.empty:
        return pd.DataFrame()
    row = agg.iloc[0]

    rows = []
    for base, label in display_metrics:
        c1 = _fp(base, "п1")
        c2 = _fp(base, "п2")
        da = f"F_{base} Δабс"
        dp = f"F_{base} Δ%"
        rows.append({
            "Метрика": label,
            "п1": row.get(c1, np.nan),
            "п2": row.get(c2, np.nan),
            "Δабс": row.get(da, np.nan),
            "Δ%": row.get(dp, np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Витрина B: Воронка × Экономика диагностика
# ---------------------------------------------------------------------------

def build_funnel_economics_diag(df: pd.DataFrame) -> pd.DataFrame:
    """Модели: экономика + воронка + флаги."""
    if df.empty:
        return pd.DataFrame()

    # Агрегируем по моделям
    agg_df = funnel_agg_by_model(df)
    if agg_df.empty:
        return pd.DataFrame()

    cols = ["Модель"]
    econ_cols = [
        col_p("Маржа, ₽", "п2"), "Маржа, ₽ Δабс",
        col_p("Ср.чек заказа после СПП, ₽", "п2"), "Ср.чек заказа после СПП, ₽ Δабс",
        col_p("Реклама внутр., ₽", "п2"), col_p("Реклама внеш., ₽", "п2"),
        col_p("ДРР от заказов (до СПП), %", "п2"), "ДРР от заказов (до СПП), % Δабс",
    ]
    funnel_cols = [
        _fp("Показы", "п2"), f"F_Показы Δабс",
        _fp("Переходы в карточку", "п2"), f"F_Переходы в карточку Δабс",
        _fp(CALC_CONV_CART, "п2"), f"F_{CALC_CONV_CART} Δабс",
        _fp(CALC_CONV_ORDER, "п2"), f"F_{CALC_CONV_ORDER} Δабс",
    ]
    for c in econ_cols + funnel_cols:
        if c in agg_df.columns:
            cols.append(c)

    if len(cols) <= 1:
        return pd.DataFrame()

    result = agg_df[[c for c in cols if c in agg_df.columns]].copy()

    # Флаги
    _add_diag_flags(result)

    # Сортировка по марже
    sort_c = col_p("Маржа, ₽", "п2")
    if sort_c in result.columns:
        result = result.sort_values(sort_c, ascending=False, na_position="last")

    return result.reset_index(drop=True)


def _add_diag_flags(df: pd.DataFrame) -> None:
    """Добавляет диагностические флаги в DataFrame (inplace).

    Все Δ = п2 (текущий) − п1 (прошлый). Пороги значимости встроены.
    """
    flags = pd.Series("", index=df.index)

    # 🔴 Реклама ↑ (≥500₽), Переходы ↓
    clicks_d = "F_Переходы в карточку Δабс"
    if clicks_d in df.columns:
        total_ad_d = df.get("Реклама внутр., ₽ Δабс", pd.Series(0, index=df.index)).fillna(0)
        if "Реклама внеш., ₽ Δабс" in df.columns:
            total_ad_d = total_ad_d + df["Реклама внеш., ₽ Δабс"].fillna(0)
        clicks_delta = df[clicks_d].fillna(0)
        mask = (total_ad_d >= 500) & (clicks_delta <= 0)
        flags = flags.where(~mask, flags + "🔴Реклама↑Трафик↓ ")

    # 🟡 Цена после СПП ↑ (≥50₽) и конверсия ↓
    price_d = "Ср.чек заказа после СПП, ₽ Δабс"
    conv_d = f"F_{CALC_CONV_ORDER} Δабс"
    if price_d in df.columns and conv_d in df.columns:
        mask = (df[price_d].fillna(0) >= 50) & (df[conv_d].fillna(0) < -0.5)
        flags = flags.where(~mask, flags + "🟡Цена↑Конверсия↓ ")

    # 🟠 Показы ↓ (≥10% падение)
    shows_d = "F_Показы Δабс"
    shows_pct = "F_Показы Δ%"
    if shows_pct in df.columns:
        mask = df[shows_pct].fillna(0) <= -10
        flags = flags.where(~mask, flags + "🟠Показы↓ ")
    elif shows_d in df.columns:
        shows_p1 = _fp("Показы", "п1")
        if shows_p1 in df.columns:
            shows_delta_pct = safe_pct_change(df.get(_fp("Показы", "п2"), 0), df[shows_p1])
            mask = shows_delta_pct.fillna(0) <= -10
            flags = flags.where(~mask, flags + "🟠Показы↓ ")

    df["Флаги"] = flags.str.strip()


# ---------------------------------------------------------------------------
# Витрина C: Точки роста конверсий
# ---------------------------------------------------------------------------

def build_conversion_growth_points(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Ранжирование моделей по потенциалу роста конверсий."""
    if df.empty:
        return pd.DataFrame()

    # Агрегируем по моделям
    agg_df = funnel_agg_by_model(df)
    if agg_df.empty:
        return pd.DataFrame()

    clicks_c = _fp("Переходы в карточку", "п2")
    shows_c = _fp("Показы", "п2")
    conv_cart_c = _fp(CALC_CONV_CART, "п2")
    conv_order_c = _fp(CALC_CONV_ORDER, "п2")

    needed = [clicks_c, conv_cart_c, conv_order_c]
    if not all(c in agg_df.columns for c in needed):
        return pd.DataFrame()

    result = agg_df.copy()
    result = result[result[clicks_c].notna() & (result[clicks_c] > 0)]

    if result.empty:
        return pd.DataFrame()

    # Медианы конверсий
    med_cart = result[conv_cart_c].median()
    med_order = result[conv_order_c].median()

    # Потенциал: высокий трафик + низкая конверсия
    opportunity = pd.Series("", index=result.index)

    # Высокие показы/переходы, но низкая Conv2Cart
    high_traffic = result[clicks_c] >= result[clicks_c].quantile(0.5)
    low_cart = result[conv_cart_c] < med_cart
    mask1 = high_traffic & low_cart
    opportunity = opportunity.where(~mask1, opportunity + "Проблема карточки/цены ")

    # Высокая Conv2Cart, но низкая Conv2Order
    high_cart = result[conv_cart_c] >= med_cart
    low_order = result[conv_order_c] < med_order
    mask2 = high_cart & low_order
    opportunity = opportunity.where(~mask2, opportunity + "Проблема заказа/доставки ")

    # Конверсии растут при стабильной цене
    conv_d = f"F_{CALC_CONV_ORDER} Δабс"
    price_d = "Ср.чек заказа после СПП, ₽ Δабс"
    if conv_d in result.columns and price_d in result.columns:
        mask3 = (result[conv_d].fillna(0) > 0) & (result[price_d].fillna(0).abs() < 50)
        opportunity = opportunity.where(~mask3, opportunity + "✅Контент/позиция работает ")

    result["Потенциал"] = opportunity.str.strip()

    cols = ["Модель"]
    for c in [shows_c, clicks_c, conv_cart_c, conv_order_c,
              _fp(CALC_CONV_CART, "п1"), f"F_{CALC_CONV_CART} Δабс",
              _fp(CALC_CONV_ORDER, "п1"), f"F_{CALC_CONV_ORDER} Δабс",
              col_p("Маржа, ₽", "п2"), "Потенциал"]:
        if c in result.columns:
            cols.append(c)

    result = result[[c for c in cols if c in result.columns]]
    
    # Переименование колонок для понятности
    rename_dict = {
        shows_c: "Показы п2",
        clicks_c: "Переходы п2", 
        conv_cart_c: "Конверсия в корзину, % п2",
        conv_order_c: "Конверсия в заказ, % п2",
        _fp(CALC_CONV_CART, "п1"): "Конверсия в корзину, % п1",
        f"F_{CALC_CONV_CART} Δабс": "Δ Конверсия в корзину, %",
        _fp(CALC_CONV_ORDER, "п1"): "Конверсия в заказ, % п1", 
        f"F_{CALC_CONV_ORDER} Δабс": "Δ Конверсия в заказ, %",
        col_p("Маржа, ₽", "п2"): "Маржа, ₽ п2"
    }
    result = result.rename(columns=rename_dict)
    # Сортировка: сначала по марже п2 (убывание), потом по трафику/конверсии
    sort_cols = []
    sort_asc = []
    if "Маржа, ₽ п2" in result.columns:
        sort_cols.append("Маржа, ₽ п2")
        sort_asc.append(False)  # маржа по убыванию
    if "Конверсия в заказ, % п2" in result.columns:
        sort_cols.append("Конверсия в заказ, % п2")
        sort_asc.append(True)   # конверсия по возрастанию (хуже = выше)
    if "Переходы п2" in result.columns:
        sort_cols.append("Переходы п2")
        sort_asc.append(False)  # трафик по убыванию
    
    if sort_cols:
        result = result.sort_values(sort_cols, ascending=sort_asc, na_position="last")
    
    return result.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Витрина D: Дополнение склеек воронкой
# ---------------------------------------------------------------------------

def enrich_glue_with_funnel(glue_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет в таблицу склеек метрики воронки."""
    if glue_df.empty or merged_df.empty:
        return glue_df
    if "Склейка на WB" not in merged_df.columns:
        return glue_df

    funnel_glue = aggregate_funnel(merged_df, ["Склейка на WB", "Артикул"])
    if funnel_glue.empty:
        return glue_df

    # Выбираем ключевые funnel-колонки
    funnel_cols = ["Склейка на WB", "Артикул"]
    for c in funnel_glue.columns:
        if c.startswith("F_") and ("Показы" in c or "Переходы" in c or "Conv2Cart" in c or "Conv2Order" in c):
            if "п2" in c:
                funnel_cols.append(c)

    funnel_cols = [c for c in funnel_cols if c in funnel_glue.columns]
    if len(funnel_cols) <= 2:
        return glue_df

    funnel_subset = funnel_glue[funnel_cols]

    # Merge
    result = glue_df.merge(funnel_subset, on=["Склейка на WB", "Артикул"], how="left")

    # Паразит по конверсии: высокий трафик, низкая конверсия
    clicks_c = _fp("Переходы в карточку", "п2")
    conv_c = _fp(CALC_CONV_ORDER, "п2")
    if clicks_c in result.columns and conv_c in result.columns:
        med_conv = result[conv_c].median()
        q75_clicks = result[clicks_c].quantile(0.75)
        mask = (result[clicks_c] >= q75_clicks) & (result[conv_c] < med_conv * 0.5)
        if "Роль" in result.columns:
            result.loc[mask & (result["Роль"] == ""), "Роль"] = "паразит (конверсия)"
        else:
            result["Роль конверсии"] = ""
            result.loc[mask, "Роль конверсии"] = "паразит (конверсия)"

    return result
