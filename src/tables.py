"""Формирование витрин (таблиц) для UI: KPI, ТОП/анти-ТОП, реклама, склад и т.д."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.schema import (
    CALC_ANNUAL_YIELD,
    CALC_MARGIN_RATE,
    col_p,
)
from src.wb_funnel_io import CALC_CONV_CART, CALC_CONV_ORDER
from src.aggregations import (
    agg_by_article,
    agg_by_color_collection,
    agg_by_glue,
    agg_by_glue_article,
    agg_by_model,
    agg_by_status,
    aggregate,
    total_kpi,
)
from src.utils import safe_div, safe_pct_change


# ---------------------------------------------------------------------------
# 1) Итоговые KPI
# ---------------------------------------------------------------------------

KPI_DISPLAY = [
    ("Маржа, ₽", "₽"),
    (CALC_MARGIN_RATE, "%"),
    (CALC_ANNUAL_YIELD, "%"),
    ("Продажи до СПП, ₽", "₽"),
    ("Продажи, шт", "шт"),
    ("Заказы до СПП, ₽", "₽"),
    ("Заказы, шт", "шт"),
    ("Реклама внутр., ₽", "₽"),
    ("Реклама внеш., ₽", "₽"),
    ("ДРР от заказов (до СПП), %", "%"),
    ("ДРР от продаж (до СПП), %", "%"),
    ("Остатки, шт", "шт"),
    ("Оборот продаж, дни", "дн"),
    ("Оборот заказов, дни", "дн"),
]


def build_kpi_table(df: pd.DataFrame) -> pd.DataFrame:
    """Строит таблицу KPI (п1, п2, Δабс, Δ%)."""
    kpi = total_kpi(df)
    if kpi.empty:
        return pd.DataFrame()

    rows = []
    for base, unit in KPI_DISPLAY:
        c1 = col_p(base, "п1")
        c2 = col_p(base, "п2")
        da = f"{base} Δабс"
        dp = f"{base} Δ%"
        rows.append({
            "Метрика": f"{base} ({unit})",
            "п1": kpi.get(c1, np.nan),
            "п2": kpi.get(c2, np.nan),
            "Δабс": kpi.get(da, np.nan),
            "Δ%": kpi.get(dp, np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2) ТОП-10 / анти-ТОП-10
# ---------------------------------------------------------------------------

def _sort_col(metric: str) -> str:
    """Возвращает колонку для сортировки (преимущественно п2)."""
    return col_p(metric, "п2")


_EXCLUDED_STATUSES_ANTI = {"выводим", "архив"}


def build_top_articles(df: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ТОП-10 и анти-ТОП-10 артикулов по Марже п2.

    Анти-ТОП: только Маржа п2 < 0, исключены статусы Выводим/Архив.
    """
    agg = agg_by_article(df)
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame()
    sort_c = _sort_col("Маржа, ₽")
    if sort_c not in agg.columns:
        return pd.DataFrame(), pd.DataFrame()

    cols = _pick_display_cols(agg, "Артикул")
    if "Статус" in agg.columns and "Статус" not in cols:
        cols.insert(1, "Статус")
    cols = [c for c in cols if c in agg.columns]

    agg_sorted = agg.sort_values(sort_c, ascending=False, na_position="last")
    top = agg_sorted.head(n)[cols].reset_index(drop=True)

    # Анти-ТОП: маржа п2 < 0, исключить Выводим/Архив
    anti_mask = agg[sort_c] < 0
    if "Статус" in agg.columns:
        anti_mask = anti_mask & ~agg["Статус"].str.strip().str.lower().isin(_EXCLUDED_STATUSES_ANTI)
    anti = agg[anti_mask].sort_values(sort_c, ascending=True, na_position="last")
    bottom = anti.head(n)[cols].reset_index(drop=True)

    return top, bottom


def build_top_models(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """ТОП моделей по Марже п2 + Годовая доходность + кол-во артикулов."""
    agg = agg_by_model(df)
    if agg.empty:
        return pd.DataFrame()
    sort_c = _sort_col("Маржа, ₽")
    if sort_c not in agg.columns:
        return pd.DataFrame()

    # Кол-во артикулов (SKU) в каждой модели
    if "Модель" in df.columns and "Артикул" in df.columns:
        sku_count = df.groupby("Модель")["Артикул"].nunique().reset_index(name="SKU шт")
        agg = agg.merge(sku_count, on="Модель", how="left")

    # Исключить модели, где все SKU = Архив
    if "Модель" in df.columns and "Статус" in df.columns:
        all_archive = df.groupby("Модель")["Статус"].apply(
            lambda s: s.str.strip().str.lower().eq("архив").all()
        ).reset_index(name="_all_archive")
        agg = agg.merge(all_archive, on="Модель", how="left")
        agg = agg[~agg["_all_archive"].fillna(False)]
        agg = agg.drop(columns=["_all_archive"], errors="ignore")

    agg_sorted = agg.sort_values(sort_c, ascending=False, na_position="last")
    cols = _pick_display_cols(agg, "Модель")
    if "SKU шт" in agg.columns and "SKU шт" not in cols:
        cols.insert(1, "SKU шт")
    cols = [c for c in cols if c in agg_sorted.columns]
    return agg_sorted.head(n)[cols].reset_index(drop=True)


def build_top_models_with_funnel(df: pd.DataFrame, funnel_df: pd.DataFrame | None = None, n: int = 10) -> pd.DataFrame:
    """ТОП моделей по Марже п2 + Годовая доходность + кол-во артикулов + конверсии воронки.

    Включает в конец таблицы упорядоченные колонки воронки: Показы, Положили в корзину,
    Конверсия в корзину, % и Конверсия в заказ, % (для каждой — п1, п2, Δабс, Δ%).
    """
    # Базовые данные моделей
    top_models = build_top_models(df, n=n * 2)  # Берем больше с запасом

    if top_models.empty or funnel_df is None:
        return top_models.head(n)

    # Агрегируем данные воронки по моделям
    from src.wb_funnel_metrics import funnel_agg_by_model
    funnel_agg = funnel_agg_by_model(funnel_df)

    if funnel_agg.empty:
        return top_models.head(n)

    # Объединяем с данными воронки
    result = top_models.merge(funnel_agg, on="Модель", how="left")

    # Переименование F_ колонок в читаемые имена и Δ-колонок
    rename_dict = {}
    needed_bases = ["Показы", "Переходы в карточку", "Положили в корзину"]
    for period in ["п1", "п2"]:
        for base in needed_bases:
            rename_dict[f"F_{base} {period}"] = f"{base} {period}"

        rename_dict[f"F_{CALC_CONV_CART} {period}"] = f"Конверсия в корзину, % {period}"
        rename_dict[f"F_{CALC_CONV_ORDER} {period}"] = f"Конверсия в заказ, % {period}"

    # Δабс и Δ% переименование
    for base in needed_bases:
        rename_dict[f"F_{base} Δабс"] = f"{base} Δабс"
        rename_dict[f"F_{base} Δ%"] = f"{base} Δ%"

    rename_dict[f"F_{CALC_CONV_CART} Δабс"] = "Конверсия в корзину, % Δабс"
    rename_dict[f"F_{CALC_CONV_CART} Δ%"] = "Конверсия в корзину, % Δ%"
    rename_dict[f"F_{CALC_CONV_ORDER} Δабс"] = "Конверсия в заказ, % Δабс"
    rename_dict[f"F_{CALC_CONV_ORDER} Δ%"] = "Конверсия в заказ, % Δ%"

    result = result.rename(columns=rename_dict)

    # Явно вычислим Δабс и Δ% для нужных метрик (если п1/п2 присутствуют)
    metrics_for_deltas = [
        "Показы",
        "Переходы в карточку",
        "Положили в корзину",
        "Конверсия в корзину, %",
        "Конверсия в заказ, %",
    ]
    for m in metrics_for_deltas:
        p1 = f"{m} п1"
        p2 = f"{m} п2"
        da = f"{m} Δабс"
        dp = f"{m} Δ%"
        if p1 in result.columns and p2 in result.columns:
            result[da] = result[p2] - result[p1]
            result[dp] = safe_pct_change(result[p2], result[p1])

    # Собираем финальный порядок колонок: базовые колонки (всё, что не F_ и не периодные),
    # затем по каждому метрике — п1, п2, Δабс, Δ%
    funnel_order = []
    for m in ["Показы", "Переходы в карточку", "Положили в корзину", "Конверсия в корзину, %", "Конверсия в заказ, %"]:
        funnel_order.extend([f"{m} п1", f"{m} п2", f"{m} Δабс", f"{m} Δ%"])

    # Сохраняем все колонки из исходного топа (чтобы не потерять дополнительные метрики)
    top_cols = [c for c in top_models.columns if c in result.columns]
    # Затем добавляем воронковые колонки в нужном порядке
    funnel_cols_in_order = [c for c in funnel_order if c in result.columns]
    final_cols = top_cols + funnel_cols_in_order
    # На всякий случай добавляем остальные колонки, которые могут присутствовать (без дублирования),
    # но исключаем ненужные сырые F_ колонки и широкие наборы показателей, которые пользователь просил убрать.
    unwanted_substr = [
        "Добавили в отложенные",
        "Заказали, шт",
        "Заказали ВБ клуб, шт",
        "Выкупили, шт",
        "Выкупили ВБ клуб, шт",
        "Отменили, шт",
        "Отменили ВБ клуб, шт",
        "Заказали на сумму",
        "Заказали на сумму ВБ клуб",
        "Выкупили на сумму",
        "Выкупили на сумму ВБ клуб",
        "Отменили на сумму",
        "Отменили на сумму ВБ клуб",
        "CTR",
        "PurchaseRate",
        "Переходы в карточку",
    ]
    remaining = []
    for c in result.columns:
        if c in final_cols:
            continue
        # Exclude raw F_ columns
        if c.startswith("F_"):
            continue
        # Exclude columns matching unwanted substrings
        if any(sub in c for sub in unwanted_substr):
            continue
        remaining.append(c)

    final_cols.extend(remaining)
    result = result[[c for c in final_cols if c in result.columns]]

    # Сортировка по марже п2
    sort_c = "Маржа, ₽ п2"
    if sort_c in result.columns:
        result = result.sort_values(sort_c, ascending=False, na_position="last")

    return result.head(n).reset_index(drop=True)


_EXCLUDED_STATUSES_SCALE = {"выводим", "архив"}


def build_scale_candidates(
    df: pd.DataFrame,
    n: int = 10,
    *,
    stock_min: int = 200,
    turnover_min: float = 10.0,
    days_cover_min: float = 90.0,
    period_days: int = 14,
) -> pd.DataFrame:
    """Кандидаты на масштабирование (п2 = текущий период).

    Фильтры (все применяются к п2):
    - Статус ∉ {Выводим, Архив}
    - Маржа до СПП, % п2 > 25%, Маржа п2 > 0
    - Остатки п2 ≥ stock_min
    - Оборот продаж п2 ≥ turnover_min
    - Оборот продаж дни > 60
    - ДРР от продаж (до СПП) < 5%
    - DaysCover п2 = Остатки / (Продажи шт / period_days) ≥ days_cover_min
    """
    agg = agg_by_article(df)
    if agg.empty:
        return pd.DataFrame()

    margin_c = _sort_col("Маржа, ₽")
    ay_c = _sort_col(CALC_ANNUAL_YIELD)
    margin_rate_c = _sort_col(CALC_MARGIN_RATE)
    stock_c = col_p("Остатки, шт", "п2")
    turn_c = col_p("Оборот продаж, дни", "п2")
    sales_qty_c = col_p("Продажи, шт", "п2")

    needed = [margin_c, ay_c]
    if not all(c in agg.columns for c in needed):
        return pd.DataFrame()

    # DaysCover п2
    if stock_c in agg.columns and sales_qty_c in agg.columns:
        daily_sales = safe_div(agg[sales_qty_c], period_days)
        agg["DaysCover п2"] = safe_div(agg[stock_c], daily_sales)
    else:
        agg["DaysCover п2"] = np.nan

    # Базовый фильтр: маржа и маржа до СПП, %
    mask = (agg[margin_c] > 0)
    if margin_rate_c in agg.columns:
        mask = mask & (agg[margin_rate_c] > 25)
    if ay_c in agg.columns:
        mask = mask & (agg[ay_c] > 0)

    # Статус
    if "Статус" in agg.columns:
        mask = mask & ~agg["Статус"].str.strip().str.lower().isin(_EXCLUDED_STATUSES_SCALE)

    # Остатки
    if stock_c in agg.columns:
        mask = mask & (agg[stock_c] >= stock_min)

    # Оборот
    if turn_c in agg.columns:
        mask = mask & (agg[turn_c] >= turnover_min) & (agg[turn_c] > 60)

    # ДРР от продаж < 5%
    drr_c = col_p("ДРР от продаж (до СПП), %", "п2")
    if drr_c in agg.columns:
        mask = mask & (agg[drr_c] < 5)

    # DaysCover
    mask = mask & (agg["DaysCover п2"].fillna(0) >= days_cover_min)

    filtered = agg[mask].sort_values(margin_c, ascending=False, na_position="last")

    # Колонки для отображения
    cols = _pick_display_cols(agg, "Артикул")
    extra = ["Статус", stock_c, turn_c, "DaysCover п2", sales_qty_c]
    for ec in extra:
        if ec in agg.columns and ec not in cols:
            cols.insert(1 if ec == "Статус" else len(cols), ec)

    cols = [c for c in cols if c in filtered.columns]
    return filtered.head(n)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) Реклама и лаг
# ---------------------------------------------------------------------------

def build_ad_future(
    df: pd.DataFrame,
    *,
    ad_delta_pct_thr: float = 10.0,
    ad_delta_abs_thr: float = 500.0,
) -> pd.DataFrame:
    """Реклама — будущее (заказы): реклама, Заказы, DRR заказов, Δ заказов."""
    agg = agg_by_article(df)
    if agg.empty:
        return pd.DataFrame()
    cols_needed = ["Артикул"]
    metric_cols = []
    for base in ["Реклама внутр., ₽", "Реклама внеш., ₽", "Заказы до СПП, ₽",
                  "Заказы, шт", "ДРР от заказов (до СПП), %"]:
        for suf in ["п1", "п2", "Δабс", "Δ%"]:
            c = f"{base} {suf}" if suf in ("п1", "п2") else f"{base} {suf}"
            if c in agg.columns:
                metric_cols.append(c)
    cols_needed += metric_cols
    cols_present = [c for c in cols_needed if c in agg.columns]
    result = agg[cols_present].copy()

    # Подсветка: реклама ↑ (с порогом значимости), заказы ↓
    ad1 = col_p("Реклама внутр., ₽", "п1")
    ad2 = col_p("Реклама внутр., ₽", "п2")
    ord1 = col_p("Заказы до СПП, ₽", "п1")
    ord2 = col_p("Заказы до СПП, ₽", "п2")
    if all(c in result.columns for c in [ad1, ad2, ord1, ord2]):
        ad_ext1 = col_p("Реклама внеш., ₽", "п1")
        ad_ext2 = col_p("Реклама внеш., ₽", "п2")
        total_ad_p1 = result[ad1].fillna(0)
        total_ad_p2 = result[ad2].fillna(0)
        if ad_ext1 in result.columns:
            total_ad_p1 = total_ad_p1 + result[ad_ext1].fillna(0)
        if ad_ext2 in result.columns:
            total_ad_p2 = total_ad_p2 + result[ad_ext2].fillna(0)
        ad_delta = total_ad_p2 - total_ad_p1
        ad_delta_pct = safe_div(ad_delta, total_ad_p1.abs()) * 100
        significant = (ad_delta_pct.fillna(0).abs() >= ad_delta_pct_thr) & (ad_delta.abs() >= ad_delta_abs_thr)
        result["⚠️ Реклама↑ Заказы↓"] = (
            significant & (ad_delta > 0) & (result[ord2] <= result[ord1])
        ).map({True: "⚠️", False: ""})

    # Фильтр: убираем строки где реклама п1=0 и реклама п2=0 (внутренняя+внешняя)
    ad1_total = total_ad_p1
    ad2_total = total_ad_p2
    result = result[(ad1_total > 0) | (ad2_total > 0)]

    # Сортировка по марже п2
    margin_c = col_p("Маржа, ₽", "п2")
    if margin_c in result.columns:
        result = result.sort_values(margin_c, ascending=False, na_position="last")

    return result


def build_ad_current(df: pd.DataFrame) -> pd.DataFrame:
    """Реклама — настоящее (продажи): реклама, Продажи, DRR продаж, Маржа."""
    agg = agg_by_article(df)
    if agg.empty:
        return pd.DataFrame()
    metric_cols = []
    for base in ["Реклама внутр., ₽", "Реклама внеш., ₽", "Продажи до СПП, ₽",
                  "Продажи, шт", "ДРР от продаж (до СПП), %", "Маржа, ₽"]:
        for suf in ["п1", "п2", "Δабс", "Δ%"]:
            c = f"{base} {suf}" if suf in ("п1", "п2") else f"{base} {suf}"
            if c in agg.columns:
                metric_cols.append(c)
    cols_present = ["Артикул"] + [c for c in metric_cols if c in agg.columns]
    result = agg[[c for c in cols_present if c in agg.columns]].copy()

    # Подсветка: DRR продаж ↑ и маржа ↓ (Δ = п2 − п1)
    drr_d = "ДРР от продаж (до СПП), % Δабс"
    m_d = "Маржа, ₽ Δабс"
    if drr_d in result.columns and m_d in result.columns:
        result["⚠️ DRR↑ Маржа↓"] = (
            (result[drr_d].fillna(0) > 0) & (result[m_d].fillna(0) < 0)
        ).map({True: "⚠️", False: ""})

    # Фильтр: убираем строки где реклама п1=0 и реклама п2=0 (внутренняя+внешняя)
    ad_int1 = col_p("Реклама внутр., ₽", "п1")
    ad_int2 = col_p("Реклама внутр., ₽", "п2")
    ad_ext1 = col_p("Реклама внеш., ₽", "п1")
    ad_ext2 = col_p("Реклама внеш., ₽", "п2")

    # Суммарная реклама п1 и п2
    ad1_total = result.get(ad_int1, pd.Series(0, index=result.index)).fillna(0)
    ad2_total = result.get(ad_int2, pd.Series(0, index=result.index)).fillna(0)
    if ad_ext1 in result.columns:
        ad1_total = ad1_total + result[ad_ext1].fillna(0)
    if ad_ext2 in result.columns:
        ad2_total = ad2_total + result[ad_ext2].fillna(0)

    result = result[(ad1_total > 0) | (ad2_total > 0)]

    # Сортировка по марже п2
    margin_c = col_p("Маржа, ₽", "п2")
    if margin_c in result.columns:
        result = result.sort_values(margin_c, ascending=False, na_position="last")

    return result


# ---------------------------------------------------------------------------
# 4) Склад / оборотка
# ---------------------------------------------------------------------------

def build_warehouse(
    df: pd.DataFrame,
    *,
    stock_thr: int = 100,
    turnover_thr: float = 200.0,
) -> pd.DataFrame:
    """Остатки, Оборот, Хранение, Маржа — красные флаги (п2 = текущий).

    🔴 = Остатки п2 ≥ stock_thr И Оборот п2 ≥ turnover_thr.
    """
    agg = agg_by_article(df)
    if agg.empty:
        return pd.DataFrame()
    cols = ["Артикул"]
    if "Статус" in agg.columns:
        cols.append("Статус")
    for base in ["Остатки, шт", "Оборот продаж, дни", "Хранение на ед, ₽", "Маржа, ₽"]:
        for suf in ["п1", "п2", "Δабс"]:
            c = f"{base} {suf}" if suf in ("п1", "п2") else f"{base} {suf}"
            if c in agg.columns:
                cols.append(c)
    result = agg[[c for c in cols if c in agg.columns]].copy()

    # Red flag: абсолютные пороги на п2
    ost = col_p("Остатки, шт", "п2")
    turn = col_p("Оборот продаж, дни", "п2")
    if ost in result.columns and turn in result.columns:
        flag_mask = (result[ost].fillna(0) >= stock_thr) & (result[turn].fillna(0) >= turnover_thr)
        result["🔴 Флаг"] = flag_mask.map({True: "🔴", False: ""})
        result["Почему флаг"] = ""
        result.loc[flag_mask, "Почему флаг"] = (
            "Остатки≥" + str(stock_thr) + " и оборот≥" + str(turnover_thr) + "д"
        )

    # Фильтр: убираем строки где остатки п2 < 30 шт
    if ost in result.columns:
        result = result[result[ost].fillna(0) >= 30]

    sort_c = col_p("Маржа, ₽", "п2")
    if sort_c in result.columns:
        result = result.sort_values(sort_c, ascending=False, na_position="last")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5) Лаг цен
# ---------------------------------------------------------------------------

def build_price_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Ср.чек заказа vs Ср.чек продажи (до СПП)."""
    # Агрегация по моделям
    agg = agg_by_model(df)
    if agg.empty:
        return pd.DataFrame()
    cols = ["Модель"]
    for p in ("п1", "п2"):
        co = col_p("Ср.чек заказа до СПП, ₽", p)
        cs = col_p("Ср.чек продажи до СПП, ₽", p)
        if co in agg.columns:
            cols.append(co)
        if cs in agg.columns:
            cols.append(cs)

    result = agg[[c for c in cols if c in agg.columns]].copy()

    # Разница п2 (текущий период)
    co2 = col_p("Ср.чек заказа до СПП, ₽", "п2")
    cs2 = col_p("Ср.чек продажи до СПП, ₽", "п2")
    if co2 in result.columns and cs2 in result.columns:
        result["Разница чеков п2"] = result[co2] - result[cs2]
        result["Разница чеков п2, %"] = safe_div(result["Разница чеков п2"], result[cs2].abs()) * 100
        result["⚠️ Сильный лаг"] = (
            result["Разница чеков п2, %"].abs() > 10
        ).map({True: "⚠️", False: ""})

    # Если колонок чеков нет — нечего показывать
    if co2 not in result.columns or cs2 not in result.columns:
        return pd.DataFrame()

    # Фильтры: убираем строки где либо цен нет, либо разница чеков < 1%
    has_prices = (result[co2].notna() & result[cs2].notna() &
                  (result[co2] > 0) & (result[cs2] > 0))

    diff_pct_col = "Разница чеков п2, %"
    has_diff = result[diff_pct_col].abs() >= 1.0 if diff_pct_col in result.columns else pd.Series(False, index=result.index)

    result = result[has_prices & has_diff]

    # Сортировка по марже п2
    margin_c = col_p("Маржа, ₽", "п2")
    if margin_c in result.columns:
        result = result.sort_values(margin_c, ascending=False, na_position="last")

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6) Цветовые аномалии
# ---------------------------------------------------------------------------

def build_color_anomalies(
    df: pd.DataFrame,
    *,
    anomaly_ratio: float = 0.3,
    min_top_margin: float = 1000.0,
) -> pd.DataFrame:
    """Агрегация по (Color code, Коллекция, Модель) — поиск аномалий.

    ⚠️ = в одном цвете+коллекции есть модель с маржой < anomaly_ratio × лучшей,
    при условии что лучшая маржа ≥ min_top_margin ₽.
    """
    if not all(c in df.columns for c in ["Color code", "Коллекция", "Модель"]):
        return pd.DataFrame()

    agg = aggregate(df, ["Color code", "Коллекция", "Модель"])
    if agg.empty:
        return pd.DataFrame()

    margin_c = col_p("Маржа, ₽", "п2")
    orders_c = col_p("Заказы, шт", "п2")
    ay_c = col_p(CALC_ANNUAL_YIELD, "п2")

    display = ["Color code", "Коллекция", "Модель"]
    for c in [margin_c, orders_c, ay_c,
              col_p("Продажи, шт", "п2"), col_p("Маржа, ₽", "п1"),
              "Маржа, ₽ Δ%"]:
        if c in agg.columns:
            display.append(c)

    result = agg[[c for c in display if c in agg.columns]].copy()

    if margin_c in result.columns:
        grp = result.groupby(["Color code", "Коллекция"])
        anomalies = []
        for (cc, coll), g in grp:
            if len(g) < 2:
                continue
            top_m = g[margin_c].max()
            bot_m = g[margin_c].min()
            if (pd.notna(top_m) and pd.notna(bot_m)
                    and top_m >= min_top_margin
                    and bot_m < top_m * anomaly_ratio):
                anomalies.extend(g.index.tolist())
        result["⚠️ Аномалия"] = ""
        if anomalies:
            result.loc[anomalies, "⚠️ Аномалия"] = "⚠️"

    if margin_c in result.columns:
        result = result.sort_values(margin_c, ascending=False, na_position="last")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7) Склейки WB
# ---------------------------------------------------------------------------

def build_glue_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Анализ склеек WB: доли артикулов внутри склейки."""
    if "Склейка на WB" not in df.columns:
        return pd.DataFrame()

    agg_glue = agg_by_glue_article(df)
    if agg_glue.empty:
        return pd.DataFrame()

    margin_c = col_p("Маржа, ₽", "п2")
    orders_c = col_p("Заказы, шт", "п2")
    sales_c = col_p("Продажи, шт", "п2")
    orders_rub_c = col_p("Заказы до СПП, ₽", "п2")

    # Доли внутри склейки
    for metric, label in [(margin_c, "Доля маржи в склейке, %"),
                           (orders_c, "Доля заказов шт в склейке, %"),
                           (sales_c, "Доля продаж шт в склейке, %")]:
        if metric in agg_glue.columns:
            totals = agg_glue.groupby("Склейка на WB")[metric].transform("sum")
            agg_glue[label] = safe_div(agg_glue[metric], totals) * 100

    # Метки донор/паразит
    margin_share = "Доля маржи в склейке, %"
    orders_share = "Доля заказов шт в склейке, %"
    if margin_share in agg_glue.columns and orders_share in agg_glue.columns:
        conditions = []
        for idx, row in agg_glue.iterrows():
            ms = row.get(margin_share, 0)
            os = row.get(orders_share, 0)
            if pd.isna(ms):
                ms = 0
            if pd.isna(os):
                os = 0
            if ms > 40 or os > 40:
                conditions.append("донор")
            elif ms < 10 and os > 15:
                conditions.append("паразит")
            else:
                conditions.append("")
        agg_glue["Роль"] = conditions

    # Каннибализация
    orders_d = "Заказы, шт Δабс"
    chek_spp = col_p("Ср.чек заказа после СПП, ₽", "п2")
    if orders_d in agg_glue.columns:
        agg_glue["Каннибализация"] = ""
        for glue, g in agg_glue.groupby("Склейка на WB", dropna=False):
            if len(g) < 2:
                continue
            has_up = (g[orders_d] > 0).any()
            has_down = (g[orders_d] < 0).any()
            if has_up and has_down:
                agg_glue.loc[g.index, "Каннибализация"] = "возможная каннибализация"

    cols_display = ["Склейка на WB", "Артикул"]
    for c in [margin_c, orders_c, sales_c,
              "Доля маржи в склейке, %", "Доля заказов шт в склейке, %",
              "Доля продаж шт в склейке, %", "Роль", "Каннибализация",
              orders_d]:
        if c in agg_glue.columns:
            cols_display.append(c)

    result = agg_glue[[c for c in cols_display if c in agg_glue.columns]]
    if margin_c in result.columns:
        result = result.sort_values(["Склейка на WB", margin_c], ascending=[True, False])
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Хелпер: выбор колонок для отображения
# ---------------------------------------------------------------------------

def _pick_display_cols(agg: pd.DataFrame, key_col: str) -> list[str]:
    """Выбирает ключевые колонки для отображения."""
    cols = [key_col]
    for base in ["Маржа, ₽", CALC_MARGIN_RATE, CALC_ANNUAL_YIELD, "Продажи до СПП, ₽",
                  "Заказы до СПП, ₽", "ДРР от продаж (до СПП), %", "Остатки, шт",
                  "Реклама внутр., ₽", "Реклама внеш., ₽"]:
        for suf in ["п1", "п2", "Δабс", "Δ%"]:
            c = f"{base} {suf}" if suf in ("п1", "п2") else f"{base} {suf}"
            if c in agg.columns:
                cols.append(c)
    return [c for c in cols if c in agg.columns]
