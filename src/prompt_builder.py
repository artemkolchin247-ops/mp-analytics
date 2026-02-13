"""Сборка PROMPT_КРАТКИЙ и PROMPT_ДЕТАЛЬНЫЙ для внешнего ИИ."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.schema import CALC_ANNUAL_YIELD, CALC_MARGIN_RATE, col_p
from src.tables import (
    build_ad_current,
    build_ad_future,
    build_color_anomalies,
    build_glue_analysis,
    build_kpi_table,
    build_price_lag,
    build_scale_candidates,
    build_top_articles,
    build_top_models,
    build_warehouse,
)
from src.aggregations import (
    agg_by_article,
    agg_by_color_collection,
    agg_by_model,
    agg_by_status,
    agg_by_glue,
    total_kpi,
)
from src.wb_funnel_metrics import (
    build_funnel_kpi,
    build_funnel_economics_diag,
    build_conversion_growth_points,
    funnel_agg_by_model,
    funnel_agg_by_status,
    funnel_agg_by_glue,
    _fp,
)
from src.wb_ads_metrics import (
    has_ads_cols,
    build_ads_kpi,
    build_ads_by_article,
    build_ads_funnel_price,
    ads_agg_by_model,
    ads_agg_by_status,
    ads_agg_by_color_collection,
    ads_agg_by_glue,
)


# ---------------------------------------------------------------------------
# Общие определения
# ---------------------------------------------------------------------------

DEFINITIONS = """\
## Definitions

- **п2** — текущий период анализа; **п1** — прошлый аналогичный период.
- **СПП** (Скидка Постоянного Покупателя) — влияет на цену для покупателя (спрос), но НЕ влияет на нашу выручку/маржу до СПП. НДС считается от выручки до СПП.
- **«Продажи до СПП, ₽»** = выручка до СПП (база для комиссии и НДС).
- **Комиссия до СПП, %** — данность маркетплейса, берётся из данных.
- **Лаг заказ → продажа**: «Ср.чек заказа …» влияет на *будущие* продажи; «Ср.чек продажи …» отражает *прошлые* заказы.
- **DRR от заказов (до СПП), %** = рекламные расходы / сумма заказов до СПП → показывает *будущую* эффективность рекламы; на текущую маржу напрямую НЕ влияет.
- **DRR от продаж (до СПП), %** = рекламные расходы / выручка (продажи) до СПП → влияет на *текущую* маржу до СПП.
- **Лаг рекламы**: большие рекламные расходы сегодня не обязаны увеличить выручку сегодня, но должны увеличивать заказы; если реклама не растит заказы — она убыточна.
- **Маржа до СПП, %** = (Маржа, ₽ / Продажи до СПП, ₽) × 100.
- **Годовая доходность, %** = Маржа до СПП, % / Оборот продаж, дни × 365.
- **Δабс** = п2 − п1 (положительное = рост, отрицательное = падение).
- **Δ%** = (п2 − п1) / |п1| × 100. Если п1 = 0 → «—» (нет базы для сравнения).
- **Приоритет**: 1) Маржа ₽, 2) Годовая доходность %, 3) Маржа до СПП %.

### Воронка WB (если доступна)
- **Показы** → **Переходы в карточку** (CTR) → **Положили в корзину** (Conv2Cart) → **Заказали** (Conv2Order) → **Выкупили** (PurchaseRate).
- **CTR, %** = Переходы / Показы × 100.
- **Conv2Cart, %** = Корзина / Переходы × 100 — качество карточки/цены.
- **Conv2Order, %** = Заказали / Переходы × 100 — итоговая конверсия.
- **PurchaseRate, %** = Выкупили / Заказали × 100.
- Связь с экономикой: «Ср.чек заказа после СПП» = прокси цены для покупателя; реклама влияет на верх воронки (показы/переходы), а цена — на конверсию.

### True Ads Stats WB (если доступна, источник: WB API)
- **ВАЖНО**: Funnel CTR (из Воронки) ≠ True Ads CTR (из рекламного кабинета). Funnel = общий трафик (органика+реклама+прямые). Ads = только рекламные показы/клики.
- **True_CTR_ads, %** = Клики(ads) / Показы(ads) × 100 — рекламный CTR.
- **CPC** = Затраты / Клики(ads) — стоимость клика.
- **CPM** = Затраты / Показы(ads) × 1000 — стоимость 1000 показов.
- **CR_ads, %** = Заказы из рекламы / Клики(ads) × 100 — конверсия рекламы в заказ.
- **CartRate_ads, %** = Корзина из рекламы / Клики(ads) × 100.
- Сопоставление: nmId (WB API) → vendorCode (через воронку) → Артикул (экономика).
"""


# ---------------------------------------------------------------------------
# Форматирование таблиц в CSV-блоки
# ---------------------------------------------------------------------------

def _df_to_csv_block(df: pd.DataFrame, title: str) -> str:
    """Превращает DataFrame в Markdown CSV-блок (без обрезки — все строки)."""
    if df.empty:
        return f"### {title}\n\n*Нет данных*\n\n"
    out = df.copy()
    for col in out.select_dtypes(include=["float"]).columns:
        out[col] = out[col].round(1)
    csv = out.to_csv(index=False)
    return f"### {title}\n\n```csv\n{csv}```\n\n"


def _series_to_md(s: pd.Series, title: str) -> str:
    """Series → Markdown таблица."""
    if s.empty:
        return f"### {title}\n\n*Нет данных*\n\n"
    lines = [f"### {title}\n", "| Метрика | Значение |", "|---|---|"]
    for k, v in s.items():
        if pd.isna(v):
            lines.append(f"| {k} | — |")
        elif isinstance(v, float):
            lines.append(f"| {k} | {v:,.1f} |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n\n"


def _kpi_to_md(df: pd.DataFrame, title: str) -> str:
    """KPI DataFrame → Markdown таблица."""
    if df.empty:
        return f"### {title}\n\n*Нет данных*\n\n"
    lines = [f"### {title}\n", "| Метрика | п1 | п2 | Δабс | Δ% |", "|---|---|---|---|---|"]
    for _, row in df.iterrows():
        vals = []
        for c in ["Метрика", "п1", "п2", "Δабс", "Δ%"]:
            v = row.get(c, "")
            if pd.isna(v):
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:,.1f}")
            else:
                vals.append(str(v))
        lines.append(f"| {' | '.join(vals)} |")
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Хелперы для воронки
# ---------------------------------------------------------------------------

def _has_funnel_cols(df: pd.DataFrame) -> bool:
    """Проверяет наличие funnel-колонок в DataFrame."""
    if df is None or df.empty:
        return False
    return any(c.startswith("F_") for c in df.columns)


def _coverage_md(coverage: dict) -> str:
    """Coverage-информация в Markdown."""
    econ = coverage.get("econ_total", 0)
    m1 = coverage.get("matched_p1", 0)
    m2 = coverage.get("matched_p2", 0)
    pct1 = f"{m1/econ*100:.0f}%" if econ else "—"
    pct2 = f"{m2/econ*100:.0f}%" if econ else "—"
    return (
        f"**Funnel coverage**: экономика {econ} артикулов, "
        f"совпало п1: {m1} ({pct1}), совпало п2: {m2} ({pct2}).\n\n"
    )


def _ads_coverage_md(ads_cov: dict) -> str:
    """Ads coverage-информация в Markdown."""
    total = ads_cov.get("ads_nm_total", 0)
    mf = ads_cov.get("matched_via_funnel", 0)
    mo = ads_cov.get("matched_via_override", 0)
    unm = ads_cov.get("unmatched", 0)
    sp_pct = ads_cov.get("spend_matched_pct", 0)
    return (
        f"**Ads coverage**: {total} nmId, matched via funnel: {mf}, "
        f"override: {mo}, unmatched: {unm}, spend matched: {sp_pct:.0f}%.\n\n"
    )


def _source_info_md(funnel_src: str = "Excel", ads_src: str = "Off") -> str:
    """Информация об источниках данных."""
    return (
        f"**Data sources**: Funnel = {funnel_src}, Ads = {ads_src}.\n"
        f"**ВАЖНО**: Funnel CTR ≠ True Ads CTR (разные источники).\n\n"
    )


# ---------------------------------------------------------------------------
# Краткий промпт
# ---------------------------------------------------------------------------

def build_prompt_brief(
    df_wb: Optional[pd.DataFrame],
    df_ozon: Optional[pd.DataFrame],
    wb_merged: Optional[pd.DataFrame] = None,
    coverage: Optional[dict] = None,
    ads_cov: Optional[dict] = None,
    funnel_src: str = "Excel",
    ads_src: str = "Off",
) -> str:
    """Генерирует PROMPT_КРАТКИЙ — только данные (таблицы), как в интерфейсе."""
    parts: list[str] = []

    if df_wb is not None and not df_wb.empty:
        parts.append("# WB\n\n")
        kpi = build_kpi_table(df_wb)
        parts.append(_kpi_to_md(kpi, "KPI WB"))
        top, bottom = build_top_articles(df_wb, n=10)
        parts.append(_df_to_csv_block(top, "ТОП-10 артикулов WB (по марже п2)"))
        parts.append(_df_to_csv_block(bottom, "Анти-ТОП-10 артикулов WB"))
        scale = build_scale_candidates(df_wb, n=10)
        parts.append(_df_to_csv_block(scale, "Кандидаты на масштабирование WB"))

        src_f = wb_merged if wb_merged is not None else df_wb
        if _has_funnel_cols(src_f):
            parts.append("## WB Funnel (краткий)\n\n")
            fkpi = build_funnel_kpi(src_f)
            parts.append(_kpi_to_md(fkpi, "Воронка WB — KPI"))
            cgp = build_conversion_growth_points(src_f, n=5)
            parts.append(_df_to_csv_block(cgp, "Точки роста конверсий WB (ТОП-5)"))

        if has_ads_cols(src_f):
            parts.append("## WB True Ads (краткий)\n\n")
            akpi = build_ads_kpi(src_f)
            parts.append(_kpi_to_md(akpi, "True Ads WB — KPI"))
            ads_art = build_ads_by_article(src_f, n=5)
            parts.append(_df_to_csv_block(ads_art, "ТОП-5 по затратам рекламы (деградация)"))
        elif ads_src != "Off":
            parts.append("## WB True Ads\n\n### True Ads WB\n\n*Нет данных*\n\n")

    if df_ozon is not None and not df_ozon.empty:
        parts.append("# Ozon\n\n")
        kpi = build_kpi_table(df_ozon)
        parts.append(_kpi_to_md(kpi, "KPI Ozon"))
        top, bottom = build_top_articles(df_ozon, n=10)
        parts.append(_df_to_csv_block(top, "ТОП-10 артикулов Ozon (по марже п2)"))
        parts.append(_df_to_csv_block(bottom, "Анти-ТОП-10 артикулов Ozon"))
        scale = build_scale_candidates(df_ozon, n=10)
        parts.append(_df_to_csv_block(scale, "Кандидаты на масштабирование Ozon"))

    if df_wb is not None and df_ozon is not None and not df_wb.empty and not df_ozon.empty:
        parts.append("# WB + Ozon (общий итог)\n\n")
        _wb_copy = df_wb.copy()
        _wb_copy["_platform"] = "WB"
        _oz_copy = df_ozon.copy()
        _oz_copy["_platform"] = "Ozon"
        combined = pd.concat([_wb_copy, _oz_copy], ignore_index=True)
        kpi = build_kpi_table(combined)
        parts.append(_kpi_to_md(kpi, "Общий KPI WB+Ozon"))
        top_m = build_top_models(combined, n=10)
        parts.append(_df_to_csv_block(top_m, "ТОП-10 моделей WB+Ozon"))

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Детальный промпт
# ---------------------------------------------------------------------------

def build_prompt_detailed(
    df: pd.DataFrame,
    platform: str,
    is_wb: bool = False,
    wb_merged: Optional[pd.DataFrame] = None,
    coverage: Optional[dict] = None,
    ads_cov: Optional[dict] = None,
    funnel_src: str = "Excel",
    ads_src: str = "Off",
) -> str:
    """Генерирует PROMPT_ДЕТАЛЬНЫЙ для одной площадки — только данные (таблицы), как в интерфейсе."""
    parts: list[str] = [f"# {platform}\n\n"]

    kpi = build_kpi_table(df)
    parts.append(_kpi_to_md(kpi, f"KPI {platform}"))

    # По артикулу
    agg_art = agg_by_article(df)
    parts.append(_df_to_csv_block(agg_art, f"Агрегация по Артикулу ({platform})"))

    # По модели
    agg_mod = agg_by_model(df)
    parts.append(_df_to_csv_block(agg_mod, f"Агрегация по Модели ({platform})"))

    # По Color code + Коллекция
    agg_cc = agg_by_color_collection(df)
    parts.append(_df_to_csv_block(agg_cc, f"Агрегация по Color code+Коллекция ({platform})"))

    # По статусу
    agg_st = agg_by_status(df)
    parts.append(_df_to_csv_block(agg_st, f"Агрегация по Статусу ({platform})"))

    # Склейка WB
    if is_wb:
        agg_gl = agg_by_glue(df)
        parts.append(_df_to_csv_block(agg_gl, f"Агрегация по Склейке ({platform})"))
        glue_detail = build_glue_analysis(df)
        parts.append(_df_to_csv_block(glue_detail, f"Детализация склеек ({platform})"))

    # Реклама будущее
    ad_f = build_ad_future(df)
    parts.append(_df_to_csv_block(ad_f, f"Реклама → Будущее (заказы) ({platform})"))

    # Реклама настоящее
    ad_c = build_ad_current(df)
    parts.append(_df_to_csv_block(ad_c, f"Реклама → Настоящее (продажи) ({platform})"))

    # Склад
    wh = build_warehouse(df)
    parts.append(_df_to_csv_block(wh, f"Склад/Оборотка ({platform})"))

    # Лаг цен
    pl = build_price_lag(df)
    parts.append(_df_to_csv_block(pl, f"Лаг цен ({platform})"))

    # Цветовые аномалии
    ca = build_color_anomalies(df)
    parts.append(_df_to_csv_block(ca, f"Цветовые аномалии ({platform})"))

    _has_fn = False
    if is_wb:
        src_f = wb_merged if wb_merged is not None else df
        if _has_funnel_cols(src_f):
            _has_fn = True
            parts.append("\n## WB Funnel (детальный)\n\n")
            fkpi = build_funnel_kpi(src_f)
            parts.append(_kpi_to_md(fkpi, "Воронка WB — KPI"))
            diag = build_funnel_economics_diag(src_f)
            parts.append(_df_to_csv_block(diag, "Воронка × Экономика: диагностика"))
            cgp = build_conversion_growth_points(src_f, n=15)
            parts.append(_df_to_csv_block(cgp, "Точки роста конверсий"))
            f_mod = funnel_agg_by_model(src_f)
            parts.append(_df_to_csv_block(f_mod, "Воронка: агрегация по Модели"))
            f_st = funnel_agg_by_status(src_f)
            parts.append(_df_to_csv_block(f_st, "Воронка: агрегация по Статусу"))
            f_gl = funnel_agg_by_glue(src_f)
            parts.append(_df_to_csv_block(f_gl, "Воронка: агрегация по Склейке"))

    # True Ads WB (детальный)
    _has_ad = False
    if is_wb:
        src_a = wb_merged if wb_merged is not None else df
        if has_ads_cols(src_a):
            _has_ad = True
            parts.append("\n## WB True Ads (детальный)\n\n")
            akpi = build_ads_kpi(src_a)
            parts.append(_kpi_to_md(akpi, "True Ads WB — KPI"))
            ads_art = build_ads_by_article(src_a, n=20)
            parts.append(_df_to_csv_block(ads_art, "True Ads эффективность по артикулам"))
            afp = build_ads_funnel_price(src_a, n=15)
            parts.append(_df_to_csv_block(afp, "Ads × Funnel × Price"))
            a_mod = ads_agg_by_model(src_a)
            parts.append(_df_to_csv_block(a_mod, "True Ads: агрегация по Модели"))
            a_st = ads_agg_by_status(src_a)
            parts.append(_df_to_csv_block(a_st, "True Ads: агрегация по Статусу"))
            a_gl = ads_agg_by_glue(src_a)
            parts.append(_df_to_csv_block(a_gl, "True Ads: агрегация по Склейке"))
        elif ads_src != "Off":
            parts.append("\n## WB True Ads\n\n### True Ads WB\n\n*Нет данных*\n\n")

    return "\n".join(parts).strip()


def _build_detailed_tables_only(
    df: pd.DataFrame,
    platform: str,
    is_wb: bool = False,
    wb_merged: Optional[pd.DataFrame] = None,
) -> list[str]:
    parts: list[str] = []

    kpi = build_kpi_table(df)
    parts.append(_df_to_csv_block(kpi, f"KPI {platform}"))

    agg_art = agg_by_article(df)
    parts.append(_df_to_csv_block(agg_art, f"Агрегация по Артикулу ({platform})"))

    agg_mod = agg_by_model(df)
    parts.append(_df_to_csv_block(agg_mod, f"Агрегация по Модели ({platform})"))

    agg_cc = agg_by_color_collection(df)
    parts.append(_df_to_csv_block(agg_cc, f"Агрегация по Color code+Коллекция ({platform})"))

    agg_st = agg_by_status(df)
    parts.append(_df_to_csv_block(agg_st, f"Агрегация по Статусу ({platform})"))

    if is_wb:
        agg_gl = agg_by_glue(df)
        parts.append(_df_to_csv_block(agg_gl, f"Агрегация по Склейке ({platform})"))
        glue_detail = build_glue_analysis(df)
        parts.append(_df_to_csv_block(glue_detail, f"Детализация склеек ({platform})"))

    ad_f = build_ad_future(df)
    parts.append(_df_to_csv_block(ad_f, f"Реклама → Будущее (заказы) ({platform})"))

    ad_c = build_ad_current(df)
    parts.append(_df_to_csv_block(ad_c, f"Реклама → Настоящее (продажи) ({platform})"))

    wh = build_warehouse(df)
    parts.append(_df_to_csv_block(wh, f"Склад/Оборотка ({platform})"))

    pl = build_price_lag(df)
    parts.append(_df_to_csv_block(pl, f"Лаг цен ({platform})"))

    ca = build_color_anomalies(df)
    parts.append(_df_to_csv_block(ca, f"Цветовые аномалии ({platform})"))

    if is_wb:
        src_f = wb_merged if wb_merged is not None else df
        if _has_funnel_cols(src_f):
            fkpi = build_funnel_kpi(src_f)
            parts.append(_df_to_csv_block(fkpi, "Воронка WB — KPI"))
            diag = build_funnel_economics_diag(src_f)
            parts.append(_df_to_csv_block(diag, "Воронка × Экономика: диагностика"))
            cgp = build_conversion_growth_points(src_f, n=15)
            parts.append(_df_to_csv_block(cgp, "Точки роста конверсий"))
            f_mod = funnel_agg_by_model(src_f)
            parts.append(_df_to_csv_block(f_mod, "Воронка: агрегация по Модели"))
            f_st = funnel_agg_by_status(src_f)
            parts.append(_df_to_csv_block(f_st, "Воронка: агрегация по Статусу"))
            f_gl = funnel_agg_by_glue(src_f)
            parts.append(_df_to_csv_block(f_gl, "Воронка: агрегация по Склейке"))

        src_a = wb_merged if wb_merged is not None else df
        if has_ads_cols(src_a):
            akpi = build_ads_kpi(src_a)
            parts.append(_df_to_csv_block(akpi, "True Ads WB — KPI"))
            ads_art = build_ads_by_article(src_a, n=20)
            parts.append(_df_to_csv_block(ads_art, "True Ads эффективность по артикулам"))
            afp = build_ads_funnel_price(src_a, n=15)
            parts.append(_df_to_csv_block(afp, "Ads × Funnel × Price"))
            a_mod = ads_agg_by_model(src_a)
            parts.append(_df_to_csv_block(a_mod, "True Ads: агрегация по Модели"))
            a_st = ads_agg_by_status(src_a)
            parts.append(_df_to_csv_block(a_st, "True Ads: агрегация по Статусу"))
            a_gl = ads_agg_by_glue(src_a)
            parts.append(_df_to_csv_block(a_gl, "True Ads: агрегация по Склейке"))

    return parts


def build_prompt_detailed_all_platforms(
    df_wb: Optional[pd.DataFrame],
    df_ozon: Optional[pd.DataFrame],
    wb_merged: Optional[pd.DataFrame] = None,
) -> str:
    parts: list[str] = []

    if df_wb is not None and not df_wb.empty:
        parts.extend(_build_detailed_tables_only(df_wb, "WB", is_wb=True, wb_merged=wb_merged))

    if df_ozon is not None and not df_ozon.empty:
        parts.extend(_build_detailed_tables_only(df_ozon, "Ozon", is_wb=False))

    return "\n".join(parts).strip()


def _build_tasks_detailed(platform: str, has_funnel: bool = False, has_ads: bool = False) -> str:
    funnel_tasks = ""
    if has_funnel:
        funnel_tasks = """
8. Для блока «Воронка × Экономика: диагностика»:
   - выдели артикулы с флагом «Реклама↑Трафик↓» — рекомендуй сменить креативы, ключевые слова или снизить бюджет.
   - выдели артикулы с «Цена↑Конверсия↓» — оцени ценовую чувствительность и предложи тест цены.
   - выдели артикулы с «Показы↓» — гипотезы: смена позиции, алгоритм МП, потеря рекламы, SEO.
9. Для блока «Точки роста конверсий»:
   - где Conv2Cart низкая при высоком трафике — предложи улучшения карточки (фото, описание, отзывы, цена).
   - где Conv2Cart высокая, но Conv2Order низкая — предложи проверить условия доставки, цену, конкурентов.
   - где конверсии растут при стабильной цене — выдели как успешный кейс, предложи масштабировать.
10. Для воронки по Склейкам:
   - выдели склейки, где высокий трафик и низкая конверсия — проблема ассортимента или «паразитов» в склейке.
"""
    ads_tasks = ""
    if has_ads:
        ads_tasks = """
11. Для блока «True Ads Stats»:
   - выдели артикулы с флагом «Spend↑Orders↓» — предложи снизить ставки, сменить креативы или отключить кампанию.
   - выдели артикулы с «CTR↓» — проблема релевантности / позиции / конкурентов.
   - выдели артикулы с «CPC↑CR↓» — предложи снизить ставку или улучшить карточку.
12. Сопоставь True Ads CTR с Funnel CTR (если есть):
   - если Ads CTR высокий, но Funnel Conv2Order низкий — проблема не в рекламе, а в карточке/цене.
   - если Ads CTR низкий и Funnel CTR низкий — проблема в карточке / фото / SEO.
13. Для блока «Ads × Funnel × Price»:
   - выдели артикулы, где реклама эффективна (CR_ads высокий, CPC низкий) — масштабировать.
   - выдели артикулы, где цена выросла и CR_ads упал — тест цены.
14. Для агрегаций Ads по Модели/Статусу/Склейке:
   - выдели группы с неэффективными рекламными затратами и предложи перераспределить бюджет.
"""
    return f"""\
---

## Tasks for AI ({platform})

1. Проанализируй каждую агрегацию (Артикул, Модель, Color code+Коллекция, Статус{", Склейка" if platform == "WB" else ""}) и дай конкретные рекомендации.
2. Для блока «Реклама → Будущее (заказы)»:
   - выдели артикулы, где реклама растёт, а заказы не растут — рекомендуй снизить бюджет или сменить стратегию.
   - выдели артикулы, где заказы растут при умеренном DRR — рекомендуй масштабировать.
3. Для блока «Реклама → Настоящее (продажи)»:
   - выдели артикулы, где DRR продаж вырос, а маржа упала — предложи корректировку.
4. Для блока «Склад/Оборотка»:
   - выдели красные флаги (высокий остаток + долгий оборот + высокое хранение).
   - предложи действия: распродажа, перемещение, вывод.
5. Для блока «Лаг цен»:
   - объясни расхождения Ср.чек заказа vs продажи и предложи, менять ли цены.
6. Для блока «Цветовые аномалии»:
   - если цвет топ в одной модели и провал в другой — предложи гипотезы почему.
{"7. Для блока «Склейки WB»:" if platform == "WB" else ""}
{"   - выдели доноров и паразитов, предложи действия." if platform == "WB" else ""}
{"   - при каннибализации — предложи разделить склейку или изменить цены." if platform == "WB" else ""}
{funnel_tasks}{ads_tasks}
Для каждого действия укажи:
- **Приоритет**: P0 / P1 / P2
- **Ожидаемый эффект**: Маржа ₽ и Годовая доходность %
- **Гипотезы / что проверить**, если данных недостаточно.

## Questions (≤5)

Задай до 5 уточняющих вопросов, только если без них невозможно принять ключевое решение.
"""
