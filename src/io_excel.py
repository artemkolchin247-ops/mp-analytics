"""Загрузка Excel, нормализация заголовков, преобразование типов, валидация."""
from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
import pandas as pd

from src.schema import (
    ID_COLS,
    METRIC_BASES,
    UploadValidation,
    all_metric_cols,
    col_p,
    normalize_header,
)


# ---------------------------------------------------------------------------
# Загрузка и нормализация
# ---------------------------------------------------------------------------

def load_excel(file: BytesIO) -> pd.DataFrame:
    """Читаем первый лист Excel-файла."""
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = [normalize_header(c) for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------

TOTAL_ROW_PATTERNS = {"total", "итого", "итог", "всего", "grand total"}


def _drop_total_rows(df: pd.DataFrame, v: UploadValidation) -> pd.DataFrame:
    """Удаляет строки-итоги (total/итого) из колонки Артикул."""
    if "Артикул" not in df.columns:
        return df
    mask = df["Артикул"].astype(str).str.strip().str.lower().isin(TOTAL_ROW_PATTERNS)
    n_dropped = mask.sum()
    if n_dropped > 0:
        v.info_warnings.append(
            f"Удалены строки-итоги из колонки «Артикул»: {n_dropped} шт. "
            f"(значения: {', '.join(df.loc[mask, 'Артикул'].astype(str).unique())}). "
            f"Приложение рассчитывает итоги самостоятельно."
        )
        df = df[~mask].reset_index(drop=True)
    return df


def validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, UploadValidation]:
    """Проверка колонок, преобразование типов, детект аномалий."""
    v = UploadValidation()

    # --- Удаление строк-итогов ---
    df = _drop_total_rows(df, v)

    existing = set(df.columns)

    # --- Проверка ID-колонок ---
    for c in ID_COLS:
        if c not in existing:
            v.missing_id_cols.append(c)

    # --- Проверка метрик ---
    metric_cols = all_metric_cols()
    for c in metric_cols:
        if c not in existing:
            v.missing_metric_cols.append(c)

    # --- Преобразование числовых колонок ---
    for c in metric_cols:
        if c in existing:
            original = df[c].copy()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            bad = original.notna() & df[c].isna()
            n_bad = bad.sum()
            if n_bad > 0:
                v.type_conversion_errors.append(
                    f"'{c}': {n_bad} значений не удалось привести к числу"
                )

    # --- Детект аномалий ---
    _detect_anomalies(df, v, existing)

    return df, v


def _detect_anomalies(df: pd.DataFrame, v: UploadValidation, existing: set[str]) -> None:
    """Поиск аномалий в данных. Разделяет на критичные и информационные."""
    for p in ("п1", "п2"):
        ost = col_p("Остатки, шт", p)
        sales_qty = col_p("Продажи, шт", p)
        orders_qty = col_p("Заказы, шт", p)
        turn_sales = col_p("Оборот продаж, дни", p)

        # Критичное: отрицательные остатки — скорее всего ошибка источника
        if ost in existing:
            neg = (df[ost] < 0).sum()
            if neg > 0:
                v.critical_warnings.append(
                    f"Отрицательные остатки ({p}): {neg} строк. "
                    f"Проверьте корректность выгрузки — остатки не должны быть < 0."
                )

        # Информационное: оборот = 0 при наличии продаж
        if turn_sales in existing and sales_qty in existing:
            mask = (df[turn_sales] == 0) & (df[sales_qty] > 0)
            cnt = mask.sum()
            if cnt > 0:
                v.info_warnings.append(
                    f"Оборот продаж = 0 дней при наличии продаж ({p}): {cnt} строк. "
                    f"Вероятно, товары с очень быстрым оборотом (<1 дня) или особенность расчёта МП. "
                    f"На анализ не влияет, но годовая доходность для этих строк будет NaN."
                )

        # Информационное: продажи > заказов — нормальный лаг
        if sales_qty in existing and orders_qty in existing:
            mask = df[sales_qty] > df[orders_qty]
            mask = mask & df[sales_qty].notna() & df[orders_qty].notna()
            cnt = mask.sum()
            if cnt > 0:
                v.info_warnings.append(
                    f"Продажи шт > Заказы шт ({p}): {cnt} строк. "
                    f"Это нормально: продажи текущего периода включают заказы из прошлого "
                    f"(лаг заказ → продажа). Не является ошибкой."
                )
