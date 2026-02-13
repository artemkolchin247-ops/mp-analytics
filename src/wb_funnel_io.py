"""Загрузка и обработка файлов воронки WB (лист «Товары»)."""
from __future__ import annotations

import re
from io import BytesIO
from typing import Tuple, List

import numpy as np
import pandas as pd

from src.schema import normalize_header
from src.utils import safe_div


# ---------------------------------------------------------------------------
# Нормализация ключа артикула для join
# ---------------------------------------------------------------------------

def normalize_article_key(val) -> str:
    """Мягкая нормализация артикула для join экономики и воронки.

    - trim, lower, множественные пробелы → один,
    - ё → е,
    - пробелы вокруг дефисов/слэшей убираются.
    """
    s = str(val).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("ё", "е")
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return s


# ---------------------------------------------------------------------------
# Колонки воронки
# ---------------------------------------------------------------------------

FUNNEL_SHEET = "Товары"

FUNNEL_ID_COL = "Артикул продавца"

FUNNEL_SUM_COLS: list[str] = [
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

FUNNEL_RATE_COLS: list[str] = [
    "CTR",
    "Конверсия в корзину, %",
    "Конверсия в заказ, %",
    "Процент выкупа",
    "Процент выкупа ВБ клуб",
]

# Вычисляемые конверсии
CALC_CTR = "CTR, %"
CALC_CONV_CART = "Conv2Cart, %"
CALC_CONV_ORDER = "Conv2Order, %"
CALC_PURCHASE_RATE = "PurchaseRate, %"


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def load_funnel_excel(file: BytesIO) -> Tuple[pd.DataFrame, List[str]]:
    """Читает лист 'Товары' из Excel-файла воронки WB.

    Returns:
        (df, warnings) — DataFrame и список предупреждений.
    """
    warnings: List[str] = []
    try:
        xl = pd.ExcelFile(file, engine="openpyxl")
    except Exception as e:
        warnings.append(f"Ошибка чтения файла воронки: {e}")
        return pd.DataFrame(), warnings

    if FUNNEL_SHEET not in xl.sheet_names:
        available = ", ".join(xl.sheet_names[:10])
        warnings.append(
            f"Лист «{FUNNEL_SHEET}» не найден. Доступные листы: {available}"
        )
        return pd.DataFrame(), warnings

    df = xl.parse(FUNNEL_SHEET, header=1)
    df.columns = [normalize_header(c) for c in df.columns]

    # Проверка ключевой колонки
    if FUNNEL_ID_COL not in df.columns:
        warnings.append(f"Колонка «{FUNNEL_ID_COL}» не найдена в листе «{FUNNEL_SHEET}».")
        return pd.DataFrame(), warnings

    # Проверка наличия числовых колонок
    existing = set(df.columns)
    missing = [c for c in FUNNEL_SUM_COLS if c not in existing]
    if missing:
        warnings.append(
            f"Отсутствуют колонки воронки ({len(missing)}): {', '.join(missing[:8])}"
            + ("…" if len(missing) > 8 else "")
        )

    # Преобразование типов
    for c in FUNNEL_SUM_COLS + FUNNEL_RATE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, warnings


# ---------------------------------------------------------------------------
# Агрегация по артикулу (суммы по датам + пересчёт конверсий)
# ---------------------------------------------------------------------------

def aggregate_funnel_by_article(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует воронку по артикулу: суммы счётчиков + пересчёт конверсий."""
    if df.empty or FUNNEL_ID_COL not in df.columns:
        return pd.DataFrame()

    # Нормализованный ключ
    df = df.copy()
    df["_art_key"] = df[FUNNEL_ID_COL].apply(normalize_article_key)

    # Суммируемые колонки
    existing_sum = [c for c in FUNNEL_SUM_COLS if c in df.columns]
    agg_dict = {c: "sum" for c in existing_sum}
    # Сохраняем оригинальный артикул (первое значение)
    agg_dict[FUNNEL_ID_COL] = "first"

    result = df.groupby("_art_key", dropna=False).agg(agg_dict).reset_index()

    # Пересчёт конверсий из агрегированных сумм
    result = _recompute_rates(result)

    return result


def _recompute_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Пересчитывает конверсии из агрегированных абсолютных показателей."""
    df = df.copy()

    shows = "Показы"
    clicks = "Переходы в карточку"
    cart = "Положили в корзину"
    ordered = "Заказали, шт"
    bought = "Выкупили, шт"

    if shows in df.columns and clicks in df.columns:
        df[CALC_CTR] = safe_div(df[clicks], df[shows]) * 100

    if clicks in df.columns and cart in df.columns:
        df[CALC_CONV_CART] = safe_div(df[cart], df[clicks]) * 100

    if clicks in df.columns and ordered in df.columns:
        df[CALC_CONV_ORDER] = safe_div(df[ordered], df[clicks]) * 100

    if ordered in df.columns and bought in df.columns:
        df[CALC_PURCHASE_RATE] = safe_div(df[bought], df[ordered]) * 100

    return df
