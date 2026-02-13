"""Утилиты отображения: форматирование чисел, display DF, легенды, CSS."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Каноническое определение периодов (единый источник истины)
# ---------------------------------------------------------------------------

PERIOD_DEFINITIONS = (
    "**п2** — текущий период; **п1** — прошлый аналогичный период. "
    "**Δабс** = п2 − п1; **Δ%** = (п2 − п1) / |п1| × 100."
)


# ---------------------------------------------------------------------------
# Форматирование отдельных значений
# ---------------------------------------------------------------------------

def fmt_int(val, decimals: int | None = None, thousand_sep: str = " ") -> str:
    """Число с пробелами-разделителями. Если `decimals` задан — показывает дробную часть.
    NaN → '—'."""
    if pd.isna(val):
        return "—"
    if decimals is None or decimals == 0:
        return f"{int(round(val)):,}".replace(",", thousand_sep)
    return f"{val:,.{decimals}f}".replace(",", thousand_sep).replace(".", ",")


def fmt_rub(val, decimals: int | None = None, thousand_sep: str = " ") -> str:
    """Рубли с суффиксом ₽. `decimals` управляет количеством знаков после запятой.
    NaN → '—'."""
    if pd.isna(val):
        return "—"
    if decimals is None or decimals == 0:
        return f"{int(round(val)):,}".replace(",", thousand_sep) + " ₽"
    num = f"{val:,.{decimals}f}".replace(",", thousand_sep).replace(".", ",")
    return f"{num} ₽"


def fmt_pct(val, decimals: int | None = 1, thousand_sep: str = " ") -> str:
    """Проценты с заданным числом знаков. NaN → '—'."""
    if pd.isna(val):
        return "—"
    if decimals is None:
        return f"{val}%"
    return f"{val:.{decimals}f}%".replace(".", ",")


def fmt_days(val, decimals: int | None = 1, thousand_sep: str = " ") -> str:
    """Дни/дни оборота: формат с заданным числом знаков. NaN → '—'."""
    if pd.isna(val):
        return "—"
    if decimals is None:
        return f"{val}"
    return f"{val:.{decimals}f}".replace(",", thousand_sep).replace(".", ",")


def fmt_delta_pct(val, decimals: int | None = 1, thousand_sep: str = " ") -> str:
    """Δ% со знаком ±. NaN → '—'."""
    if pd.isna(val):
        return "—"
    sign = "+" if val > 0 else ""
    if decimals is None:
        return f"{sign}{val}%"
    return f"{sign}{val:.{decimals}f}%".replace(".", ",")


def fmt_delta_abs(val, decimals: int | None = 0, thousand_sep: str = " ") -> str:
    """Δабс со знаком ±. NaN → '—'."""
    if pd.isna(val):
        return "—"
    sign = "+" if val > 0 else ""
    if decimals is None or decimals == 0:
        return f"{sign}{int(round(val)):,}".replace(",", thousand_sep)
    return f"{sign}{val:,.{decimals}f}".replace(",", thousand_sep).replace(".", ",")


# ---------------------------------------------------------------------------
# Автоформатирование колонки по имени
# ---------------------------------------------------------------------------

def _guess_formatter(col_name: str):
    """Возвращает функцию форматирования по имени колонки."""
    cn = col_name.lower()
    # Все возвращаемые функции должны принимать сигнатуру (val, decimals, thousand_sep)
    if "δ%" in cn or "δ%" in col_name:
        return fmt_delta_pct
    if "δабс" in cn or "δабс" in col_name:
        if "%" in cn:
            return lambda v, decimals=None, thousand_sep=" ": fmt_delta_abs(v, decimals=1, thousand_sep=thousand_sep)
        if "дни" in cn or "дн" in cn:
            return lambda v, decimals=None, thousand_sep=" ": fmt_delta_abs(v, decimals=1, thousand_sep=thousand_sep)
        return fmt_delta_abs
    if "₽" in cn:
        return fmt_rub
    if "%" in cn:
        return fmt_pct
    if "дни" in cn or "дн" in cn or "оборот" in cn:
        return fmt_days
    if "шт" in cn or "остатки" in cn:
        return fmt_int
    return None


def format_df_for_display(df: pd.DataFrame, *, decimals: int | None = None, thousand_sep: str = " ") -> pd.DataFrame:
    """Создаёт display-копию DF с отформатированными строками.

    Исходный DF не изменяется. Возвращает DF со строковыми значениями.
    """
    if df.empty:
        return df

    # Копируем чтобы не изменять оригинал
    out = df.copy()

    for col in out.columns:
        fmt = _guess_formatter(col)
        if fmt is not None:
            out[col] = out[col].apply(lambda v: fmt(v, decimals=decimals, thousand_sep=thousand_sep))
        else:
            # Для числовых колонок без явного формата — хотя бы NaN → '—'
            if hasattr(out[col], 'dtype') and str(out[col].dtype).startswith("float"):
                if decimals is None:
                    out[col] = out[col].apply(lambda v: "—" if pd.isna(v) else v)
                else:
                    out[col] = out[col].apply(lambda v: "—" if pd.isna(v) else (f"{v:,.{decimals}f}".replace(",", thousand_sep).replace(".", ",")))
    return out


def display_copyable_table(
    df: pd.DataFrame,
    use_container_width: bool = True,
    hide_index: bool = True,
    key: Optional[str] = None,
    format_display: bool = True,
    format_options: Optional[Dict] = None,
) -> None:
    """Отображает таблицу с возможностью выделения и копирования всех данных включая заголовки.

    Args:
        df: DataFrame для отображения
        use_container_width: Использовать ли всю ширину контейнера
        hide_index: Скрывать ли индекс
        key: Уникальный ключ для виджета (опционально)
        format_display: Применять ли форматирование к DataFrame (по умолчанию True)
    """
    if df.empty:
        st.info("Нет данных для отображения")
        return

    # Форматируем DataFrame если нужно
    if format_display:
        if format_options is None:
            display_df = format_df_for_display(df)
        else:
            display_df = format_df_for_display(
                df,
                decimals=format_options.get("decimals"),
                thousand_sep=format_options.get("thousand_sep", " "),
            )
    else:
        display_df = df

    # Используем st.dataframe с настройками для копирования
    st.dataframe(
        display_df,
        use_container_width=use_container_width,
        hide_index=hide_index,
        key=key,
    )


# ---------------------------------------------------------------------------
# CSS для Streamlit: wrap заголовков, компактные таблицы
# ---------------------------------------------------------------------------

TABLE_CSS = """\
<style>
/* Перенос длинных заголовков в st.dataframe */
div[data-testid="stDataFrame"] th {
    white-space: normal !important;
    word-wrap: break-word !important;
    max-width: 160px;
    text-align: center;
    user-select: text !important;
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
}
div[data-testid="stDataFrame"] td {
    white-space: nowrap;
    user-select: text !important;
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
}
/* Включаем выделение для всех элементов таблицы */
div[data-testid="stDataFrame"] * {
    user-select: text !important;
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
}
/* Позволяем копировать заголовки столбцов */
div[data-testid="stDataFrame"] thead th {
    cursor: text !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Легенды (markdown)
# ---------------------------------------------------------------------------

LEGEND_WAREHOUSE = """\
**Легенда «Склад / Оборотка»**
| Символ | Значение | Правило | Действие |
|--------|----------|---------|----------|
| 🔴 | Красный флаг | Остатки п2 ≥ {stock_thr} шт **И** Оборот п2 ≥ {turn_thr} дн | Распродажа / вывод / перемещение |
"""

LEGEND_SCALE = """\
**Легенда «Кандидаты на масштабирование»**
Товары с высокой доходностью, достаточными остатками и запасом дней.
Фильтры (п2 = текущий период):
- Статус ≠ «Выводим» / «Архив»
- Остатки п2 ≥ {stock_thr} шт
- Оборот продаж п2 ≥ {turn_thr} дн
- Оборот продаж дни > 60
- ДРР от продаж (до СПП) < 5%
- Запас дней (DaysCover) п2 ≥ {cover_thr} дн
- Маржа до СПП, % п2 > 25%, Маржа п2 > 0
**Масштабировать** = увеличить закупку / поставку / рекламный бюджет.
"""

LEGEND_ANTI_TOP = """\
**Легенда «Анти-ТОП»**
Артикулы с **отрицательной маржой** (п2 < 0).
Статусы «Выводим» и «Архив» исключены.
Действие: пересмотреть цену, снизить расходы или вывести из ассортимента.
"""

LEGEND_AD_FUTURE = """\
**Легенда «Реклама → Будущее (заказы)»**
| Символ | Значение | Правило |
|--------|----------|---------|
| ⚠️ | Реклама↑ Заказы↓ | Δ рекламы ≥ {ad_pct_thr}% **И** ≥ {ad_abs_thr} ₽, при этом заказы не выросли |
⚠️ **Лаг 7–14 дней**: рост рекламы сегодня → рост заказов через 1–2 недели. Если лаг прошёл, а заказы не выросли — реклама неэффективна.
"""

LEGEND_AD_CURRENT = """\
**Легенда «Реклама → Настоящее (продажи)»**
| Символ | Значение | Правило |
|--------|----------|---------|
| ⚠️ | DRR↑ Маржа↓ | DRR продаж вырос, маржа упала |
DRR может расти из-за: (а) роста рекламных расходов, (б) падения продаж. Проверьте оба фактора.
"""

LEGEND_DIAG_FLAGS = """\
**Легенда «Воронка × Экономика: флаги»**
| Символ | Значение | Правило | Действие |
|--------|----------|---------|----------|
| 🔴 | Реклама↑ Трафик↓ | Δ рекламы > 0 **И** Δ переходов ≤ 0 (с порогом значимости) | Сменить креативы / ключевые слова |
| 🟡 | Цена↑ Конверсия↓ | Δ цены после СПП > 0 **И** Δ Conv2Order < 0 | Тест цены |
| 🟠 | Показы↓ | Δ показов < 0 при стабильной цене | Проверить позицию / SEO / рекламу |
"""

LEGEND_COLOR_ANOMALY = """\
**Легенда «Цветовые аномалии»**
Анализ по (Color code, Коллекция, Модель).
⚠️ = в одном цвете+коллекции есть модель с маржой < {ratio}× лучшей (при лучшей марже ≥ {min_margin}₽).
Действие: проверить цены, скидки или позиционирование аномальной модели.
"""

LEGEND_CONVERSION_GROWTH = """\
**Легенда «Точки роста конверсий»**
Анализ моделей с высоким трафиком, но низкой конверсией.
- **Проблема карточки/цены**: высокие показы/переходы, но низкая Conv2Cart (конверсия в корзину)
- **Проблема заказа/доставки**: высокая Conv2Cart, но низкая Conv2Order (конверсия в заказ)
- **✅Контент/позиция работает**: конверсии растут при стабильной цене
Действие: оптимизировать карточки, цены, доставку или усилить работающие каналы.
"""

LEGEND_ICONS_GENERAL = """\
**Общие обозначения**
| Символ | Значение |
|--------|----------|
| 🔴 | Критическая проблема — требует немедленного действия |
| 🟡 | Предупреждение — проверить и скорректировать |
| 🟠 | Внимание — мониторить, возможна проблема |
| ⚠️ | Флаг — аномалия или потенциальная проблема |
| ✅ | Позитивный сигнал — контент/позиция работает |
"""
