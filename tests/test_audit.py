"""Тесты аудита: дельты consistency, масштабирование, склад, форматирование."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.schema import col_p, CALC_MARGIN_RATE, CALC_ANNUAL_YIELD
from src.metrics import compute_deltas, add_calculated_metrics
from src.aggregations import aggregate, agg_by_article
from src.tables import build_scale_candidates, build_warehouse, build_top_articles
from src.display_utils import (
    fmt_int, fmt_rub, fmt_pct, fmt_days, fmt_delta_pct, fmt_delta_abs,
    format_df_for_display,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(**overrides):
    """Минимальный DataFrame с п1/п2 для тестов."""
    base = {
        "Артикул": ["A1", "A2", "A3"],
        "Статус": ["Продается", "Выводим", "Архив"],
        "Модель": ["M1", "M1", "M2"],
        "Склейка на WB": ["G1", "G1", "G2"],
        "Color code": ["C1", "C1", "C2"],
        "Коллекция": ["K1", "K1", "K2"],
        col_p("Маржа, ₽", "п1"): [100.0, 50.0, -30.0],
        col_p("Маржа, ₽", "п2"): [120.0, -10.0, -50.0],
        col_p("Себес-ть, ₽", "п1"): [200.0, 100.0, 80.0],
        col_p("Себес-ть, ₽", "п2"): [200.0, 100.0, 80.0],
        col_p("Продажи, шт", "п1"): [50.0, 20.0, 10.0],
        col_p("Продажи, шт", "п2"): [60.0, 15.0, 5.0],
        col_p("Остатки, шт", "п1"): [300.0, 50.0, 10.0],
        col_p("Остатки, шт", "п2"): [300.0, 50.0, 10.0],
        col_p("Оборот продаж, дни", "п1"): [20.0, 30.0, 5.0],
        col_p("Оборот продаж, дни", "п2"): [25.0, 35.0, 3.0],
        col_p("Заказы, шт", "п1"): [60.0, 25.0, 12.0],
        col_p("Заказы, шт", "п2"): [70.0, 18.0, 6.0],
        col_p("Заказы до СПП, ₽", "п1"): [6000.0, 2500.0, 1200.0],
        col_p("Заказы до СПП, ₽", "п2"): [7000.0, 1800.0, 600.0],
        col_p("Продажи до СПП, ₽", "п1"): [5000.0, 2000.0, 1000.0],
        col_p("Продажи до СПП, ₽", "п2"): [6000.0, 1500.0, 500.0],
        col_p("Реклама внутр., ₽", "п1"): [100.0, 50.0, 0.0],
        col_p("Реклама внутр., ₽", "п2"): [120.0, 60.0, 0.0],
        col_p("Реклама внеш., ₽", "п1"): [0.0, 0.0, 0.0],
        col_p("Реклама внеш., ₽", "п2"): [0.0, 0.0, 0.0],
        col_p("Хранение на ед, ₽", "п1"): [5.0, 3.0, 2.0],
        col_p("Хранение на ед, ₽", "п2"): [5.0, 3.0, 2.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _prepared_df(**overrides):
    """DataFrame с вычисленными метриками и дельтами."""
    df = _make_df(**overrides)
    df = add_calculated_metrics(df)
    df = compute_deltas(df)
    return df


# ---------------------------------------------------------------------------
# 1) Дельты consistency: п2 − п1 везде
# ---------------------------------------------------------------------------

class TestDeltasConsistency:
    """Δабс = п2 − п1, Δ% = (п2−п1)/|п1|×100 — одинаково в metrics и aggregations."""

    def test_compute_deltas_sign(self):
        df = pd.DataFrame({
            col_p("Маржа, ₽", "п1"): [100.0],
            col_p("Маржа, ₽", "п2"): [120.0],
        })
        result = compute_deltas(df)
        # п2 − п1 = 120 − 100 = +20
        assert result["Маржа, ₽ Δабс"].iloc[0] == pytest.approx(20.0)
        # (120−100)/|100|×100 = 20%
        assert result["Маржа, ₽ Δ%"].iloc[0] == pytest.approx(20.0)

    def test_compute_deltas_negative(self):
        df = pd.DataFrame({
            col_p("Маржа, ₽", "п1"): [100.0],
            col_p("Маржа, ₽", "п2"): [80.0],
        })
        result = compute_deltas(df)
        assert result["Маржа, ₽ Δабс"].iloc[0] == pytest.approx(-20.0)
        assert result["Маржа, ₽ Δ%"].iloc[0] == pytest.approx(-20.0)

    def test_compute_deltas_p1_zero(self):
        df = pd.DataFrame({
            col_p("Маржа, ₽", "п1"): [0.0],
            col_p("Маржа, ₽", "п2"): [50.0],
        })
        result = compute_deltas(df)
        assert result["Маржа, ₽ Δабс"].iloc[0] == pytest.approx(50.0)
        assert np.isnan(result["Маржа, ₽ Δ%"].iloc[0])

    def test_compute_deltas_nan(self):
        df = pd.DataFrame({
            col_p("Маржа, ₽", "п1"): [np.nan],
            col_p("Маржа, ₽", "п2"): [50.0],
        })
        result = compute_deltas(df)
        assert np.isnan(result["Маржа, ₽ Δ%"].iloc[0])

    def test_compute_deltas_negative_base(self):
        df = pd.DataFrame({
            col_p("Маржа, ₽", "п1"): [-100.0],
            col_p("Маржа, ₽", "п2"): [50.0],
        })
        result = compute_deltas(df)
        # Δабс = 50 − (−100) = 150
        assert result["Маржа, ₽ Δабс"].iloc[0] == pytest.approx(150.0)
        # Δ% = 150 / |−100| × 100 = 150%
        assert result["Маржа, ₽ Δ%"].iloc[0] == pytest.approx(150.0)

    def test_aggregate_deltas_match_compute_deltas(self):
        """aggregate() и compute_deltas() дают одинаковый знак."""
        df = _prepared_df()
        agg = agg_by_article(df)
        # Для артикула A1: Маржа п2=120, п1=100 → Δабс=+20
        row = agg[agg["Артикул"] == "A1"].iloc[0]
        assert row["Маржа, ₽ Δабс"] == pytest.approx(20.0)
        assert row["Маржа, ₽ Δ%"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 2) Масштабирование: фильтры
# ---------------------------------------------------------------------------

class TestScaleCandidates:
    def test_excludes_vyvodim_and_archive(self):
        df = _prepared_df()
        result = build_scale_candidates(df, n=10, stock_min=0, turnover_min=0, days_cover_min=0)
        if not result.empty and "Статус" in result.columns:
            statuses = result["Статус"].str.strip().str.lower().tolist()
            assert "выводим" not in statuses
            assert "архив" not in statuses

    def test_excludes_zero_stock(self):
        df = _prepared_df(**{col_p("Остатки, шт", "п2"): [0.0, 0.0, 0.0]})
        result = build_scale_candidates(df, n=10, stock_min=1)
        assert result.empty

    def test_excludes_low_turnover(self):
        df = _prepared_df(**{col_p("Оборот продаж, дни", "п2"): [5.0, 5.0, 5.0]})
        result = build_scale_candidates(df, n=10, stock_min=0, turnover_min=10, days_cover_min=0)
        assert result.empty

    def test_positive_margin_only(self):
        df = _prepared_df()
        result = build_scale_candidates(df, n=10, stock_min=0, turnover_min=0, days_cover_min=0)
        margin_c = col_p("Маржа, ₽", "п2")
        if not result.empty and margin_c in result.columns:
            assert (result[margin_c] > 0).all()


# ---------------------------------------------------------------------------
# 3) Склад: абсолютные пороги
# ---------------------------------------------------------------------------

class TestWarehouse:
    def test_red_flag_absolute_thresholds(self):
        df = _prepared_df(**{
            col_p("Остатки, шт", "п2"): [500.0, 50.0, 10.0],
            col_p("Оборот продаж, дни", "п2"): [300.0, 100.0, 5.0],
        })
        result = build_warehouse(df, stock_thr=100, turnover_thr=200)
        if "🔴 Флаг" in result.columns:
            flags = result["🔴 Флаг"].tolist()
            # A1: 500≥100 и 300≥200 → 🔴
            assert "🔴" in flags
            # Должен быть ровно 1 красный флаг (A1)
            assert flags.count("🔴") == 1

    def test_no_quantile_dependency(self):
        """Флаг не зависит от распределения данных (нет квантилей)."""
        df1 = _prepared_df(**{
            col_p("Остатки, шт", "п2"): [150.0, 50.0, 10.0],
            col_p("Оборот продаж, дни", "п2"): [250.0, 100.0, 5.0],
        })
        df2 = _prepared_df(**{
            col_p("Остатки, шт", "п2"): [150.0, 150.0, 150.0],
            col_p("Оборот продаж, дни", "п2"): [250.0, 250.0, 250.0],
        })
        r1 = build_warehouse(df1, stock_thr=100, turnover_thr=200)
        r2 = build_warehouse(df2, stock_thr=100, turnover_thr=200)
        # В df1 A1 = 🔴; в df2 все три = 🔴 (абсолютные пороги, не квантили)
        if "🔴 Флаг" in r1.columns and "🔴 Флаг" in r2.columns:
            assert r1["🔴 Флаг"].tolist().count("🔴") == 1
            assert r2["🔴 Флаг"].tolist().count("🔴") == 3

    def test_why_flag_column(self):
        df = _prepared_df(**{
            col_p("Остатки, шт", "п2"): [500.0, 50.0, 10.0],
            col_p("Оборот продаж, дни", "п2"): [300.0, 100.0, 5.0],
        })
        result = build_warehouse(df, stock_thr=100, turnover_thr=200)
        assert "Почему флаг" in result.columns


# ---------------------------------------------------------------------------
# 4) Анти-ТОП: маржа < 0, исключить Выводим/Архив
# ---------------------------------------------------------------------------

class TestAntiTop:
    def test_anti_top_negative_margin_only(self):
        df = _prepared_df()
        _, bottom = build_top_articles(df, n=10)
        margin_c = col_p("Маржа, ₽", "п2")
        if not bottom.empty and margin_c in bottom.columns:
            assert (bottom[margin_c] < 0).all()

    def test_anti_top_excludes_vyvodim_archive(self):
        df = _prepared_df()
        _, bottom = build_top_articles(df, n=10)
        if not bottom.empty and "Статус" in bottom.columns:
            statuses = bottom["Статус"].str.strip().str.lower().tolist()
            assert "выводим" not in statuses
            assert "архив" not in statuses


# ---------------------------------------------------------------------------
# 5) Форматирование чисел
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_fmt_int_normal(self):
        assert fmt_int(1234567) == "1 234 567"

    def test_fmt_int_nan(self):
        assert fmt_int(np.nan) == "—"

    def test_fmt_rub(self):
        assert fmt_rub(1234567) == "1 234 567 ₽"

    def test_fmt_rub_nan(self):
        assert fmt_rub(np.nan) == "—"

    def test_fmt_pct(self):
        assert fmt_pct(12.345) == "12.3%"

    def test_fmt_pct_nan(self):
        assert fmt_pct(np.nan) == "—"

    def test_fmt_delta_pct_positive(self):
        assert fmt_delta_pct(15.5) == "+15.5%"

    def test_fmt_delta_pct_negative(self):
        assert fmt_delta_pct(-10.3) == "-10.3%"

    def test_fmt_delta_pct_nan(self):
        assert fmt_delta_pct(np.nan) == "—"

    def test_fmt_delta_abs_positive(self):
        assert fmt_delta_abs(1500) == "+1 500"

    def test_fmt_delta_abs_negative(self):
        assert fmt_delta_abs(-500) == "-500"

    def test_format_df_for_display_nan_to_dash(self):
        df = pd.DataFrame({"Маржа, ₽ п2": [1000.0, np.nan]})
        display = format_df_for_display(df)
        assert display["Маржа, ₽ п2"].iloc[1] == "—"

    def test_format_df_for_display_delta_pct(self):
        df = pd.DataFrame({"Маржа, ₽ Δ%": [15.5, -10.3]})
        display = format_df_for_display(df)
        assert display["Маржа, ₽ Δ%"].iloc[0] == "+15.5%"
        assert display["Маржа, ₽ Δ%"].iloc[1] == "-10.3%"
