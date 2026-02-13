"""Тесты расчётов маржи до СПП, годовой доходности, дельт — включая edge cases."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import compute_margin_rate, compute_annual_yield, add_calculated_metrics, compute_deltas
from src.schema import col_p, CALC_MARGIN_RATE, CALC_ANNUAL_YIELD
from src.utils import safe_div, safe_pct_change


# ---------------------------------------------------------------------------
# Маржа до СПП, %
# ---------------------------------------------------------------------------

class TestComputeMarginRate:
    def test_normal(self):
        assert compute_margin_rate(100.0, 200.0) == pytest.approx(50.0)

    def test_zero_sales(self):
        result = compute_margin_rate(100.0, 0.0)
        assert np.isnan(result)

    def test_zero_margin(self):
        assert compute_margin_rate(0.0, 200.0) == 0.0

    def test_negative_sales(self):
        assert compute_margin_rate(100.0, -200.0) == pytest.approx(-50.0)

    def test_series(self):
        margin = pd.Series([100.0, 200.0])
        sales = pd.Series([200.0, 400.0])
        result = compute_margin_rate(margin, sales)
        expected = pd.Series([50.0, 50.0])
        pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# Годовая доходность
# ---------------------------------------------------------------------------

class TestComputeAnnualYield:
    def test_normal(self):
        # Маржа до СПП, %=50%, turnover=30 дней → 50/30*365 = 608.33
        result = compute_annual_yield(50.0, 30.0)
        assert result == pytest.approx(608.333, rel=1e-2)

    def test_zero_turnover(self):
        result = compute_annual_yield(50.0, 0.0)
        assert np.isnan(result)

    def test_nan_turnover(self):
        result = compute_annual_yield(50.0, np.nan)
        assert np.isnan(result)

    def test_nan_roi(self):
        result = compute_annual_yield(np.nan, 30.0)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# safe_div
# ---------------------------------------------------------------------------

class TestSafeDiv:
    def test_normal(self):
        assert safe_div(10, 5) == 2.0

    def test_zero_denom(self):
        assert np.isnan(safe_div(10, 0))

    def test_nan_denom(self):
        assert np.isnan(safe_div(10, np.nan))

    def test_nan_num(self):
        assert np.isnan(safe_div(np.nan, 5))

    def test_custom_default(self):
        assert safe_div(10, 0, default=0) == 0


# ---------------------------------------------------------------------------
# safe_pct_change
# ---------------------------------------------------------------------------

class TestSafePctChange:
    def test_normal(self):
        # (120 - 100) / |100| * 100 = 20%
        assert safe_pct_change(120.0, 100.0) == pytest.approx(20.0)

    def test_decrease(self):
        assert safe_pct_change(80.0, 100.0) == pytest.approx(-20.0)

    def test_zero_base(self):
        result = safe_pct_change(100.0, 0.0)
        assert np.isnan(result)

    def test_negative_base(self):
        # (50 - (-100)) / |-100| * 100 = 150%
        assert safe_pct_change(50.0, -100.0) == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# add_calculated_metrics
# ---------------------------------------------------------------------------

class TestAddCalculatedMetrics:
    def _make_df(self):
        return pd.DataFrame({
            col_p("Маржа, ₽", "п1"): [100, -50],
            col_p("Маржа, ₽", "п2"): [80, -30],
            col_p("Продажи до СПП, ₽", "п1"): [200, 0],
            col_p("Продажи до СПП, ₽", "п2"): [150, 100],
            col_p("Оборот продаж, дни", "п1"): [30, 0],
            col_p("Оборот продаж, дни", "п2"): [25, 15],
        })

    def test_columns_added(self):
        df = add_calculated_metrics(self._make_df())
        assert col_p(CALC_MARGIN_RATE, "п1") in df.columns
        assert col_p(CALC_ANNUAL_YIELD, "п1") in df.columns
        assert col_p(CALC_MARGIN_RATE, "п2") in df.columns
        assert col_p(CALC_ANNUAL_YIELD, "п2") in df.columns

    def test_margin_rate_values(self):
        df = add_calculated_metrics(self._make_df())
        mr_p1 = df[col_p(CALC_MARGIN_RATE, "п1")]
        assert mr_p1.iloc[0] == pytest.approx(50.0)  # 100/200*100
        assert np.isnan(mr_p1.iloc[1])  # sales=0

    def test_annual_yield_zero_turnover(self):
        df = add_calculated_metrics(self._make_df())
        ay_p1 = df[col_p(CALC_ANNUAL_YIELD, "п1")]
        assert np.isnan(ay_p1.iloc[1])  # turnover=0


# ---------------------------------------------------------------------------
# compute_deltas
# ---------------------------------------------------------------------------

class TestComputeDeltas:
    def test_delta_columns(self):
        df = pd.DataFrame({
            col_p("Маржа, ₽", "п1"): [120.0],
            col_p("Маржа, ₽", "п2"): [100.0],
        })
        result = compute_deltas(df)
        assert "Маржа, ₽ Δабс" in result.columns
        assert "Маржа, ₽ Δ%" in result.columns
        # Δ = п2 (текущий) − п1 (прошлый) = 100 − 120 = −20
        assert result["Маржа, ₽ Δабс"].iloc[0] == pytest.approx(-20.0)
        # Δ% = (п2 − п1) / |п1| * 100 = −20 / 120 * 100 ≈ −16.67
        assert result["Маржа, ₽ Δ%"].iloc[0] == pytest.approx(-16.6667, rel=1e-3)
