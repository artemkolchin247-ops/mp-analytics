"""Тесты для модулей воронки WB: нормализация артикула, агрегация, конверсии, coverage."""
import numpy as np
import pandas as pd
import pytest

from src.wb_funnel_io import (
    CALC_CONV_CART,
    CALC_CONV_ORDER,
    CALC_CTR,
    CALC_PURCHASE_RATE,
    FUNNEL_ID_COL,
    aggregate_funnel_by_article,
    normalize_article_key,
)
from src.wb_funnel_metrics import (
    join_funnel_to_economics,
    build_funnel_kpi,
    build_funnel_economics_diag,
    build_conversion_growth_points,
    aggregate_funnel,
    _fp,
)
from src.schema import col_p


# ---------------------------------------------------------------------------
# Нормализация артикула
# ---------------------------------------------------------------------------

class TestNormalizeArticleKey:
    def test_trim_lower(self):
        assert normalize_article_key("  ABC-123  ") == "abc-123"

    def test_multiple_spaces(self):
        assert normalize_article_key("abc  def") == "abc def"

    def test_yo_replacement(self):
        assert normalize_article_key("Артикулё") == "артикуле"

    def test_dash_spaces(self):
        assert normalize_article_key("abc - 123") == "abc-123"

    def test_slash_spaces(self):
        assert normalize_article_key("abc / 123") == "abc/123"

    def test_numeric(self):
        assert normalize_article_key(12345) == "12345"


# ---------------------------------------------------------------------------
# Агрегация по артикулу
# ---------------------------------------------------------------------------

def _make_funnel_df():
    """Тестовый DataFrame воронки с несколькими датами для одного артикула."""
    return pd.DataFrame({
        FUNNEL_ID_COL: ["ART-1", "ART-1", "ART-2"],
        "Дата": ["2024-01-01", "2024-01-02", "2024-01-01"],
        "Показы": [1000, 2000, 500],
        "Переходы в карточку": [100, 200, 50],
        "Положили в корзину": [30, 60, 20],
        "Заказали, шт": [10, 20, 5],
        "Выкупили, шт": [8, 16, 4],
        "Отменили, шт": [2, 4, 1],
        "Заказали на сумму, ₽": [5000, 10000, 2500],
        "Выкупили на сумму, ₽": [4000, 8000, 2000],
        "Отменили на сумму, ₽": [1000, 2000, 500],
    })


class TestAggregateFunnelByArticle:
    def test_sums_correct(self):
        df = _make_funnel_df()
        result = aggregate_funnel_by_article(df)
        art1 = result[result["_art_key"] == "art-1"].iloc[0]
        assert art1["Показы"] == 3000
        assert art1["Переходы в карточку"] == 300
        assert art1["Положили в корзину"] == 90
        assert art1["Заказали, шт"] == 30

    def test_ctr_recomputed(self):
        df = _make_funnel_df()
        result = aggregate_funnel_by_article(df)
        art1 = result[result["_art_key"] == "art-1"].iloc[0]
        # CTR = 300 / 3000 * 100 = 10%
        assert art1[CALC_CTR] == pytest.approx(10.0)

    def test_conv2cart_recomputed(self):
        df = _make_funnel_df()
        result = aggregate_funnel_by_article(df)
        art1 = result[result["_art_key"] == "art-1"].iloc[0]
        # Conv2Cart = 90 / 300 * 100 = 30%
        assert art1[CALC_CONV_CART] == pytest.approx(30.0)

    def test_conv2order_recomputed(self):
        df = _make_funnel_df()
        result = aggregate_funnel_by_article(df)
        art1 = result[result["_art_key"] == "art-1"].iloc[0]
        # Conv2Order = 30 / 300 * 100 = 10%
        assert art1[CALC_CONV_ORDER] == pytest.approx(10.0)

    def test_purchase_rate_recomputed(self):
        df = _make_funnel_df()
        result = aggregate_funnel_by_article(df)
        art1 = result[result["_art_key"] == "art-1"].iloc[0]
        # PurchaseRate = 24 / 30 * 100 = 80%
        assert art1[CALC_PURCHASE_RATE] == pytest.approx(80.0)

    def test_two_articles(self):
        df = _make_funnel_df()
        result = aggregate_funnel_by_article(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Конверсии при нулевых знаменателях
# ---------------------------------------------------------------------------

class TestZeroDenominatorConversions:
    def test_zero_shows(self):
        df = pd.DataFrame({
            FUNNEL_ID_COL: ["A"],
            "Показы": [0],
            "Переходы в карточку": [0],
            "Положили в корзину": [0],
            "Заказали, шт": [0],
            "Выкупили, шт": [0],
        })
        result = aggregate_funnel_by_article(df)
        assert np.isnan(result[CALC_CTR].iloc[0])
        assert np.isnan(result[CALC_CONV_CART].iloc[0])
        assert np.isnan(result[CALC_CONV_ORDER].iloc[0])
        assert np.isnan(result[CALC_PURCHASE_RATE].iloc[0])

    def test_zero_clicks_nonzero_shows(self):
        df = pd.DataFrame({
            FUNNEL_ID_COL: ["A"],
            "Показы": [100],
            "Переходы в карточку": [0],
            "Положили в корзину": [0],
            "Заказали, шт": [0],
            "Выкупили, шт": [0],
        })
        result = aggregate_funnel_by_article(df)
        assert result[CALC_CTR].iloc[0] == pytest.approx(0.0)
        assert np.isnan(result[CALC_CONV_CART].iloc[0])


# ---------------------------------------------------------------------------
# Join и coverage
# ---------------------------------------------------------------------------

def _make_econ_df():
    return pd.DataFrame({
        "Артикул": ["ART-1", "ART-2", "ART-3"],
        "Модель": ["M1", "M1", "M2"],
        "Статус": ["Продается", "Продается", "Новый"],
        "Склейка на WB": ["S1", "S1", "S2"],
        "Color code": ["RED", "BLUE", "RED"],
        "Коллекция": ["FW24", "FW24", "SS25"],
        col_p("Маржа, ₽", "п1"): [1000, 2000, -500],
        col_p("Маржа, ₽", "п2"): [800, 1500, -300],
    })


class TestJoinFunnelToEconomics:
    def test_coverage_counts(self):
        econ = _make_econ_df()
        funnel = _make_funnel_df()  # has ART-1 and ART-2
        merged, cov = join_funnel_to_economics(econ, funnel, None)
        assert cov["econ_total"] == 3
        assert cov["matched_p1"] == 2  # ART-1 and ART-2 matched
        assert cov["matched_p2"] == 0  # no p2 funnel

    def test_merged_has_funnel_cols(self):
        econ = _make_econ_df()
        funnel = _make_funnel_df()
        merged, _ = join_funnel_to_economics(econ, funnel, None)
        assert _fp("Показы", "п1") in merged.columns
        assert _fp(CALC_CTR, "п1") in merged.columns

    def test_unmatched_article_has_nan(self):
        econ = _make_econ_df()
        funnel = _make_funnel_df()
        merged, _ = join_funnel_to_economics(econ, funnel, None)
        art3 = merged[merged["Артикул"] == "ART-3"].iloc[0]
        assert np.isnan(art3[_fp("Показы", "п1")])


# ---------------------------------------------------------------------------
# Агрегация воронки на уровне
# ---------------------------------------------------------------------------

class TestAggregateFunnel:
    def test_aggregate_by_model(self):
        econ = _make_econ_df()
        funnel = _make_funnel_df()
        merged, _ = join_funnel_to_economics(econ, funnel, None)

        result = aggregate_funnel(merged, ["Модель"])
        assert len(result) == 2  # M1, M2

        m1 = result[result["Модель"] == "M1"]
        if not m1.empty:
            shows = _fp("Показы", "п1")
            if shows in m1.columns:
                # ART-1: 3000, ART-2: 500 = 3500
                assert m1.iloc[0][shows] == pytest.approx(3500)
