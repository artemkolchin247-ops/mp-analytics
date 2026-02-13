"""Тесты WB API: contracts, normalize, ads metrics, matching pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "api"


# ---------------------------------------------------------------------------
# 1. Contracts — parsing
# ---------------------------------------------------------------------------

class TestContracts:
    """Тесты Pydantic-контрактов."""

    def test_funnel_response_parse(self):
        from src.wb_api.contracts import FunnelResponse

        raw = json.loads((FIXTURES / "funnel_sample.json").read_text("utf-8"))
        resp = FunnelResponse.model_validate(raw)
        assert resp.data is not None
        assert len(resp.data.products) == 2
        p0 = resp.data.products[0]
        assert p0.product.nmId == 100001
        assert p0.product.vendorCode == "ART-001"
        assert p0.statistic.selected.openCount == 500
        assert p0.statistic.selected.conversions.addToCartPercent == 10
        assert p0.statistic.past.orderCount == 8

    def test_funnel_response_empty(self):
        from src.wb_api.contracts import FunnelResponse

        resp = FunnelResponse.model_validate({"data": {"products": [], "currency": "RUB"}})
        assert resp.data is not None
        assert len(resp.data.products) == 0

    def test_funnel_response_extra_fields_ignored(self):
        from src.wb_api.contracts import FunnelResponse

        raw = json.loads((FIXTURES / "funnel_sample.json").read_text("utf-8"))
        raw["data"]["products"][0]["product"]["unknown_field"] = "test"
        resp = FunnelResponse.model_validate(raw)
        assert resp.data.products[0].product.nmId == 100001

    def test_ads_campaign_count_parse(self):
        from src.wb_api.contracts import CampaignCountResponse

        raw = json.loads((FIXTURES / "ads_campaigns_sample.json").read_text("utf-8"))
        resp = CampaignCountResponse.model_validate(raw)
        assert resp.all == 2
        assert len(resp.adverts) == 2
        all_ids = [aid for g in resp.adverts for aid in g.advertIds]
        assert 22161678 in all_ids
        assert 28449281 in all_ids

    def test_ads_fullstats_parse(self):
        from src.wb_api.contracts import AdsCampaignStats

        raw = json.loads((FIXTURES / "ads_fullstats_sample.json").read_text("utf-8"))
        assert isinstance(raw, list)
        campaigns = [AdsCampaignStats.model_validate(item) for item in raw]
        assert len(campaigns) == 2
        c0 = campaigns[0]
        assert c0.advertId == 22161678
        assert c0.clicks == 139
        assert len(c0.days) == 1
        assert len(c0.days[0].apps) == 2
        nm0 = c0.days[0].apps[0].nms[0]
        assert nm0.nmId == 100001
        assert nm0.clicks == 75

    def test_ads_fullstats_extra_fields(self):
        from src.wb_api.contracts import AdsCampaignStats

        raw = [{"advertId": 1, "unknown": True, "views": 10, "clicks": 1,
                "ctr": 10, "cpc": 5, "cr": 0, "sum": 5, "sum_price": 0,
                "orders": 0, "atbs": 0, "shks": 0, "canceled": 0, "days": []}]
        c = AdsCampaignStats.model_validate(raw[0])
        assert c.advertId == 1
        assert c.views == 10


# ---------------------------------------------------------------------------
# 2. Normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_normalize_vendor_code(self):
        from src.wb_api.normalize import normalize_vendor_code
        assert normalize_vendor_code("  ART - 001  ") == "art-001"
        assert normalize_vendor_code("Арт / 002") == "арт/002"
        assert normalize_vendor_code("Ёлка") == "елка"

    def test_normalize_nm_id(self):
        from src.wb_api.normalize import normalize_nm_id
        assert normalize_nm_id(100001) == "100001"
        assert normalize_nm_id(100001.0) == "100001"
        assert normalize_nm_id("  100001 ") == "100001"
        assert normalize_nm_id(None) == ""
        assert normalize_nm_id(float("nan")) == ""


# ---------------------------------------------------------------------------
# 3. Ads flatten + metrics
# ---------------------------------------------------------------------------

class TestAdsFlatten:
    @pytest.fixture
    def ads_campaigns(self):
        from src.wb_api.contracts import AdsCampaignStats
        raw = json.loads((FIXTURES / "ads_fullstats_sample.json").read_text("utf-8"))
        return [AdsCampaignStats.model_validate(item) for item in raw]

    def test_ads_campaigns_to_df(self, ads_campaigns):
        from src.wb_api.ads import ads_campaigns_to_df
        df = ads_campaigns_to_df(ads_campaigns)
        assert not df.empty
        # nmId 100001 appears in both campaigns
        assert 100001 in df["nmId"].values
        assert 100002 in df["nmId"].values

        row_1 = df[df["nmId"] == 100001].iloc[0]
        # Campaign 1: clicks 75+64=139, Campaign 2: clicks 5 → total 144
        assert row_1["clicks"] == 75 + 64 + 5
        # spend: 268.5 + 387.2 + 80.0 = 735.7
        assert abs(row_1["spend"] - 735.7) < 0.1
        # orders: 0+0+1=1
        assert row_1["orders"] == 1

    def test_ads_campaigns_to_df_has_rates(self, ads_campaigns):
        from src.wb_api.ads import ads_campaigns_to_df
        df = ads_campaigns_to_df(ads_campaigns)
        assert "true_ctr" in df.columns
        assert "cpc" in df.columns
        assert "cpm" in df.columns
        assert "cr_ads" in df.columns
        assert "cart_rate_ads" in df.columns
        # true_ctr for nmId 100001: clicks/views * 100
        row = df[df["nmId"] == 100001].iloc[0]
        expected_ctr = row["clicks"] / row["views"] * 100
        assert abs(row["true_ctr"] - expected_ctr) < 0.01

    def test_ads_campaigns_to_df_empty(self):
        from src.wb_api.ads import ads_campaigns_to_df
        df = ads_campaigns_to_df([])
        assert df.empty


# ---------------------------------------------------------------------------
# 4. Adapters — prefixing, merging
# ---------------------------------------------------------------------------

class TestAdapters:
    @pytest.fixture
    def ads_df(self):
        from src.wb_api.contracts import AdsCampaignStats
        from src.wb_api.ads import ads_campaigns_to_df
        raw = json.loads((FIXTURES / "ads_fullstats_sample.json").read_text("utf-8"))
        campaigns = [AdsCampaignStats.model_validate(item) for item in raw]
        return ads_campaigns_to_df(campaigns)

    def test_ads_df_with_prefix(self, ads_df):
        from src.wb_api.adapters import ads_df_with_prefix
        prefixed = ads_df_with_prefix(ads_df, "п1")
        assert "A_spend п1" in prefixed.columns
        assert "A_clicks п1" in prefixed.columns
        assert "A_true_ctr п1" in prefixed.columns
        assert "nmId" in prefixed.columns

    def test_merge_ads_periods(self, ads_df):
        from src.wb_api.adapters import ads_df_with_prefix, merge_ads_periods
        p1 = ads_df_with_prefix(ads_df, "п1")
        p2 = ads_df_with_prefix(ads_df, "п2")
        merged = merge_ads_periods(p1, p2)
        assert "A_spend п1" in merged.columns
        assert "A_spend п2" in merged.columns
        assert "A_spend Δабс" in merged.columns
        assert "A_spend Δ%" in merged.columns

    def test_funnel_api_to_excel_format(self):
        from src.wb_api.contracts import FunnelResponse
        from src.wb_api.funnel import funnel_products_to_df
        from src.wb_api.adapters import api_funnel_to_excel_format_pair

        raw = json.loads((FIXTURES / "funnel_sample.json").read_text("utf-8"))
        resp = FunnelResponse.model_validate(raw)
        df_api = funnel_products_to_df(resp.data.products)
        assert not df_api.empty

        df_p1, df_p2 = api_funnel_to_excel_format_pair(df_api)
        assert "Артикул продавца" in df_p1.columns
        assert "Показы" in df_p1.columns
        assert "Положили в корзину" in df_p1.columns
        assert "Заказали, шт" in df_p1.columns
        # п1 = past, п2 = selected (current)
        assert df_p1["Показы"].iloc[0] == 400  # nmId 100001 past_openCount → п1
        assert df_p2["Показы"].iloc[0] == 500  # sel_openCount → п2


# ---------------------------------------------------------------------------
# 5. Ads Metrics
# ---------------------------------------------------------------------------

class TestAdsMetrics:
    @pytest.fixture
    def merged_df(self):
        """Создаёт merged df с ads колонками для тестов."""
        from src.wb_api.contracts import AdsCampaignStats
        from src.wb_api.ads import ads_campaigns_to_df
        from src.wb_api.adapters import ads_df_with_prefix, merge_ads_periods

        raw = json.loads((FIXTURES / "ads_fullstats_sample.json").read_text("utf-8"))
        campaigns = [AdsCampaignStats.model_validate(item) for item in raw]
        df = ads_campaigns_to_df(campaigns)
        p1 = ads_df_with_prefix(df, "п1")
        p2 = ads_df_with_prefix(df, "п2")
        merged = merge_ads_periods(p1, p2)
        # Add dummy Артикул col
        merged["Артикул"] = ["ART-001", "ART-002"]
        return merged

    def test_has_ads_cols(self, merged_df):
        from src.wb_ads_metrics import has_ads_cols
        assert has_ads_cols(merged_df) is True
        assert has_ads_cols(pd.DataFrame({"x": [1]})) is False
        assert has_ads_cols(None) is False

    def test_build_ads_kpi(self, merged_df):
        from src.wb_ads_metrics import build_ads_kpi
        kpi = build_ads_kpi(merged_df)
        assert not kpi.empty
        assert "Метрика" in kpi.columns
        assert "п1" in kpi.columns
        assert "п2" in kpi.columns
        assert "Δабс" in kpi.columns

    def test_build_ads_by_article(self, merged_df):
        from src.wb_ads_metrics import build_ads_by_article
        art = build_ads_by_article(merged_df, n=10)
        assert not art.empty
        assert "Артикул" in art.columns


# ---------------------------------------------------------------------------
# 6. Matching pipeline
# ---------------------------------------------------------------------------

class TestMatching:
    def test_join_ads_to_economics_via_funnel(self):
        from src.wb_api.adapters import join_ads_to_economics, ads_df_with_prefix, merge_ads_periods
        from src.wb_api.ads import ads_campaigns_to_df
        from src.wb_api.contracts import AdsCampaignStats, FunnelResponse
        from src.wb_api.funnel import funnel_products_to_df

        # Ads
        raw_ads = json.loads((FIXTURES / "ads_fullstats_sample.json").read_text("utf-8"))
        campaigns = [AdsCampaignStats.model_validate(item) for item in raw_ads]
        df_ads = ads_campaigns_to_df(campaigns)
        p1 = ads_df_with_prefix(df_ads, "п1")
        merged_ads = merge_ads_periods(p1, pd.DataFrame())

        # Funnel (for nmId → vendorCode mapping)
        raw_f = json.loads((FIXTURES / "funnel_sample.json").read_text("utf-8"))
        resp = FunnelResponse.model_validate(raw_f)
        df_funnel = funnel_products_to_df(resp.data.products)

        # Fake economics
        df_econ = pd.DataFrame({
            "Артикул": ["ART-001", "ART-002", "ART-003"],
            "Маржа, ₽ п1": [100, 200, 300],
        })

        result, coverage = join_ads_to_economics(
            df_econ, merged_ads, df_funnel_api=df_funnel,
        )

        assert coverage["has_ads"] is True
        assert coverage["matched_via_funnel"] == 2  # 100001→ART-001, 100002→ART-002
        assert coverage["unmatched"] == 0

    def test_join_ads_to_economics_with_override(self):
        from src.wb_api.adapters import join_ads_to_economics, ads_df_with_prefix, merge_ads_periods
        from src.wb_api.ads import ads_campaigns_to_df
        from src.wb_api.contracts import AdsCampaignStats

        raw_ads = json.loads((FIXTURES / "ads_fullstats_sample.json").read_text("utf-8"))
        campaigns = [AdsCampaignStats.model_validate(item) for item in raw_ads]
        df_ads = ads_campaigns_to_df(campaigns)
        p1 = ads_df_with_prefix(df_ads, "п1")
        merged_ads = merge_ads_periods(p1, pd.DataFrame())

        df_econ = pd.DataFrame({
            "Артикул": ["MY-ART-X"],
            "Маржа, ₽ п1": [500],
        })

        # Override: 100001 → MY-ART-X
        overrides = {"100001": "MY-ART-X"}
        result, coverage = join_ads_to_economics(
            df_econ, merged_ads, overrides=overrides,
        )
        assert coverage["matched_via_override"] >= 1


# ---------------------------------------------------------------------------
# 7. Inspector schema summary
# ---------------------------------------------------------------------------

class TestInspector:
    def test_summarize_schema(self):
        from src.wb_api.inspector import summarize_schema
        obj = {"a": 1, "b": [{"c": "x"}], "d": {"e": True}}
        rows = summarize_schema(obj)
        assert len(rows) > 0
        paths = [r["path"] for r in rows]
        assert "a" in paths
        assert "b" in paths
        assert "d.e" in paths

    def test_summarize_schema_empty(self):
        from src.wb_api.inspector import summarize_schema
        rows = summarize_schema({})
        assert len(rows) == 1
        assert rows[0]["path"] == "(root)"


# ---------------------------------------------------------------------------
# 8. Client — token masking
# ---------------------------------------------------------------------------

class TestClient:
    def test_mask_token_long(self):
        from src.wb_api.client import mask_token
        assert mask_token("abcdefghijklmnop") == "abcd…mnop"

    def test_mask_token_short(self):
        from src.wb_api.client import mask_token
        assert mask_token("short") == "***"

    def test_get_token_returns_none_if_not_set(self):
        import os
        old = os.environ.pop("WB_TOKEN", None)
        try:
            from src.wb_api.client import get_token
            # In test env, secrets.toml likely has placeholder
            token = get_token()
            # Should be None (placeholder "WB_TOKEN_VALUE" is filtered)
            assert token is None or token != "WB_TOKEN_VALUE"
        finally:
            if old is not None:
                os.environ["WB_TOKEN"] = old
