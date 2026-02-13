"""Получение данных воронки WB через API v3."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.wb_api.client import WBClient, BASE_ANALYTICS, WBAPIError
from src.wb_api.contracts import FunnelProduct, FunnelResponse, FunnelResponseData
from src.utils import safe_div


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

_PAGE_LIMIT = 100  # макс. кол-во карточек за запрос


def _fetch_funnel_page(
    client: WBClient,
    sel_start: str,
    sel_end: str,
    past_start: str,
    past_end: str,
    offset: int = 0,
    limit: int = _PAGE_LIMIT,
) -> FunnelResponseData:
    """Один запрос к /api/analytics/v3/sales-funnel/products."""
    body = {
        "selectedPeriod": {"start": sel_start, "end": sel_end},
        "pastPeriod": {"start": past_start, "end": past_end},
        "nmIds": [],
        "brandNames": [],
        "subjectIds": [],
        "tagIds": [],
        "skipDeletedNm": False,
        "orderBy": {"field": "openCard", "mode": "desc"},
        "limit": limit,
        "offset": offset,
    }
    resp = client.post(BASE_ANALYTICS, "/api/analytics/v3/sales-funnel/products", json_body=body)
    raw = resp.json()
    parsed = FunnelResponse.model_validate(raw)
    return parsed.data or FunnelResponseData()


def fetch_funnel_all(
    client: WBClient,
    sel_start: str,
    sel_end: str,
    past_start: str,
    past_end: str,
    max_pages: int = 50,
) -> List[FunnelProduct]:
    """Пагинация: забирает все карточки funnel."""
    all_products: List[FunnelProduct] = []
    for page in range(max_pages):
        data = _fetch_funnel_page(
            client, sel_start, sel_end, past_start, past_end,
            offset=page * _PAGE_LIMIT, limit=_PAGE_LIMIT,
        )
        all_products.extend(data.products)
        if len(data.products) < _PAGE_LIMIT:
            break
    return all_products


# ---------------------------------------------------------------------------
# Нормализация в DataFrame
# ---------------------------------------------------------------------------

def funnel_products_to_df(products: List[FunnelProduct]) -> pd.DataFrame:
    """Конвертирует список FunnelProduct в плоский DataFrame.

    Колонки: nmId, vendorCode, title, brandName,
             + selected_* и past_* для всех метрик.
    """
    if not products:
        return pd.DataFrame()

    rows = []
    for fp in products:
        p = fp.product
        sel = fp.statistic.selected
        past = fp.statistic.past
        row: Dict[str, Any] = {
            "nmId": p.nmId,
            "vendorCode": p.vendorCode,
            "title": p.title,
            "brandName": p.brandName,
            "subjectName": p.subjectName,
        }
        # Selected period
        for prefix, stats in [("sel", sel), ("past", past)]:
            row[f"{prefix}_openCount"] = stats.openCount
            row[f"{prefix}_cartCount"] = stats.cartCount
            row[f"{prefix}_orderCount"] = stats.orderCount
            row[f"{prefix}_orderSum"] = stats.orderSum
            row[f"{prefix}_buyoutCount"] = stats.buyoutCount
            row[f"{prefix}_buyoutSum"] = stats.buyoutSum
            row[f"{prefix}_cancelCount"] = stats.cancelCount
            row[f"{prefix}_cancelSum"] = stats.cancelSum
            row[f"{prefix}_avgPrice"] = stats.avgPrice
            row[f"{prefix}_addToWishlist"] = stats.addToWishlist
            row[f"{prefix}_addToCartPct"] = stats.conversions.addToCartPercent
            row[f"{prefix}_cartToOrderPct"] = stats.conversions.cartToOrderPercent
            row[f"{prefix}_buyoutPct"] = stats.conversions.buyoutPercent
        rows.append(row)

    return pd.DataFrame(rows)


def fetch_funnel(
    client: WBClient,
    sel_start: str,
    sel_end: str,
    past_start: str,
    past_end: str,
) -> pd.DataFrame:
    """Полный fetch + нормализация в DataFrame."""
    products = fetch_funnel_all(client, sel_start, sel_end, past_start, past_end)
    return funnel_products_to_df(products)
