"""Получение данных рекламы WB через API v3 (true ads)."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.wb_api.client import WBClient, BASE_ADVERT, WBAPIError
from src.wb_api.contracts import (
    AdsCampaignStats,
    AdsNmStats,
    CampaignCountResponse,
)
from src.utils import safe_div


# ---------------------------------------------------------------------------
# Campaign list
# ---------------------------------------------------------------------------

# Статусы кампаний, для которых доступна статистика (7=завершена, 9=активна, 11=приостановлена)
_STATS_STATUSES = {7, 9, 11}


def fetch_campaign_ids(client: WBClient) -> List[int]:
    """Получает список ID всех кампаний с доступной статистикой."""
    resp = client.get(BASE_ADVERT, "/adv/v1/promotion/count")
    raw = resp.json()
    parsed = CampaignCountResponse.model_validate(raw)
    ids: List[int] = []
    for group in parsed.adverts:
        if group.status in _STATS_STATUSES:
            ids.extend(group.advertIds)
    return ids


# ---------------------------------------------------------------------------
# Fullstats
# ---------------------------------------------------------------------------

_MAX_IDS_PER_REQUEST = 50
_RATE_LIMIT_PAUSE = 1.0  # секунда между batch-запросами


def _fetch_fullstats_batch(
    client: WBClient,
    campaign_ids: List[int],
    start: str,
    end: str,
) -> List[AdsCampaignStats]:
    """Один запрос к /adv/v3/fullstats (до 50 ID)."""
    ids_str = ",".join(str(i) for i in campaign_ids[:_MAX_IDS_PER_REQUEST])
    params = {"ids": ids_str, "beginDate": start, "endDate": end}
    resp = client.get(BASE_ADVERT, "/adv/v3/fullstats", params=params)
    raw = resp.json()
    if not isinstance(raw, list):
        return []
    return [AdsCampaignStats.model_validate(item) for item in raw]


def fetch_ads_all(
    client: WBClient,
    start: str,
    end: str,
    campaign_ids: Optional[List[int]] = None,
) -> List[AdsCampaignStats]:
    """Получает статистику всех кампаний с пагинацией по 50 ID."""
    if campaign_ids is None:
        campaign_ids = fetch_campaign_ids(client)
    if not campaign_ids:
        return []

    all_stats: List[AdsCampaignStats] = []
    for i in range(0, len(campaign_ids), _MAX_IDS_PER_REQUEST):
        batch = campaign_ids[i:i + _MAX_IDS_PER_REQUEST]
        stats = _fetch_fullstats_batch(client, batch, start, end)
        all_stats.extend(stats)
        if i + _MAX_IDS_PER_REQUEST < len(campaign_ids):
            time.sleep(_RATE_LIMIT_PAUSE)

    return all_stats


# ---------------------------------------------------------------------------
# Flatten: campaign stats → per-nmId DataFrame
# ---------------------------------------------------------------------------

def ads_campaigns_to_df(campaigns: List[AdsCampaignStats]) -> pd.DataFrame:
    """Разворачивает вложенную структуру кампаний в плоский DataFrame по nmId.

    Агрегирует по nmId (один товар может быть в нескольких кампаниях/днях).
    """
    if not campaigns:
        return pd.DataFrame()

    nm_rows: Dict[int, Dict[str, Any]] = {}

    for camp in campaigns:
        for day in camp.days:
            for app in day.apps:
                for nm in app.nms:
                    key = nm.nmId
                    if key not in nm_rows:
                        nm_rows[key] = {
                            "nmId": nm.nmId,
                            "name": nm.name,
                            "views": 0,
                            "clicks": 0,
                            "spend": 0.0,
                            "orders": 0,
                            "atbs": 0,
                            "shks": 0,
                            "canceled": 0,
                            "sum_price": 0.0,
                        }
                    r = nm_rows[key]
                    r["views"] += nm.views
                    r["clicks"] += nm.clicks
                    r["spend"] += nm.sum
                    r["orders"] += nm.orders
                    r["atbs"] += nm.atbs
                    r["shks"] += nm.shks
                    r["canceled"] += nm.canceled
                    r["sum_price"] += nm.sum_price
                    # Обновляем name если пустое
                    if not r["name"] and nm.name:
                        r["name"] = nm.name

    if not nm_rows:
        return pd.DataFrame()

    df = pd.DataFrame(list(nm_rows.values()))
    df = _recompute_ads_rates(df)
    return df


def _recompute_ads_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Пересчитывает rate-метрики из агрегированных абсолютных значений."""
    df = df.copy()
    df["true_ctr"] = safe_div(df["clicks"], df["views"]) * 100
    df["cpc"] = safe_div(df["spend"], df["clicks"])
    df["cpm"] = safe_div(df["spend"], df["views"]) * 1000
    df["cr_ads"] = safe_div(df["orders"], df["clicks"]) * 100
    df["cart_rate_ads"] = safe_div(df["atbs"], df["clicks"]) * 100
    return df


def fetch_ads(
    client: WBClient,
    start: str,
    end: str,
    campaign_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Полный fetch + нормализация в DataFrame."""
    campaigns = fetch_ads_all(client, start, end, campaign_ids)
    return ads_campaigns_to_df(campaigns)
