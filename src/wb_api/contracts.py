"""Pydantic-контракты WB API — минимальные поля + безопасная валидация."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Funnel API v3 — /api/analytics/v3/sales-funnel/products
# ---------------------------------------------------------------------------

class FunnelConversions(BaseModel):
    """Конверсии воронки."""
    addToCartPercent: float = 0
    cartToOrderPercent: float = 0
    buyoutPercent: float = 0

    model_config = {"extra": "ignore"}


class FunnelPeriodStats(BaseModel):
    """Статистика за один период (selected / past)."""
    openCount: int = 0
    cartCount: int = 0
    orderCount: int = 0
    orderSum: float = 0
    buyoutCount: int = 0
    buyoutSum: float = 0
    cancelCount: int = 0
    cancelSum: float = 0
    avgPrice: float = 0
    avgOrdersCountPerDay: float = 0
    addToWishlist: int = 0
    conversions: FunnelConversions = Field(default_factory=FunnelConversions)

    model_config = {"extra": "ignore"}


class FunnelProductInfo(BaseModel):
    """Информация о продукте из funnel API."""
    nmId: int
    vendorCode: str = ""
    title: str = ""
    brandName: str = ""
    subjectId: int = 0
    subjectName: str = ""

    model_config = {"extra": "ignore"}


class FunnelStatistic(BaseModel):
    """Статистика продукта (selected + past)."""
    selected: FunnelPeriodStats = Field(default_factory=FunnelPeriodStats)
    past: FunnelPeriodStats = Field(default_factory=FunnelPeriodStats)

    model_config = {"extra": "ignore"}


class FunnelProduct(BaseModel):
    """Один продукт из ответа funnel API."""
    product: FunnelProductInfo
    statistic: FunnelStatistic = Field(default_factory=FunnelStatistic)

    model_config = {"extra": "ignore"}


class FunnelResponseData(BaseModel):
    """Данные ответа funnel API."""
    products: List[FunnelProduct] = []
    currency: str = "RUB"

    model_config = {"extra": "ignore"}


class FunnelResponse(BaseModel):
    """Корневой ответ funnel API."""
    data: Optional[FunnelResponseData] = None

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Ads API v3 — /adv/v3/fullstats
# ---------------------------------------------------------------------------

class AdsNmStats(BaseModel):
    """Статистика одного nmId внутри рекламной кампании."""
    nmId: int
    name: str = ""
    views: int = 0
    clicks: int = 0
    ctr: float = 0
    cpc: float = 0
    cr: float = 0
    sum: float = 0           # spend (₽)
    sum_price: float = 0     # order sum (₽)
    orders: int = 0
    atbs: int = 0            # add to basket
    shks: int = 0            # shipped items
    canceled: int = 0

    model_config = {"extra": "ignore"}


class AdsAppStats(BaseModel):
    """Статистика по типу приложения за день."""
    appType: int = 0
    nms: List[AdsNmStats] = []
    views: int = 0
    clicks: int = 0
    orders: int = 0
    sum: float = 0
    atbs: int = 0

    model_config = {"extra": "ignore"}


class AdsDayStats(BaseModel):
    """Статистика за один день кампании."""
    date: str = ""
    apps: List[AdsAppStats] = []
    views: int = 0
    clicks: int = 0
    orders: int = 0
    sum: float = 0
    atbs: int = 0

    model_config = {"extra": "ignore"}


class AdsCampaignStats(BaseModel):
    """Статистика одной рекламной кампании."""
    advertId: int
    views: int = 0
    clicks: int = 0
    ctr: float = 0
    cpc: float = 0
    cr: float = 0
    sum: float = 0
    sum_price: float = 0
    orders: int = 0
    atbs: int = 0
    shks: int = 0
    canceled: int = 0
    days: List[AdsDayStats] = []

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Campaign list — /adv/v1/promotion/count
# ---------------------------------------------------------------------------

class CampaignGroup(BaseModel):
    """Группа кампаний по типу/статусу."""
    type: int = 0
    status: int = 0
    count: int = 0
    advertIds: List[int] = []

    model_config = {"extra": "ignore"}


class CampaignCountResponse(BaseModel):
    """Ответ /adv/v1/promotion/count."""
    all: int = 0
    adverts: List[CampaignGroup] = []

    model_config = {"extra": "ignore"}
