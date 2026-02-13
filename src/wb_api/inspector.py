"""API Inspector — тестовые вызовы, schema summary, сохранение samples."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.wb_api.client import WBClient, BASE_ANALYTICS, BASE_ADVERT, WBAPIError


SAMPLES_DIR = Path("data/api_samples")


# ---------------------------------------------------------------------------
# Schema summary
# ---------------------------------------------------------------------------

def summarize_schema(obj: Any, prefix: str = "", max_depth: int = 4, _depth: int = 0) -> List[Dict[str, str]]:
    """Рекурсивно собирает схему JSON-объекта.

    Returns:
        list of {"path": ..., "type": ..., "example": ...}
    """
    rows: List[Dict[str, str]] = []
    if _depth >= max_depth:
        rows.append({"path": prefix or "(root)", "type": f"{type(obj).__name__} (truncated)", "example": ""})
        return rows

    if isinstance(obj, dict):
        if not prefix:
            rows.append({"path": "(root)", "type": "object", "example": f"{len(obj)} keys"})
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            rows.extend(summarize_schema(v, prefix=p, max_depth=max_depth, _depth=_depth + 1))
    elif isinstance(obj, list):
        rows.append({"path": prefix or "(root)", "type": f"array[{len(obj)}]", "example": ""})
        if obj:
            rows.extend(summarize_schema(obj[0], prefix=f"{prefix}[0]", max_depth=max_depth, _depth=_depth + 1))
    else:
        example = str(obj)[:80] if obj is not None else "null"
        rows.append({"path": prefix or "(root)", "type": type(obj).__name__, "example": example})
    return rows


# ---------------------------------------------------------------------------
# Test calls
# ---------------------------------------------------------------------------

def test_funnel_call(
    client: WBClient,
    start: str,
    end: str,
    limit: int = 5,
) -> Tuple[Dict[str, Any], float]:
    """Тестовый вызов Funnel API. Возвращает (raw_json, latency_sec)."""
    from datetime import date as _date, timedelta as _td
    _sel_s = _date.fromisoformat(start)
    _sel_e = _date.fromisoformat(end)
    _plen = (_sel_e - _sel_s).days + 1
    _past_e = _sel_s - _td(days=1)
    _past_s = _past_e - _td(days=_plen - 1)
    body = {
        "selectedPeriod": {"start": start, "end": end},
        "pastPeriod": {"start": str(_past_s), "end": str(_past_e)},
        "nmIds": [],
        "brandNames": [],
        "subjectIds": [],
        "tagIds": [],
        "skipDeletedNm": False,
        "orderBy": {"field": "openCard", "mode": "desc"},
        "limit": limit,
        "offset": 0,
    }
    t0 = time.monotonic()
    resp = client.post(BASE_ANALYTICS, "/api/analytics/v3/sales-funnel/products", json_body=body)
    latency = time.monotonic() - t0
    return resp.json(), latency


def test_ads_campaign_list(client: WBClient) -> Tuple[Dict[str, Any], float]:
    """Тестовый вызов списка кампаний."""
    t0 = time.monotonic()
    resp = client.get(BASE_ADVERT, "/adv/v1/promotion/count")
    latency = time.monotonic() - t0
    return resp.json(), latency


def test_ads_fullstats_call(
    client: WBClient,
    campaign_ids: List[int],
    start: str,
    end: str,
) -> Tuple[Any, float]:
    """Тестовый вызов Ads fullstats API."""
    ids_str = ",".join(str(i) for i in campaign_ids[:50])
    params = {"ids": ids_str, "beginDate": start, "endDate": end}
    t0 = time.monotonic()
    resp = client.get(BASE_ADVERT, "/adv/v3/fullstats", params=params)
    latency = time.monotonic() - t0
    return resp.json(), latency


# ---------------------------------------------------------------------------
# Save / load samples
# ---------------------------------------------------------------------------

def save_sample(endpoint_name: str, start: str, end: str, raw_json: Any, summary: List[Dict]) -> Path:
    """Сохраняет raw sample и schema summary в gitignored папку."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{endpoint_name}_{start}_{end}_{ts}"

    raw_path = SAMPLES_DIR / f"{base}_raw.json"
    summary_path = SAMPLES_DIR / f"{base}_schema.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_json, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return raw_path
