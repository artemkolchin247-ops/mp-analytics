"""Нормализация ключей из WB API: nmId, vendorCode, title/name."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def normalize_vendor_code(val) -> str:
    """Мягкая нормализация vendorCode для join с Артикул экономики.

    Аналогична normalize_article_key из wb_funnel_io:
    trim, lower, ё→е, пробелы вокруг дефисов/слэшей.
    """
    s = str(val).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("ё", "е")
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return s


def normalize_nm_id(val) -> str:
    """Приводит nmId к строке без пробелов."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip()
    s = re.sub(r"\.0$", "", s)
    return s
