"""HTTP-клиент WB API с ретраями, backoff, маскированием токена."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Базовые URL
# ---------------------------------------------------------------------------

BASE_ANALYTICS = "https://seller-analytics-api.wildberries.ru"
BASE_ADVERT = "https://advert-api.wildberries.ru"

_TIMEOUT = (10, 60)  # (connect, read)
_MAX_RETRIES = 5
_BACKOFF_FACTOR = 3.0  # 3s, 6s, 12s, 24s, 48s


# ---------------------------------------------------------------------------
# Получение токена
# ---------------------------------------------------------------------------

def get_token() -> Optional[str]:
    """Получает WB API токен из Streamlit secrets или .env.

    Приоритет:
    1. st.secrets["wb"]["token"]
    2. os.environ["WB_TOKEN"]
    3. None
    """
    # Streamlit secrets
    try:
        import streamlit as st
        token = st.secrets.get("wb", {}).get("token")
        if token and token != "WB_TOKEN_VALUE":
            return str(token)
    except Exception:
        pass

    # Environment / .env
    token = os.environ.get("WB_TOKEN")
    if token and token != "WB_TOKEN_VALUE":
        return token

    return None


def get_token_2() -> Optional[str]:
    """Получает второй WB API токен.

    Приоритет:
    1. st.secrets["wb"]["token2"]
    2. os.environ["WB_TOKEN_2"]
    3. None
    """
    try:
        import streamlit as st
        token = st.secrets.get("wb", {}).get("token2")
        if token and token != "WB_TOKEN_VALUE":
            return str(token)
    except Exception:
        pass

    token = os.environ.get("WB_TOKEN_2")
    if token and token != "WB_TOKEN_VALUE":
        return token

    return None


def get_all_tokens() -> list[tuple[str, str]]:
    """Возвращает список (label, token) для всех настроенных токенов."""
    result: list[tuple[str, str]] = []
    t1 = get_token()
    if t1:
        result.append(("Token 1", t1))
    t2 = get_token_2()
    if t2:
        result.append(("Token 2", t2))
    return result


def mask_token(token: str) -> str:
    """Маскирует токен для безопасного вывода: первые 4 и последние 4 символа."""
    if len(token) <= 12:
        return "***"
    return f"{token[:4]}…{token[-4:]}"


# ---------------------------------------------------------------------------
# Сессия с ретраями
# ---------------------------------------------------------------------------

def _build_session(token: str) -> requests.Session:
    """Создаёт requests.Session с retry-стратегией и авторизацией."""
    session = requests.Session()
    session.headers.update({
        "Authorization": token,
        "Content-Type": "application/json",
    })

    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Публичный API-клиент
# ---------------------------------------------------------------------------

class WBAPIError(Exception):
    """Ошибка WB API (без утечки токена)."""

    def __init__(self, message: str, status_code: int = 0, url: str = ""):
        # Гарантируем: токен НЕ попадает в сообщение
        safe_msg = message
        super().__init__(safe_msg)
        self.status_code = status_code
        self.url = url


class WBClient:
    """Обёртка над WB API с ретраями и безопасными ошибками."""

    def __init__(self, token: Optional[str] = None):
        self._token = token or get_token()
        if not self._token:
            raise WBAPIError("WB API token not configured. Set it in .streamlit/secrets.toml or WB_TOKEN env var.")
        self._session = _build_session(self._token)

    @property
    def token_info(self) -> str:
        """Безопасная информация о токене для UI."""
        if not self._token:
            return "not set"
        return f"set ({len(self._token)} chars, {mask_token(self._token)})"

    def get(self, base_url: str, path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """GET-запрос с обработкой ошибок."""
        url = f"{base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=_TIMEOUT)
        except requests.RequestException as e:
            raise WBAPIError(f"Network error: {type(e).__name__}", url=url) from e
        self._check_response(resp, url)
        return resp

    def post(self, base_url: str, path: str, json_body: Any = None) -> requests.Response:
        """POST-запрос с обработкой ошибок."""
        url = f"{base_url}{path}"
        try:
            resp = self._session.post(url, json=json_body, timeout=_TIMEOUT)
        except requests.RequestException as e:
            raise WBAPIError(f"Network error: {type(e).__name__}", url=url) from e
        self._check_response(resp, url)
        return resp

    @staticmethod
    def _check_response(resp: requests.Response, url: str) -> None:
        """Проверяет HTTP-ответ и бросает WBAPIError при ошибке."""
        if resp.status_code == 200:
            return
        msg = f"HTTP {resp.status_code}"
        if resp.status_code == 401:
            msg += " — Unauthorized (check token)"
        elif resp.status_code == 403:
            msg += " — Forbidden (token lacks permission)"
        elif resp.status_code == 429:
            msg += " — Rate limited (try later)"
        else:
            # Первые 200 символов тела для диагностики
            body_preview = resp.text[:200] if resp.text else ""
            msg += f" — {body_preview}"
        raise WBAPIError(msg, status_code=resp.status_code, url=url)
