"""Streamlit entrypoint — MP Analytics (WB + Ozon)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np

from src.io_excel import load_excel, validate_dataframe
from src.metrics import add_calculated_metrics, compute_deltas
from src.tables import (
    build_ad_current,
    build_ad_future,
    build_color_anomalies,
    build_glue_analysis,
    build_kpi_table,
    build_price_lag,
    build_scale_candidates,
    build_top_articles,
    build_top_models,
    build_top_models_with_funnel,
    build_warehouse,
)
from src.aggregations import (
    agg_by_article,
    agg_by_model,
    agg_by_color_collection,
    agg_by_status,
    agg_by_glue,
)
from src.prompt_builder import build_prompt_brief, build_prompt_detailed, build_prompt_detailed_all_platforms
from src.export import export_tables_to_xlsx, export_brief_to_csv
from src.schema import col_p, CALC_MARGIN_RATE, CALC_ANNUAL_YIELD
from src.wb_funnel_io import load_funnel_excel, CALC_CTR, CALC_CONV_CART, CALC_CONV_ORDER, CALC_PURCHASE_RATE
from src.wb_funnel_metrics import (
    join_funnel_to_economics,
    build_funnel_kpi,
    build_funnel_economics_diag,
    build_conversion_growth_points,
    enrich_glue_with_funnel,
    funnel_agg_by_model,
    funnel_agg_by_status,
    funnel_agg_by_glue,
    _fp,
)
from src.wb_api.client import WBClient, WBAPIError, get_token, get_token_2, get_all_tokens, mask_token
from src.wb_api.inspector import (
    summarize_schema, test_funnel_call, test_ads_campaign_list,
    test_ads_fullstats_call, save_sample,
)
from src.wb_api.funnel import fetch_funnel
from src.wb_api.ads import fetch_ads, fetch_campaign_ids
from src.wb_api.adapters import (
    api_funnel_to_excel_format_pair, ads_df_with_prefix,
    merge_ads_periods, join_ads_to_economics, get_unmatched_ads, ADS_PREFIX,
)
from src.wb_ads_metrics import (
    has_ads_cols, build_ads_kpi, build_ads_by_article,
    build_ads_funnel_price, ads_agg_by_model, ads_agg_by_status,
    ads_agg_by_color_collection, ads_agg_by_glue, _ap,
)
from src.display_utils import (
    TABLE_CSS, PERIOD_DEFINITIONS,
    LEGEND_WAREHOUSE, LEGEND_SCALE, LEGEND_ANTI_TOP,
    LEGEND_AD_FUTURE, LEGEND_AD_CURRENT, LEGEND_DIAG_FLAGS,
    LEGEND_COLOR_ANOMALY, LEGEND_CONVERSION_GROWTH, LEGEND_ICONS_GENERAL,
    format_df_for_display,
    display_copyable_table,
)

# ---------------------------------------------------------------------------
# Настройки страницы
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MP Analytics — WB & Ozon",
    page_icon="📊",
    layout="wide",
)

st.title("📊 MP Analytics — WB & Ozon")
st.markdown(TABLE_CSS, unsafe_allow_html=True)
st.caption(PERIOD_DEFINITIONS)

# ---------------------------------------------------------------------------
# Sidebar: загрузка файлов
# ---------------------------------------------------------------------------

st.sidebar.header("Загрузка данных")

file_wb = st.sidebar.file_uploader("Файл WB (.xlsx)", type=["xlsx"], key="wb")
file_ozon = st.sidebar.file_uploader("Файл Ozon (.xlsx)", type=["xlsx"], key="ozon")

# Ручной переключатель площадки
swap = st.sidebar.checkbox("Поменять файлы местами (WB ↔ Ozon)", value=False)

if swap:
    file_wb, file_ozon = file_ozon, file_wb

st.sidebar.divider()
st.sidebar.subheader("Воронка WB")
file_funnel_p1 = st.sidebar.file_uploader(
    "WB Funnel п1 (прошлый период)", type=["xlsx"], key="funnel_p1",
    accept_multiple_files=True,
)
file_funnel_p2 = st.sidebar.file_uploader(
    "WB Funnel п2 (текущий период)", type=["xlsx"], key="funnel_p2",
    accept_multiple_files=True,
)

# ---------------------------------------------------------------------------
# WB API — переключатели и даты
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("WB API")

_all_tokens = get_all_tokens()
_wb_token_ok = len(_all_tokens) > 0
if _wb_token_ok:
    for _lbl, _tok in _all_tokens:
        st.sidebar.success(f"{_lbl}: {mask_token(_tok)}")
else:
    st.sidebar.warning("Token not set → .streamlit/secrets.toml")

funnel_source = st.sidebar.radio(
    "WB Funnel source", ["Excel", "API"], index=0, key="funnel_src", horizontal=True,
    disabled=not _wb_token_ok,
)
ads_enabled = st.sidebar.toggle(
    "WB Ads (API)", value=False, key="ads_on",
    disabled=not _wb_token_ok,
)

# Даты (для API режима)
_api_needs_dates = (funnel_source == "API") or ads_enabled
if _api_needs_dates and _wb_token_ok:
    st.sidebar.markdown("**Периоды (API)**")
    _today = date.today()
    _def_p2_start = _today - timedelta(days=14)
    _def_p2_end = _today - timedelta(days=1)

    p2_start = st.sidebar.date_input("п2 начало (текущий)", value=_def_p2_start, key="p2s")
    p2_end = st.sidebar.date_input("п2 конец", value=_def_p2_end, key="p2e")

    auto_p1 = st.sidebar.checkbox("Авто-п1 (та же длина перед п2)", value=True, key="auto_p1")
    if auto_p1:
        _p2_len = (p2_end - p2_start).days + 1
        p1_end = p2_start - timedelta(days=1)
        p1_start = p1_end - timedelta(days=_p2_len - 1)
    else:
        p1_start = st.sidebar.date_input("п1 начало (прошлый)", value=_def_p2_start - timedelta(days=15), key="p1s")
        p1_end = st.sidebar.date_input("п1 конец", value=_def_p2_start - timedelta(days=1), key="p1e")

    st.sidebar.caption(f"п2 (текущий): {p2_start} → {p2_end} | п1 (прошлый): {p1_start} → {p1_end}")
else:
    p1_start = p1_end = p2_start = p2_end = None


# ---------------------------------------------------------------------------
# Обработка файлов
# ---------------------------------------------------------------------------

def process_file(file, platform: str) -> tuple[pd.DataFrame | None, dict]:
    """Загрузка, валидация, расчёт метрик."""
    if file is None:
        return None, {}
    try:
        df = load_excel(file)
    except Exception as e:
        st.error(f"Ошибка чтения файла {platform}: {e}")
        return None, {}

    df, validation = validate_dataframe(df)
    info = {"validation": validation, "rows": len(df), "cols": len(df.columns)}

    if validation.is_critical:
        st.error(
            f"**{platform}**: Критичные колонки отсутствуют: "
            f"{', '.join(validation.missing_id_cols)}. Анализ невозможен."
        )
        return None, info

    df = add_calculated_metrics(df)
    df = compute_deltas(df)
    return df, info


df_wb, info_wb = process_file(file_wb, "WB")
df_ozon, info_ozon = process_file(file_ozon, "Ozon")

has_wb = df_wb is not None and not df_wb.empty
has_ozon = df_ozon is not None and not df_ozon.empty


# ---------------------------------------------------------------------------
# Обработка воронки WB
# ---------------------------------------------------------------------------

funnel_p1_raw = None
funnel_p2_raw = None
funnel_warnings: list[str] = []
df_wb_merged = None  # WB economics + funnel (если доступен)
funnel_coverage: dict = {}
has_funnel = False

def _load_funnel(files: list, label: str):
    """Загружает один или несколько файлов воронки и объединяет их."""
    if not files:
        return None, []
    all_dfs: list[pd.DataFrame] = []
    all_warns: list[str] = []
    for i, f in enumerate(files):
        suffix = f" [{i+1}]" if len(files) > 1 else ""
        df, warns = load_funnel_excel(f)
        all_warns.extend(f"[{label}{suffix}] {w}" for w in warns)
        if not df.empty:
            all_dfs.append(df)
    if not all_dfs:
        return None, all_warns
    combined = pd.concat(all_dfs, ignore_index=True)
    if len(all_dfs) > 1:
        all_warns.append(f"[{label}] Объединено {len(all_dfs)} файл(ов), итого строк: {len(combined)}")
    return combined, all_warns

funnel_source_label = "Excel"  # для промптов
df_funnel_api_raw = None  # сырой API funnel DF (для join ads → vendorCode)

# --- Кэшированные обёртки API (TTL 5 мин) ---
@st.cache_data(ttl=300, show_spinner="Загрузка воронки WB…")
def _cached_fetch_funnel(token: str, sel_start: str, sel_end: str, past_start: str, past_end: str) -> pd.DataFrame:
    client = WBClient(token=token)
    return fetch_funnel(client, sel_start, sel_end, past_start, past_end)

@st.cache_data(ttl=300, show_spinner="Загрузка кампаний WB…")
def _cached_fetch_campaign_ids(token: str) -> list[int]:
    client = WBClient(token=token)
    return fetch_campaign_ids(client)

@st.cache_data(ttl=300, show_spinner="Загрузка рекламы WB…")
def _cached_fetch_ads(token: str, start: str, end: str, campaign_ids: tuple[int, ...] | None = None) -> pd.DataFrame:
    client = WBClient(token=token)
    ids_list = list(campaign_ids) if campaign_ids else None
    return fetch_ads(client, start, end, campaign_ids=ids_list)


if funnel_source == "API" and _wb_token_ok and has_wb and p2_start is not None:
    # --- Funnel через API (все токены, кэшировано) ---
    funnel_source_label = "API"
    _funnel_parts: list[pd.DataFrame] = []
    _funnel_api_error = False
    for _t_lbl, _t_val in _all_tokens:
        try:
            _df_part = _cached_fetch_funnel(
                _t_val,
                sel_start=str(p2_start), sel_end=str(p2_end),
                past_start=str(p1_start), past_end=str(p1_end),
            )
            if not _df_part.empty:
                _funnel_parts.append(_df_part)
                funnel_warnings.append(f"[API {_t_lbl}] Funnel: {len(_df_part)} карточек")
            else:
                funnel_warnings.append(f"[API {_t_lbl}] Funnel: пустой ответ")
        except WBAPIError as e:
            funnel_warnings.append(f"[API {_t_lbl}] Funnel ошибка: {e}")
            _funnel_api_error = True

    if _funnel_parts:
        df_funnel_api_raw = pd.concat(_funnel_parts, ignore_index=True)
        df_funnel_api_raw = df_funnel_api_raw.drop_duplicates(subset=["nmId"], keep="first")
        funnel_p1_raw, funnel_p2_raw = api_funnel_to_excel_format_pair(df_funnel_api_raw)
        funnel_warnings.append(f"[API] Funnel итого: {len(df_funnel_api_raw)} уникальных карточек")
    elif _funnel_api_error:
        funnel_p1_raw, w1 = _load_funnel(file_funnel_p1, "Funnel п1")
        funnel_p2_raw, w2 = _load_funnel(file_funnel_p2, "Funnel п2")
        funnel_warnings.extend(w1 + w2)
        funnel_source_label = "Excel (fallback)"
    else:
        funnel_warnings.append("[API] Funnel: пустой ответ от всех токенов")
else:
    # --- Funnel через Excel ---
    funnel_p1_raw, w1 = _load_funnel(file_funnel_p1, "Funnel п1")
    funnel_p2_raw, w2 = _load_funnel(file_funnel_p2, "Funnel п2")
    funnel_warnings = w1 + w2

if has_wb and (funnel_p1_raw is not None or funnel_p2_raw is not None):
    df_wb_merged, funnel_coverage = join_funnel_to_economics(
        df_wb, funnel_p1_raw, funnel_p2_raw
    )
    has_funnel = True
elif has_wb:
    df_wb_merged = df_wb


# ---------------------------------------------------------------------------
# Обработка рекламы WB (API only)
# ---------------------------------------------------------------------------

ads_coverage: dict = {}
has_ads = False
ads_source_label = "Off"
df_ads_merged_wide: pd.DataFrame = pd.DataFrame()

if "ads_overrides" not in st.session_state:
    st.session_state["ads_overrides"] = {}

if ads_enabled and _wb_token_ok and has_wb and p2_start is not None:
    ads_source_label = "API"
    ads_warnings_list: list[str] = []

    # Собираем ads по всем токенам (кэшировано)
    _ads_p2_parts: list[pd.DataFrame] = []
    _ads_p1_parts: list[pd.DataFrame] = []
    _ads_api_error = False
    for _t_lbl, _t_val in _all_tokens:
        try:
            _cids = _cached_fetch_campaign_ids(_t_val)
            ads_warnings_list.append(f"[API {_t_lbl}] Campaigns: {len(_cids)} ids")
            if not _cids:
                ads_warnings_list.append(f"[API {_t_lbl}] Ads: нет активных кампаний")
                continue
            _cids_tuple = tuple(_cids)
            _a_p2 = _cached_fetch_ads(_t_val, str(p2_start), str(p2_end), campaign_ids=_cids_tuple)
            _a_p1 = _cached_fetch_ads(_t_val, str(p1_start), str(p1_end), campaign_ids=_cids_tuple)
            if not _a_p2.empty:
                _ads_p2_parts.append(_a_p2)
            if not _a_p1.empty:
                _ads_p1_parts.append(_a_p1)
            ads_warnings_list.append(
                f"[API {_t_lbl}] Ads: п2={len(_a_p2)} / п1={len(_a_p1)} nmId"
            )
        except WBAPIError as e:
            ads_warnings_list.append(f"[API {_t_lbl}] Ads ошибка: {e}")
            _ads_api_error = True

    # Concat + dedup по nmId (суммируем метрики для одного nmId из разных токенов)
    df_ads_p2 = pd.concat(_ads_p2_parts, ignore_index=True) if _ads_p2_parts else pd.DataFrame()
    df_ads_p1 = pd.concat(_ads_p1_parts, ignore_index=True) if _ads_p1_parts else pd.DataFrame()

    # Дедупликация: один nmId может прийти из обоих токенов — оставляем первый
    if not df_ads_p2.empty and "nmId" in df_ads_p2.columns:
        df_ads_p2 = df_ads_p2.drop_duplicates(subset=["nmId"], keep="first")
    if not df_ads_p1.empty and "nmId" in df_ads_p1.columns:
        df_ads_p1 = df_ads_p1.drop_duplicates(subset=["nmId"], keep="first")

    if df_ads_p1.empty and df_ads_p2.empty:
        ads_warnings_list.append("[API] Ads: нет данных за выбранные периоды")
        if _ads_api_error:
            ads_source_label = "API (error)"
    else:
        ads_p2_wide = ads_df_with_prefix(df_ads_p2, "п2") if not df_ads_p2.empty else pd.DataFrame()
        ads_p1_wide = ads_df_with_prefix(df_ads_p1, "п1") if not df_ads_p1.empty else pd.DataFrame()
        df_ads_merged_wide = merge_ads_periods(ads_p1_wide, ads_p2_wide)

        cnt = len(df_ads_merged_wide)
        ads_warnings_list.append(f"[API] Ads итого: {cnt} уникальных nmId")

        # Join ads → economics
        df_wb_merged, ads_coverage = join_ads_to_economics(
            df_wb_merged if df_wb_merged is not None else df_wb,
            df_ads_merged_wide,
            df_funnel_api=df_funnel_api_raw,
            overrides=st.session_state.get("ads_overrides"),
        )
        has_ads = ads_coverage.get("has_ads", False)

    funnel_warnings.extend(ads_warnings_list)


# ---------------------------------------------------------------------------
# Data Quality Warnings
# ---------------------------------------------------------------------------

def show_warnings(info: dict, platform: str):
    """Показывает предупреждения о качестве данных с разделением на критичные и информационные."""
    v = info.get("validation")
    if v is None:
        return
    if not v.has_warnings:
        return

    # Определяем иконку заголовка по наличию критичных предупреждений
    icon = "🔴" if v.has_critical_warnings else "ℹ️"
    with st.expander(f"{icon} Качество данных — {platform}", expanded=v.has_critical_warnings):

        # --- Критичные ---
        if v.has_critical_warnings:
            st.markdown("#### 🔴 Требуют внимания")
            if v.type_conversion_errors:
                for err in v.type_conversion_errors:
                    st.error(f"**Ошибка типов:** {err}")
            if v.critical_warnings:
                for w in v.critical_warnings:
                    st.error(w)

        # --- Информационные ---
        if v.has_info_warnings:
            st.markdown("#### ℹ️ К сведению")
            if v.missing_metric_cols:
                st.info(
                    f"Не найдены некоторые колонки метрик ({len(v.missing_metric_cols)}): "
                    f"{', '.join(v.missing_metric_cols[:10])}"
                    + ("…" if len(v.missing_metric_cols) > 10 else "")
                    + ". Анализ продолжится по доступным данным."
                )
            if v.info_warnings:
                for w in v.info_warnings:
                    st.info(w)


if has_wb:
    show_warnings(info_wb, "WB")
if has_ozon:
    show_warnings(info_ozon, "Ozon")

# Предупреждения воронки
if funnel_warnings:
    with st.expander("ℹ️ Воронка WB — предупреждения", expanded=False):
        for w in funnel_warnings:
            st.warning(w)

if has_funnel and funnel_coverage:
    with st.expander("📊 Funnel coverage (WB)", expanded=False):
        c = funnel_coverage
        cols_cov = st.columns(3)
        with cols_cov[0]:
            st.metric("Артикулов в экономике", c.get("econ_total", 0))
        with cols_cov[1]:
            matched_p1 = c.get("matched_p1", 0)
            total_p1 = c.get("funnel_p1_total", 0)
            pct_p1 = f"{matched_p1 / c['econ_total'] * 100:.0f}%" if c.get("econ_total") else "—"
            st.metric("Совпало п1", f"{matched_p1} / {c.get('econ_total', 0)} ({pct_p1})")
        with cols_cov[2]:
            matched_p2 = c.get("matched_p2", 0)
            total_p2 = c.get("funnel_p2_total", 0)
            pct_p2 = f"{matched_p2 / c['econ_total'] * 100:.0f}%" if c.get("econ_total") else "—"
            st.metric("Совпало п2", f"{matched_p2} / {c.get('econ_total', 0)} ({pct_p2})")

# Ads coverage + override UI
if has_ads and ads_coverage:
    with st.expander("📊 Ads coverage (WB API)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("nmId в Ads", ads_coverage.get("ads_nm_total", 0))
        with c2:
            mf = ads_coverage.get("matched_via_funnel", 0)
            mo = ads_coverage.get("matched_via_override", 0)
            st.metric("Matched (funnel)", mf)
            st.metric("Matched (override)", mo)
        with c3:
            unm = ads_coverage.get("unmatched", 0)
            sp = ads_coverage.get("spend_matched_pct", 0)
            st.metric("Unmatched", unm)
            st.metric("Spend matched, %", f"{sp:.0f}%")

    with st.expander("🔧 Ads Matching: overrides", expanded=False):
        st.markdown("**Ручное сопоставление nmId → Артикул (экономика)**")
        uploaded_map = st.file_uploader("Upload mapping.json", type=["json"], key="upload_mapping")
        if uploaded_map is not None:
            try:
                loaded = json.loads(uploaded_map.read().decode("utf-8"))
                if isinstance(loaded, dict):
                    st.session_state["ads_overrides"].update(loaded)
                    st.success(f"Загружено {len(loaded)} override(s).")
                    st.rerun()
            except Exception as e:
                st.error(f"Ошибка JSON: {e}")

        current_overrides = st.session_state.get("ads_overrides", {})
        if current_overrides:
            st.download_button(
                "📥 Download mapping.json",
                data=json.dumps(current_overrides, ensure_ascii=False, indent=2),
                file_name="ads_mapping.json",
                mime="application/json",
                key="download_mapping",
            )
            st.caption(f"Текущие overrides: {len(current_overrides)}")

        if has_wb and df_wb is not None:
            econ_articles = sorted(df_wb["Артикул"].dropna().unique().tolist())
            nm_input = st.text_input("nmId (число)", key="override_nm_id")
            econ_sel = st.selectbox("→ Артикул экономики", [""] + econ_articles, key="override_econ")
            if st.button("Добавить override", key="add_override"):
                if nm_input.strip() and econ_sel:
                    st.session_state["ads_overrides"][nm_input.strip()] = econ_sel
                    st.success(f"Override: {nm_input.strip()} → {econ_sel}")
                    st.rerun()

        # Таблица unmatched
        if not df_ads_merged_wide.empty:
            unmatched_df = get_unmatched_ads(
                df_ads_merged_wide, df_funnel_api_raw, st.session_state.get("ads_overrides"),
            )
            if not unmatched_df.empty:
                st.markdown("**Непривязанные nmId (по spend):**")
                display_copyable_table(unmatched_df.head(20), format_display=False)


# ---------------------------------------------------------------------------
# Настройки порогов (sidebar)
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("⚙️ Пороги анализа")

with st.sidebar.expander("Масштабирование", expanded=False):
    _scale_stock = st.number_input("Остатки п2 ≥", value=200, min_value=0, key="thr_scale_stock")
    _scale_turn = st.number_input("Оборот п2 ≥ дн", value=10.0, min_value=0.0, key="thr_scale_turn")
    _scale_cover = st.number_input("DaysCover п2 ≥ дн", value=90.0, min_value=0.0, key="thr_scale_cover")

with st.sidebar.expander("Склад 🔴", expanded=False):
    _wh_stock = st.number_input("Остатки п2 ≥", value=100, min_value=0, key="thr_wh_stock")
    _wh_turn = st.number_input("Оборот п2 ≥ дн", value=200.0, min_value=0.0, key="thr_wh_turn")

with st.sidebar.expander("Реклама ⚠️", expanded=False):
    _ad_pct = st.number_input("Δ рекламы ≥ %", value=10.0, min_value=0.0, key="thr_ad_pct")
    _ad_abs = st.number_input("Δ рекламы ≥ ₽", value=500.0, min_value=0.0, key="thr_ad_abs")

with st.sidebar.expander("Цветовые аномалии", expanded=False):
    _anom_ratio = st.number_input("Порог ratio", value=0.3, min_value=0.0, max_value=1.0, step=0.05, key="thr_anom_ratio")
    _anom_margin = st.number_input("Мин. макс. маржа ₽", value=1000.0, min_value=0.0, key="thr_anom_margin")

# Длина периода (для DaysCover)
_period_days = 14
if _api_needs_dates and _wb_token_ok:
    _period_days = (p2_end - p2_start).days + 1


# ---------------------------------------------------------------------------
# Фильтры (sidebar)
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """Фильтрация по статусу, модели, артикулу."""
    filtered = df.copy()

    if "Статус" in filtered.columns:
        statuses = sorted(filtered["Статус"].dropna().unique().tolist())
        default_statuses = [s for s in statuses if s.strip().lower() != "архив"]
        selected = st.sidebar.multiselect(
            f"Статус ({platform})", statuses, default=default_statuses, key=f"status_{platform}"
        )
        if selected:
            filtered = filtered[filtered["Статус"].isin(selected)]

    if "Модель" in filtered.columns:
        models = sorted(filtered["Модель"].dropna().unique().tolist())
        selected_models = st.sidebar.multiselect(
            f"Модель ({platform})", models, default=[], key=f"model_{platform}"
        )
        if selected_models:
            filtered = filtered[filtered["Модель"].isin(selected_models)]

    search_art = st.sidebar.text_input(f"Поиск артикула ({platform})", key=f"search_{platform}")
    if search_art and "Артикул" in filtered.columns:
        filtered = filtered[
            filtered["Артикул"].astype(str).str.contains(search_art, case=False, na=False)
        ]

    return filtered

# ---------------------------------------------------------------------------
# Рендеринг таблиц площадки
# ---------------------------------------------------------------------------

def render_platform_tab(df_raw: pd.DataFrame, platform: str, is_wb: bool,
                        wb_has_funnel: bool = False, wb_has_ads: bool = False,
                        funnel_df: pd.DataFrame | None = None):
    """Рендерит содержимое вкладки площадки."""

    df = apply_filters(df_raw, platform)
    if df.empty:
        st.info("Нет данных после фильтрации.")
        return

    mode = st.radio(
        "Режим отчёта",
        ["Краткая", "Детальная"],
        key=f"mode_{platform}",
        horizontal=True,
    )

    if mode == "Краткая":
        _render_brief(df, platform, is_wb, wb_has_funnel, wb_has_ads)
    else:
        _render_detailed(df, platform, is_wb, wb_has_funnel, wb_has_ads, funnel_df)

    # Кнопка генерации промпта
    st.divider()
    if st.button(f"🤖 Сгенерировать промпт для ИИ ({mode})", key=f"prompt_btn_{platform}_{mode}"):
        if mode == "Краткая":
            if is_wb:
                prompt = build_prompt_brief(
                    df, None, wb_merged=df if wb_has_funnel else None,
                    coverage=funnel_coverage if wb_has_funnel else None,
                    ads_cov=ads_coverage if wb_has_ads else None,
                    funnel_src=funnel_source_label, ads_src=ads_source_label,
                )
            else:
                prompt = build_prompt_brief(None, df)
        else:
            prompt = build_prompt_detailed(
                df, platform, is_wb=is_wb,
                wb_merged=df if (is_wb and wb_has_funnel) else None,
                coverage=funnel_coverage if (is_wb and wb_has_funnel) else None,
                ads_cov=ads_coverage if wb_has_ads else None,
                funnel_src=funnel_source_label, ads_src=ads_source_label,
            )

        st.text_area(
            f"Промпт ({mode}) — {platform}",
            value=prompt,
            height=400,
            key=f"prompt_area_{platform}_{mode}",
        )
        st.caption("Выделите весь текст (Ctrl/Cmd+A в поле) и скопируйте (Ctrl/Cmd+C).")


def _render_brief(df: pd.DataFrame, platform: str, is_wb: bool,
                  wb_has_funnel: bool = False, wb_has_ads: bool = False):
    """Краткий режим: KPI + ТОП/анти-ТОП."""
    st.subheader("Итоговые KPI")
    kpi = build_kpi_table(df)
    if not kpi.empty:
        display_copyable_table(kpi)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ТОП-10 артикулов (по марже п2)")
        top, _ = build_top_articles(df, n=10)
        if not top.empty:
            display_copyable_table(top)

    with col2:
        st.subheader("Анти-ТОП-10 артикулов")
        _, bottom = build_top_articles(df, n=10)
        if not bottom.empty:
            display_copyable_table(bottom)
        with st.expander("ℹ️ Легенда", expanded=False):
            st.markdown(LEGEND_ANTI_TOP)

    st.subheader("Кандидаты на масштабирование")
    scale = build_scale_candidates(
        df, n=10,
        stock_min=_scale_stock, turnover_min=_scale_turn,
        days_cover_min=_scale_cover, period_days=_period_days,
    )
    if not scale.empty:
        display_copyable_table(scale)
    else:
        st.info("Нет артикулов, прошедших все фильтры масштабирования.")
    with st.expander("ℹ️ Легенда", expanded=False):
        st.markdown(LEGEND_SCALE.format(
            stock_thr=_scale_stock, turn_thr=_scale_turn, cover_thr=_scale_cover,
        ))

    st.subheader("ТОП моделей")
    top_m = build_top_models(df, n=10)
    if not top_m.empty:
        display_copyable_table(top_m)

    # --- Воронка WB (краткая) ---
    if is_wb and wb_has_funnel:
        _render_funnel_brief(df)

    # --- Реклама WB (краткая) ---
    if is_wb and wb_has_ads:
        _render_ads_brief(df)


def _render_detailed(df: pd.DataFrame, platform: str, is_wb: bool,
                    wb_has_funnel: bool = False, wb_has_ads: bool = False,
                    funnel_df: pd.DataFrame | None = None):
    """Детальный режим: все таблицы + экспорт."""
    tables_for_export: dict[str, pd.DataFrame] = {}

    # Общая легенда иконок
    with st.expander("ℹ️ Общие обозначения иконок", expanded=False):
        st.markdown(LEGEND_ICONS_GENERAL)

    # 1) KPI
    st.subheader("1. Итоговые KPI")
    kpi = build_kpi_table(df)
    if not kpi.empty:
        display_copyable_table(kpi)
        tables_for_export["KPI"] = kpi

    # 2) ТОП/анти-ТОП артикулов
    st.subheader("2. ТОП / анти-ТОП артикулов")
    top, bottom = build_top_articles(df, n=10)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ТОП-10**")
        if not top.empty:
            display_copyable_table(top)
            tables_for_export["ТОП артикулов"] = top
    with col2:
        st.markdown("**Анти-ТОП-10**")
        if not bottom.empty:
            display_copyable_table(bottom)
            tables_for_export["Анти-ТОП артикулов"] = bottom
        with st.expander("ℹ️ Легенда", expanded=False):
            st.markdown(LEGEND_ANTI_TOP)

    scale = build_scale_candidates(
        df, n=10,
        stock_min=_scale_stock, turnover_min=_scale_turn,
        days_cover_min=_scale_cover, period_days=_period_days,
    )
    st.markdown("**Кандидаты на масштабирование**")
    if not scale.empty:
        display_copyable_table(scale)
        tables_for_export["Масштабирование"] = scale
    else:
        st.info("Нет артикулов, прошедших все фильтры масштабирования.")
    with st.expander("ℹ️ Легенда", expanded=False):
        st.markdown(LEGEND_SCALE.format(
            stock_thr=_scale_stock, turn_thr=_scale_turn, cover_thr=_scale_cover,
        ))

    # ТОП моделей
    st.subheader("2b. ТОП моделей")
    if is_wb and funnel_df is not None:
        top_m = build_top_models_with_funnel(df, funnel_df, n=10)
    else:
        top_m = build_top_models(df, n=10)
    if not top_m.empty:
        display_copyable_table(top_m)
        tables_for_export["ТОП моделей"] = top_m

    # 3) Реклама и лаг
    st.subheader("3. Реклама и лаг")
    st.markdown("**A) Будущее (заказы)**")
    ad_f = build_ad_future(df, ad_delta_pct_thr=_ad_pct, ad_delta_abs_thr=_ad_abs)
    if not ad_f.empty:
        display_copyable_table(ad_f)
        tables_for_export["Реклама-заказы"] = ad_f
    with st.expander("ℹ️ Легенда", expanded=False):
        st.markdown(LEGEND_AD_FUTURE.format(ad_pct_thr=_ad_pct, ad_abs_thr=_ad_abs))

    st.markdown("**B) Настоящее (продажи)**")
    ad_c = build_ad_current(df)
    if not ad_c.empty:
        display_copyable_table(ad_c)
        tables_for_export["Реклама-продажи"] = ad_c
    with st.expander("ℹ️ Легенда", expanded=False):
        st.markdown(LEGEND_AD_CURRENT)

    # 4) Склад/оборотка
    st.subheader("4. Склад / оборотка")
    wh = build_warehouse(df, stock_thr=_wh_stock, turnover_thr=_wh_turn)
    if not wh.empty:
        display_copyable_table(wh)
        tables_for_export["Склад"] = wh
    with st.expander("ℹ️ Легенда", expanded=False):
        st.markdown(LEGEND_WAREHOUSE.format(stock_thr=_wh_stock, turn_thr=_wh_turn))

    # 5) Лаг цен
    st.subheader("5. Лаг цен")
    pl = build_price_lag(df)
    if not pl.empty:
        display_copyable_table(pl)
        tables_for_export["Лаг цен"] = pl

    # 6) Цветовые аномалии
    st.subheader("6. Цветовые аномалии (Color code + Коллекция)")
    ca = build_color_anomalies(df, anomaly_ratio=_anom_ratio, min_top_margin=_anom_margin)
    if not ca.empty:
        display_copyable_table(ca)
        tables_for_export["Цветовые аномалии"] = ca
    with st.expander("ℹ️ Легенда", expanded=False):
        st.markdown(LEGEND_COLOR_ANOMALY.format(ratio=_anom_ratio, min_margin=_anom_margin))

    # 7) Склейки WB
    if is_wb:
        st.subheader("7. Склейки WB")
        gl = build_glue_analysis(df)
        if not gl.empty:
            if wb_has_funnel:
                gl = enrich_glue_with_funnel(gl, df)
            display_copyable_table(gl)
            tables_for_export["Склейки WB"] = gl

    # 8) Воронка WB
    if is_wb and wb_has_funnel:
        _render_funnel_detailed(df, tables_for_export)
    elif is_wb and not wb_has_funnel:
        st.info("Загрузите файлы воронки WB (Funnel п1/п2) в боковой панели для анализа конверсий.")

    # 9) Реклама WB (True Ads API)
    if is_wb and wb_has_ads:
        _render_ads_detailed(df, tables_for_export)
    elif is_wb and not wb_has_ads:
        st.info("Включите WB Ads (API) в боковой панели для анализа рекламы.")

    # Агрегации
    with st.expander("Агрегация по Модели"):
        agg_m = agg_by_model(df)
        if not agg_m.empty:
            display_copyable_table(agg_m, format_display=False)
            tables_for_export["Агр. по Модели"] = agg_m

    with st.expander("Агрегация по Color code + Коллекция"):
        agg_cc = agg_by_color_collection(df)
        if not agg_cc.empty:
            display_copyable_table(agg_cc, format_display=False)
            tables_for_export["Агр. Color+Коллекция"] = agg_cc

    with st.expander("Агрегация по Статусу"):
        agg_s = agg_by_status(df)
        if not agg_s.empty:
            display_copyable_table(agg_s, format_display=False)
            tables_for_export["Агр. по Статусу"] = agg_s

    if is_wb:
        with st.expander("Агрегация по Склейке"):
            agg_g = agg_by_glue(df)
            if not agg_g.empty:
                display_copyable_table(agg_g, format_display=False)
                tables_for_export["Агр. по Склейке"] = agg_g

    # Экспорт
    st.divider()
    if tables_for_export:
        xlsx_buf = export_tables_to_xlsx(tables_for_export)
        st.download_button(
            label="📥 Export to Excel",
            data=xlsx_buf,
            file_name=f"{platform}_detailed_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"export_{platform}",
        )


# ---------------------------------------------------------------------------
# Funnel-рендер (WB only)
# ---------------------------------------------------------------------------

def _render_funnel_brief(df: pd.DataFrame, format_options: dict | None = None):
    """Краткие funnel-секции для WB.

    Аргумент `format_options` прокидывается в `display_copyable_table`.
    """
    st.divider()
    st.subheader("Воронка WB (общая)")

    fkpi = build_funnel_kpi(df)
    if not fkpi.empty:
        display_copyable_table(fkpi, format_display=True, format_options=format_options)

    # (Точки роста конверсий и легенда удалены по запросу)


def _render_funnel_detailed(df: pd.DataFrame, tables_for_export: dict):
    """Детальные funnel-секции для WB."""
    st.divider()
    st.subheader("8. Воронка WB")

    # A) Итог воронки
    st.markdown("**A) Итог воронки WB**")
    fkpi = build_funnel_kpi(df)
    if not fkpi.empty:
        display_copyable_table(fkpi)
        tables_for_export["Воронка KPI"] = fkpi

    # B) Воронка × Экономика диагностика
    st.markdown("**B) Воронка × Экономика: диагностика**")
    diag = build_funnel_economics_diag(df)
    if not diag.empty:
        display_copyable_table(diag)
        tables_for_export["Воронка×Экономика"] = diag
    with st.expander("ℹ️ Легенда флагов", expanded=False):
        st.markdown(LEGEND_DIAG_FLAGS)

    # C) Точки роста конверсий
    st.markdown("**C) Точки роста конверсий**")
    cgp = build_conversion_growth_points(df, n=15)
    if not cgp.empty:
        display_copyable_table(cgp)
        tables_for_export["Точки роста конв."] = cgp
        with st.expander("ℹ️ Легенда", expanded=False):
            st.markdown(LEGEND_CONVERSION_GROWTH)

    # Агрегации воронки
    with st.expander("Воронка: агрегация по Модели"):
        f_mod = funnel_agg_by_model(df)
        if not f_mod.empty:
            display_copyable_table(f_mod, format_display=False)
            tables_for_export["Воронка по Модели"] = f_mod

    with st.expander("Воронка: агрегация по Статусу"):
        f_st = funnel_agg_by_status(df)
        if not f_st.empty:
            display_copyable_table(f_st, format_display=False)
            tables_for_export["Воронка по Статусу"] = f_st

    with st.expander("Воронка: агрегация по Склейке"):
        f_gl = funnel_agg_by_glue(df)
        if not f_gl.empty:
            display_copyable_table(f_gl, format_display=False)
            tables_for_export["Воронка по Склейке"] = f_gl


# ---------------------------------------------------------------------------
# Ads-рендер (WB only, API)
# ---------------------------------------------------------------------------

def _render_ads_brief(df: pd.DataFrame):
    """Краткие ads-секции для WB."""
    st.divider()
    st.subheader("📢 Реклама WB — True Ads (краткая)")

    akpi = build_ads_kpi(df)
    if not akpi.empty:
        display_copyable_table(akpi)

    st.markdown("**ТОП-5 по затратам (деградация)**")
    ads_art = build_ads_by_article(df, n=5)
    if not ads_art.empty:
        display_copyable_table(ads_art)
    else:
        st.caption("Нет данных.")


def _render_ads_detailed(df: pd.DataFrame, tables_for_export: dict):
    """Детальные ads-секции для WB."""
    st.divider()
    st.subheader("9. Реклама WB — True Ads (API)")

    # A) KPI
    st.markdown("**A) WB Ads Summary (п1/п2/Δ)**")
    akpi = build_ads_kpi(df)
    if not akpi.empty:
        display_copyable_table(akpi)
        tables_for_export["WB_Ads_Summary"] = akpi

    # B) По артикулам
    st.markdown("**B) True Ads по артикулам**")
    ads_art = build_ads_by_article(df, n=30)
    if not ads_art.empty:
        display_copyable_table(ads_art)
        tables_for_export["WB_Ads_ByProduct"] = ads_art

    # C) Ads × Funnel × Price
    st.markdown("**C) Ads × Funnel × Price**")
    afp = build_ads_funnel_price(df, n=15)
    if not afp.empty:
        display_copyable_table(afp)
        tables_for_export["Ads×Funnel×Price"] = afp

    # Агрегации ads
    with st.expander("Ads: агрегация по Модели"):
        a_mod = ads_agg_by_model(df)
        if not a_mod.empty:
            display_copyable_table(a_mod, format_display=False)
            tables_for_export["Ads по Модели"] = a_mod

    with st.expander("Ads: агрегация по Статусу"):
        a_st = ads_agg_by_status(df)
        if not a_st.empty:
            display_copyable_table(a_st, format_display=False)
            tables_for_export["Ads по Статусу"] = a_st

    with st.expander("Ads: агрегация по Склейке"):
        a_gl = ads_agg_by_glue(df)
        if not a_gl.empty:
            display_copyable_table(a_gl, format_display=False)
            tables_for_export["Ads по Склейке"] = a_gl


# ---------------------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------------------

if not has_wb and not has_ozon:
    st.info("Загрузите хотя бы один Excel-файл (WB или Ozon) в боковой панели.")
    st.stop()

tab_names: list[str] = []
if has_wb:
    tab_names.append("WB")
if has_ozon:
    tab_names.append("Ozon")
tab_names.append("Краткая сводка")
tab_names.append("Промпты для ИИ")
if _wb_token_ok:
    tab_names.append("WB API → Инспектор")

tabs = st.tabs(tab_names)
tab_idx = 0

# --- WB ---
if has_wb:
    with tabs[tab_idx]:
        render_platform_tab(
            df_wb_merged if df_wb_merged is not None else df_wb,
            "WB", is_wb=True, wb_has_funnel=has_funnel, wb_has_ads=has_ads,
            funnel_df=df_wb_merged if has_funnel else None,
        )
    tab_idx += 1

# --- Ozon ---
if has_ozon:
    with tabs[tab_idx]:
        render_platform_tab(df_ozon, "Ozon", is_wb=False)
    tab_idx += 1

# --- Краткая сводка ---
with tabs[tab_idx]:
    st.header("Краткая сводка")
    tables_for_export: dict[str, pd.DataFrame] = {}
    if has_wb and has_ozon:
        # Общий итог сначала
        st.subheader("Общий итог WB + Ozon")
        combined = pd.concat([df_wb, df_ozon], ignore_index=True)
        combined = add_calculated_metrics(combined)
        combined = compute_deltas(combined)
        kpi_combined = build_kpi_table(combined)
        if not kpi_combined.empty:
            display_copyable_table(
                kpi_combined,
                format_display=True,
                format_options={"decimals": 1, "thousand_sep": " "},
            )
            tables_for_export["Общий итог WB + Ozon"] = format_df_for_display(
                kpi_combined, decimals=1, thousand_sep=" "
            )

        # Затем индивидуальные итоги по площадкам
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Итог по WB")
            kpi_wb = build_kpi_table(df_wb)
            if not kpi_wb.empty:
                display_copyable_table(
                    kpi_wb,
                    format_display=True,
                    format_options={"decimals": 1, "thousand_sep": " "},
                )
                tables_for_export["Итог по WB"] = format_df_for_display(kpi_wb, decimals=1, thousand_sep=" ")
        with col2:
            st.subheader("Итог по Ozon")
            kpi_ozon = build_kpi_table(df_ozon)
            if not kpi_ozon.empty:
                display_copyable_table(
                    kpi_ozon,
                    format_display=True,
                    format_options={"decimals": 1, "thousand_sep": " "},
                )
                tables_for_export["Итог по Ozon"] = format_df_for_display(kpi_ozon, decimals=1, thousand_sep=" ")

        st.subheader("Модели WB")
        # Показать все модели (не ограничивать ТОП-10)
        if "Модель" in df_wb.columns:
            n_models_wb = int(df_wb["Модель"].nunique())
        else:
            n_models_wb = len(df_wb)

        if has_funnel and df_wb_merged is not None:
            top_wb = build_top_models_with_funnel(df_wb, df_wb_merged, n=n_models_wb)
        else:
            top_wb = build_top_models(df_wb, n=n_models_wb)

        if not top_wb.empty:
            display_copyable_table(
                top_wb,
                format_display=True,
                format_options={"decimals": 1, "thousand_sep": " "},
            )
            tables_for_export["Модели WB"] = format_df_for_display(top_wb, decimals=1, thousand_sep=" ")

        st.subheader("Модели Ozon")
        if "Модель" in df_ozon.columns:
            n_models_oz = int(df_ozon["Модель"].nunique())
        else:
            n_models_oz = len(df_ozon)
        top_oz = build_top_models(df_ozon, n=n_models_oz)
        if not top_oz.empty:
            display_copyable_table(
                top_oz,
                format_display=True,
                format_options={"decimals": 1, "thousand_sep": " "},
            )
            tables_for_export["Модели Ozon"] = format_df_for_display(top_oz, decimals=1, thousand_sep=" ")

        # Воронка WB (краткая) в сводке
        if has_funnel:
            # Используем объединённый DF (экономика + воронка), если он есть
            funnel_df_to_show = df_wb_merged if df_wb_merged is not None else df_wb
            _render_funnel_brief(funnel_df_to_show, format_options={"decimals": 1, "thousand_sep": " "})
            # Добавляем итоговую таблицу воронки в экспорт (в том виде, как показано)
            fkpi = build_funnel_kpi(funnel_df_to_show)
            if not fkpi.empty:
                tables_for_export["Воронка WB (общая)"] = format_df_for_display(fkpi, decimals=1, thousand_sep=" ")

        # Внизу: кнопка экспорта всего, что отображено в сводке
        if tables_for_export:
            # CSV export (multi-block single CSV for brief summary)
            try:
                csv_bytes = export_brief_to_csv(tables_for_export)
                csv_name = f"Краткая сводка_{pd.Timestamp('today').strftime('%Y%m%d')}.csv"
                st.download_button(
                    label="📥 Скачать Краткая сводка (CSV)",
                    data=csv_bytes,
                    file_name=csv_name,
                    mime="text/csv; charset=utf-8",
                    key="export_brief_summary_csv",
                )
            except Exception as e:
                st.warning(f"Не удалось подготовить CSV экспорт: {e}")

            # Excel export (оставляем для совместимости)
            xlsx_buf = export_tables_to_xlsx(tables_for_export)
            file_name = f"Краткая сводка_{pd.Timestamp('today').strftime('%Y%m%d')}.xlsx"
            st.download_button(
                label="📥 Скачать Краткая сводка (Excel)",
                data=xlsx_buf,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="export_brief_summary",
            )
    elif has_wb:
        st.subheader("WB — Краткая")
        # Передаём информацию о том, что есть воронка
        _render_brief(df_wb if df_wb is not None else pd.DataFrame(), "WB", is_wb=True, wb_has_funnel=has_funnel)
        if has_funnel:
            funnel_df_to_show = df_wb_merged if df_wb_merged is not None else df_wb
            _render_funnel_brief(funnel_df_to_show, format_options={"decimals": 1, "thousand_sep": " "})
    elif has_ozon:
        st.subheader("Ozon — Краткая")
        _render_brief(df_ozon, "Ozon", is_wb=False)

tab_idx += 1

# --- Промпты для ИИ ---
with tabs[tab_idx]:
    st.header("Промпты для ИИ")
    prompt_mode = st.radio(
        "Выберите тип промпта",
        ["Краткий (все площадки)", "Детальный (все площадки)", "Детальный WB", "Детальный Ozon"],
        horizontal=True,
        key="prompt_global_mode",
    )

    if prompt_mode == "Краткий (все площадки)":
        if st.button("🤖 Сгенерировать краткий промпт", key="gen_brief_global"):
            prompt = build_prompt_brief(
                df_wb if has_wb else None,
                df_ozon if has_ozon else None,
                wb_merged=df_wb_merged if has_funnel else None,
                coverage=funnel_coverage if has_funnel else None,
                ads_cov=ads_coverage if has_ads else None,
                funnel_src=funnel_source_label, ads_src=ads_source_label,
            )
            st.text_area("Промпт (краткий)", value=prompt, height=500, key="prompt_brief_global")
            st.caption("Выделите весь текст (Ctrl/Cmd+A) и скопируйте (Ctrl/Cmd+C).")

    elif prompt_mode == "Детальный (все площадки)":
        if not has_wb and not has_ozon:
            st.warning("Файлы WB и Ozon не загружены.")
        elif st.button("🤖 Сгенерировать детальный промпт (все площадки)", key="gen_det_all"):
            prompt = build_prompt_detailed_all_platforms(
                df_wb if has_wb else None,
                df_ozon if has_ozon else None,
                wb_merged=df_wb_merged if has_funnel else None,
            )
            st.text_area("Промпт (детальный, все площадки)", value=prompt, height=500, key="prompt_det_all")
            st.caption("Выделите весь текст (Ctrl/Cmd+A) и скопируйте (Ctrl/Cmd+C).")

    elif prompt_mode == "Детальный WB":
        if not has_wb:
            st.warning("Файл WB не загружен.")
        elif st.button("🤖 Сгенерировать детальный промпт WB", key="gen_det_wb"):
            wb_src = df_wb_merged if df_wb_merged is not None else df_wb
            prompt = build_prompt_detailed(
                wb_src, "WB", is_wb=True,
                wb_merged=df_wb_merged if has_funnel else None,
                coverage=funnel_coverage if has_funnel else None,
                ads_cov=ads_coverage if has_ads else None,
                funnel_src=funnel_source_label, ads_src=ads_source_label,
            )
            st.text_area("Промпт (детальный WB)", value=prompt, height=500, key="prompt_det_wb")
            st.caption("Выделите весь текст (Ctrl/Cmd+A) и скопируйте (Ctrl/Cmd+C).")

    elif prompt_mode == "Детальный Ozon":
        if not has_ozon:
            st.warning("Файл Ozon не загружен.")
        elif st.button("🤖 Сгенерировать детальный промпт Ozon", key="gen_det_ozon"):
            prompt = build_prompt_detailed(df_ozon, "Ozon", is_wb=False)
            st.text_area("Промпт (детальный Ozon)", value=prompt, height=500, key="prompt_det_ozon")
            st.caption("Выделите весь текст (Ctrl/Cmd+A) и скопируйте (Ctrl/Cmd+C).")

tab_idx += 1

# --- WB API → Инспектор ---
if _wb_token_ok:
    with tabs[tab_idx]:
        st.header("WB API → Инспектор")

        # Выбор токена для инспектора
        _token_labels = [lbl for lbl, _ in _all_tokens]
        _insp_token_label = st.selectbox("Токен", _token_labels, key="insp_token_sel")
        _insp_token_val = dict(_all_tokens).get(_insp_token_label, _all_tokens[0][1])
        st.caption(f"Token: {mask_token(_insp_token_val)}")

        insp_type = st.radio("Endpoint", ["Funnel", "Ads — Campaign List", "Ads — Fullstats"], horizontal=True, key="insp_type")
        insp_col1, insp_col2 = st.columns(2)
        with insp_col1:
            insp_start = st.date_input("Начало", value=date.today() - timedelta(days=3), key="insp_s")
        with insp_col2:
            insp_end = st.date_input("Конец", value=date.today() - timedelta(days=1), key="insp_e")

        if st.button("📡 Test call", key="insp_call"):
            try:
                insp_client = WBClient(token=_insp_token_val)
                if insp_type == "Funnel":
                    raw, lat = test_funnel_call(insp_client, str(insp_start), str(insp_end), limit=5)
                    ep_name = "funnel"
                elif insp_type == "Ads — Campaign List":
                    raw, lat = test_ads_campaign_list(insp_client)
                    ep_name = "ads_campaigns"
                else:
                    # Need campaign IDs first
                    cids = fetch_campaign_ids(insp_client)
                    if not cids:
                        st.warning("Нет кампаний со статистикой.")
                        st.stop()
                    raw, lat = test_ads_fullstats_call(insp_client, cids[:5], str(insp_start), str(insp_end))
                    ep_name = "ads_fullstats"

                resp_size = len(json.dumps(raw, ensure_ascii=False))
                row_count = "?"
                if isinstance(raw, dict) and "data" in raw:
                    prods = raw.get("data", {}).get("products", [])
                    row_count = len(prods)
                elif isinstance(raw, list):
                    row_count = len(raw)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Status", "200 OK")
                mc2.metric("Latency", f"{lat:.2f}s")
                mc3.metric("Rows", row_count)
                mc4.metric("Size", f"{resp_size:,} chars")

                # Schema summary
                schema = summarize_schema(raw)
                st.subheader("Schema Summary")
                display_copyable_table(pd.DataFrame(schema), format_display=False)

                # JSON preview
                with st.expander("JSON Preview (raw)", expanded=False):
                    st.json(raw)

                # Auto-save sample
                saved = save_sample(ep_name, str(insp_start), str(insp_end), raw, schema)
                st.success(f"Sample saved: {saved}")

                if st.button("💾 Save sample again", key="insp_save"):
                    saved2 = save_sample(ep_name, str(insp_start), str(insp_end), raw, schema)
                    st.success(f"Re-saved: {saved2}")

            except WBAPIError as e:
                st.error(f"API Error: {e}")
            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")
