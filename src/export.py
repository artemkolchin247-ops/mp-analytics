"""Экспорт таблиц в Excel (xlsx) с несколькими листами."""
from __future__ import annotations

from io import BytesIO
from typing import Dict

import pandas as pd
from src.display_utils import format_df_for_display


def export_tables_to_xlsx(tables: Dict[str, pd.DataFrame]) -> BytesIO:
    """
    Принимает словарь {имя_листа: DataFrame} и возвращает BytesIO с xlsx.
    Имена листов обрезаются до 31 символа (ограничение Excel).
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, df in tables.items():
            if df.empty:
                continue
            sheet_name = name[:31]
            # Преобразуем в строковую, отформатированную версию (тысячные пробелы, десятичная запятая)
            try:
                df_to_write = format_df_for_display(df, decimals=1, thousand_sep=" ")
            except Exception:
                df_to_write = df
            df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
            # Авто-ширина колонок
            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).str.len().max(),
                    len(str(col)),
                ) + 2
                max_len = min(max_len, 50)
                worksheet.set_column(i, i, max_len)
    buf.seek(0)
    return buf


import io
import csv
import math
from typing import Iterable, Optional


def export_brief_to_csv(
    tables: Dict[str, pd.DataFrame],
    export_spec: Optional[Dict[str, Dict]] = None,
    meta: Optional[str] = None,
    legends: Optional[Dict[str, str]] = None,
) -> bytes:
    """
    Экспортирует словарь таблиц в один CSV (UTF-8) с маркерами начала/конца таблиц.

    Формат файла:
    ###META
    <глобальный контекст для ИИ>
    ###END_META

    ###TABLE: <name>
    ###LEGEND
    <пояснение к таблице>
    ###END_LEGEND
    ###COLUMNS
    col1,col2,...
    <data rows>
    ###END_TABLE

    Параметры:
    - meta: строка с глобальным контекстом (определения периодов, формулы, флаги)
    - legends: dict {table_name: legend_text} — пояснение к каждой таблице
    - export_spec: dict {table_name: {'percent_cols': [...]}} — явное указание процентных колонок

    Числа: проценты — до 1 знака с запятой-разделителем (8,5); остальные — целые.
    """
    if export_spec is None:
        export_spec = {}
    if legends is None:
        legends = {}

    # Токены для автодетекта процентов
    pct_tokens = ["%", "процент", "доля", "ctr", "cvr", "cr", "conv", "conversion", "dr", "drr"]

    sio = io.StringIO()

    # Глобальный мета-блок (определения, формулы, флаги)
    if meta:
        sio.write("###META\n")
        sio.write(meta.strip() + "\n")
        sio.write("###END_META\n\n")

    def _is_percent_col(col: str, spec_cols: Optional[Iterable[str]] = None) -> bool:
        if spec_cols:
            return col in spec_cols
        lc = col.lower()
        return any(tok in lc for tok in pct_tokens)

    for name, df in tables.items():
        if df is None or df.empty:
            continue

        # Work on a copy
        tdf = df.copy()

        # If index is meaningful (not default RangeIndex starting at 0), reset and keep as column
        if not isinstance(tdf.index, pd.RangeIndex) or tdf.index.name is not None:
            tdf = tdf.reset_index()

        # Determine percent cols
        spec = export_spec.get(name, {}) if export_spec else {}
        spec_pct = spec.get('percent_cols') if isinstance(spec, dict) else None

        pct_cols = set()
        for col in tdf.columns:
            if spec_pct and col in spec_pct:
                pct_cols.add(col)
            elif spec_pct is None and _is_percent_col(col):
                pct_cols.add(col)

        # Prepare formatted string table (all values -> strings)
        rows = []
        for _, r in tdf.iterrows():
            out_row = []
            for col in tdf.columns:
                val = r[col]
                if pd.isna(val):
                    cell = ""
                else:
                    # Percent columns: round to 1 decimal
                    if col in pct_cols:
                        try:
                            num = float(val)
                            num = round(num, 1)
                            # Use decimal comma for fractional part
                            if math.isfinite(num):
                                cell = (f"{num:.1f}").replace('.', ',')
                            else:
                                cell = ""
                        except Exception:
                            cell = str(val)
                    else:
                        # Numeric -> round to integer
                        if isinstance(val, (int,)):
                            cell = str(val)
                        else:
                            try:
                                f = float(val)
                                if math.isfinite(f):
                                    cell = str(int(round(f)))
                                else:
                                    cell = ""
                            except Exception:
                                cell = str(val)

                # Escape rows starting with '###' — any cell beginning with '###' should be prefixed
                if isinstance(cell, str) and cell.startswith('###'):
                    cell = "'" + cell
                out_row.append(cell)
            rows.append(out_row)

        # Write block markers and columns line (raw, not CSV-quoted)
        sio.write(f"###TABLE: {name}\n")
        if name in legends and legends[name]:
            sio.write("###LEGEND\n")
            sio.write(legends[name].strip() + "\n")
            sio.write("###END_LEGEND\n")
        sio.write("###COLUMNS\n")
        # Columns line: simple comma-joined names
        col_line = ",".join([str(c) for c in tdf.columns])
        sio.write(col_line + "\n")

        # Use csv.writer to write data rows so fields with commas are quoted
        writer = csv.writer(sio, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for r in rows:
            writer.writerow(r)

        sio.write("###END_TABLE\n")

    data = sio.getvalue().encode('utf-8')
    sio.close()
    return data
