"""Тесты нормализации заголовков и схемы."""
import pytest
from src.schema import normalize_header, col_p, all_metric_cols, UploadValidation


class TestNormalizeHeader:
    def test_trim_spaces(self):
        assert normalize_header("  Артикул  ") == "Артикул"

    def test_multiple_spaces(self):
        assert normalize_header("Заказы,  шт   п1") == "Заказы, шт п1"

    def test_tabs(self):
        assert normalize_header("Маржа,\t₽\tп1") == "Маржа, ₽ п1"

    def test_mixed_whitespace(self):
        assert normalize_header(" \t Остатки, \t шт  п2 \t ") == "Остатки, шт п2"

    def test_already_clean(self):
        assert normalize_header("Модель") == "Модель"

    def test_empty(self):
        assert normalize_header("") == ""

    def test_numeric_input(self):
        assert normalize_header(123) == "123"


class TestColP:
    def test_p1(self):
        assert col_p("Маржа, ₽", "п1") == "Маржа, ₽ п1"

    def test_p2(self):
        assert col_p("Заказы, шт", "п2") == "Заказы, шт п2"


class TestAllMetricCols:
    def test_count(self):
        from src.schema import METRIC_BASES
        cols = all_metric_cols()
        assert len(cols) == len(METRIC_BASES) * 2

    def test_pairs(self):
        cols = all_metric_cols()
        assert "Маржа, ₽ п1" in cols
        assert "Маржа, ₽ п2" in cols


class TestUploadValidation:
    def test_not_critical_when_empty(self):
        v = UploadValidation()
        assert not v.is_critical
        assert not v.has_warnings

    def test_critical_when_missing_id(self):
        v = UploadValidation(missing_id_cols=["Артикул"])
        assert v.is_critical

    def test_has_warnings_from_info(self):
        v = UploadValidation(info_warnings=["test"])
        assert v.has_warnings
        assert v.has_info_warnings
        assert not v.has_critical_warnings

    def test_has_warnings_from_critical(self):
        v = UploadValidation(critical_warnings=["test"])
        assert v.has_warnings
        assert v.has_critical_warnings
        assert not v.has_info_warnings

    def test_has_warnings_from_type_errors(self):
        v = UploadValidation(type_conversion_errors=["err"])
        assert v.has_warnings
        assert v.has_critical_warnings
