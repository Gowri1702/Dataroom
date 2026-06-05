"""Tests for src/csv_analyst.py (keyword-based analyst)"""
import pandas as pd
import pytest
from src.csv_analyst import answer_csv_question, find_matching_column, summarize_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "region":   ["North", "South", "East", "West"],
        "revenue":  [10000.0, 20000.0, 15000.0, 5000.0],
        "arr":      [120000.0, 240000.0, 180000.0, 60000.0],
        "mrr":      [10000.0, 20000.0, 15000.0, 5000.0],
        "customers": [10, 20, 15, 5],
    })


@pytest.fixture
def missing_df():
    return pd.DataFrame({
        "revenue": [1000.0, None, 3000.0],
        "region":  ["A", None, "C"],
    })


# ── summarize_dataset ─────────────────────────────────────────────────────────

class TestSummarizeDataset:
    def test_returns_dict(self, sample_df):
        summary = summarize_dataset(sample_df)
        assert isinstance(summary, dict)

    def test_row_col_count(self, sample_df):
        summary = summarize_dataset(sample_df)
        assert summary["rows"] == 4
        assert summary["columns"] == 5

    def test_numeric_columns_identified(self, sample_df):
        summary = summarize_dataset(sample_df)
        assert "revenue" in summary["numeric_columns"]
        assert "arr" in summary["numeric_columns"]

    def test_categorical_columns_identified(self, sample_df):
        summary = summarize_dataset(sample_df)
        assert "region" in summary["categorical_columns"]

    def test_missing_values_counted(self, missing_df):
        summary = summarize_dataset(missing_df)
        assert summary["missing_values"]["revenue"] == 1
        assert summary["missing_values"]["region"] == 1


# ── find_matching_column ──────────────────────────────────────────────────────

class TestFindMatchingColumn:
    def test_exact_match(self):
        col = find_matching_column("total revenue by region", ["revenue", "arr"])
        assert col == "revenue"

    def test_underscore_normalised(self):
        col = find_matching_column("annual arr value", ["annual_arr", "mrr"])
        assert col == "annual_arr"

    def test_no_match_returns_none(self):
        col = find_matching_column("completely unrelated", ["revenue", "arr"])
        assert col is None

    def test_first_matching_col_returned(self):
        col = find_matching_column("revenue arr mrr", ["revenue", "arr", "mrr"])
        assert col == "revenue"


# ── answer_csv_question ───────────────────────────────────────────────────────

class TestAnswerCsvQuestion:
    def test_summary_question(self, sample_df):
        answer, fig = answer_csv_question("Give me a summary", sample_df)
        assert "rows" in answer.lower() or "columns" in answer.lower()
        assert fig is None

    def test_missing_values_question(self, missing_df):
        answer, fig = answer_csv_question("Which columns have missing values?", missing_df)
        assert "revenue" in answer.lower() or "region" in answer.lower()
        assert fig is None

    def test_no_missing_values_message(self, sample_df):
        answer, fig = answer_csv_question("Are there missing values?", sample_df)
        assert "no missing" in answer.lower()

    def test_columns_question(self, sample_df):
        answer, fig = answer_csv_question("What are the columns?", sample_df)
        assert "revenue" in answer.lower()
        assert fig is None

    def test_numeric_columns_question(self, sample_df):
        answer, fig = answer_csv_question("What are the numeric columns?", sample_df)
        assert "revenue" in answer.lower()

    def test_total_question(self, sample_df):
        answer, fig = answer_csv_question("What is the total revenue?", sample_df)
        assert "50000" in answer.replace(",", "")
        assert fig is None

    def test_average_question(self, sample_df):
        answer, fig = answer_csv_question("What is the average revenue?", sample_df)
        assert "12500" in answer.replace(",", "")

    def test_max_question(self, sample_df):
        answer, fig = answer_csv_question("What is the highest revenue?", sample_df)
        assert "20000" in answer.replace(",", "")

    def test_min_question(self, sample_df):
        answer, fig = answer_csv_question("What is the lowest revenue?", sample_df)
        assert "5000" in answer.replace(",", "")

    def test_grouped_by_returns_chart(self, sample_df):
        answer, fig = answer_csv_question("Show revenue by region", sample_df)
        assert fig is not None

    def test_fallback_returns_string(self, sample_df):
        answer, fig = answer_csv_question("xyz completely unrecognised question", sample_df)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_returns_tuple(self, sample_df):
        result = answer_csv_question("summary", sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 2
