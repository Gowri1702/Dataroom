import pandas as pd
import pytest
from src.claim_checker import (
    extract_claims_from_text,
    verify_claim_against_csv,
    _extract_change_claim,
    _find_column,
    _extract_time_window,
)


class TestExtractClaims:
    def test_basic_increase_claim(self):
        text = "Revenue increased by 20% this quarter."
        claims = extract_claims_from_text(text)
        assert len(claims) == 1
        assert "20%" in claims[0]

    def test_decrease_claim(self):
        text = "Costs declined by 10% year on year."
        claims = extract_claims_from_text(text)
        assert len(claims) == 1

    def test_no_percentage_excluded(self):
        text = "Revenue increased significantly this year."
        claims = extract_claims_from_text(text)
        assert len(claims) == 0

    def test_no_direction_excluded(self):
        text = "Revenue was 20% of total sales."
        claims = extract_claims_from_text(text)
        assert len(claims) == 0

    def test_deduplication(self):
        text = "Revenue increased by 20%. Revenue increased by 20%."
        claims = extract_claims_from_text(text)
        assert len(claims) == 1

    def test_cap_at_eight(self):
        sentences = [f"Metric{i} grew by {i+5}%." for i in range(20)]
        claims = extract_claims_from_text(" ".join(sentences))
        assert len(claims) <= 8

    def test_multiple_claims(self):
        text = (
            "Revenue increased by 20%. "
            "Costs declined by 5%. "
            "Profit grew by 35%."
        )
        claims = extract_claims_from_text(text)
        assert len(claims) == 3

    def test_too_short_sentence_excluded(self):
        text = "Up 5%."
        claims = extract_claims_from_text(text)
        assert len(claims) == 0


class TestExtractChangeClaim:
    def test_increase(self):
        pct, direction, tol, mode = _extract_change_claim("Revenue increased by 20%")
        assert pct == 20.0
        assert direction == "increase"
        assert mode is None

    def test_decrease(self):
        pct, direction, tol, mode = _extract_change_claim("Costs declined by 7.5%")
        assert pct == 7.5
        assert direction == "decrease"

    def test_no_percentage_no_multiplier(self):
        pct, direction, tol, mode = _extract_change_claim("Revenue grew significantly")
        assert pct is None

    def test_no_direction(self):
        pct, direction, tol, mode = _extract_change_claim("Revenue was 15% of total")
        assert pct == 15.0
        assert direction is None

    def test_decimal_percentage(self):
        pct, direction, tol, mode = _extract_change_claim("ARR grew by 3.14%")
        assert abs(pct - 3.14) < 0.001

    def test_doubled(self):
        pct, direction, tol, mode = _extract_change_claim("Revenue doubled last year")
        assert pct == 100.0
        assert direction == "increase"
        assert tol == 20.0
        assert mode is None

    def test_tripled(self):
        pct, direction, tol, mode = _extract_change_claim("Users tripled in Q3")
        assert pct == 200.0
        assert tol == 25.0

    def test_halved(self):
        pct, direction, tol, mode = _extract_change_claim("Churn halved this year")
        assert pct == 50.0
        assert direction == "decrease"

    def test_more_than_doubled(self):
        pct, direction, tol, mode = _extract_change_claim("Revenue more than doubled")
        assert pct == 100.0
        assert mode == "more_than_doubled"

    def test_nearly_doubled(self):
        pct, direction, tol, mode = _extract_change_claim("Sales nearly doubled")
        assert pct == 100.0
        assert tol == 30.0


class TestFindColumn:
    def test_exact_match(self):
        col, score, method = _find_column("revenue increased by 20%", ["revenue", "profit"])
        assert col == "revenue"
        assert method == "exact"
        assert score == 1.0

    def test_alias_match(self):
        col, score, method = _find_column("sales grew by 10%", ["revenue", "profit"])
        assert col == "revenue"
        assert method in ("business-term", "alias")

    def test_fuzzy_match(self):
        col, score, method = _find_column("revenu increased", ["revenue", "profit"])
        assert col == "revenue"
        assert score > 0.0

    def test_no_match_returns_none(self):
        col, score, method = _find_column("completely unrelated text xyz", ["revenue"])
        if col is None:
            assert score == 0.0

    def test_mrr_alias(self):
        col, score, method = _find_column("monthly recurring revenue grew 5%", ["mrr", "arr"])
        assert col == "mrr"

    def test_nps_maps_to_nps_score(self):
        col, score, method = _find_column("nps improved by 10%", ["nps_score", "revenue"])
        assert col == "nps_score"
        assert method == "business-term"

    def test_net_promoter_score_maps_to_nps(self):
        col, score, method = _find_column(
            "net promoter score increased by 5%", ["nps_score", "revenue"]
        )
        assert col == "nps_score"
        assert method == "business-term"

    def test_headcount_maps_to_headcount(self):
        col, score, method = _find_column("headcount grew 15%", ["headcount", "revenue"])
        assert col == "headcount"
        assert score == 1.0

    def test_employees_maps_to_headcount(self):
        col, score, method = _find_column("employees grew by 10%", ["headcount", "revenue"])
        assert col == "headcount"

    def test_churn_maps_to_churn_rate_pct(self):
        col, score, method = _find_column("churn declined by 8%", ["churn_rate_pct", "revenue"])
        assert col == "churn_rate_pct"
        assert method == "business-term"

    def test_total_revenue_prefers_revenue_usd_thousands(self):
        col, score, method = _find_column(
            "total revenue grew by 20%",
            ["revenue_usd_thousands", "revenue", "arr"],
        )
        assert col == "revenue_usd_thousands"
        assert method == "business-term"

    def test_longer_phrase_beats_shorter(self):
        col, _, method = _find_column(
            "total revenue increased by 5%",
            ["revenue_usd_thousands", "revenue"],
        )
        assert col == "revenue_usd_thousands"


class TestExtractTimeWindow:
    def test_quarter(self):
        tw = _extract_time_window("Revenue grew 20% in Q3")
        assert tw == {"type": "quarter", "value": 3}

    def test_half(self):
        tw = _extract_time_window("Costs declined 10% in H1")
        assert tw == {"type": "half", "value": 1}

    def test_year(self):
        tw = _extract_time_window("ARR increased 30% in 2023")
        assert tw == {"type": "year", "value": 2023}

    def test_no_window(self):
        tw = _extract_time_window("Revenue grew by 20%")
        assert tw is None

    def test_from_with_two_years_returns_none(self):
        tw = _extract_time_window("Revenue grew 20% from FY2024 to 2025")
        assert tw is None

    def test_from_with_year_and_quarter_returns_none(self):
        tw = _extract_time_window("ARR increased 15% from Q1 2024 to Q4 2024")
        assert tw is None

    def test_single_year_no_from_still_extracted(self):
        tw = _extract_time_window("Revenue grew 20% in 2025")
        assert tw == {"type": "year", "value": 2025}

    def test_fy_prefix_parsed(self):
        tw = _extract_time_window("ARR grew 30% in FY2023")
        assert tw == {"type": "year", "value": 2023}


class TestVerifyClaim:
    def _df(self, **cols):
        return pd.DataFrame(cols)

    def test_supported_increase(self):
        df = self._df(revenue=[1000, 1050])
        result = verify_claim_against_csv("Revenue increased by 5%", df)
        assert result["verdict"] == "Supported"
        assert result["csv_metric"] == "revenue"

    def test_contradicted_wrong_direction(self):
        df = self._df(revenue=[1000, 900])
        result = verify_claim_against_csv("Revenue increased by 10%", df)
        assert result["verdict"] == "Contradicted"

    def test_contradicted_wrong_magnitude(self):
        df = self._df(revenue=[1000, 1050])
        result = verify_claim_against_csv("Revenue increased by 50%", df)
        assert result["verdict"] == "Contradicted"

    def test_unverifiable_zero_first(self):
        df = self._df(revenue=[0, 100])
        result = verify_claim_against_csv("Revenue increased by 5%", df)
        assert result["verdict"] == "Unverifiable"

    def test_unverifiable_no_numeric_cols(self):
        df = pd.DataFrame({"name": ["a", "b"]})
        result = verify_claim_against_csv("Revenue increased by 5%", df)
        assert result["verdict"] == "Unverifiable"

    def test_unverifiable_weak_column_match(self):
        df = pd.DataFrame({"irrelevant_col": [10, 20]})
        result = verify_claim_against_csv("Revenue increased by 5%", df)
        assert result["verdict"] == "Unverifiable"

    def test_unverifiable_single_row(self):
        df = self._df(revenue=[1000])
        result = verify_claim_against_csv("Revenue increased by 5%", df)
        assert result["verdict"] == "Unverifiable"

    def test_confidence_present(self):
        df = self._df(revenue=[1000, 1050])
        result = verify_claim_against_csv("Revenue increased by 5%", df)
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_match_method_present(self):
        df = self._df(revenue=[1000, 1050])
        result = verify_claim_against_csv("Revenue increased by 5%", df)
        assert "match_method" in result

    def test_time_window_extracted(self):
        df = self._df(revenue=[1000, 1050])
        result = verify_claim_against_csv("Revenue increased by 5% in Q3", df)
        assert result.get("time_window") == {"type": "quarter", "value": 3}

    def test_decrease_supported(self):
        df = self._df(cost=[200, 170])
        result = verify_claim_against_csv("Cost reduced by 15%", df)
        assert result["verdict"] == "Supported"

    def test_profit_alias(self):
        df = self._df(profit=[100, 120])
        result = verify_claim_against_csv("Earnings grew by 20%", df)
        assert result["verdict"] == "Supported"

    def test_result_keys_complete(self):
        df = self._df(revenue=[1000, 1050])
        result = verify_claim_against_csv("Revenue grew 5%", df)
        for key in ["claim", "verdict", "csv_metric", "actual_percent",
                    "first_val", "last_val", "reason", "confidence",
                    "time_window", "match_method"]:
            assert key in result, f"Missing key: {key}"
