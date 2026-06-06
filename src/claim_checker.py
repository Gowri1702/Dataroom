
from __future__ import annotations
import re
from typing import Optional, Tuple

import pandas as pd
from rapidfuzz import process as fuzz_process, fuzz


MATCH_CONFIDENCE_THRESHOLD = 0.80


def extract_claims_from_text(text: str) -> list[str]:
    direction_words = [
        "increased", "increase", "grew", "growth", "up", "rose", "higher", "surged",
        "jumped", "improved", "expanded", "gained",
        "decreased", "decrease", "declined", "decline", "down", "fell", "lower",
        "dropped", "reduced", "contracted", "shrunk",
    ]
    multiplier_words = ["doubled", "tripled", "halved", "quadrupled",
                        "nearly doubled", "nearly tripled",
                        "more than doubled", "more than tripled"]

    _DOLLAR_RE   = re.compile(r"\$\s*\d+(?:\.\d+)?\s*(?:m|k|b|million|billion)?", re.IGNORECASE)
    _FROM_TO_RE  = re.compile(r"\bfrom\s+[\$\d][\d,\.]*\s+to\s+[\$\d][\d,\.]*", re.IGNORECASE)
    _DIR_NUM_RE  = re.compile(
        r"\b(?:grew|increased|declined|decreased|fell|rose|dropped|reduced)\s+(?:by\s+)?\d+(?:\.\d+)?(?!\s*%)",
        re.IGNORECASE,
    )
    _PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")

    raw_sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    claims: list[str] = []
    seen: set[str] = set()

    for sent in raw_sentences:
        sent = sent.strip()
        if len(sent) < 15 or len(sent) > 250:
            continue

        s_lower = sent.lower()

        has_pct        = bool(_PCT_RE.search(sent))
        has_multiplier = any(w in s_lower for w in multiplier_words)
        has_dollar     = bool(_DOLLAR_RE.search(sent))
        has_from_to    = bool(_FROM_TO_RE.search(sent))
        has_dir_num    = bool(_DIR_NUM_RE.search(sent))
        has_direction  = any(w in s_lower for w in direction_words)

        qualifies = (
            (has_pct and has_direction)
            or has_multiplier
            or (has_dollar and has_direction)
            or has_from_to
            or has_dir_num
        )
        if not qualifies:
            continue

        norm = re.sub(r"\s+", " ", s_lower)
        if norm in seen:
            continue
        seen.add(norm)
        clean = sent[0].upper() + sent[1:] if sent else sent
        claims.append(clean)

    return claims[:8]


def _detect_direction(claim_lower: str) -> Optional[str]:
    if any(w in claim_lower for w in [
        "increased", "increase", "grew", "growth", "up", "rose", "higher",
        "surged", "jumped", "improved", "expanded", "gained",
    ]):
        return "increase"
    if any(w in claim_lower for w in [
        "decreased", "decrease", "declined", "decline", "down", "fell", "lower",
        "dropped", "reduced", "contracted", "shrunk",
    ]):
        return "decrease"
    return None


def _extract_change_claim(claim: str) -> Tuple[Optional[float], Optional[str], float, Optional[str]]:
    claim_lower = claim.lower()

    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", claim_lower)
    if pct_match:
        return float(pct_match.group(1)), _detect_direction(claim_lower), 5.0, None

    if re.search(r"\bmore\s+than\s+doubled\b", claim_lower):
        return 100.0, "increase", 0.0, "more_than_doubled"
    if re.search(r"\bmore\s+than\s+tripled\b", claim_lower):
        return 200.0, "increase", 0.0, "more_than_tripled"
    if re.search(r"\bnearly\s+doubled\b", claim_lower):
        return 100.0, "increase", 30.0, None
    if re.search(r"\bnearly\s+tripled\b", claim_lower):
        return 200.0, "increase", 30.0, None
    if re.search(r"\bdoubled\b", claim_lower):
        return 100.0, "increase", 20.0, None
    if re.search(r"\btripled\b", claim_lower):
        return 200.0, "increase", 25.0, None
    if re.search(r"\bhalved\b", claim_lower):
        return 50.0, "decrease", 10.0, None

    return None, None, 5.0, None


_BUSINESS_TERM_MAP: dict[str, list[str]] = {
    "net promoter score": ["nps_score", "nps"],
    "nps":                ["nps_score", "nps"],
    "headcount":          ["headcount", "head_count", "employees", "staff"],
    "employees":          ["employees", "headcount", "head_count", "staff"],
    "staff":              ["staff", "headcount", "employees"],
    "workforce":          ["headcount", "employees", "workforce"],
    "paying customers":   ["customers", "paying_customers", "customer"],
    "customer base":      ["customers", "customer_base", "customer"],
    "customers":          ["customers", "customer"],
    "clients":            ["customers", "clients", "client"],
    "accounts":           ["customers", "accounts"],
    "churn rate":         ["churn_rate_pct", "churn_rate", "churn"],
    "churn":              ["churn_rate_pct", "churn_rate", "churn"],
    "attrition":          ["churn_rate_pct", "attrition", "churn"],
    "total revenue":      ["revenue_usd_thousands", "total_revenue", "revenue", "sales"],
    "net revenue":        ["net_revenue", "revenue_usd_thousands", "revenue"],
    "revenue":            ["revenue_usd_thousands", "revenue", "sales"],
    "sales":              ["sales", "revenue_usd_thousands", "revenue"],
    "gross profit":       ["gross_profit", "gross_margin", "profit"],
    "net income":         ["net_income", "net_profit", "profit"],
    "profit margin":      ["profit_margin", "gross_margin", "margin"],
    "profit":             ["profit", "net_income", "earnings"],
    "earnings":           ["earnings", "net_income", "profit"],
    "annual recurring revenue":  ["arr", "annual_recurring_revenue"],
    "monthly recurring revenue": ["mrr", "monthly_recurring_revenue"],
    "arr":                ["arr", "annual_recurring_revenue"],
    "mrr":                ["mrr", "monthly_recurring_revenue"],
    "operating costs":    ["operating_cost", "opex", "cost", "expenses"],
    "cost of goods sold": ["cogs_usd_m", "cogs", "cost_of_goods"],
    "cost of goods":      ["cogs_usd_m", "cogs", "cost_of_goods", "cost"],
    "cogs":               ["cogs_usd_m", "cogs"],
    "expenses":           ["expenses", "opex", "cost"],
    "cost":               ["cost", "costs", "expenses"],
    "research & development":  ["rnd_spend_usd_m", "rnd", "r_and_d", "research"],
    "research and development": ["rnd_spend_usd_m", "rnd", "r_and_d", "research"],
    "r and d":            ["rnd_spend_usd_m", "rnd"],
    "r&d":                ["rnd_spend_usd_m", "rnd"],
    "customer lifetime value": ["ltv_usd", "ltv", "lifetime_value"],
    "lifetime value":     ["ltv_usd", "ltv", "lifetime_value"],
    "ltv":                ["ltv_usd", "ltv", "lifetime_value"],
    "customer acquisition cost": ["cac_usd", "cac", "customer_acquisition_cost"],
    "cac":                ["cac_usd", "cac", "customer_acquisition_cost"],
    "seats":              ["seats", "licenses", "users"],
    "growth rate":        ["growth_rate", "growth"],
    "retention":          ["retention_rate", "retention"],
    "conversion":         ["conversion_rate", "conversion"],
}

_DOMAIN_ALIASES: dict[str, list[str]] = {
    "revenue":   ["revenue", "sales", "income", "turnover"],
    "profit":    ["profit", "net income", "earnings"],
    "cost":      ["cost", "expense", "expenditure"],
    "arr":       ["arr", "annual recurring revenue"],
    "mrr":       ["mrr", "monthly recurring revenue"],
    "seats":     ["seats", "licenses", "users"],
    "customers": ["customers", "clients", "accounts"],
    "churn":     ["churn", "attrition"],
    "nps":       ["nps", "net promoter score", "promoter"],
    "headcount": ["headcount", "employees", "staff", "workforce"],
    "growth":    ["growth", "growth rate"],
    "cogs":      ["cogs", "cost of goods sold", "cost of goods"],
    "rnd":       ["r&d", "research and development", "research & development", "r and d"],
    "ltv":       ["ltv", "lifetime value", "customer lifetime value"],
    "cac":       ["cac", "customer acquisition cost"],
}


def _col_contains(col: str, pattern: str) -> bool:
    norm_col = col.lower().replace("_", " ").replace("-", " ")
    norm_pat = pattern.lower().replace("_", " ").replace("-", " ")
    return norm_pat in norm_col or norm_col == norm_pat


def _find_column(
    claim_lower: str, numeric_cols: list[str]
) -> Tuple[Optional[str], float, str]:
    norm_claim = claim_lower.replace("_", " ").replace("-", " ")

    for phrase in sorted(
        (p for p in _BUSINESS_TERM_MAP if " " in p or "&" in p), key=len, reverse=True
    ):
        norm_phrase = phrase.replace("_", " ").replace("-", " ")
        if norm_phrase in norm_claim:
            for candidate in _BUSINESS_TERM_MAP[phrase]:
                for col in numeric_cols:
                    if _col_contains(col, candidate):
                        return col, 1.0, "business-term"

    for col in numeric_cols:
        norm_col = col.lower().replace("_", " ").replace("-", " ")
        if norm_col in norm_claim:
            return col, 1.0, "exact"

    for phrase in sorted(
        (p for p in _BUSINESS_TERM_MAP if " " not in p and "&" not in p), key=len, reverse=True
    ):
        if re.search(r"\b" + re.escape(phrase) + r"\b", norm_claim):
            for candidate in _BUSINESS_TERM_MAP[phrase]:
                for col in numeric_cols:
                    if _col_contains(col, candidate):
                        return col, 1.0, "business-term"

    for col in numeric_cols:
        col_lower = col.lower()
        for canonical, aliases in _DOMAIN_ALIASES.items():
            if col_lower == canonical or col_lower.startswith(canonical):
                if any(alias in norm_claim for alias in aliases):
                    return col, 0.9, "alias"

    if numeric_cols:
        col_names = [c.lower().replace("_", " ").replace("-", " ") for c in numeric_cols]
        best = fuzz_process.extractOne(
            norm_claim, col_names,
            scorer=fuzz.WRatio,
            score_cutoff=60,
        )
        if best is not None:
            matched_name, score, idx = best
            return numeric_cols[idx], score / 100.0, f"fuzzy({score:.0f})"

    return None, 0.0, "none"


_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_HALF_RE    = re.compile(r"\bH([12])\b",  re.IGNORECASE)
_YEAR_RE    = re.compile(r"\b(FY)?(20\d{2}|19\d{2})\b")


def _extract_time_window(claim: str) -> Optional[dict]:
    q_matches = _QUARTER_RE.findall(claim)
    h_matches = _HALF_RE.findall(claim)
    y_matches = _YEAR_RE.findall(claim)
    year_values = [int(y) for _, y in y_matches]

    has_from = bool(re.search(r"\bfrom\b", claim, re.IGNORECASE))
    n_time_refs = len(q_matches) + len(h_matches) + len(year_values)

    if has_from and n_time_refs >= 2:
        return None

    if q_matches:
        return {"type": "quarter", "value": int(q_matches[0])}
    if h_matches:
        return {"type": "half", "value": int(h_matches[0])}
    if year_values:
        return {"type": "year", "value": year_values[0]}
    return None


def _filter_by_time_window(df: pd.DataFrame, window: dict) -> pd.DataFrame:
    date_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("date", "period", "quarter", "month", "year", "time"))
    ]
    if not date_cols:
        return df

    date_col = date_cols[0]
    col_vals = df[date_col].astype(str).str.lower()

    if window["type"] == "quarter":
        mask = col_vals.str.contains(f"q{window['value']}", na=False)
    elif window["type"] == "half":
        mask = col_vals.str.contains(f"h{window['value']}", na=False)
    elif window["type"] == "year":
        mask = col_vals.str.contains(str(window["value"]), na=False)
    else:
        return df

    filtered = df[mask]
    return filtered if len(filtered) >= 2 else df


def _confidence(
    match_score: float,
    direction_match: bool,
    magnitude_match: bool,
    time_filtered: bool,
) -> float:
    base = match_score * 0.5
    base += 0.25 if direction_match else 0.0
    base += 0.15 if magnitude_match else 0.0
    base += 0.10 if time_filtered else 0.0
    return round(min(base, 1.0), 2)


def verify_claim_against_csv(claim: str, df: pd.DataFrame) -> dict:
    claim_lower = claim.lower()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    _base = {
        "claim":        claim,
        "first_val":    None,
        "last_val":     None,
        "time_window":  None,
        "match_method": "none",
        "confidence":   0.0,
    }

    if not numeric_cols:
        return {
            **_base,
            "verdict":        "Unverifiable",
            "csv_metric":     None,
            "actual_percent": None,
            "reason":         "No numeric columns found in the CSV to verify against.",
        }

    matched_col, match_score, match_method = _find_column(claim_lower, numeric_cols)

    if matched_col is None or match_score < MATCH_CONFIDENCE_THRESHOLD:
        best_candidate = matched_col or numeric_cols[0]
        return {
            **_base,
            "verdict":        "Unverifiable",
            "csv_metric":     None,
            "actual_percent": None,
            "match_method":   match_method,
            "confidence":     round(1.0 - match_score, 2),
            "reason": (
                f"No CSV column closely matches this claim. "
                f"Best candidate was '{best_candidate}' "
                f"(confidence {match_score:.0%}), "
                f"below the {MATCH_CONFIDENCE_THRESHOLD:.0%} threshold required to verify."
            ),
        }

    claimed_pct, direction, tolerance, special_mode = _extract_change_claim(claim)
    time_window = _extract_time_window(claim)

    if claimed_pct is None:
        return {
            **_base,
            "verdict":        "Unverifiable",
            "csv_metric":     None,
            "actual_percent": None,
            "time_window":    time_window,
            "match_method":   match_method,
            "confidence":     0.0,
            "reason": (
                "No percentage value or multiplier word "
                "(doubled / tripled / halved) detected in the claim."
            ),
        }

    time_filtered = False
    reason_prefix = ""

    if time_window:
        filtered = _filter_by_time_window(df, time_window)
        if len(filtered) >= 2 and len(filtered) < len(df):
            df = filtered
            time_filtered = True
            reason_prefix += (
                f"Filtered to {time_window['type'].upper()} {time_window['value']} "
                f"({len(filtered)} rows). "
            )

    series = df[matched_col].dropna()

    if len(series) < 2:
        return {
            **_base,
            "verdict":        "Unverifiable",
            "csv_metric":     matched_col,
            "actual_percent": None,
            "time_window":    time_window,
            "match_method":   match_method,
            "confidence":     _confidence(match_score, False, False, time_filtered),
            "reason":         f"{reason_prefix}Column '{matched_col}' has fewer than 2 non-null values.",
        }

    first_val = float(series.iloc[0])
    last_val  = float(series.iloc[-1])

    if first_val == 0:
        return {
            **_base,
            "verdict":        "Unverifiable",
            "csv_metric":     matched_col,
            "actual_percent": None,
            "first_val":      first_val,
            "last_val":       last_val,
            "time_window":    time_window,
            "match_method":   match_method,
            "confidence":     _confidence(match_score, False, False, time_filtered),
            "reason":         f"{reason_prefix}First value in '{matched_col}' is 0; cannot compute % change.",
        }

    actual_pct       = ((last_val - first_val) / abs(first_val)) * 100
    actual_direction = "increase" if actual_pct >= 0 else "decrease"

    direction_match = (direction is None) or (direction == actual_direction)

    if special_mode == "more_than_doubled":
        magnitude_match = actual_pct > 100.0
    elif special_mode == "more_than_tripled":
        magnitude_match = actual_pct > 200.0
    else:
        magnitude_match = abs(abs(actual_pct) - claimed_pct) <= tolerance

    confidence = _confidence(match_score, direction_match, magnitude_match, time_filtered)

    if special_mode == "more_than_doubled":
        claimed_label = "more than doubled (>100% increase)"
    elif special_mode == "more_than_tripled":
        claimed_label = "more than tripled (>200% increase)"
    elif claimed_pct == 100.0 and direction == "increase":
        claimed_label = f"doubled (~100% increase, ±{tolerance:.0f}% tolerance)"
    elif claimed_pct == 200.0 and direction == "increase":
        claimed_label = f"tripled (~200% increase, ±{tolerance:.0f}% tolerance)"
    elif claimed_pct == 50.0 and direction == "decrease":
        claimed_label = f"halved (~50% decrease, ±{tolerance:.0f}% tolerance)"
    else:
        claimed_label = f"{claimed_pct:.1f}% {direction or 'change'}"

    if direction_match and magnitude_match:
        verdict = "Supported"
        reason = (
            f"{reason_prefix}Claim: {claimed_label} in '{matched_col}'. "
            f"Actual: {actual_pct:+.2f}% "
            f"(from {first_val:,.2f} to {last_val:,.2f}). "
            f"Column matched via {match_method}."
        )
    elif direction_match and not magnitude_match:
        verdict = "Contradicted"
        reason = (
            f"{reason_prefix}Direction matches ({actual_direction}), but magnitude differs. "
            f"Claimed {claimed_label}, actual {actual_pct:+.2f}% in '{matched_col}' "
            f"(from {first_val:,.2f} to {last_val:,.2f}). Column matched via {match_method}."
        )
    elif not direction_match:
        verdict = "Contradicted"
        reason = (
            f"{reason_prefix}Claim says '{direction}', but '{matched_col}' actually "
            f"{'increased' if actual_pct >= 0 else 'decreased'} by {abs(actual_pct):.2f}% "
            f"(from {first_val:,.2f} to {last_val:,.2f}). Column matched via {match_method}."
        )
    else:
        verdict = "Unverifiable"
        reason = f"{reason_prefix}Could not determine a clear verdict for '{matched_col}'."

    return {
        "claim":          claim,
        "verdict":        verdict,
        "csv_metric":     matched_col,
        "actual_percent": actual_pct,
        "first_val":      first_val,
        "last_val":       last_val,
        "time_window":    time_window,
        "match_method":   match_method,
        "confidence":     confidence,
        "reason":         reason,
    }
