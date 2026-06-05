"""
Keyword-based question router.

Uses word-boundary matching (\b) for single-word keywords so that short tokens
like "arr", "sum", "max", "min", "match" don't fire inside longer words
(e.g. "narrative", "summarize", "matching").

Multi-word phrases (e.g. "by region", "against the csv") still use plain
substring matching because they inherently avoid false positives.
"""

import re


def _any_keyword(text: str, keywords: list[str]) -> bool:
    for kw in keywords:
        if " " in kw:
            if kw in text:
                return True
        else:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                return True
    return False


def route_question(question: str) -> str:
    """
    Route the user question to the correct analysis system.

    Returns:
        "pdf"  -> PDF/document question
        "csv"  -> spreadsheet/data question
        "both" -> needs PDF + CSV comparison
    """
    q = question.lower()

    pdf_keywords = [
        "document", "pdf", "report", "page", "section", "mentioned", "states",
        "says", "according to", "summarize", "summary", "takeaways", "risks",
        "highlights", "narrative", "financial highlights", "main points",
    ]

    csv_keywords = [
        "csv", "data", "dataset", "spreadsheet", "table", "columns", "rows",
        "missing", "null", "total", "sum", "average", "mean", "maximum",
        "minimum", "max", "min", "highest", "lowest", "count", "group",
        "by region", "by category", "revenue by", "sales by", "profit by",
        "arr", "mrr", "seats", "customers",
    ]

    both_keywords = [
        "match", "matches", "verify", "compare", "comparison",
        "contradict", "contradicted", "supported", "claim", "claims",
        "consistent", "consistency", "does the report match", "does the pdf match",
        "document vs data", "pdf vs csv", "against the csv", "against the data",
        "validate", "validation",
    ]

    if _any_keyword(q, both_keywords):
        return "both"

    if _any_keyword(q, csv_keywords):
        return "csv"

    if _any_keyword(q, pdf_keywords):
        return "pdf"

    return "pdf"
