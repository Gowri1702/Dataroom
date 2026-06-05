"""
LLM-based intent classifier for question routing.

Returns:
    {
        "intent":     "pdf" | "csv" | "both",
        "confidence": 0.0 – 1.0,
        "reasoning":  "short explanation"
    }

Falls back to keyword router if the OpenAI API key is absent or the LLM
response cannot be parsed.
"""

from __future__ import annotations
import json
import re
from typing import TypedDict

from src.llm_utils import get_openai_client
from src.router import route_question as _keyword_route


class RouteResult(TypedDict):
    intent: str
    confidence: float
    reasoning: str


_SYSTEM = """You are a question-routing classifier for a document analytics tool.

The tool has three data sources:
  - PDF: a business report or document
  - CSV: a spreadsheet with numeric business data
  - BOTH: requires comparing the PDF narrative against CSV data (e.g. fact-checking)

Given a user question, output a JSON object with exactly these keys:
  {
    "intent": "pdf" | "csv" | "both",
    "confidence": <float 0.0 to 1.0>,
    "reasoning": "<one sentence>"
  }

Rules:
- Use "pdf"  for questions about document content, narrative, risk, strategy, summaries.
- Use "csv"  for questions about data, totals, averages, missing values, numeric columns.
- Use "both" for questions that compare, verify, validate, or cross-check PDF claims with data.
- Confidence should reflect how clear-cut the routing decision is.
- Output ONLY the JSON object. No markdown, no extra text.
"""


def route_question_llm(question: str) -> RouteResult:
    """
    Classify a question using the LLM.  Falls back to the keyword router
    with confidence=0.5 if the API key is missing or parsing fails.
    """
    client = get_openai_client()

    if client is None:
        intent = _keyword_route(question)
        return RouteResult(intent=intent, confidence=0.5, reasoning="Keyword fallback (no API key).")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        parsed = json.loads(raw)

        intent = str(parsed.get("intent", "pdf")).lower()
        if intent not in ("pdf", "csv", "both"):
            intent = _keyword_route(question)

        confidence = float(parsed.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(parsed.get("reasoning", ""))

        return RouteResult(intent=intent, confidence=confidence, reasoning=reasoning)

    except Exception:
        intent = _keyword_route(question)
        return RouteResult(
            intent=intent,
            confidence=0.5,
            reasoning="Keyword fallback (LLM parse error).",
        )
