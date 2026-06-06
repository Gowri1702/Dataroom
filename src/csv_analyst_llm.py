
from __future__ import annotations
import re
import traceback
from typing import Tuple, Optional, Any

import pandas as pd
import plotly.express as px

from src.llm_utils import get_openai_client


_BLOCKED = re.compile(
    r"\b(import|__import__|open|exec\b|eval\b|os\b|sys\b|subprocess|"
    r"write|shutil|socket|builtins|globals|locals|vars|getattr|setattr|"
    r"delattr|compile|memoryview|breakpoint|input|print)\s*[\(\.]",
    re.IGNORECASE,
)
_SAFE_ASSIGN = re.compile(r"^\s*result\s*=", re.MULTILINE)


def _is_safe(code: str) -> Tuple[bool, str]:
    if _BLOCKED.search(code):
        return False, "Generated code contains blocked operations."
    if not _SAFE_ASSIGN.search(code):
        return False, "Generated code must assign to 'result'."
    if "__" in code and "result" not in code.split("__")[0]:
        return False, "Dunder usage detected outside result assignment."
    return True, ""


def _build_schema(df: pd.DataFrame) -> str:
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(3).tolist()
        lines.append(f"  - {col} ({dtype}): sample = {sample}")
    return "\n".join(lines)


_SYSTEM = (
    "You are a Pandas code generator. "
    "Given a DataFrame schema and a user question, write a single Python expression "
    "that assigns the answer to a variable named `result`. "
    "Rules:\n"
    "1. The DataFrame is already available as `df`.\n"
    "2. You may use `pd` (pandas) and `df` only.\n"
    "3. Do NOT use import, open, os, sys, subprocess, exec, eval, or any file I/O.\n"
    "4. Output ONLY the Python code — no explanation, no markdown fences.\n"
    "5. The code must be a single `result = ...` statement (multi-line is ok).\n"
    "6. If you need a chart, set `result` to a grouped DataFrame with two columns.\n"
)


def _generate_pandas_code(question: str, schema: str) -> str:
    client = get_openai_client()
    if client is None:
        return ""

    prompt = f"DataFrame schema:\n{schema}\n\nUser question: {question}"
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    code = response.choices[0].message.content.strip()
    code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.MULTILINE)
    code = re.sub(r"\s*```$", "", code, flags=re.MULTILINE)
    return code.strip()


def _execute_code(code: str, df: pd.DataFrame) -> Any:
    namespace = {"df": df, "pd": pd}
    exec(code, {"__builtins__": {}}, namespace)
    return namespace.get("result")


def _maybe_chart(result: Any, question: str):
    if not isinstance(result, pd.DataFrame):
        return None
    if result.shape[1] < 2:
        return None
    cols = result.columns.tolist()
    x_col, y_col = cols[0], cols[1]
    if not pd.api.types.is_numeric_dtype(result[y_col]):
        return None
    fig = px.bar(result, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
    return fig


_FORMAT_SYSTEM = (
    "You are a data analyst writing a one-sentence answer for a business user. "
    "Rules:\n"
    "1. Write exactly one clear sentence — no bullet points, no extra commentary.\n"
    "2. Format dollar amounts as $X.XM (millions) or $X.XK (thousands) where appropriate.\n"
    "3. Format percentages with one decimal place, e.g. 35.9%.\n"
    "4. Round large plain numbers to two decimal places.\n"
    "5. If the raw result is a table or Series, summarise the key insight, not every row.\n"
    "6. Do not say 'the result is' or 'based on the data' — just state the fact.\n"
)


def _format_result(question: str, raw_result: Any) -> str:
    client = get_openai_client()
    if client is None:
        return str(raw_result)

    raw_str = str(raw_result)
    if len(raw_str) > 1500:
        raw_str = raw_str[:1500] + "\n... (truncated)"

    prompt = f"User question: {question}\n\nRaw result from pandas:\n{raw_str}"
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": _FORMAT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return raw_str


def answer_csv_question_llm(
    question: str, df: pd.DataFrame
) -> Tuple[str, Optional[Any]]:
    client = get_openai_client()
    if client is None:
        return (
            "OpenAI API key not set. LLM-powered CSV analysis requires an API key.",
            None,
        )

    schema = _build_schema(df)
    code = _generate_pandas_code(question, schema)

    if not code:
        return "Could not generate Pandas code for this question.", None

    safe, reason = _is_safe(code)
    if not safe:
        return f"Generated code was rejected for safety: {reason}", None

    try:
        result = _execute_code(code, df)
    except Exception:
        tb = traceback.format_exc().split("\n")[-2]
        return f"Execution error: {tb}\n\nGenerated code:\n{code}", None

    if result is None:
        return "The query returned no result.", None

    if isinstance(result, pd.DataFrame):
        fig = _maybe_chart(result, question)
        raw_str = (
            result.to_string(index=False)
            if len(result) <= 20
            else result.head(20).to_string(index=False) + "\n... (truncated)"
        )
        answer = _format_result(question, raw_str)
        return answer, fig

    if isinstance(result, pd.Series):
        answer = _format_result(question, result.to_string())
        return answer, None

    answer = _format_result(question, result)
    return answer, None
