import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

from src.pdf_utils import extract_pdf_text
from src.csv_utils import load_csv, profile_csv
from src.rag_utils import (
    load_embedding_model,
    chunk_pdf_pages,
    create_faiss_index,
    retrieve_relevant_chunks,
)
from src.llm_utils import answer_pdf_question
from src.csv_analyst import answer_csv_question
from src.csv_analyst_llm import answer_csv_question_llm
from src.router import route_question
from src.router_llm import route_question_llm
from src.claim_checker import verify_claim_against_csv, extract_claims_from_text
from src.hybrid_retrieval import build_bm25_index, retrieve_hybrid
# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Dataroom AI",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# GLOBAL CSS
# ==========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,800;0,900;1,700&family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

/* ---------- Base ---------- */
html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background: #f4f6fb; color: #111827; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1440px; }

/* Force light color-scheme so Streamlit dataframes never go dark */
:root { color-scheme: light !important; }

/* ---------- Typography ---------- */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #111827 !important;
    letter-spacing: -0.01em !important;
    font-weight: 800 !important;
}
h1 { font-size: 1.65rem !important; line-height: 1.2 !important; }
h2 { font-size: 1.25rem !important; line-height: 1.25 !important; }
h3 { font-size: 1rem !important; line-height: 1.35 !important; }
p, div, span, label, li { color: #374151; font-size: 14px; line-height: 1.55; }

.section-label {
    color: #6b7280;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin: 0 0 12px 0;
}
[data-testid="stCaptionContainer"] { color: #9ca3af !important; font-size: 12px !important; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid rgba(255,255,255,0.05);
    min-width: 220px !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0; }
section[data-testid="stSidebar"] * { color: #9ca3af !important; }

/* Kill red radio dot */
section[data-testid="stSidebar"] input[type="radio"] { display: none !important; }

section[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] [role="radiogroup"] > label {
    padding: 10px 16px;
    border-radius: 8px;
    cursor: pointer;
    width: 100%;
    margin: 0;
    background: transparent !important;
    transition: background 0.12s ease;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label p {
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #9ca3af !important;
}

/* Active nav — solid purple fill like DocInsight */
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input[type="radio"]:checked) {
    background: #4f46e5 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input[type="radio"]:checked) p {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Nav icons via CSS ::before on nth-child */
section[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(1) p::before { content: "⊞  "; }
section[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(2) p::before { content: "↑  "; }
section[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(3) p::before { content: "⚡  "; }
section[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(4) p::before { content: "◈  "; }
section[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(5) p::before { content: "≡  "; }
section[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(6) p::before { content: "◉  "; }
section[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(7) p::before { content: "▤  "; }

/* ---------- Page header bar ---------- */
.page-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid #e5e7eb;
}
/* Use .welcome-title div — avoids Streamlit's h1 size override */
.welcome-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 4.2rem !important;
    font-weight: 900 !important;
    color: #0f172a !important;
    margin: 0 0 10px 0 !important;
    line-height: 1.0 !important;
    letter-spacing: -0.03em !important;
    display: block !important;
}
.welcome-subtitle { font-size: 15px; color: #6b7280; margin: 0; display: block; }

/* ---------- Dashboard cards — shadow-first, DocInsight style ---------- */
.dash-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(17,24,39,0.06), 0 4px 16px rgba(17,24,39,0.04);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    height: 100%;
}
.dash-card:hover {
    box-shadow: 0 4px 12px rgba(79,70,229,0.10), 0 8px 30px rgba(17,24,39,0.07);
    transform: translateY(-2px);
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(17,24,39,0.06), 0 4px 16px rgba(17,24,39,0.04);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(79,70,229,0.10), 0 8px 30px rgba(17,24,39,0.07);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 12px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
}
.metric-row { display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: 12px; }
.metric-value {
    font-family: 'Bebas Neue', Impact, sans-serif !important;
    font-size: 2.6rem;
    font-weight: 400;        /* Bebas Neue is a single-weight display face */
    color: #0a0f1e;
    letter-spacing: 0.02em;  /* slight open tracking suits condensed caps */
    line-height: 1;
}
.metric-spark { opacity: 0.85; flex-shrink: 0; }
.metric-delta {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 600;
}
.metric-delta-positive { background: #d1fae5; color: #065f46; }
.metric-delta-negative { background: #fee2e2; color: #991b1b; }
.metric-delta-neutral  { background: #f3f4f6; color: #4b5563; }

/* ---------- Status pills ---------- */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 20px;
    font-size: 12.5px; font-weight: 600; margin-bottom: 8px;
}
.status-success { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
.status-warning { background: #fffbeb; color: #b45309; border: 1px solid #fcd34d; }
.status-danger  { background: #fef2f2; color: #dc2626; border: 1px solid #fca5a5; }

/* ---------- Document status card (DocInsight file card) ---------- */
.doc-status-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(17,24,39,0.06), 0 4px 16px rgba(17,24,39,0.04);
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 1.5rem;
}
.doc-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.doc-info { flex: 1; }
.doc-name { font-size: 15px; font-weight: 600; color: #111827; margin-bottom: 2px; }
.doc-meta { font-size: 12px; color: #9ca3af; }
.doc-status-ok { color: #059669; font-size: 12.5px; font-weight: 600; margin-top: 6px; }
.doc-status-warn { color: #b45309; font-size: 12.5px; font-weight: 600; margin-top: 6px; }
.doc-takeaway {
    background: #f8faff;
    border: 1px solid #e0e7ff;
    border-radius: 10px;
    padding: 14px 16px;
    flex: 1.2;
}
.doc-takeaway-label { font-size: 11px; font-weight: 700; color: #6366f1; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.doc-takeaway-text { font-size: 13px; color: #374151; line-height: 1.55; }

/* ---------- Key insights list ---------- */
.insight-item {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid #f3f4f6;
}
.insight-item:last-child { border-bottom: none; }
.insight-dot {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0; margin-top: 1px;
}
.insight-dot-green  { background: #d1fae5; }
.insight-dot-blue   { background: #dbeafe; }
.insight-dot-purple { background: #ede9fe; }
.insight-dot-orange { background: #ffedd5; }
.insight-text { font-size: 13.5px; color: #374151; line-height: 1.5; }
.insight-text b { color: #111827; font-weight: 600; }

/* ---------- Answer / chat bubbles ---------- */
.answer {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 3px solid #4f46e5;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 14px;
    line-height: 1.7;
    color: #374151;
    white-space: pre-line;
    box-shadow: 0 1px 3px rgba(17,24,39,0.04);
}
.answer b { font-weight: 700; color: #4f46e5; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; }
.answer-user {
    background: #f9fafb;
    border-left: 3px solid #d1d5db;
    box-shadow: none;
}
.answer-user b { color: #6b7280; }

/* ---------- Mini summary table ---------- */
.summary-table {
    width: 100%; border-collapse: collapse;
    background: white; border-radius: 12px; overflow: hidden;
}
.summary-table th {
    background: #f9fafb; color: #6b7280;
    font-weight: 600; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.06em; padding: 11px 14px;
    text-align: left; border-bottom: 1px solid #f3f4f6;
}
.summary-table td {
    padding: 11px 14px; font-size: 13px; color: #374151;
    border-bottom: 1px solid #f9fafb;
}
.summary-table tr:last-child td { border-bottom: none; }
.summary-table tr:hover td { background: #f9fafb; }
.summary-table .col-name { font-family: 'Playfair Display', serif; color: #111827; font-weight: 700; }
.summary-table .num { font-variant-numeric: tabular-nums; color: #374151; }

/* ---------- Inputs ---------- */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    box-shadow: 0 1px 2px rgba(17,24,39,0.04) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.10) !important;
    outline: none !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.10) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #9ca3af !important; }

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important; border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important; color: #111827 !important; min-height: 40px !important;
}
.stSelectbox div[data-baseweb="select"] svg { color: #9ca3af !important; }
.stSelectbox div[data-baseweb="select"] input { color: #111827 !important; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
    background: #ffffff !important; border: 1px solid #f3f4f6 !important;
    border-radius: 12px !important; box-shadow: 0 10px 30px rgba(17,24,39,0.10) !important;
}
[role="listbox"] li, [role="option"], [data-baseweb="menu"] li {
    background: #ffffff !important; color: #374151 !important; font-size: 14px !important; padding: 8px 12px !important;
}
[role="option"]:hover, [role="option"][aria-selected="true"], [data-baseweb="menu"] li:hover {
    background: #f5f3ff !important; color: #4f46e5 !important;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
    background: #ffffff !important; border: 2px dashed #c7d2fe;
    border-radius: 14px; padding: 8px;
    box-shadow: 0 1px 3px rgba(17,24,39,0.04);
}
[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important; border: none !important; color: #6b7280 !important; padding: 1.5rem !important;
}
[data-testid="stFileUploader"] section *, [data-testid="stFileUploaderDropzone"] * {
    background: transparent !important; color: #6b7280 !important;
}
[data-testid="stFileUploader"] small { color: #9ca3af !important; }
[data-testid="stFileUploader"] button {
    background: #f97316 !important; color: #111827 !important; border: none !important;
    border-radius: 8px !important; padding: 7px 16px !important; font-weight: 700 !important; font-size: 13px !important;
    box-shadow: 0 2px 10px rgba(249,115,22,0.30) !important;
}
[data-testid="stFileUploader"] button:hover { background: #ea580c !important; color: #111827 !important; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: #ffffff; color: #374151;
    border: 1.5px solid #e5e7eb; border-radius: 10px;
    padding: 10px 14px; font-weight: 500; font-size: 13px;
    text-align: left; width: 100%; justify-content: flex-start;
    transition: all 0.15s ease; box-shadow: 0 1px 2px rgba(17,24,39,0.04);
    margin-bottom: 4px; line-height: 1.4; white-space: normal;
    height: auto; min-height: 40px;
}
.stButton > button:hover {
    background: #f5f3ff; color: #4f46e5;
    border-color: #a5b4fc; box-shadow: 0 2px 8px rgba(79,70,229,0.10);
}
.stButton > button:focus { box-shadow: none !important; outline: none !important; }
.stButton > button[kind="primary"] {
    background: #4f46e5; color: white; border: none;
    text-align: center; justify-content: center;
    box-shadow: 0 4px 14px rgba(79,70,229,0.30);
}
.stButton > button[kind="primary"]:hover { background: #4338ca; }

/* ---------- Plotly chart container ---------- */
div[data-testid="stPlotlyChart"] {
    background: #ffffff; border-radius: 14px; padding: 12px;
    box-shadow: 0 1px 3px rgba(17,24,39,0.06), 0 4px 16px rgba(17,24,39,0.04);
}

/* ---------- Dataframe ---------- */
div[data-testid="stDataFrame"] {
    border-radius: 12px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(17,24,39,0.06);
}

/* ---------- Bordered containers → shadow cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: none !important;
    border-radius: 14px;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 1px 3px rgba(17,24,39,0.06), 0 4px 16px rgba(17,24,39,0.04);
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: #ffffff; border: 1px solid #f3f4f6 !important;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(17,24,39,0.04);
}
[data-testid="stExpander"] summary { color: #374151 !important; font-weight: 500 !important; }

/* ---------- Spinner ---------- */
.stSpinner > div { border-top-color: #4f46e5 !important; }

/* ---------- Download button — red ---------- */
.stDownloadButton > button {
    background: #dc2626 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 10px 20px !important;
    box-shadow: 0 4px 14px rgba(220,38,38,0.30) !important;
    transition: background 0.15s ease, box-shadow 0.15s ease !important;
}
.stDownloadButton > button:hover {
    background: #b91c1c !important;
    box-shadow: 0 6px 20px rgba(220,38,38,0.40) !important;
}

/* ---------- Metric card — subtle left accent on hover ---------- */
.metric-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #4f46e5, #7c3aed);
    border-radius: 14px 0 0 14px;
    opacity: 0;
    transition: opacity 0.2s ease;
}
.metric-card:hover::before { opacity: 1; }

/* ---------- Section label — stronger visual anchor ---------- */
.section-label {
    color: #4f46e5;
    font-weight: 700;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 0 0 12px 0;
}

/* ---------- Hero (kept for inner pages) ---------- */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #3730a3 50%, #4f46e5 100%);
    border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(79,70,229,0.25);
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(167,139,250,0.30) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-eyebrow {
    color: #a5b4fc !important; font-size: 10.5px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.12em !important; margin-bottom: 8px !important;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #ffffff !important; font-size: 2.9rem !important; font-weight: 900 !important;
    letter-spacing: -0.02em !important; margin: 0 0 10px 0 !important; line-height: 1.0 !important;
    text-shadow: 0 2px 24px rgba(0,0,0,0.20);
}
.hero-subtitle { color: rgba(255,255,255,0.75) !important; font-size: 14px !important; line-height: 1.55 !important; margin: 0 !important; }

</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# PLOTLY THEME (refined, professional)
# ==========================================================

# Vibrant chart palette — intentionally distinct from the indigo UI chrome
CHART_PRIMARY = "#0ea5e9"   # sky blue as default single-series colour
CHART_PALETTE = [
    "#0ea5e9",  # sky
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#f43f5e",  # rose
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#fb923c",  # orange
]


def style_fig(fig, height=380, title=None):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#1e293b", size=12),
        title=dict(
            text=title or (fig.layout.title.text or ""),
            font=dict(size=14, color="#0f172a", family="Inter, sans-serif"),
            x=0,
            xanchor="left",
            pad=dict(b=12),
        ),
        margin=dict(l=10, r=10, t=55, b=20),
        height=height,
        colorway=CHART_PALETTE,
        xaxis=dict(
            showgrid=False,
            linecolor="#e5e7eb",
            tickfont=dict(color="#64748b", size=11),
            title_font=dict(color="#475569", size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            linecolor="#e5e7eb",
            tickfont=dict(color="#64748b", size=11),
            title_font=dict(color="#475569", size=12),
        ),
        legend=dict(font=dict(color="#475569", size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="DM Sans, sans-serif"),
    )
    return fig


def style_bars(fig, with_labels=False):
    fig.update_traces(marker_color=CHART_PRIMARY, marker_line_width=0, opacity=0.88)
    if with_labels:
        fig.update_traces(
            texttemplate="%{y:,.0f}", textposition="outside",
            textfont=dict(size=10, color="#6b7280", family="DM Sans, sans-serif"),
            cliponaxis=False,
        )
    return fig


def make_chart(df, chart_type, x_col, y_col, title=""):
    """Build a Plotly figure of the requested type and apply the app theme."""
    if chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=CHART_PALETTE)
        style_bars(fig, with_labels=True)
    elif chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=CHART_PALETTE,
                      markers=True)
        fig.update_traces(line=dict(width=2.5))
    elif chart_type == "Pie":
        fig = px.pie(df, names=x_col, values=y_col, color_discrete_sequence=CHART_PALETTE,
                     hole=0.45)
        fig.update_traces(textinfo="percent+label", textfont_size=12)
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=x_col,
                         color_discrete_sequence=CHART_PALETTE, size_max=18)
    else:
        fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=CHART_PALETTE)
    return style_fig(fig, height=340, title=title)


# ---------- Sparkline SVG helper ----------
def make_sparkline(values, color="#4f46e5", width=80, height=34):
    """Return a tiny inline SVG polyline from a list of numbers."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = 2 + i * (width - 4) / (n - 1)
        y = (height - 4) - (v - mn) / rng * (height - 8) + 2
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="overflow:visible">'
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


# ==========================================================
# UI HELPERS
# ==========================================================

def render_hero(eyebrow, title, subtitle):
    """Hero uses divs, not h1, so Streamlit's h1 styles can't override the white text."""
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label, value, delta, kind="positive", sparkline_svg=""):
    klass = {
        "positive": "metric-delta-positive",
        "negative": "metric-delta-negative",
        "neutral":  "metric-delta-neutral",
    }.get(kind, "metric-delta-neutral")
    icon = {"positive": "↑", "negative": "↓", "neutral": "•"}.get(kind, "•")
    spark = f'<div class="metric-spark">{sparkline_svg}</div>' if sparkline_svg else ""
    # st.html() guarantees HTML renders without markdown interference
    st.html(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-row">
                <div class="metric-value">{value}</div>
                {spark}
            </div>
            <span class="metric-delta {klass}">{icon}&nbsp;{delta}</span>
        </div>
    """)


def render_status(text, kind="success"):
    klass = {"success": "status-success", "warning": "status-warning", "danger": "status-danger"}.get(kind, "status-success")
    icon  = {"success": "✓", "warning": "!", "danger": "✕"}.get(kind, "•")
    st.markdown(
        f'<div class="status-pill {klass}">{icon}&nbsp;&nbsp;{text}</div>',
        unsafe_allow_html=True,
    )


def render_section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def get_numeric_and_categorical_columns(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def get_chart_categorical_cols(df, max_unique=50):
    """Filter out high-cardinality columns (like contract_id) so bar charts show real variation."""
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [c for c in cats if 1 < df[c].nunique() <= max_unique]


def build_summary_table(df, numeric_cols):
    """Clean HTML stats table — replaces dark df.describe()."""
    rows = []
    for c in numeric_cols:
        s = df[c]
        rows.append(
            f"""
            <tr>
                <td class="col-name">{c}</td>
                <td class="num">{s.count():,}</td>
                <td class="num">{s.mean():,.2f}</td>
                <td class="num">{s.median():,.2f}</td>
                <td class="num">{s.min():,.2f}</td>
                <td class="num">{s.max():,.2f}</td>
                <td class="num">{s.std():,.2f}</td>
            </tr>
            """
        )
    return f"""
    <table class="summary-table">
        <thead>
            <tr>
                <th>Column</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Min</th>
                <th>Max</th>
                <th>Std</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


# ==========================================================
# CALLBACKS for clickable suggested questions
# ==========================================================

def relevance_label(distance):
    sim = 1 / (1 + distance)
    if sim >= 0.45:
        return "high relevance"
    elif sim >= 0.30:
        return "medium relevance"
    else:
        return "low relevance"


def retrieve_chunks(question, top_k=4):
    """Use hybrid retrieval when BM25 is available, else fall back to FAISS-only."""
    chunks = st.session_state["pdf_chunks"]
    faiss_index = st.session_state["faiss_index"]
    model = st.session_state["embedding_model"]

    if "bm25_index" in st.session_state:
        return retrieve_hybrid(
            question=question,
            chunks=chunks,
            faiss_index=faiss_index,
            bm25_index=st.session_state["bm25_index"],
            model=model,
            top_k=top_k,
        )
    from src.rag_utils import retrieve_relevant_chunks
    return retrieve_relevant_chunks(
        question=question, chunks=chunks, index=faiss_index, model=model, top_k=top_k
    )

def set_smart_question(q):
    st.session_state["smart_question"] = q


# ==========================================================
# CLAIM CHART + CARD HELPERS
# ==========================================================

def make_claim_chart(result):
    """Compact before → after bar chart for a single verified claim."""
    fv = result.get("first_val")
    lv = result.get("last_val")
    if fv is None or lv is None or result.get("actual_percent") is None:
        return None
    went_up = lv >= fv
    bar_color = "#10b981" if went_up else "#f43f5e"
    fig = go.Figure(data=[
        go.Bar(
            x=["Start", "End"],
            y=[fv, lv],
            marker_color=["#e2e8f0", bar_color],
            text=[f"{fv:,.0f}", f"{lv:,.0f}"],
            textposition="outside",
            width=0.45,
        )
    ])
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
        font=dict(family="DM Sans, sans-serif", size=11, color="#374151"),
        title=dict(
            text=f"{result['csv_metric']} · actual {result['actual_percent']:+.1f}%",
            font=dict(size=11, color="#6b7280"), x=0, xanchor="left",
        ),
        xaxis=dict(showgrid=False, linecolor="#f3f4f6"),
        yaxis=dict(showgrid=True, gridcolor="#f9fafb", zeroline=False),
    )
    return fig


def render_verified_claim_card(result, show_chart=True):
    """Full claim verification result card with coloured verdict, metrics, optional chart."""
    verdict      = result["verdict"]
    v_color  = {"Supported": "#059669", "Contradicted": "#dc2626", "Unverifiable": "#b45309"}.get(verdict, "#6b7280")
    v_bg     = {"Supported": "#ecfdf5", "Contradicted": "#fef2f2", "Unverifiable": "#fffbeb"}.get(verdict, "#f9fafb")
    v_border = {"Supported": "#a7f3d0", "Contradicted": "#fca5a5", "Unverifiable": "#fcd34d"}.get(verdict, "#e5e7eb")
    v_icon   = {"Supported": "✓", "Contradicted": "✕", "Unverifiable": "?"}.get(verdict, "•")

    with st.container(border=True):
        hc1, hc2 = st.columns([5, 1])
        with hc1:
            st.markdown(f"**{result['claim']}**")
        with hc2:
            st.html(
                f'<span style="background:{v_bg};color:{v_color};padding:5px 12px;'
                f'border-radius:20px;font-size:12px;font-weight:700;border:1px solid {v_border};'
                f'white-space:nowrap;">{v_icon} {verdict}</span>'
            )

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.caption("MATCHED COLUMN")
            method = result.get("match_method", "")
            method_label = f" `{method}`" if method and method not in ("none",) else ""
            st.markdown(f"**{result['csv_metric'] or '—'}**{method_label}")
        with mc2:
            st.caption("ACTUAL CHANGE")
            pct = result["actual_percent"]
            c   = "#059669" if pct and pct >= 0 else "#dc2626"
            st.html(f'<span style="font-size:1.15rem;font-weight:700;color:{c}">'
                    f'{"—" if pct is None else f"{pct:+.2f}%"}</span>')
        with mc3:
            fv, lv = result.get("first_val"), result.get("last_val")
            if fv is not None and lv is not None:
                st.caption("VALUE RANGE")
                st.markdown(f"**{fv:,.0f}** → **{lv:,.0f}**")
        with mc4:
            confidence = result.get("confidence")
            if confidence is not None:
                st.caption("CONFIDENCE")
                conf_color = "#059669" if confidence >= 0.7 else "#b45309" if confidence >= 0.4 else "#dc2626"
                st.html(f'<span style="font-size:1.1rem;font-weight:700;color:{conf_color}">'
                        f'{confidence:.0%}</span>')
            tw = result.get("time_window")
            if tw:
                st.caption("TIME WINDOW")
                st.markdown(f"**{tw['type'].upper()} {tw['value']}**")

        if show_chart:
            fig = make_claim_chart(result)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Reasoning & trace"):
            st.write(result["reason"])


# ==========================================================
# MARKDOWN REPORT GENERATOR
# ==========================================================

def generate_markdown_report():
    """Build a plain-text Markdown executive report from session state."""
    lines = [
        "# Dataroom AI — Executive Report",
        f"Generated: {date.today().strftime('%B %d, %Y')}",
        "",
    ]

    # PDF section
    if "pdf_text" in st.session_state:
        pages = st.session_state.get("pdf_pages", [])
        chunks = st.session_state.get("pdf_chunks", [])
        lines += [
            "## Document Overview",
            f"- Pages: {len(pages)}",
            f"- Characters indexed: {len(st.session_state['pdf_text']):,}",
            f"- Searchable chunks: {len(chunks)}",
            "",
        ]

    # CSV section
    if "df" in st.session_state:
        df = st.session_state["df"]
        num_cols, cat_cols = get_numeric_and_categorical_columns(df)
        lines += [
            "## Dataset Summary",
            f"- Rows: {df.shape[0]:,}",
            f"- Columns: {df.shape[1]}",
            f"- Numeric columns: {', '.join(num_cols)}",
            f"- Categorical columns: {', '.join(cat_cols)}",
            "",
            "## Key Metrics",
            "| Column | Total | Mean | Max |",
            "|--------|-------|------|-----|",
        ]
        for c in num_cols[:8]:
            lines.append(f"| {c} | {df[c].sum():,.2f} | {df[c].mean():,.2f} | {df[c].max():,.2f} |")
        lines.append("")

    # Claim verification
    verified = st.session_state.get("report_verified_claims", [])
    if verified:
        lines += [
            "## Claim Verification Results",
            "| Claim | Verdict | Column | Actual Change |",
            "|-------|---------|--------|---------------|",
        ]
        for r in verified:
            pct_str = f"{r['actual_percent']:+.2f}%" if r["actual_percent"] is not None else "—"
            lines.append(f"| {r['claim']} | {r['verdict']} | {r['csv_metric'] or '—'} | {pct_str} |")
        lines.append("")

    lines += [
        "---",
        "_Report generated by Dataroom AI. Verify all claims independently before use._",
    ]
    return "\n".join(lines)

def set_pdf_question(q):
    st.session_state["pdf_q"] = q

def set_csv_question(q):
    st.session_state["csv_q"] = q


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    # Brand header
    st.markdown(
        """
        <div style="padding: 1.25rem 16px 1rem 16px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom: 4px;">
                <div style="width:32px;height:32px;background:linear-gradient(135deg,#4f46e5,#7c3aed);
                            border-radius:8px;display:flex;align-items:center;justify-content:center;
                            font-size:16px;">◈</div>
                <div style="font-size:15px;font-weight:700;color:#ffffff;font-family:'Fraunces',serif;">Dataroom AI</div>
            </div>
            <div style="font-size:11.5px;color:#6b7280;padding-left:42px;">PDF + CSV analyst</div>
        </div>
        <div style="height:1px;background:rgba(255,255,255,0.07);margin:0 16px 12px 16px;"></div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Dashboard", "Upload", "Smart Analyst", "AI Analyst", "CSV Analyst", "Verify Claims", "Reports"],
        label_visibility="collapsed",
    )

    # Workspace file status
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.07);margin:12px 16px;"></div>', unsafe_allow_html=True)
    pdf_loaded = "pdf_text" in st.session_state
    csv_loaded = "df" in st.session_state
    pdf_color  = "#10b981" if pdf_loaded else "#4b5563"
    csv_color  = "#10b981" if csv_loaded else "#4b5563"
    pdf_label  = st.session_state.get("pdf_pages", [])
    pdf_meta   = f"{len(pdf_label)} pages" if pdf_loaded and pdf_label else "Not loaded"
    csv_meta   = f"{st.session_state['df'].shape[0]:,} rows" if csv_loaded else "Not loaded"
    st.markdown(
        f"""
        <div style="padding: 0 16px 1rem 16px;">
            <div style="font-size:10px;font-weight:700;color:#4b5563;text-transform:uppercase;
                        letter-spacing:0.09em;margin-bottom:10px;">Files</div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <div style="width:8px;height:8px;border-radius:50%;background:{pdf_color};flex-shrink:0;"></div>
                <div>
                    <div style="font-size:12.5px;color:#d1d5db;font-weight:500;">PDF Document</div>
                    <div style="font-size:11px;color:#6b7280;">{pdf_meta}</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:8px;height:8px;border-radius:50%;background:{csv_color};flex-shrink:0;"></div>
                <div>
                    <div style="font-size:12.5px;color:#d1d5db;font-weight:500;">CSV Dataset</div>
                    <div style="font-size:11px;color:#6b7280;">{csv_meta}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":
    # ── Page header (DocInsight style — no big gradient hero) ─────────────────
    has_pdf = "pdf_text" in st.session_state
    has_csv = "df" in st.session_state

    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-left">
                <span class="welcome-title">Welcome back 👋</span>
                <span class="welcome-subtitle">Upload a PDF and CSV to get instant insights with evidence-grounded analysis.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── DocInsight-style document status card ─────────────────────────────────
    if has_pdf or has_csv:
        pdf_pages = st.session_state.get("pdf_pages", [])
        pdf_chars = len(st.session_state.get("pdf_text", ""))
        df_shape  = st.session_state["df"].shape if has_csv else (0, 0)

        pdf_block = (
            f'<div class="doc-status-ok">✓ PDF indexed — {len(pdf_pages)} pages · {pdf_chars:,} chars</div>'
            if has_pdf else
            '<div class="doc-status-warn">⚠ No PDF uploaded</div>'
        )
        csv_block = (
            f'<div class="doc-status-ok">✓ CSV loaded — {df_shape[0]:,} rows × {df_shape[1]} columns</div>'
            if has_csv else
            '<div class="doc-status-warn">⚠ No CSV uploaded</div>'
        )

        # Auto key takeaway
        takeaway = "Upload a CSV to see an auto-generated takeaway."
        if has_csv:
            df = st.session_state["df"]
            numeric_cols, cat_cols = get_numeric_and_categorical_columns(df)
            if numeric_cols:
                c0 = numeric_cols[0]
                total = df[c0].sum()
                avg   = df[c0].mean()
                mx    = df[c0].max()
                takeaway = (
                    f"<b>{c0.title()}</b> totals <b>{total:,.0f}</b> across {df.shape[0]:,} records "
                    f"(avg {avg:,.1f}, peak {mx:,.0f})."
                )
                if cat_cols and len(numeric_cols) > 1:
                    c1_col = numeric_cols[1]
                    takeaway += (
                        f" <b>{c1_col.title()}</b> averages <b>{df[c1_col].mean():,.1f}</b>."
                    )

        st.markdown(
            f"""
            <div class="doc-status-card">
                <div class="doc-icon">◈</div>
                <div class="doc-info">
                    <div class="doc-name">Dataroom Workspace</div>
                    <div class="doc-meta">Active analyst session</div>
                    {pdf_block}
                    {csv_block}
                </div>
                <div class="doc-takeaway">
                    <div class="doc-takeaway-label">⚡ Key Takeaway</div>
                    <div class="doc-takeaway-text">{takeaway}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="doc-status-card">
                <div class="doc-icon">◈</div>
                <div class="doc-info">
                    <div class="doc-name">No files loaded</div>
                    <div class="doc-meta">Go to Upload to add a PDF and CSV</div>
                    <div class="doc-status-warn">⚠ Upload files to unlock all analytics</div>
                </div>
                <div class="doc-takeaway">
                    <div class="doc-takeaway-label">Getting started</div>
                    <div class="doc-takeaway-text">Upload a business PDF and a CSV dataset. The AI will instantly extract insights, surface metrics, and let you ask questions in plain English.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Metric cards with sparklines ──────────────────────────────────────────
    if has_csv:
        df = st.session_state["df"]
        numeric_cols, _ = get_numeric_and_categorical_columns(df)

        if numeric_cols:
            render_section_label("Business metrics")
            cols = st.columns(min(4, len(numeric_cols)))
            spark_colors = ["#4f46e5", "#10b981", "#f59e0b", "#ef4444"]

            for i, c in enumerate(numeric_cols[:4]):
                with cols[i]:
                    total  = df[c].sum()
                    n      = df[c].count()
                    vals   = df[c].dropna().tolist()
                    color  = spark_colors[i % len(spark_colors)]
                    spark  = make_sparkline(vals, color=color)
                    render_metric(c, f"{total:,.0f}", f"across {n:,} rows", "neutral", spark)

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    if has_csv:
        df = st.session_state["df"]
        numeric_cols, _ = get_numeric_and_categorical_columns(df)
        chart_cats = get_chart_categorical_cols(df)

        c1, c2 = st.columns(2)

        with c1:
            with st.container(border=True):
                render_section_label("Distribution")
                if numeric_cols:
                    rc1, rc2 = st.columns([3, 1])
                    with rc1:
                        sel = st.selectbox("Column", numeric_cols, key="dash_num", label_visibility="collapsed")
                    with rc2:
                        dist_type = st.selectbox("Type", ["Histogram", "Box", "Violin"], key="dash_dist_type", label_visibility="collapsed")

                    if dist_type == "Histogram":
                        fig = px.histogram(df, x=sel, nbins=25, color_discrete_sequence=[CHART_PALETTE[0]])
                        style_bars(fig)
                        style_fig(fig, height=320, title=f"Distribution · {sel}")
                    elif dist_type == "Box":
                        fig = px.box(df, y=sel, color_discrete_sequence=[CHART_PALETTE[1]])
                        style_fig(fig, height=320, title=f"Box plot · {sel}")
                    else:
                        fig = px.violin(df, y=sel, color_discrete_sequence=[CHART_PALETTE[2]], box=True)
                        style_fig(fig, height=320, title=f"Violin · {sel}")
                    st.plotly_chart(fig, use_container_width=True)

        with c2:
            with st.container(border=True):
                render_section_label("Breakdown")
                if chart_cats and numeric_cols:
                    rc1, rc2 = st.columns([2, 1])
                    with rc1:
                        cat = st.selectbox("Category", chart_cats,  key="dash_cat", label_visibility="collapsed")
                        val = st.selectbox("Value",    numeric_cols, key="dash_val", label_visibility="collapsed")
                    with rc2:
                        chart_type = st.selectbox("Chart", ["Bar", "Line", "Pie", "Scatter"], key="dash_chart_type", label_visibility="collapsed")

                    grouped = (
                        df.groupby(cat)[val].sum()
                          .reset_index().sort_values(val, ascending=False).head(8)
                    )
                    fig = make_chart(grouped, chart_type, cat, val, title=f"{val} by {cat}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    render_status("Need a categorical + numeric column for breakdown chart", "warning")

        # ── Key Insights panel (like DocInsight's bottom section) ─────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            render_section_label("Key insights")
            if numeric_cols:
                icons   = ["📈", "📊", "🎯", "💡"]
                colors  = ["insight-dot-green", "insight-dot-blue", "insight-dot-purple", "insight-dot-orange"]
                insights = []

                c0 = numeric_cols[0]
                insights.append((icons[0], colors[0],
                    f"<b>{c0.title()}</b> totals <b>{df[c0].sum():,.0f}</b> with an average of "
                    f"<b>{df[c0].mean():,.1f}</b> per record."))

                if len(numeric_cols) > 1:
                    c1_col = numeric_cols[1]
                    insights.append((icons[1], colors[1],
                        f"<b>{c1_col.title()}</b> ranges from <b>{df[c1_col].min():,.1f}</b> to "
                        f"<b>{df[c1_col].max():,.1f}</b>."))

                if chart_cats:
                    top_cat  = chart_cats[0]
                    top_val  = numeric_cols[0]
                    top_grp  = df.groupby(top_cat)[top_val].sum().idxmax()
                    top_amt  = df.groupby(top_cat)[top_val].sum().max()
                    insights.append((icons[2], colors[2],
                        f"<b>{top_grp}</b> is the top <b>{top_cat}</b> by <b>{top_val}</b> "
                        f"({top_amt:,.0f})."))

                missing = df.isnull().sum().sum()
                missing_str = "no" if missing == 0 else f"{missing:,}"
                insights.append((icons[3], colors[3],
                    f"Dataset has <b>{df.shape[0]:,} rows × {df.shape[1]} columns</b> with "
                    f"<b>{missing_str} missing values</b>."))

                html = ""
                for icon, color_cls, text in insights:
                    html += (
                        f'<div class="insight-item">'
                        f'<div class="insight-dot {color_cls}">{icon}</div>'
                        f'<div class="insight-text">{text}</div>'
                        f'</div>'
                    )
                st.markdown(html, unsafe_allow_html=True)
    else:
        render_status("Upload a CSV to view metrics and charts", "warning")

# ==========================================================
# SMART ANALYST — ROUTED PDF / CSV / BOTH
# ==========================================================

if page == "Smart Analyst":
    render_hero(
        "Unified analyst",
        "Smart Analyst",
        "Ask one question and DataRoom AI will decide whether to use the PDF, CSV, or both.",
    )

    c1, c2 = st.columns([2.2, 1])

    with c1:
        with st.container(border=True):
            render_section_label("Ask anything")
            smart_question = st.text_input(
                "Question",
                placeholder="e.g. Does the report's revenue claim match the CSV?",
                label_visibility="collapsed",
                key="smart_question",
            )

        if smart_question:
            with st.spinner("Classifying question..."):
                route_result = route_question_llm(smart_question)
            route = route_result["intent"]
            confidence = route_result["confidence"]
            reasoning = route_result["reasoning"]

            st.markdown(
                f'<div class="answer answer-user"><b>You asked:</b><br>{smart_question}</div>',
                unsafe_allow_html=True,
            )

            st.html(
                f'<div style="display:flex;gap:10px;align-items:center;margin-bottom:12px;">'
                f'<span style="background:#ede9fe;color:#4f46e5;padding:5px 12px;border-radius:20px;'
                f'font-size:12px;font-weight:700;">Route: {route.upper()}</span>'
                f'<span style="background:#f0fdf4;color:#059669;padding:5px 12px;border-radius:20px;'
                f'font-size:12px;font-weight:700;">Confidence: {confidence:.0%}</span>'
                f'<span style="font-size:12px;color:#6b7280;font-style:italic;">{reasoning}</span>'
                f'</div>'
            )

            # -----------------------------
            # PDF ROUTE
            # -----------------------------
            if route == "pdf":
                if "faiss_index" in st.session_state and "pdf_chunks" in st.session_state:
                    retrieval_label = "hybrid (FAISS+BM25)" if "bm25_index" in st.session_state else "semantic (FAISS)"
                    with st.spinner(f"Retrieving PDF evidence [{retrieval_label}]..."):
                        results = retrieve_chunks(smart_question, top_k=4)

                    with st.spinner("Generating PDF-grounded answer..."):
                        answer = answer_pdf_question(smart_question, results)

                    if answer.startswith("OpenAI API key not found"):
                        render_status(answer, "warning")
                    else:
                        st.markdown(
                            f'<div class="answer"><b>PDF Answer</b><br><br>{answer}</div>',
                            unsafe_allow_html=True,
                        )

                    with st.container(border=True):
                        render_section_label("Retrieved PDF evidence")
                        pages_seen = {}
                        for r in results:
                            pg = r["page_number"]
                            if pg not in pages_seen:
                                pages_seen[pg] = {"texts": [], "best_distance": r["distance"]}
                            pages_seen[pg]["texts"].append(r["text"])
                            pages_seen[pg]["best_distance"] = min(pages_seen[pg]["best_distance"], r["distance"])
                        for pg, info in sorted(pages_seen.items()):
                            label = relevance_label(info["best_distance"])
                            with st.expander(f"Page {pg}  ·  {label}"):
                                for i, txt in enumerate(info["texts"]):
                                    if i > 0: st.divider()
                                    st.write(txt)
                else:
                    render_status("Upload a PDF first to answer document questions.", "warning")

            # -----------------------------
            # CSV ROUTE
            # -----------------------------
            elif route == "csv":
                if "df" in st.session_state:
                    df = st.session_state["df"]

                    with st.spinner("Analyzing CSV..."):
                        answer, fig = answer_csv_question(smart_question, df)

                    st.markdown(
                        f'<div class="answer"><b>CSV Answer</b><br><br>{answer}</div>',
                        unsafe_allow_html=True,
                    )

                    if fig is not None:
                        style_fig(fig, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    render_status("Upload a CSV first to answer spreadsheet questions.", "warning")

            # -----------------------------
            # BOTH ROUTE — live claim verification
            # -----------------------------
            elif route == "both":
                pdf_ready = "faiss_index" in st.session_state and "pdf_chunks" in st.session_state
                csv_ready = "df" in st.session_state

                if not pdf_ready:
                    render_status("Upload a PDF first so claims can be extracted.", "warning")
                if not csv_ready:
                    render_status("Upload a CSV first so claims can be verified.", "warning")

                if pdf_ready and csv_ready:
                    df_both  = st.session_state["df"]
                    pdf_text = st.session_state.get("pdf_text", "")

                    with st.spinner("Scanning PDF for business claims…"):
                        all_claims = extract_claims_from_text(pdf_text)

                    q_keywords = {w for w in smart_question.lower().split() if len(w) > 3}
                    relevant   = [c for c in all_claims if any(kw in c.lower() for kw in q_keywords)]
                    to_verify  = relevant or all_claims[:4]

                    if to_verify:
                        st.markdown(
                            f'<div class="answer"><b>Found {len(to_verify)} PDF claim(s) — '
                            f'verifying against your CSV</b></div>',
                            unsafe_allow_html=True,
                        )
                        with st.spinner("Verifying claims…"):
                            claim_results = [verify_claim_against_csv(c, df_both) for c in to_verify]
                        for res in claim_results:
                            render_verified_claim_card(res, show_chart=True)
                    else:
                        render_status(
                            "No percentage-change claims found in the PDF. "
                            "Try AI Analyst or CSV Analyst for other questions.",
                            "warning",
                        )

                    # Supporting PDF evidence
                    with st.spinner("Retrieving supporting PDF evidence..."):
                        evi = retrieve_chunks(smart_question, top_k=3)
                    with st.container(border=True):
                        render_section_label("Supporting PDF evidence")
                        pages_seen = {}
                        for r in evi:
                            pg = r["page_number"]
                            if pg not in pages_seen:
                                pages_seen[pg] = {"texts": [], "best_distance": r["distance"]}
                            pages_seen[pg]["texts"].append(r["text"])
                            pages_seen[pg]["best_distance"] = min(pages_seen[pg]["best_distance"], r["distance"])
                        for pg, info in sorted(pages_seen.items()):
                            with st.expander(f"Page {pg}  ·  {relevance_label(info['best_distance'])}"):
                                for i, txt in enumerate(info["texts"]):
                                    if i > 0: st.divider()
                                    st.write(txt)

    with c2:
        with st.container(border=True):
            render_section_label("Examples")

            examples = [
                "What are the key risks in the PDF?",
                "Give me a summary of the dataset",
                "Which columns have missing values?",
                "What is the total revenue?",
                "Does the report's revenue claim match the CSV?",
            ]

            for i, q in enumerate(examples):
                st.button(
                    q,
                    key=f"sq_smart_{i}",
                    on_click=set_smart_question,
                    args=(q,),
                    use_container_width=True,
                )



# ==========================================================
# UPLOAD
# ==========================================================

if page == "Upload":
    render_hero(
        "File workspace",
        "Upload files",
        "Add a PDF and a CSV to unlock the full analyst workspace.",
    )

    # ── Custom upload zone cards ───────────────────────────────────────────────
    st.markdown("""
    <style>
    .upload-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px 28px 16px 28px;
        box-shadow: 0 1px 3px rgba(17,24,39,0.06), 0 4px 16px rgba(17,24,39,0.04);
        margin-bottom: 0;
    }
    .upload-card-icon {
        font-size: 2.2rem;
        margin-bottom: 12px;
        display: block;
    }
    .upload-card-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.3rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        margin: 0 0 6px 0 !important;
        display: block;
    }
    .upload-card-desc {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 14px;
        line-height: 1.5;
        display: block;
    }
    .upload-powers {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-bottom: 14px;
    }
    .upload-power-tag {
        background: #ede9fe;
        color: #4f46e5;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="upload-card">
            <span class="upload-card-icon">📄</span>
            <span class="upload-card-title">PDF Document</span>
            <span class="upload-card-desc">Upload a business report, financial statement, or research document. The AI will extract and index every page for instant Q&A.</span>
            <div class="upload-powers">
                <span class="upload-power-tag">AI Analyst</span>
                <span class="upload-power-tag">Smart Analyst</span>
                <span class="upload-power-tag">Claim Extraction</span>
                <span class="upload-power-tag">Reports</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed",
        )

    with c2:
        st.markdown("""
        <div class="upload-card">
            <span class="upload-card-icon">📊</span>
            <span class="upload-card-title">CSV Dataset</span>
            <span class="upload-card-desc">Upload a spreadsheet with your business data — revenue, customers, contracts, or any metrics. All analytics run locally with no LLM costs.</span>
            <div class="upload-powers">
                <span class="upload-power-tag">CSV Analyst</span>
                <span class="upload-power-tag">Dashboard</span>
                <span class="upload-power-tag">Claim Verification</span>
                <span class="upload-power-tag">Reports</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_csv = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            key="csv_uploader",
            label_visibility="collapsed",
        )

    if uploaded_pdf is not None:
        render_status(f"PDF uploaded: {uploaded_pdf.name}", "success")

        with st.spinner("Extracting PDF text..."):
            pdf_text, pdf_pages = extract_pdf_text(uploaded_pdf)

        st.session_state["pdf_text"] = pdf_text
        st.session_state["pdf_pages"] = pdf_pages

        with st.container(border=True):
            render_section_label("PDF preview")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                render_metric("Pages", f"{len(pdf_pages):,}", "extracted", "positive")
            with cc2:
                render_metric("Characters", f"{len(pdf_text):,}", "indexed", "positive")
            with cc3:
                render_metric("Status", "Ready", "for Q&A", "positive")

            with st.expander("Preview extracted text"):
                st.text_area(
                    "PDF preview",
                    pdf_text[:3000],
                    height=240,
                    label_visibility="collapsed",
                )

        if len(pdf_pages) > 0:
            with st.spinner("Creating searchable index..."):
                model = load_embedding_model()
                chunks = chunk_pdf_pages(pdf_pages)

                if len(chunks) > 0:
                    index, embeddings = create_faiss_index(chunks, model)
                    bm25_index = build_bm25_index(chunks)
                    st.session_state["embedding_model"] = model
                    st.session_state["pdf_chunks"] = chunks
                    st.session_state["faiss_index"] = index
                    st.session_state["pdf_embeddings"] = embeddings
                    st.session_state["bm25_index"] = bm25_index
                    render_status(f"Indexed {len(chunks)} chunks (FAISS + BM25 hybrid)", "success")
                else:
                    render_status("No text chunks created — PDF may be scanned/image-based", "warning")

    if uploaded_csv is not None:
        render_status(f"CSV uploaded: {uploaded_csv.name}", "success")

        with st.spinner("Loading and profiling CSV..."):
            df = load_csv(uploaded_csv)
            profile = profile_csv(df)

        st.session_state["df"] = df
        st.session_state["csv_profile"] = profile

        with st.container(border=True):
            render_section_label("CSV summary")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                render_metric("Rows", f"{profile['rows']:,}", "loaded", "positive")
            with cc2:
                render_metric("Columns", f"{profile['columns']:,}", "detected", "positive")
            with cc3:
                render_metric("Numeric columns", f"{len(profile['numeric_columns']):,}", "ready", "positive")

            st.markdown("**Preview**")
            st.dataframe(df.head(8), use_container_width=True, height=300)

        with st.container(border=True):
            render_section_label("Column profile")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Column types**")
                ctype = pd.DataFrame(
                    list(profile["column_types"].items()),
                    columns=["Column", "Type"],
                )
                st.dataframe(ctype, use_container_width=True, height=280, hide_index=True)
            with cc2:
                st.markdown("**Missing values**")
                cmiss = pd.DataFrame(
                    list(profile["missing_values"].items()),
                    columns=["Column", "Missing"],
                )
                st.dataframe(cmiss, use_container_width=True, height=280, hide_index=True)


# ==========================================================
# AI ANALYST (PDF RAG)
# ==========================================================

if page == "AI Analyst":
    render_hero(
        "PDF RAG assistant",
        "AI Analyst",
        "Ask citation-grounded questions about your uploaded PDF.",
    )

    if "pdf_q" not in st.session_state:
        st.session_state["pdf_q"] = ""

    c1, c2 = st.columns([2.2, 1])

    with c1:
        with st.container(border=True):
            render_section_label("Ask a question")
            question = st.text_input(
                "Question",
                key="pdf_q",
                placeholder="e.g. What were the main risks called out in Q3?",
                label_visibility="collapsed",
            )

        if question:
            st.markdown(
                f'<div class="answer answer-user"><b>You asked:</b><br>{question}</div>',
                unsafe_allow_html=True,
            )

            if "faiss_index" in st.session_state and "pdf_chunks" in st.session_state:
                with st.spinner("Retrieving relevant evidence (hybrid)..."):
                    results = retrieve_chunks(question, top_k=4)

                with st.spinner("Generating citation-grounded answer..."):
                    answer = answer_pdf_question(question, results)

                if answer.startswith("OpenAI API key not found"):
                    render_status(answer, "warning")
                else:
                    st.markdown(
                        f'<div class="answer"><b>Answer</b><br><br>{answer}</div>',
                        unsafe_allow_html=True,
                    )

                with st.container(border=True):
                    render_section_label("Retrieved evidence")
                    # Group chunks by page
                    pages_seen = {}
                    for r in results:
                        pg = r["page_number"]
                        if pg not in pages_seen:
                            pages_seen[pg] = {"texts": [], "best_distance": r["distance"]}
                        pages_seen[pg]["texts"].append(r["text"])
                        pages_seen[pg]["best_distance"] = min(pages_seen[pg]["best_distance"], r["distance"])

                    for pg, info in sorted(pages_seen.items()):
                        label = relevance_label(info["best_distance"])
                        with st.expander(f"Page {pg}  ·  {label}  ({len(info['texts'])} chunk{'s' if len(info['texts'])>1 else ''})"):
                            for i, txt in enumerate(info["texts"]):
                                if i > 0:
                                    st.divider()
                                st.write(txt)
            else:
                render_status("Upload a PDF first so the analyst can search it", "warning")

    with c2:
        with st.container(border=True):
            render_section_label("Suggested questions")
            st.caption("Click any question to ask it instantly")
            pdf_suggestions = [
                "What are the key financial highlights?",
                "What are the main risks mentioned?",
                "What does the document say about revenue?",
                "Summarize the main takeaways.",
            ]
            for i, q in enumerate(pdf_suggestions):
                st.button(
                    q,
                    key=f"sq_pdf_{i}",
                    on_click=set_pdf_question,
                    args=(q,),
                    use_container_width=True,
                )


# ==========================================================
# CSV ANALYST
# ==========================================================

if page == "CSV Analyst":
    render_hero(
        "Spreadsheet intelligence",
        "CSV Analyst",
        "Ask natural-language questions about your uploaded CSV.",
    )

    if "df" not in st.session_state:
        render_status("Upload a CSV first to use the CSV Analyst", "warning")
    else:
        df = st.session_state["df"]

        if "csv_q" not in st.session_state:
            st.session_state["csv_q"] = ""

        c1, c2 = st.columns([2.2, 1])

        with c1:
            with st.container(border=True):
                render_section_label("Ask about your dataset")
                mode_col, q_col = st.columns([1, 3])
                with mode_col:
                    csv_mode = st.radio(
                        "Mode",
                        ["Keyword", "AI (Text-to-Pandas)"],
                        key="csv_analyst_mode",
                        help="Keyword: fast, local. AI: uses LLM to generate Pandas code for any question.",
                    )
                with q_col:
                    csv_question = st.text_input(
                        "Question",
                        key="csv_q",
                        placeholder="e.g. What is the total revenue by region?",
                        label_visibility="collapsed",
                    )

            if csv_question:
                use_llm = csv_mode == "AI (Text-to-Pandas)"
                spinner_msg = "Generating Pandas code with AI..." if use_llm else "Analyzing CSV..."

                with st.spinner(spinner_msg):
                    if use_llm:
                        answer, fig = answer_csv_question_llm(csv_question, df)
                    else:
                        answer, fig = answer_csv_question(csv_question, df)

                clean_answer = answer.replace("`", "")

                st.markdown(
                    f'<div class="answer answer-user"><b>You asked:</b><br>{csv_question}</div>',
                    unsafe_allow_html=True,
                )
                mode_label = "AI Answer (Text-to-Pandas)" if use_llm else "Answer"
                st.markdown(
                    f'<div class="answer"><b>{mode_label}</b><br><br>{clean_answer}</div>',
                    unsafe_allow_html=True,
                )

                if fig is not None:
                    chart_type_csv = st.radio(
                        "Chart type",
                        ["Bar", "Line", "Pie", "Scatter"],
                        horizontal=True,
                        key="csv_chart_type",
                    )
                    try:
                        x_data = list(fig.data[0].x)
                        y_data = list(fig.data[0].y)
                        x_name = fig.layout.xaxis.title.text or "category"
                        y_name = fig.layout.yaxis.title.text or "value"
                        df_fig = pd.DataFrame({x_name: x_data, y_name: y_data})
                        fig = make_chart(df_fig, chart_type_csv, x_name, y_name)
                    except Exception:
                        style_bars(fig, with_labels=True)
                        style_fig(fig, height=400)
                    st.plotly_chart(fig, use_container_width=True)

            with st.container(border=True):
                render_section_label("Dataset preview")
                st.dataframe(df.head(10), use_container_width=True, height=320, hide_index=True)

        with c2:
            with st.container(border=True):
                render_section_label("Try these")
                st.caption("Click any question to ask it instantly")
                csv_suggestions = [
                    "Give me a summary of the dataset",
                    "Which columns have missing values?",
                    "What are the numeric columns?",
                    "Total ARR by region",
                    "Top 10 customers by MRR",
                    "Show me customers where revenue > 50000",
                    "What is the correlation between ARR and MRR?",
                ]
                for i, q in enumerate(csv_suggestions):
                    st.button(
                        q,
                        key=f"sq_csv_{i}",
                        on_click=set_csv_question,
                        args=(q,),
                        use_container_width=True,
                    )


# ==========================================================
# VERIFY CLAIMS
# ==========================================================

if page == "Verify Claims":
    render_hero(
        "Document-data consistency",
        "Claim verification board",
        "Automatically extract business claims from the PDF and verify them against your CSV.",
    )

    if "df" not in st.session_state:
        render_status("Upload a CSV first to verify claims.", "warning")
    else:
        df = st.session_state["df"]

        # ── Section 1: Auto-extract from PDF ──────────────────────────────────
        has_pdf = "pdf_text" in st.session_state
        with st.container(border=True):
            hcol1, hcol2 = st.columns([4, 1])
            with hcol1:
                render_section_label("Auto extraction")
                st.markdown("**Extract claims from the uploaded PDF and verify all at once**")
                st.caption("Scans for sentences like 'Revenue increased by 20%' and checks each against your CSV.")
            with hcol2:
                st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
                extract_btn = st.button(
                    "⚡ Extract from PDF",
                    key="extract_claims_btn",
                    disabled=not has_pdf,
                    use_container_width=True,
                )

            if not has_pdf:
                render_status("Upload a PDF first to enable auto-extraction.", "warning")

            if extract_btn:
                with st.spinner("Scanning PDF for percentage-change claims…"):
                    found = extract_claims_from_text(st.session_state["pdf_text"])
                st.session_state["extracted_claims"]  = found
                st.session_state.pop("verified_claims", None)   # reset old results

            if "extracted_claims" in st.session_state:
                found = st.session_state["extracted_claims"]
                if not found:
                    render_status("No percentage-change claims found in the PDF text.", "warning")
                else:
                    render_status(f"Found {len(found)} claim(s) in the PDF", "success")

                    # Show the list with Verify All button
                    for i, c in enumerate(found):
                        st.markdown(f"**{i+1}.** {c}")

                    if st.button("✓ Verify All Claims", key="verify_all_btn", use_container_width=False):
                        with st.spinner("Verifying all claims against CSV…"):
                            st.session_state["verified_claims"] = [
                                verify_claim_against_csv(c, df) for c in found
                            ]
                            # Also store for Reports page
                            st.session_state["report_verified_claims"] = st.session_state["verified_claims"]

                    if "verified_claims" in st.session_state:
                        st.markdown("---")
                        # Summary row
                        vc = st.session_state["verified_claims"]
                        supported    = sum(1 for r in vc if r["verdict"] == "Supported")
                        contradicted = sum(1 for r in vc if r["verdict"] == "Contradicted")
                        unverifiable = sum(1 for r in vc if r["verdict"] == "Unverifiable")
                        s1, s2, s3 = st.columns(3)
                        s1.metric("✓ Supported",    supported)
                        s2.metric("✕ Contradicted", contradicted)
                        s3.metric("? Unverifiable", unverifiable)
                        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                        for res in vc:
                            render_verified_claim_card(res, show_chart=True)

        # ── Section 2: Manual single claim ────────────────────────────────────
        with st.container(border=True):
            render_section_label("Manual claim check")
            claim_text = st.text_input(
                "Enter a claim",
                placeholder="e.g. Revenue increased by 20%",
                label_visibility="collapsed",
                key="manual_claim_input",
            )

            if claim_text:
                result = verify_claim_against_csv(claim_text, df)
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                render_verified_claim_card(result, show_chart=True)

        with st.expander("Example claims to try"):
            st.markdown(
                "- Revenue increased by 20%\n"
                "- Sales decreased by 10%\n"
                "- Profit increased by 5%\n\n"
                "_Tip: the claim checker compares the first row value to the last row value in the matched column._"
            )


# ==========================================================
# REPORTS (redesigned — clean HTML summary table, no dark df.describe)
# ==========================================================

if page == "Reports":
    render_hero(
        "Executive output",
        "Executive report",
        "Generate a cited business report from PDF evidence and CSV insights.",
    )

    has_pdf = "pdf_text" in st.session_state
    has_csv = "df" in st.session_state

    render_section_label("Report inputs")
    st.markdown("### At a glance")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        render_metric(
            "PDF pages",
            f"{len(st.session_state['pdf_pages']):,}" if has_pdf else "—",
            "extracted" if has_pdf else "not loaded",
            "positive" if has_pdf else "neutral",
        )
    with mc2:
        render_metric(
            "PDF characters",
            f"{len(st.session_state['pdf_text']):,}" if has_pdf else "—",
            "indexed" if has_pdf else "not loaded",
            "positive" if has_pdf else "neutral",
        )
    with mc3:
        render_metric(
            "Dataset rows",
            f"{st.session_state['df'].shape[0]:,}" if has_csv else "—",
            "loaded" if has_csv else "not loaded",
            "positive" if has_csv else "neutral",
        )
    with mc4:
        render_metric(
            "Dataset columns",
            f"{st.session_state['df'].shape[1]:,}" if has_csv else "—",
            "detected" if has_csv else "not loaded",
            "positive" if has_csv else "neutral",
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    main, side = st.columns([3, 1])

    with main:
        if has_csv:
            df = st.session_state["df"]
            numeric_cols, _ = get_numeric_and_categorical_columns(df)

            with st.container(border=True):
                render_section_label("Numeric summary")
                st.markdown("### Dataset statistics")
                if numeric_cols:
                    st.html(build_summary_table(df, numeric_cols[:8]))
                    if len(numeric_cols) > 8:
                        st.caption(f"Showing first 8 of {len(numeric_cols)} numeric columns")
                else:
                    render_status("No numeric columns found in the CSV", "warning")

            chart_cats = get_chart_categorical_cols(df)
            if numeric_cols and chart_cats:
                with st.container(border=True):
                    render_section_label("Headline chart")
                    rca, rcb, rcc = st.columns([2, 2, 1])
                    with rca:
                        rep_cat = st.selectbox("Category", chart_cats, key="rep_cat", label_visibility="collapsed")
                    with rcb:
                        rep_val = st.selectbox("Value", numeric_cols, key="rep_val", label_visibility="collapsed")
                    with rcc:
                        rep_type = st.selectbox("Chart", ["Bar", "Pie", "Line", "Scatter"], key="rep_type", label_visibility="collapsed")
                    grouped = (
                        df.groupby(rep_cat)[rep_val]
                          .sum().reset_index()
                          .sort_values(rep_val, ascending=False).head(8)
                    )
                    fig = make_chart(grouped, rep_type, rep_cat, rep_val, title=f"{rep_val} by {rep_cat}")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            with st.container(border=True):
                render_status("Upload a CSV to include spreadsheet insights", "warning")

    with side:
        with st.container(border=True):
            render_section_label("Document evidence")
            st.markdown("### PDF status")
            if has_pdf:
                render_status(
                    f"{len(st.session_state['pdf_pages'])} pages indexed",
                    "success",
                )
                st.markdown(
                    f"<div style='color:#64748b; font-size:13px; line-height:1.7;'>"
                    f"Characters: <b style='color:#0f172a;'>{len(st.session_state['pdf_text']):,}</b><br/>"
                    f"Chunks: <b style='color:#0f172a;'>{len(st.session_state.get('pdf_chunks', []))}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                render_status("No PDF loaded", "warning")

        with st.container(border=True):
            render_section_label("Download")
            st.markdown("### Export report")
            st.caption("Downloads a Markdown file with PDF overview, CSV metrics, and claim verification results.")
            st.download_button(
                label="⬇ Download as Markdown",
                data=generate_markdown_report(),
                file_name=f"dataroom_report_{date.today().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ── Claim verification section (below the two-column layout) ─────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        render_section_label("Claim verification")
        st.markdown("### PDF claims vs CSV data")

        if "pdf_text" in st.session_state and has_csv:
            col_run, col_status = st.columns([2, 3])
            with col_run:
                run_btn = st.button("⚡ Run claim verification", key="report_run_claims", use_container_width=True)
            if run_btn:
                with st.spinner("Extracting and verifying claims from PDF…"):
                    found = extract_claims_from_text(st.session_state["pdf_text"])
                    st.session_state["report_verified_claims"] = [
                        verify_claim_against_csv(c, st.session_state["df"]) for c in found
                    ]

            verified = st.session_state.get("report_verified_claims", [])
            if verified:
                sv = sum(1 for r in verified if r["verdict"] == "Supported")
                cv = sum(1 for r in verified if r["verdict"] == "Contradicted")
                uv = sum(1 for r in verified if r["verdict"] == "Unverifiable")
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("✓ Supported",    sv)
                rc2.metric("✕ Contradicted", cv)
                rc3.metric("? Unverifiable", uv)
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                for res in verified:
                    render_verified_claim_card(res, show_chart=False)
            elif not run_btn:
                st.caption("Click 'Run claim verification' to extract and verify PDF claims automatically.")
        else:
            render_status("Upload both a PDF and a CSV to enable claim verification.", "warning")