# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate the virtual environment (required before any command)
source venv/bin/activate

# Run the app
streamlit run app.py

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup

Copy ` .env` (note the leading space in the filename) and set your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

The app silently degrades when no key is set — `get_openai_client()` in `src/llm_utils.py` returns `None` and PDF Q&A returns an explanatory string rather than crashing.

## Architecture

**Dataroom AI** is a Streamlit single-file app (`app.py`) that lets users upload a PDF and a CSV then query both with AI.

### Data flow

```
User uploads PDF  →  pdf_utils.extract_pdf_text()
                  →  rag_utils.chunk_pdf_pages()
                  →  rag_utils.create_faiss_index()   (cached in st.session_state)
                  →  rag_utils.retrieve_relevant_chunks()
                  →  llm_utils.answer_pdf_question()  (OpenAI gpt-4.1-mini)

User uploads CSV  →  csv_utils.load_csv()
                  →  csv_utils.profile_csv()           (cached in st.session_state)
                  →  csv_analyst.answer_csv_question() (pure Pandas, no LLM)

Smart Analyst     →  router.route_question()           (keyword-based routing)
                  →  delegates to pdf, csv, or both paths above

Verify Claims     →  claim_checker.verify_claim_against_csv()
```

### Module responsibilities (`src/`)

| File | Role |
|---|---|
| `pdf_utils.py` | PyMuPDF (`fitz`) extraction — returns `(full_text, pages[])` |
| `rag_utils.py` | Chunking (800-char with 150-char overlap), FAISS L2 index, retrieval. Embedding model (`all-MiniLM-L6-v2`) is `@st.cache_resource`. |
| `llm_utils.py` | OpenAI client wrapper + prompt builder for PDF Q&A |
| `csv_utils.py` | Pandas load + profile (types, missing values, numeric/categorical split) |
| `csv_analyst.py` | Keyword-matched Pandas analytics + Plotly bar charts. No LLM. |
| `router.py` | Classifies a question as `"pdf"`, `"csv"`, or `"both"` via keyword lists |
| `claim_checker.py` | Verifies percentage-change claims against the first/last values of matched CSV columns |
| `report_utils.py` | (stub — empty) |

### Streamlit pages (sidebar navigation)

- **Dashboard** — workspace status, auto metrics from CSV, auto charts
- **Upload** — file uploaders; triggers PDF chunking/indexing and CSV profiling on upload
- **Smart Analyst** — single question box routed to pdf/csv/both
- **AI Analyst** — PDF-only RAG with cited answers and evidence expanders
- **CSV Analyst** — Pandas-based Q&A with suggested questions
- **Verify Claims** — manual percentage-change claim verification against CSV
- **Reports** — HTML stats table + headline chart; downloadable report is a planned feature

### Key session state keys

`pdf_text`, `pdf_pages`, `embedding_model`, `pdf_chunks`, `faiss_index`, `pdf_embeddings`, `df`, `csv_profile`, `pdf_q`, `csv_q`

### LLM

Only `llm_utils.answer_pdf_question` calls OpenAI (`gpt-4.1-mini`, temperature 0.2). All CSV analytics are local Pandas — no external calls.

### Styling

All custom CSS is injected as a single `st.markdown(..., unsafe_allow_html=True)` block at the top of `app.py`. The chart theme (`style_fig`, `style_bars`) uses Inter font and a fixed indigo palette. Plotly figures are always passed through `style_fig()` before rendering.
