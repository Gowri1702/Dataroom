# DataRoom AI

**An evidence-grounded business analyst for PDFs and CSVs.**

Upload a company report and a data file. Ask questions in plain English. DataRoom AI reads both, answers with citations, and tells you whether the numbers in the report actually match the data — or not.

---

## What It Does

Most AI chat tools will confidently answer questions about documents even when they're guessing. DataRoom AI is built differently: every answer either comes with a source citation or the system says it cannot verify. No hallucinated figures. No made-up page references.

The three things it does well:

**1. Document Q&A with citations**
Ask any question about a PDF and get an answer with the exact page it came from. The system retrieves the most relevant passages first, then sends only those to the language model — so the answer is grounded in what the document actually says.

**2. Data analysis without writing code**
Upload a CSV and ask questions in plain English: totals, averages, comparisons, grouped breakdowns. The system generates and runs the analysis for you, with charts where useful.

**3. Claim verification**
Point the system at a specific claim from the report — "Revenue grew 40% year over year" — and it will check it against your CSV, returning one of three honest verdicts: **Supported**, **Contradicted**, or **Unverifiable**. If the column match is uncertain, it abstains rather than guessing.

---

## How It Works

### Document Pipeline (PDF)

When you upload a PDF, the text is split into overlapping chunks. Each chunk is embedded using a local sentence-transformer model (`all-MiniLM-L6-v2`) and indexed in FAISS. When you ask a question, two retrievers run in parallel:

- **Dense retrieval (FAISS)** — finds semantically similar chunks using vector similarity
- **Sparse retrieval (BM25)** — finds chunks with exact keyword matches

The results from both are merged using **Reciprocal Rank Fusion (RRF)**, a technique that combines rankings without needing to tune any weights. The top chunks are passed to GPT-4.1-mini, which generates an answer with page-level citations.

### Data Pipeline (CSV)

CSV files are loaded and profiled automatically. A keyword-based router classifies your question — is it asking about the document, the data, or both? Data questions are answered using Pandas directly (no LLM involved), which means the math is exact. If you ask a question that needs both the PDF and the CSV together, the system handles that routing too.

### Claim Verifier

The verifier extracts the claimed percentage (or multiplier word — "doubled", "tripled", "halved") from the sentence, finds the matching column in the CSV using a domain alias map plus fuzzy matching, and computes the actual change. The column match must exceed an 80% confidence threshold before any verdict is issued — weak matches return Unverifiable immediately, before any math runs.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| UI | Streamlit | Fast to build, built-in file uploaders and charts |
| PDF parsing | PyMuPDF | Fast, accurate text extraction with page metadata |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) | Free, runs locally, no API calls for indexing |
| Dense retrieval | FAISS | Fast vector search, runs fully local |
| Sparse retrieval | BM25 (rank-bm25) | Catches exact keyword matches dense search misses |
| Retrieval fusion | Reciprocal Rank Fusion | Blends rankings without hyperparameter tuning |
| Language model | OpenAI GPT-4.1-mini | Cheap, fast, supports JSON-mode structured output |
| Data analysis | Pandas + Plotly | Exact computation, no LLM for math |
| Fuzzy matching | rapidfuzz | Handles messy column names in real CSVs |
| Testing | pytest | 90+ unit tests across all modules |

---

## Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repo
git clone https://github.com/Gowri1702/Dataroom.git
cd Dataroom

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a file named `.env` in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

> If no API key is set, the app still runs — PDF Q&A will return an explanatory message instead of crashing, and all CSV analytics (which don't use the LLM) continue to work normally.

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

**Step 1 — Upload files**
Go to the **Upload** page in the sidebar. Upload your PDF (e.g., an annual report) and your CSV (e.g., the financial data backing it). The system indexes the PDF automatically on upload.

**Step 2 — Ask questions**
- **Smart Analyst** — one box that routes to the right system automatically
- **AI Analyst** — PDF-only Q&A with cited answers
- **CSV Analyst** — data questions with suggested prompts and charts

**Step 3 — Verify claims**
Go to **Verify Claims**, paste a sentence from the report (e.g., *"Revenue increased by 42% in 2024"*), and the system checks it against your CSV.

**Dashboard**
The Dashboard page shows an automatic summary of your CSV — key metrics, distributions, and charts — as soon as a file is uploaded.

---

## Project Structure

```
dataroom-ai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── pdf_utils.py        # PDF text extraction (PyMuPDF)
│   ├── rag_utils.py        # Chunking, FAISS indexing, retrieval
│   ├── hybrid_retrieval.py # BM25 + FAISS fusion via RRF
│   ├── llm_utils.py        # OpenAI wrapper + prompt builder
│   ├── csv_utils.py        # CSV loading and profiling
│   ├── csv_analyst.py      # Pandas-based Q&A + Plotly charts
│   ├── csv_analyst_llm.py  # LLM-assisted CSV analysis
│   ├── router.py           # Keyword-based query router
│   ├── router_llm.py       # GPT-4.1-mini JSON-mode router
│   ├── claim_checker.py    # Claim extraction + verification engine
│   └── report_utils.py     # Report generation utilities
│
├── tests/
│   ├── test_claim_checker.py
│   ├── test_csv_analyst.py
│   ├── test_rag_utils.py
│   └── test_router.py
│
└── evals/
    ├── dataset.json        # Labeled test cases
    ├── metrics.py          # Precision@k, faithfulness, claim F1
    └── run_eval.py         # Evaluation runner
```

---

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific module
pytest tests/test_claim_checker.py -v
```

## Running Evaluations

```bash
# Router + claim verification evals (no PDF/CSV upload needed)
python evals/run_eval.py

# Full eval including retrieval and citation (requires a PDF)
python evals/run_eval.py --full --pdf path/to/your.pdf
```

The eval harness measures:
- **Router accuracy** — correct classification of pdf / csv / both
- **Precision@k** — fraction of retrieved chunks that are actually relevant
- **Answer faithfulness** — whether answers contradict the retrieved context
- **Claim verification F1** — precision and recall of Supported / Contradicted / Unverifiable verdicts

Results are saved to `evals/results_latest.json`.

---

## Known Limitations

- **Derived metrics are not verifiable.** Composite figures like EBITDA, revenue-per-employee, or net new ARR require calculations across multiple columns. The verifier returns Unverifiable for these rather than attempting to compute them.
- **Time-window parsing uses the first date reference.** For claims with multiple date anchors (e.g., "grew 20% in 2025 from FY2024"), the parser defaults to the full dataset range rather than risk using the wrong slice.
- **Fuzzy matching has a hard threshold.** Column names using unusual abbreviations or non-standard terms may fall below the 80% match threshold and return Unverifiable even when a matching column exists.
- **No re-ranking step.** Retrieved chunks are fused by rank but not re-ranked by a cross-encoder. A re-ranker would improve answer quality on long, complex documents.

See [`LIMITATIONS.md`](LIMITATIONS.md) for full details.

---

## License

MIT License — free to use, modify, and distribute.

---

*Built by [Gowri Sriram Lakshmanan](https://linkedin.com/in/gowrisriram)*
