from fpdf import FPDF
import os

PDF_OUT = "dataroom_ai_explained.pdf"

# ── Palette ──────────────────────────────────────────────────────────────────
INDIGO = (79, 70, 229)
DARK   = (15, 23, 42)
GRAY   = (107, 114, 128)
LIGHT  = (243, 244, 246)
WHITE  = (255, 255, 255)
GREEN  = (5, 150, 105)
AMBER  = (180, 83, 9)
RED    = (220, 38, 38)


class PDF(FPDF):
    def header(self):
        # Coloured top bar
        self.set_fill_color(*INDIGO)
        self.rect(0, 0, 210, 8, "F")

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 6, f"Dataroom AI  -  Project Documentation  |  Page {self.page_no()}", align="C")

    # ── Generic helpers ───────────────────────────────────────────────────────
    def h1(self, text):
        self.ln(6)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*DARK)
        self.multi_cell(0, 9, text)
        self.set_draw_color(*INDIGO)
        self.set_line_width(0.8)
        self.line(14, self.get_y() + 1, 196, self.get_y() + 1)
        self.ln(5)

    def h2(self, text):
        self.ln(5)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*INDIGO)
        self.multi_cell(0, 7, text)
        self.ln(2)

    def h3(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*DARK)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=6):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        x_start = self.l_margin + indent
        y = self.get_y()
        self.set_fill_color(*INDIGO)
        self.ellipse(x_start - 1, y + 1.8, 2, 2, "F")
        self.set_xy(x_start + 3, y)
        text_w = self.w - self.r_margin - x_start - 3
        self.multi_cell(text_w, 5.5, text)

    def code_block(self, text):
        self.set_font("Courier", "", 8.5)
        self.set_text_color(*DARK)
        self.set_fill_color(235, 237, 245)
        self.set_draw_color(200, 205, 220)
        self.set_line_width(0.3)
        x = self.get_x()
        y = self.get_y()
        lines = text.split("\n")
        h = len(lines) * 4.5 + 6
        self.rect(x, y, 182, h, "DF")
        self.set_xy(x + 4, y + 3)
        for line in lines:
            self.cell(178, 4.5, line)
            self.ln(4.5)
        self.ln(3)

    def info_box(self, title, body_text, color=INDIGO):
        self.set_fill_color(240, 242, 255)
        self.set_draw_color(*color)
        self.set_line_width(0.5)
        x, y = self.get_x(), self.get_y()
        # Left accent bar
        self.set_fill_color(*color)
        self.rect(x, y, 3, 20, "F")
        self.set_fill_color(240, 242, 255)
        self.rect(x + 3, y, 179, 20, "F")
        self.set_xy(x + 7, y + 3)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(0, 5, title)
        self.ln(5)
        self.set_x(x + 7)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        self.multi_cell(175, 5, body_text)
        self.ln(4)

    def tag(self, text, bg=INDIGO):
        r, g, b = bg
        self.set_fill_color(r, g, b)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 7)
        tw = self.get_string_width(text) + 4
        self.cell(tw, 5, text, fill=True)
        self.set_text_color(*DARK)
        self.cell(2)

    def section_divider(self):
        self.ln(4)
        self.set_draw_color(*LIGHT[0:3] if isinstance(LIGHT[0], int) else LIGHT)
        self.set_draw_color(229, 231, 235)
        self.set_line_width(0.3)
        self.line(14, self.get_y(), 196, self.get_y())
        self.ln(4)

    def interview_q(self, num, question, answer):
        self.ln(2)
        # Question box
        self.set_fill_color(245, 243, 255)
        self.set_draw_color(*INDIGO)
        self.set_line_width(0.4)
        x, y = self.get_x(), self.get_y()

        # Draw background rect (estimate height)
        q_lines = self._estimate_lines(f"Q{num}. {question}", 174)
        a_lines = self._estimate_lines(f"A: {answer}", 174)
        box_h = (q_lines + a_lines) * 5 + 14
        self.rect(x, y, 182, box_h, "DF")

        self.set_xy(x + 4, y + 3)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*INDIGO)
        self.multi_cell(174, 5, f"Q{num}.  {question}")
        self.ln(1)
        self.set_x(x + 4)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(55, 65, 81)
        self.multi_cell(174, 5, f"A: {answer}")
        self.set_xy(x, self.get_y() + 3)

    def _estimate_lines(self, text, width_mm):
        char_width = 1.8
        chars_per_line = width_mm / char_width
        return max(1, int(len(text) / chars_per_line) + 1)

    def cover_page(self):
        self.add_page()
        # Large gradient-style header block
        self.set_fill_color(*INDIGO)
        self.rect(0, 8, 210, 80, "F")
        # Decorative circle
        self.set_fill_color(99, 102, 241)
        self.ellipse(150, -20, 120, 120, "F")
        # Title
        self.set_xy(14, 22)
        self.set_font("Helvetica", "B", 36)
        self.set_text_color(*WHITE)
        self.cell(0, 14, "Dataroom AI", align="L")
        self.ln(14)
        self.set_x(14)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(196, 181, 253)
        self.cell(0, 7, "End-to-End Project Documentation  &  Interview Guide", align="L")
        self.ln(7)
        self.set_x(14)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(165, 180, 252)
        self.cell(0, 5, "PDF + CSV Analyst · RAG · OpenAI · FAISS · Streamlit", align="L")

        # Info strip below header
        self.set_y(95)
        cols = [
            ("Type", "Full-Stack AI App"),
            ("Stack", "Python / Streamlit"),
            ("AI Layer", "OpenAI gpt-4.1-mini"),
            ("Search", "FAISS + SentenceTransformers"),
        ]
        col_w = 182 / len(cols)
        for i, (k, v) in enumerate(cols):
            self.set_xy(14 + i * col_w, 95)
            self.set_fill_color(235, 233, 255)
            self.rect(14 + i * col_w, 95, col_w - 2, 18, "F")
            self.set_xy(14 + i * col_w + 3, 97)
            self.set_font("Helvetica", "B", 7)
            self.set_text_color(*INDIGO)
            self.cell(col_w - 6, 4, k.upper())
            self.ln(4)
            self.set_x(14 + i * col_w + 3)
            self.set_font("Helvetica", "", 8.5)
            self.set_text_color(*DARK)
            self.cell(col_w - 6, 5, v)

        self.set_y(120)
        # TOC placeholder
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*DARK)
        self.cell(0, 6, "Contents")
        self.ln(8)
        toc = [
            ("1", "Project Overview"),
            ("2", "Architecture & Data Flow"),
            ("3", "Technology Stack"),
            ("4", "Module Deep-Dive"),
            ("5", "Pages & Features"),
            ("6", "LLM & RAG Pipeline"),
            ("7", "Claim Verification Engine"),
            ("8", "Session State Management"),
            ("9", "Frontend & Styling"),
            ("10", "Interview Questions"),
        ]
        for num, title in toc:
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*GRAY)
            self.cell(10, 6, num + ".")
            self.set_text_color(*DARK)
            self.cell(0, 6, title)
            self.ln(6)


# ─────────────────────────────────────────────────────────────────────────────

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=14)
pdf.cover_page()


# ── 1. Project Overview ───────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("1. Project Overview")
pdf.body(
    "Dataroom AI is an AI-powered document and data analyst built with Streamlit. It allows "
    "users to upload a PDF (business report, financial statement, research paper) alongside a "
    "CSV dataset, then ask plain-English questions against either or both sources. The system "
    "uses Retrieval-Augmented Generation (RAG) for the PDF and pure Pandas analytics for the "
    "CSV, combining them through a keyword-based question router."
)

pdf.h2("What Problem Does It Solve?")
pdf.body(
    "In a typical due-diligence or finance workflow, analysts must cross-check narrative claims "
    "in PDF reports against underlying spreadsheet data  -  a slow and error-prone manual process. "
    "Dataroom AI automates this: it extracts percentage-change claims from the PDF and "
    "immediately verifies them against the CSV, flagging inconsistencies as Supported, "
    "Contradicted, or Unverifiable."
)

pdf.h2("Key Capabilities")
bullets = [
    "PDF ingestion & full-text extraction via PyMuPDF (fitz).",
    "Semantic search over PDF pages using sentence-transformers + FAISS vector index.",
    "Citation-grounded Q&A: GPT-4.1-mini answers only from retrieved page evidence.",
    "Pandas-only CSV analytics  -  no LLM calls, so free to run at any scale.",
    "Keyword router that classifies each question as PDF, CSV, or BOTH.",
    "Automatic claim extraction (regex + direction words) and CSV verification.",
    "Interactive Plotly charts with a consistent indigo design system.",
    "Downloadable Markdown executive report summarising both sources.",
]
for b in bullets:
    pdf.bullet(b)

pdf.info_box(
    "Design Principle",
    "All CSV computation is local Pandas  -  zero LLM cost. Only PDF Q&A hits OpenAI. "
    "The app degrades gracefully: if OPENAI_API_KEY is missing, PDF Q&A returns an "
    "explanatory string rather than crashing.",
)


# ── 2. Architecture & Data Flow ───────────────────────────────────────────────
pdf.add_page()
pdf.h1("2. Architecture & Data Flow")

pdf.h2("High-Level Architecture")
pdf.body(
    "The app is a single-page Streamlit application (app.py) that imports seven src/ modules. "
    "Streamlit's session_state acts as the in-process data store  -  no database, no server-side "
    "persistence. All state lives in the browser session."
)

pdf.h2("PDF Pipeline")
pdf.code_block(
    "User uploads PDF\n"
    "  -> pdf_utils.extract_pdf_text()       # PyMuPDF extracts text per page\n"
    "  -> rag_utils.chunk_pdf_pages()        # 800-char chunks, 150-char overlap\n"
    "  -> rag_utils.create_faiss_index()     # Embeds chunks -> FAISS L2 index\n"
    "  -> rag_utils.retrieve_relevant_chunks() # Embed question, search top-k\n"
    "  -> llm_utils.answer_pdf_question()    # GPT-4.1-mini answers from context"
)

pdf.h2("CSV Pipeline")
pdf.code_block(
    "User uploads CSV\n"
    "  -> csv_utils.load_csv()              # pandas.read_csv()\n"
    "  -> csv_utils.profile_csv()          # dtype / missing / column summary\n"
    "  -> csv_analyst.answer_csv_question() # Keyword-matched Pandas analytics"
)

pdf.h2("Smart Analyst (Combined) Pipeline")
pdf.code_block(
    "User asks question\n"
    "  -> router.route_question()    # returns 'pdf', 'csv', or 'both'\n"
    "     'pdf'  -> RAG pipeline\n"
    "     'csv'  -> Pandas pipeline\n"
    "     'both' -> claim extraction + CSV verification + RAG evidence"
)

pdf.h2("Claim Verification Pipeline")
pdf.code_block(
    "claim_checker.extract_claims_from_text(pdf_text)\n"
    "  # regex: sentence with \\d+% AND direction word\n"
    "  # returns up to 8 unique claim strings\n\n"
    "claim_checker.verify_claim_against_csv(claim, df)\n"
    "  # 1. Extract percent + direction from claim\n"
    "  # 2. Find matching numeric column by name\n"
    "  # 3. Compute actual % change (first -> last row)\n"
    "  # 4. Compare direction + magnitude (±5% tolerance)\n"
    "  # 5. Return verdict: Supported | Contradicted | Unverifiable"
)


# ── 3. Technology Stack ───────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("3. Technology Stack")

stack = [
    ("Streamlit", "Web framework", "Single-file app with session_state, widgets, and sidebar navigation. No Flask/FastAPI needed."),
    ("PyMuPDF (fitz)", "PDF extraction", "Extracts raw text from PDF pages. Returns per-page dictionaries preserving page numbers."),
    ("sentence-transformers", "Embeddings", "all-MiniLM-L6-v2 loaded once via @st.cache_resource. Converts text to 384-dim float32 vectors."),
    ("FAISS (faiss-cpu)", "Vector search", "IndexFlatL2  -  exact L2 nearest-neighbour over chunk embeddings. No approximate search needed at this scale."),
    ("OpenAI SDK", "LLM inference", "gpt-4.1-mini at temperature 0.2. Only called for PDF Q&A. Returns citation-grounded answers."),
    ("Pandas", "CSV analytics", "All CSV computation. No LLM involved. Handles groupby, agg, describe, isnull."),
    ("Plotly", "Visualisation", "px.bar, px.histogram, px.line, go.Figure. Themed via style_fig() / style_bars() helpers."),
    ("python-dotenv", "Config", "Loads OPENAI_API_KEY from .env file. App silently degrades if key absent."),
    ("re (stdlib)", "Claim parsing", "Regex to detect percentage values and directional keywords in PDF sentences."),
]
for name, category, description in stack:
    pdf.h3(f"{name}  [{category}]")
    pdf.body(description)

pdf.h2("Dependencies (requirements.txt)")
pdf.code_block(
    "streamlit          # UI framework\n"
    "pandas             # CSV analytics\n"
    "numpy              # array maths for FAISS\n"
    "plotly             # interactive charts\n"
    "pymupdf            # PDF text extraction\n"
    "python-dotenv      # .env loading\n"
    "sentence-transformers  # embeddings\n"
    "faiss-cpu          # vector index\n"
    "scikit-learn       # optional utilities\n"
    "openai             # OpenAI API client\n"
    "torchvision        # required by sentence-transformers"
)


# ── 4. Module Deep-Dive ───────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("4. Module Deep-Dive")

pdf.h2("4.1  pdf_utils.py")
pdf.body(
    "Exports extract_pdf_text(uploaded_pdf). Opens the uploaded bytes with fitz.open(), "
    "iterates over each page, calls page.get_text() for raw Unicode text, and builds two "
    "return values: (1) full_text  -  a single string with page separators, and (2) pages  -  "
    "a list of dicts with keys page_number and text. This preserves page attribution "
    "so every downstream chunk knows which page it came from."
)

pdf.h2("4.2  rag_utils.py")
pdf.body(
    "Three functions drive the RAG pipeline:\n\n"
    "load_embedding_model(): decorated with @st.cache_resource so the 80 MB MiniLM model "
    "is loaded only once per Streamlit process.\n\n"
    "chunk_pdf_pages(pdf_pages, chunk_size=800, overlap=150): slides a window across each "
    "page's text. Overlap ensures a sentence split across a boundary still appears in "
    "full in at least one chunk. Returns a list of chunk dicts with chunk_id, page_number, text.\n\n"
    "create_faiss_index(chunks, model): encodes all chunk texts into float32 embeddings "
    "(384 dims), builds faiss.IndexFlatL2, adds the matrix. Returns (index, embeddings).\n\n"
    "retrieve_relevant_chunks(question, chunks, index, model, top_k=4): encodes the question, "
    "calls index.search(), returns the top-k chunk dicts with their L2 distance."
)

pdf.h2("4.3  llm_utils.py")
pdf.body(
    "get_openai_client() reads OPENAI_API_KEY from .env via dotenv, returns None if absent.\n\n"
    "build_context_from_chunks(chunks) formats retrieved chunks as [Page N] blocks.\n\n"
    "answer_pdf_question(question, retrieved_chunks) builds a structured prompt that instructs "
    "the LLM to cite page numbers and refuse to use outside knowledge. Calls GPT-4.1-mini "
    "at temperature 0.2 for consistent, factual output."
)

pdf.h2("4.4  csv_utils.py")
pdf.body(
    "load_csv(uploaded_csv): wraps pd.read_csv(). Returns a DataFrame.\n\n"
    "profile_csv(df): returns a dict with rows, columns, column_names, column_types "
    "(dtype strings), missing_values (per-column null counts), numeric_columns, and "
    "categorical_columns. Used on the Dashboard and Upload pages."
)

pdf.h2("4.5  csv_analyst.py")
pdf.body(
    "answer_csv_question(question, df) uses if/elif keyword matching on question.lower() "
    "to dispatch to the right Pandas operation: summary, missing values, totals, averages, "
    "max/min, or grouped bar charts. find_matching_column() does fuzzy column matching "
    "by comparing lowercased/normalised column names against the question. Returns (answer_str, plotly_fig_or_None)."
)

pdf.h2("4.6  router.py")
pdf.body(
    "route_question(question) classifies a question using three keyword lists:\n"
    "pdf_keywords (document, report, risks, summary, ...), csv_keywords (data, total, average, arr, ...), "
    "and both_keywords (match, verify, compare, claim, ...). Precedence: both > csv > pdf. "
    "Default fallback is 'pdf'."
)

pdf.h2("4.7  claim_checker.py")
pdf.body(
    "extract_claims_from_text(text): splits text into sentences via regex on .!?\\n, "
    "keeps only sentences that contain a percentage pattern (\\d+%) AND a directional word "
    "(increased, declined, grew, ...). Deduplicates and caps at 8.\n\n"
    "verify_claim_against_csv(claim, df): extracts (claimed_pct, direction) from the claim, "
    "finds the matching column, computes actual % change from first to last non-null value, "
    "and applies ±5% tolerance to verdict."
)


# ── 5. Pages & Features ───────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("5. Pages & Features")

pages_info = [
    ("Dashboard", "Workspace overview with document status card, metric cards with SVG sparklines, "
     "distribution/breakdown charts, and auto-generated key insights. All computed from CSV in session_state."),
    ("Upload", "File uploader widgets for PDF and CSV. On PDF upload: extracts text, chunks pages, "
     "creates FAISS index  -  all shown as progress spinners. On CSV upload: loads, profiles, shows "
     "preview dataframe and column type/missing tables."),
    ("Smart Analyst", "Single text input. Router decides pdf/csv/both. For 'both': extracts claims "
     "from PDF, verifies each against CSV, shows verdict cards + supporting evidence expanders."),
    ("AI Analyst", "PDF-only RAG. Shows cited answer with page-grouped evidence expanders. Suggested "
     "questions sidebar lets users click to pre-fill the input."),
    ("CSV Analyst", "Pandas Q&A with chart type selector (Bar/Line/Pie/Scatter) that re-renders "
     "the same grouped data in the chosen style."),
    ("Verify Claims", "Auto-extraction from PDF with 'Verify All' button, plus manual single-claim "
     "input. Results show verdict pill, matched column, actual change, before/after bar chart, and reasoning."),
    ("Reports", "HTML stats table (count/mean/median/min/max/std per numeric column), headline "
     "chart with axis/type selectors, and a downloadable Markdown report."),
]
for name, desc in pages_info:
    pdf.h2(name)
    pdf.body(desc)


# ── 6. LLM & RAG Pipeline ────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("6. LLM & RAG Pipeline")

pdf.h2("Why RAG?")
pdf.body(
    "PDF documents can be hundreds of pages long  -  far too large to fit in a single LLM context "
    "window efficiently. RAG solves this by first retrieving the most relevant passages (top-k "
    "chunks) and only sending those to the LLM. This reduces cost, latency, and hallucination risk."
)

pdf.h2("Chunking Strategy")
pdf.body(
    "chunk_size=800 characters, overlap=150 characters. The overlap prevents a sentence that "
    "straddles a chunk boundary from being split and losing context. 800 chars (~120 words) "
    "is a common sweet spot that keeps chunks semantically coherent while giving FAISS enough "
    "granularity to distinguish passages."
)

pdf.h2("Embedding Model")
pdf.body(
    "all-MiniLM-L6-v2 (SentenceTransformers) produces 384-dimensional dense vectors. It is "
    "fast, free, runs locally (no API call), and ranks highly on retrieval benchmarks for "
    "short English text. The model is cached across Streamlit reruns via @st.cache_resource."
)

pdf.h2("FAISS Index")
pdf.body(
    "IndexFlatL2 computes exact L2 (Euclidean) distances across all chunk embeddings. "
    "For documents with ~hundreds of chunks this is fast enough without approximation. "
    "L2 distance is related to cosine similarity when vectors are unit-normalised, but "
    "MiniLM outputs unnormalised vectors  -  relevance_label() converts raw distance to an "
    "informal high/medium/low label using sim = 1 / (1 + distance)."
)

pdf.h2("LLM Prompt Engineering")
pdf.code_block(
    "System: You are a careful business analyst who answers only from provided evidence.\n\n"
    "User:\n"
    "You are DataRoom AI, an evidence-grounded business analyst.\n"
    "Answer the user's question using ONLY the provided PDF evidence.\n"
    "Rules:\n"
    "1. Do not use outside knowledge.\n"
    "2. If evidence is insufficient, say: 'Not enough evidence in the uploaded PDF.'\n"
    "3. Include page citations: (Page X).\n"
    "4. Be concise, professional, and business-focused.\n\n"
    "User question: {question}\n"
    "PDF evidence:\n"
    "[Page 1]\n{chunk_text}\n..."
)
pdf.body(
    "Temperature 0.2 keeps answers deterministic and factual. The model gpt-4.1-mini "
    "balances speed and quality for retrieval-augmented tasks."
)


# ── 7. Claim Verification Engine ──────────────────────────────────────────────
pdf.add_page()
pdf.h1("7. Claim Verification Engine")

pdf.h2("Claim Extraction")
pdf.body(
    "Sentences are split from the raw PDF text using a regex that breaks on .!? followed "
    "by whitespace, or on newlines. Each sentence is kept if it:\n"
    "  - Is between 15 and 250 characters long\n"
    "  - Contains a number followed by % (e.g. 20%, 3.5%)\n"
    "  - Contains at least one directional word (increased, declined, grew, ...)\n\n"
    "Sentences are deduplicated (normalised lowercase) and capped at 8."
)

pdf.h2("Column Matching")
pdf.body(
    "The verifier tries to find a matching numeric column in the CSV using two strategies:\n"
    "1. Direct name match: lowercased column name appears in lowercased claim.\n"
    "2. Alias map: common domain terms (revenue, profit, arr, mrr) mapped to known column names.\n"
    "If no match is found, it falls back to the first numeric column and flags this in the reasoning."
)

pdf.h2("Verdict Logic")
pdf.code_block(
    "actual_pct = (last_val - first_val) / abs(first_val) * 100\n"
    "tolerance  = 5.0  # percentage points\n\n"
    "direction_match = (claimed_direction is None) or (claimed == actual_direction)\n"
    "magnitude_match = abs(abs(actual_pct) - claimed_pct) <= tolerance\n\n"
    "if direction_match and magnitude_match:  verdict = 'Supported'\n"
    "elif direction_match:                   verdict = 'Contradicted'  # magnitude off\n"
    "elif not direction_match:              verdict = 'Contradicted'  # wrong direction\n"
    "else:                                   verdict = 'Unverifiable'"
)
pdf.body(
    "The 5% tolerance acknowledges rounding, time-period mismatches, and minor definitional "
    "differences between what the PDF reports and what the CSV contains."
)


# ── 8. Session State Management ───────────────────────────────────────────────
pdf.add_page()
pdf.h1("8. Session State Management")
pdf.body(
    "Streamlit reruns the entire script on every user interaction. Session state (st.session_state) "
    "persists data across reruns within the same browser tab. The app stores:"
)

keys = [
    ("pdf_text", "Full extracted text string from the PDF."),
    ("pdf_pages", "List of page dicts [{page_number, text}, ...]"),
    ("embedding_model", "Loaded SentenceTransformer model (cached resource)"),
    ("pdf_chunks", "All text chunks after chunking"),
    ("faiss_index", "FAISS IndexFlatL2 object"),
    ("pdf_embeddings", "Numpy float32 array of all chunk embeddings"),
    ("df", "Pandas DataFrame of the uploaded CSV"),
    ("csv_profile", "Profile dict from profile_csv()"),
    ("pdf_q", "Current text in the AI Analyst question input"),
    ("csv_q", "Current text in the CSV Analyst question input"),
    ("smart_question", "Current text in the Smart Analyst input"),
    ("extracted_claims", "List of claim strings from auto-extraction"),
    ("verified_claims", "List of result dicts from Verify Claims page"),
    ("report_verified_claims", "Claim results for the Reports page"),
]
for key, desc in keys:
    pdf.bullet(f"st.session_state['{key}']   -   {desc}")

pdf.ln(3)
pdf.info_box(
    "Important",
    "All state is lost on page refresh. The app is intentionally stateless at the server  -  "
    "no database, no file system writes. Data lives only in the browser session.",
    color=AMBER,
)


# ── 9. Frontend & Styling ─────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("9. Frontend & Styling")

pdf.h2("Design System")
pdf.body(
    "All custom CSS is injected in a single st.markdown(..., unsafe_allow_html=True) block. "
    "The palette centres on indigo (#4f46e5) as primary with a dark sidebar (#111827). "
    "Typography uses Google Fonts: Playfair Display for headings (serif), DM Sans for body, "
    "and Bebas Neue for large metric numbers."
)

pdf.h2("Component Classes")
bullets2 = [
    ".dash-card / .metric-card  -  white rounded cards with subtle box-shadow and hover lift.",
    ".answer  -  left-indigo-border chat bubble for LLM answers.",
    ".status-pill  -  success/warning/danger coloured inline badges.",
    ".doc-status-card  -  flex card showing PDF+CSV load status with a key-takeaway panel.",
    ".insight-item  -  icon-dot + text rows in the Key Insights panel.",
    ".summary-table  -  custom HTML stats table replacing Streamlit's dark df.describe().",
    ".hero  -  gradient indigo hero banner used on inner pages.",
]
for b in bullets2:
    pdf.bullet(b)

pdf.h2("Plotly Theme")
pdf.body(
    "style_fig(fig) applies a white background, Inter font, hidden x-gridlines, subtle "
    "y-gridlines, and the CHART_PALETTE (sky blue, emerald, amber, rose, violet, ...). "
    "style_bars(fig) sets the primary sky-blue fill. make_sparkline() generates tiny inline "
    "SVG polylines for metric cards  -  no Plotly overhead for these micro-charts."
)

pdf.h2("Navigation")
pdf.body(
    "A st.radio() in the sidebar acts as the page router. CSS hides the native radio "
    "dots and applies active-state purple fill via :has(input:checked)  -  a modern CSS "
    "selector trick. Nav icons are injected with CSS ::before content pseudo-elements, "
    "one per :nth-child."
)


# ── 10. Interview Questions ───────────────────────────────────────────────────
pdf.add_page()
pdf.h1("10. Interview Questions & Model Answers")
pdf.body(
    "The following questions cover the technical depth an interviewer would probe "
    "for a role involving this project  -  spanning ML engineering, software design, "
    "Python proficiency, and system thinking."
)

pdf.h2("A. RAG & Embeddings")

qs = [
    (
        "What is RAG and why did you use it here instead of fine-tuning?",
        "RAG (Retrieval-Augmented Generation) retrieves relevant document chunks at query time "
        "and injects them into the LLM prompt as context. Fine-tuning would bake knowledge into "
        "model weights  -  expensive, slow, and unable to update when documents change. RAG is "
        "far cheaper, keeps the document as the ground truth, and allows page-level citations."
    ),
    (
        "Why choose all-MiniLM-L6-v2 over larger models like text-embedding-ada-002?",
        "MiniLM runs locally (no API cost), loads in under 2 seconds, and produces 384-dim "
        "vectors that are fast to index with FAISS. For short business text retrieval it "
        "performs comparably to ada-002 at a fraction of the cost. ada-002 would add latency "
        "and an OpenAI API call for every chunk and every question."
    ),
    (
        "What does IndexFlatL2 do and when would you switch to a different FAISS index?",
        "IndexFlatL2 computes exact Euclidean distance between the query vector and every "
        "stored vector  -  O(n) search. For small corpora (< 100k chunks) it is fast enough. "
        "For millions of chunks, IndexIVFFlat (inverted file with coarse quantisation) or "
        "IndexHNSW (hierarchical navigable small-world graph) would give sub-linear search "
        "at the cost of approximate results."
    ),
    (
        "How does chunk overlap help retrieval quality?",
        "Without overlap, a sentence at the boundary between two chunks is split: the first "
        "half lands in chunk N and the second half in chunk N+1. When the LLM reads only "
        "chunk N it gets an incomplete sentence. Overlap of 150 chars ensures the boundary "
        "sentence appears complete in at least one chunk, improving answer coherence."
    ),
    (
        "What is the relevance_label() function doing with the FAISS distance?",
        "FAISS returns raw L2 distance (lower = closer). We convert it to a rough cosine-like "
        "similarity with sim = 1 / (1 + distance). sim >= 0.45 is high relevance, >= 0.30 "
        "medium, else low. This gives users a qualitative label without exposing raw distances."
    ),
]
for i, (q, a) in enumerate(qs, 1):
    pdf.interview_q(i, q, a)

pdf.add_page()
pdf.h2("B. LLM & Prompt Engineering")

qs2 = [
    (
        "Why temperature=0.2 for the LLM call?",
        "Lower temperature produces more deterministic, focused responses. For factual "
        "retrieval over business documents we want consistent answers, not creative ones. "
        "0.2 allows slight variation while strongly favouring the most likely tokens."
    ),
    (
        "What happens if the API key is missing?",
        "get_openai_client() returns None. answer_pdf_question() detects this and returns "
        "an explanatory string instead of raising an exception. The UI checks if the answer "
        "starts with 'OpenAI API key not found' and shows a warning pill rather than crashing. "
        "This is a silent degradation pattern."
    ),
    (
        "How does the prompt prevent hallucination?",
        "The system prompt instructs the model to be a 'careful analyst who answers only "
        "from provided evidence'. The user prompt repeats this with an explicit rule: "
        "'Do not use outside knowledge.' A fallback phrase is specified for insufficient "
        "evidence. Temperature 0.2 further reduces invention."
    ),
    (
        "Could you replace gpt-4.1-mini with an open-source model?",
        "Yes. The answer_pdf_question function only requires a client with a chat.completions.create "
        "interface. You could swap in Ollama (llama3), a local vLLM server, or any OpenAI-compatible "
        "endpoint by changing the base_url in the OpenAI client constructor."
    ),
]
for i, (q, a) in enumerate(qs2, 6):
    pdf.interview_q(i, q, a)

pdf.add_page()
pdf.h2("C. Python & Software Design")

qs3 = [
    (
        "Why use @st.cache_resource for the embedding model?",
        "@st.cache_resource stores the return value across all sessions and reruns, sharing "
        "one model instance in memory. Without it, Streamlit would reload the 80 MB MiniLM "
        "model on every page interaction  -  adding 2-5 seconds of cold-start latency each time."
    ),
    (
        "Why is there no database? How would you add persistence?",
        "The app is intentionally stateless  -  designed for demo/single-user use. To add "
        "persistence you would: (1) write FAISS index to disk with index.write_index(), "
        "(2) store chunk metadata in SQLite, (3) replace session_state with a server-side "
        "store keyed by a user session ID (Redis or Postgres)."
    ),
    (
        "What is the purpose of find_matching_column() in csv_analyst.py?",
        "It maps a user's free-text question to an actual DataFrame column name using "
        "case-insensitive substring matching and underscore/hyphen normalisation. This lets "
        "a user type 'total arr' and match a column named 'ARR' or 'annual_arr'."
    ),
    (
        "How does the router.py keyword approach scale? What are its limitations?",
        "Keyword matching is O(n) over the keyword lists and works well for the ~30 "
        "hand-crafted terms. Limitations: brittle to paraphrasing ('what does the PDF say' "
        "vs 'as per the document'), no semantic understanding, requires manual maintenance "
        "as vocabulary grows. A better approach would be a small classifier or an LLM call "
        "to classify intent."
    ),
    (
        "How does the claim verification tolerance (±5%) work?",
        "After computing actual_pct = (last - first) / abs(first) * 100, we check if "
        "abs(abs(actual_pct) - claimed_pct) <= 5.0. This means a claim of '20% increase' "
        "is Supported if the actual change is anywhere from 15% to 25%. The tolerance "
        "handles rounding and minor definitional differences between PDF text and CSV data."
    ),
]
for i, (q, a) in enumerate(qs3, 10):
    pdf.interview_q(i, q, a)

pdf.add_page()
pdf.h2("D. Streamlit & Frontend")

qs4 = [
    (
        "How does the sidebar navigation work without multiple pages/files?",
        "A single st.radio() in the sidebar returns the selected page name. The main body "
        "uses if/elif blocks  -  one per page value  -  to render different content. This is "
        "the simplest pattern for small apps; Streamlit's multi-page app feature (pages/ "
        "directory) would be better for larger projects."
    ),
    (
        "What is st.html() vs st.markdown() for HTML rendering?",
        "st.markdown(..., unsafe_allow_html=True) renders markdown + HTML but can be "
        "interfered with by Streamlit's markdown parser. st.html() renders raw HTML "
        "directly with no markdown processing  -  safer for complex HTML structures "
        "like metric cards with nested divs and styles."
    ),
    (
        "How did you inject custom CSS without a separate stylesheet?",
        "A single st.markdown() block at the top of app.py injects a <style> tag with "
        "unsafe_allow_html=True. Streamlit appends this to the page's <head>. All custom "
        "classes are defined there and reused throughout the app via HTML strings."
    ),
    (
        "How do the suggested question buttons pre-fill the text input?",
        "Each button uses on_click=set_smart_question and args=(q,). The callback sets "
        "st.session_state['smart_question'] = q. Because the text_input widget uses "
        "key='smart_question', it reads from session_state and displays the new value "
        "on the next rerun triggered by the button click."
    ),
]
for i, (q, a) in enumerate(qs4, 15):
    pdf.interview_q(i, q, a)

pdf.add_page()
pdf.h2("E. Data Engineering & Analytics")

qs5 = [
    (
        "Why does the Dashboard auto-select the first numeric column for metrics?",
        "The app has no schema knowledge about the uploaded CSV. Using the first numeric "
        "column is a safe heuristic for demo purposes. In a production system you would "
        "ask the user to label columns (e.g., 'which column is revenue?') or infer intent "
        "from column names."
    ),
    (
        "What does profile_csv() return and where is it used?",
        "It returns a dict with rows, columns, column_names, column_types, missing_values, "
        "numeric_columns, and categorical_columns. It is used on the Upload page to display "
        "column type and missing value tables, and on other pages via get_numeric_and_categorical_columns()."
    ),
    (
        "How would you handle a very large CSV (e.g. 10 million rows)?",
        "pd.read_csv() loads everything into RAM  -  impractical at that scale. Options: "
        "(1) Use Polars instead of Pandas for lazy evaluation, (2) load with chunksize "
        "and aggregate, (3) store in DuckDB and query with SQL, (4) sample the CSV for "
        "display and compute aggregates on the fly."
    ),
    (
        "How does get_chart_categorical_cols() filter columns for charts?",
        "It excludes columns with > 50 unique values (high cardinality, e.g. contract_id, "
        "email). Such columns produce bar charts with hundreds of bars that are unreadable. "
        "The threshold of 50 is a pragmatic default."
    ),
]
for i, (q, a) in enumerate(qs5, 19):
    pdf.interview_q(i, q, a)

pdf.add_page()
pdf.h2("F. System Design & Extensions")

qs6 = [
    (
        "How would you add multi-user support?",
        "Replace st.session_state with a server-side session store (Redis keyed by UUID "
        "cookie). Store FAISS indexes and CSV DataFrames per user, with TTL expiry. Use "
        "Streamlit's authentication or a reverse proxy with login."
    ),
    (
        "How would you improve claim verification accuracy?",
        "Current approach: simple first-row to last-row % change. Improvements: (1) use "
        "the LLM to extract the time period from the claim and match it to the correct "
        "rows in the CSV, (2) use fuzzy string matching (rapidfuzz) for column matching, "
        "(3) allow multi-column aggregation."
    ),
    (
        "The router.py uses keyword lists. What ML alternative would you build?",
        "Train a three-class text classifier (pdf/csv/both) on ~500 labelled example "
        "questions using a fine-tuned distilbert or a zero-shot classifier via "
        "transformers pipeline. Alternatively, ask the LLM itself to classify intent "
        "with a short structured prompt  -  more robust but adds latency."
    ),
    (
        "How would you add streaming LLM responses to the AI Analyst page?",
        "Use the OpenAI streaming API: client.chat.completions.create(..., stream=True). "
        "In Streamlit, create a st.empty() placeholder, then iterate over the stream "
        "chunks appending text and calling placeholder.markdown(accumulated_text) on "
        "each chunk. This gives a typewriter effect."
    ),
    (
        "What security concerns exist with this app?",
        "1. The .env file with the OpenAI API key must never be committed to git. "
        "2. PDF uploads are processed server-side  -  a malicious PDF could exploit "
        "PyMuPDF vulnerabilities; pin to a patched version. "
        "3. unsafe_allow_html=True opens XSS risk if user-controlled text is rendered "
        "as HTML  -  currently only hardcoded strings are rendered this way. "
        "4. No authentication means the API key is shared among all visitors."
    ),
]
for i, (q, a) in enumerate(qs6, 23):
    pdf.interview_q(i, q, a)

# ── Final summary page ────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("Quick Reference Summary")

pdf.h2("Key Numbers to Remember")
numbers = [
    "800 chars  -  chunk size for PDF text splitting",
    "150 chars  -  overlap between adjacent chunks",
    "384 dims   -  MiniLM embedding vector size",
    "top-k = 4  -  PDF chunks retrieved per question",
    "±5%        -  claim verification tolerance",
    "8          -  max claims extracted from PDF per run",
    "0.2        -  LLM temperature for factual answers",
    "50          -  max unique values for chart category columns",
]
for n in numbers:
    pdf.bullet(n)

pdf.h2("One-liner Summaries per Module")
modules = [
    ("pdf_utils.py",    "PyMuPDF -> raw text per page -> (full_text, pages[])"),
    ("rag_utils.py",    "chunk pages -> encode with MiniLM -> FAISS index -> top-k retrieval"),
    ("llm_utils.py",    "OpenAI client -> prompt builder -> GPT-4.1-mini -> cited answer"),
    ("csv_utils.py",    "pd.read_csv -> dtype/missing profile dict"),
    ("csv_analyst.py",  "keyword if/elif on question -> Pandas operation -> (answer, fig)"),
    ("router.py",       "keyword lists -> 'pdf' | 'csv' | 'both'"),
    ("claim_checker.py","regex extract claims -> column match -> first/last % change -> verdict"),
]
for mod, desc in modules:
    pdf.h3(mod)
    pdf.body(desc)

pdf.info_box(
    "Good luck in your interview!",
    "This project demonstrates: RAG pipeline design, local embedding + vector search, "
    "LLM prompt engineering, Pandas analytics, Streamlit app architecture, and "
    "a practical claim-verification engine. Focus on the WHY behind each design choice.",
    color=GREEN,
)

pdf.output(PDF_OUT)
print(f"PDF written to {PDF_OUT}")
