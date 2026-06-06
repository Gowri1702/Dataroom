
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.router import route_question
from src.claim_checker import verify_claim_against_csv
from evals.metrics import (
    router_accuracy,
    retrieval_accuracy_at_k,
    citation_correctness,
    claim_verification_accuracy,
    aggregate_report,
)

DATASET = Path(__file__).parent / "dataset.json"


def load_dataset() -> dict:
    with open(DATASET) as f:
        return json.load(f)


def run_router_eval(data: dict) -> list[dict]:
    results = []
    for item in data["router_eval"]:
        actual = route_question(item["question"])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_route": item["expected_route"],
            "actual_route": actual,
            "correct": actual == item["expected_route"],
        })
    return results


def run_claim_eval(data: dict) -> list[dict]:
    results = []
    for item in data["claim_eval"]:
        df = pd.DataFrame(item["csv_data"])
        result = verify_claim_against_csv(item["claim"], df)
        results.append({
            "id": item["id"],
            "claim": item["claim"],
            "expected_verdict": item["expected_verdict"],
            "actual_verdict": result["verdict"],
            "actual_percent": result.get("actual_percent"),
            "matched_col": result.get("csv_metric"),
            "correct": result["verdict"] == item["expected_verdict"],
            "note": item.get("note", ""),
        })
    return results


def run_retrieval_eval(data: dict, chunks, index, model) -> list[dict]:
    from src.rag_utils import retrieve_relevant_chunks
    results = []
    for item in data["retrieval_eval"]:
        retrieved = retrieve_relevant_chunks(
            question=item["question"],
            chunks=chunks,
            index=index,
            model=model,
            top_k=4,
        )
        texts = [r["text"] for r in retrieved]
        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_keywords": item["expected_keywords"],
            "retrieved_texts": texts,
        })
    return results


def run_citation_eval(data: dict, chunks, index, model) -> list[dict]:
    from src.rag_utils import retrieve_relevant_chunks
    from src.llm_utils import answer_pdf_question
    results = []
    for item in data["retrieval_eval"]:
        retrieved = retrieve_relevant_chunks(
            question=item["question"],
            chunks=chunks,
            index=index,
            model=model,
            top_k=4,
        )
        answer = answer_pdf_question(item["question"], retrieved)
        results.append({"id": item["id"], "answer": answer})
    return results


def _print_section(title: str, rows: list[dict], cols: list[str]) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    col_w = 14
    header = "  ".join(str(c).ljust(col_w) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, bool):
                v = "PASS" if v else "FAIL"
            elif isinstance(v, float):
                v = f"{v:.2f}"
            vals.append(str(v)[:col_w].ljust(col_w))
        print("  ".join(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="Run retrieval and citation evals (requires a loaded PDF in FAISS)",
    )
    parser.add_argument(
        "--pdf", type=str, default=None,
        help="Path to a PDF file to use for retrieval/citation evals",
    )
    args = parser.parse_args()

    data = load_dataset()
    all_results: dict = {}

    print("\nRunning router eval...")
    router_results = run_router_eval(data)
    all_results["router"] = router_results
    router_metrics = router_accuracy(router_results)
    _print_section(
        "Router Eval",
        router_results,
        ["id", "expected_route", "actual_route", "correct"],
    )

    print("\nRunning claim verification eval...")
    claim_results = run_claim_eval(data)
    all_results["claims"] = claim_results
    claim_metrics = claim_verification_accuracy(claim_results)
    _print_section(
        "Claim Verification Eval",
        claim_results,
        ["id", "expected_verdict", "actual_verdict", "correct", "note"],
    )

    retrieval_metrics = {"accuracy_at_k": 0.0, "hits": 0, "total": 0}
    citation_metrics  = {"accuracy": 0.0, "cited": 0, "total": 0}

    if args.full:
        if not args.pdf:
            print("\n[SKIP] --full requires --pdf <path_to_pdf>")
        else:
            from src.rag_utils import load_embedding_model, chunk_pdf_pages, create_faiss_index
            import fitz

            print(f"\nLoading PDF: {args.pdf}")
            doc = fitz.open(args.pdf)
            pages = [
                {"page_number": i + 1, "text": page.get_text()}
                for i, page in enumerate(doc)
            ]
            model  = load_embedding_model()
            chunks = chunk_pdf_pages(pages)
            index, _ = create_faiss_index(chunks, model)

            print("Running retrieval eval...")
            retrieval_results = run_retrieval_eval(data, chunks, index, model)
            all_results["retrieval"] = retrieval_results
            retrieval_metrics = retrieval_accuracy_at_k(retrieval_results)

            print("Running citation eval (calls OpenAI)...")
            citation_results = run_citation_eval(data, chunks, index, model)
            all_results["citation"] = citation_results
            citation_metrics = citation_correctness(citation_results)

    print("\n" + aggregate_report(
        router_metrics,
        retrieval_metrics,
        citation_metrics,
        claim_metrics,
    ))

    out_path = Path(__file__).parent / "results_latest.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "router_metrics": router_metrics,
                "retrieval_metrics": retrieval_metrics,
                "citation_metrics": citation_metrics,
                "claim_metrics": claim_metrics,
                "raw": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
