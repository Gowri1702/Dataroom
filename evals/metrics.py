
from __future__ import annotations
from typing import List, Dict, Any


def router_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "by_class": {}}

    correct = sum(1 for r in results if r["actual_route"] == r["expected_route"])

    by_class: Dict[str, Dict[str, int]] = {}
    for r in results:
        cls = r["expected_route"]
        if cls not in by_class:
            by_class[cls] = {"correct": 0, "total": 0}
        by_class[cls]["total"] += 1
        if r["actual_route"] == r["expected_route"]:
            by_class[cls]["correct"] += 1

    by_class_acc = {
        cls: v["correct"] / v["total"] for cls, v in by_class.items()
    }

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "by_class": by_class_acc,
    }


def retrieval_accuracy_at_k(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(results)
    if total == 0:
        return {"accuracy_at_k": 0.0, "hits": 0, "total": 0}

    hits = 0
    for r in results:
        combined = " ".join(r.get("retrieved_texts", [])).lower()
        keywords = [kw.lower() for kw in r.get("expected_keywords", [])]
        if any(kw in combined for kw in keywords):
            hits += 1

    return {
        "accuracy_at_k": hits / total,
        "hits": hits,
        "total": total,
    }


def citation_correctness(results: List[Dict[str, Any]]) -> Dict[str, float]:
    import re
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "cited": 0, "total": 0}

    pattern = re.compile(r"\(page\s+\d+\)", re.IGNORECASE)
    cited = sum(1 for r in results if pattern.search(r.get("answer", "")))
    return {
        "accuracy": cited / total,
        "cited": cited,
        "total": total,
    }


def claim_verification_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "by_verdict": {}}

    correct = sum(
        1 for r in results if r["actual_verdict"] == r["expected_verdict"]
    )

    by_verdict: Dict[str, Dict[str, int]] = {}
    for r in results:
        v = r["expected_verdict"]
        if v not in by_verdict:
            by_verdict[v] = {"correct": 0, "total": 0}
        by_verdict[v]["total"] += 1
        if r["actual_verdict"] == r["expected_verdict"]:
            by_verdict[v]["correct"] += 1

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "by_verdict": {
            v: d["correct"] / d["total"] for v, d in by_verdict.items()
        },
    }


def aggregate_report(
    router_res: Dict[str, Any],
    retrieval_res: Dict[str, Any],
    citation_res: Dict[str, Any],
    claim_res: Dict[str, Any],
) -> str:
    lines = [
        "=" * 60,
        "  DATAROOM AI -- EVALUATION REPORT",
        "=" * 60,
        "",
        f"Router accuracy          : {router_res['accuracy']:.1%}  "
        f"({router_res['correct']}/{router_res['total']})",
    ]
    for cls, acc in router_res.get("by_class", {}).items():
        lines.append(f"  [{cls}] accuracy        : {acc:.1%}")

    lines += [
        "",
        f"Retrieval accuracy@k     : {retrieval_res['accuracy_at_k']:.1%}  "
        f"({retrieval_res['hits']}/{retrieval_res['total']})",
        "",
        f"Citation correctness     : {citation_res['accuracy']:.1%}  "
        f"({citation_res['cited']}/{citation_res['total']})",
        "",
        f"Claim verif. accuracy    : {claim_res['accuracy']:.1%}  "
        f"({claim_res['correct']}/{claim_res['total']})",
    ]
    for v, acc in claim_res.get("by_verdict", {}).items():
        lines.append(f"  [{v}] accuracy         : {acc:.1%}")

    lines += ["", "=" * 60]
    return "\n".join(lines)
