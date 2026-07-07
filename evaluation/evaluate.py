from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.logging_config import configure_logging, get_logger  # noqa: E402
from backend.modules.retrieval import retrieve  # noqa: E402

logger = get_logger(__name__)

DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"
REPORT_PATH = Path(__file__).resolve().parent / "evaluation_report.md"


def _keyword_hit(keywords: List[str], chunks: List[Dict]) -> bool:
    
    if not keywords:
        return True
    blob = " ".join(c["text"].lower() for c in chunks)
    return any(kw.lower() in blob for kw in keywords)


def evaluate(dataset_path: Path, k: int = 5) -> Dict:
    
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = data["questions"]

    n = len(questions)
    top1_hits = 0
    top3_hits = 0
    recall_sum = 0.0
    keyword_hits = 0
    per_question: List[Dict] = []

    for item in questions:
        relevant = set(item["relevant_sources"])
        results = retrieve(item["question"], top_k=k, use_reranker=False)
        sources_in_order = [r["source"] for r in results]

        top1 = bool(sources_in_order[:1]) and sources_in_order[0] in relevant
        top3 = any(s in relevant for s in sources_in_order[:3])
        found = relevant.intersection(sources_in_order[:k])
        recall = len(found) / len(relevant) if relevant else 0.0
        kw = _keyword_hit(item.get("keywords", []), results)

        top1_hits += int(top1)
        top3_hits += int(top3)
        recall_sum += recall
        keyword_hits += int(kw)

        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "top1": top1,
                "top3": top3,
                "recall@5": round(recall, 3),
                "keyword_hit": kw,
                "retrieved_sources": sources_in_order,
                "top_similarity": round(results[0]["similarity"], 3) if results else 0.0,
            }
        )

    return {
        "num_questions": n,
        "top1_accuracy": round(top1_hits / n, 4) if n else 0.0,
        "top3_accuracy": round(top3_hits / n, 4) if n else 0.0,
        "recall_at_5": round(recall_sum / n, 4) if n else 0.0,
        "keyword_hit_rate": round(keyword_hits / n, 4) if n else 0.0,
        "k": k,
        "per_question": per_question,
    }


def render_report(metrics: Dict) -> str:

    lines: List[str] = []
    lines.append("# Retrieval Evaluation Report\n")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append(f"- Questions evaluated: **{metrics['num_questions']}**")
    lines.append(f"- k (results per query): **{metrics['k']}**\n")
    lines.append("## Summary metrics\n")
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    lines.append(f"| Top-1 Retrieval Accuracy | {metrics['top1_accuracy']:.2%} |")
    lines.append(f"| Top-3 Retrieval Accuracy | {metrics['top3_accuracy']:.2%} |")
    lines.append(f"| Recall@5 | {metrics['recall_at_5']:.2%} |")
    lines.append(f"| Keyword hit-rate | {metrics['keyword_hit_rate']:.2%} |\n")
    lines.append("## Per-question results\n")
    lines.append("| ID | Top-1 | Top-3 | Recall@5 | Keyword | Top sim | Question |")
    lines.append("|---|---|---|---|---|---|---|")
    for q in metrics["per_question"]:
        lines.append(
            f"| {q['id']} | {'✓' if q['top1'] else '✗'} | "
            f"{'✓' if q['top3'] else '✗'} | {q['recall@5']:.2f} | "
            f"{'✓' if q['keyword_hit'] else '✗'} | {q['top_similarity']:.2f} | "
            f"{q['question']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    configure_logging()
    metrics = evaluate(Path(args.dataset), k=args.k)

    print("\n===== Retrieval Evaluation =====")
    print(f"Questions      : {metrics['num_questions']}")
    print(f"Top-1 Accuracy : {metrics['top1_accuracy']:.2%}")
    print(f"Top-3 Accuracy : {metrics['top3_accuracy']:.2%}")
    print(f"Recall@5       : {metrics['recall_at_5']:.2%}")
    print(f"Keyword hits   : {metrics['keyword_hit_rate']:.2%}")

    report = render_report(metrics)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")

    (REPORT_PATH.with_suffix(".json")).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
