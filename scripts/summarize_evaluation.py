#!/usr/bin/env python3
"""Create a compact Markdown summary of current evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict[str, Any] | None:
    full_path = ROOT / path
    if not full_path.exists():
        return None
    with full_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}"


def num(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def line(columns: list[str]) -> str:
    return "| " + " | ".join(columns) + " |"


def metric(path: str, key: str) -> Any:
    metrics = load(path)
    if not metrics:
        return None
    value: Any = metrics
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def final_ablation_table() -> list[str]:
    rows = [
        (
            "Baseline",
            "BM25 top-5 + FSMODQA",
            "data/squad_tr_dev_bm25_answerable_retrieval_metrics.json",
            "recall@5",
            "5",
            "checkpoints/fsmodqa_squad_tr_bm25_top5_answerable_200/expanded_metrics.json",
            "Main fixed-retrieval baseline.",
        ),
        (
            "More Contexts",
            "BM25 top-25 + FSMODQA",
            "data/squad_tr_dev_bm25_answerable_retrieval_metrics.json",
            "recall@25",
            "25",
            "checkpoints/fsmodqa_squad_tr_bm25_top25_answerable_200/expanded_metrics.json",
            "More evidence improves EM, but not F1.",
        ),
        (
            "Adaptive Retrieval",
            "Retrieval-aware 50/50/100 + FSMODQA",
            "data/squad_tr_dev_bm25_adaptive_retrieval_aware_50_50_100.metrics.json",
            "recall@100",
            "76.86 avg",
            "checkpoints/fsmodqa_squad_tr_adaptive_retrieval_aware_50_50_100_200/expanded_metrics.json",
            "Best tested reader F1 with fewer contexts than fixed top-100.",
        ),
        (
            "Question Rewriting",
            "Keyword rewrite top-5 + FSMODQA",
            "data/squad_tr_dev_bm25_rewritten_keyword_retrieval_metrics.json",
            "recall@5",
            "5",
            "checkpoints/fsmodqa_squad_tr_keyword_rewrite_top5_answerable_200/expanded_metrics.json",
            "Small retrieval gain does not transfer to reader quality.",
        ),
        (
            "Selective Rewriting",
            "Selective keyword rewrite top-5 + FSMODQA",
            "data/squad_tr_dev_bm25_selective_normscore_tuned_retrieval_metrics.json",
            "recall@5",
            "5",
            "checkpoints/fsmodqa_squad_tr_selective_rewrite_top5_answerable_200/expanded_metrics.json",
            "Selection helps slightly over pure rewrite, but still below baseline.",
        ),
        (
            "Knowledge Selector",
            "TF-IDF selector hybrid top-5 + FSMODQA",
            "data/squad_tr_dev_bm25_selector_hybrid05_top5_metrics.json",
            "recall@5",
            "5",
            "checkpoints/fsmodqa_squad_tr_selector_hybrid05_top5_answerable_200/expanded_metrics.json",
            "Close to baseline, but no reader gain.",
        ),
        (
            "Neural Selector",
            "BERTurk selector hybrid top-5 + FSMODQA",
            "data/squad_tr_dev_bm25_berturk_selector_medium_hybrid05_top5_200_metrics.json",
            "recall@5",
            "5",
            "checkpoints/fsmodqa_squad_tr_berturk_selector_medium_hybrid05_top5_200/expanded_metrics.json",
            "Improves held-out selector MRR, but not reader F1 yet.",
        ),
        (
            "Oracle Upper Bound",
            "Oracle top-5 + FSMODQA",
            "",
            "",
            "5",
            "checkpoints/fsmodqa_squad_tr_oracle_top5_answerable_200/expanded_metrics.json",
            "Not deployable; shows retrieval still bottlenecks QA.",
        ),
    ]
    out = [
        "## Final Ablation Summary",
        "",
        "Reader metrics are 200-example answerable FSMODQA smoke checks. Retrieval metrics are full-dev unless the setting name says otherwise.",
        "",
        line(
            [
                "Ablation",
                "System",
                "Retrieval Ref. (R@K)",
                "Contexts",
                "EM",
                "F1",
                "Correctness",
                "Faithfulness",
                "Takeaway",
            ]
        ),
        line(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
    ]
    for ablation, system, retrieval_path, retrieval_key, contexts, reader_path, takeaway in rows:
        retrieval_ref = "-"
        if retrieval_path and retrieval_key:
            retrieval_ref = pct(metric(retrieval_path, retrieval_key))
        out.append(
            line(
                [
                    ablation,
                    system,
                    retrieval_ref,
                    contexts,
                    pct(metric(reader_path, "em")),
                    pct(metric(reader_path, "f1")),
                    pct(metric(reader_path, "correctness_recall")),
                    pct(metric(reader_path, "faithfulness_k_precision")),
                    takeaway,
                ]
            )
        )
    return out


def retrieval_table() -> list[str]:
    rows = [
        ("BM25 top-100", "data/squad_tr_dev_bm25_answerable_retrieval_metrics.json"),
        ("Adaptive oracle 10/50/100", "data/squad_tr_dev_bm25_adaptive_oracle.metrics.json"),
        ("Adaptive question-only 25/50/100", "data/squad_tr_dev_bm25_adaptive_classifier_conservative.metrics.json"),
        ("Adaptive retrieval-aware 25/50/100", "data/squad_tr_dev_bm25_adaptive_retrieval_aware_25_50_100.metrics.json"),
        ("Adaptive retrieval-aware 50/50/100", "data/squad_tr_dev_bm25_adaptive_retrieval_aware_50_50_100.metrics.json"),
        ("TF-IDF selector hybrid top-5", "data/squad_tr_dev_bm25_selector_hybrid05_top5_metrics.json"),
        ("TF-IDF selector hybrid top-25", "data/squad_tr_dev_bm25_selector_hybrid05_top25_metrics.json"),
        ("BERTurk selector smoke top-5/50ex", "data/squad_tr_dev_bm25_berturk_selector_smoke_top5_50_metrics.json"),
        ("BERTurk selector medium hybrid top-5/200ex", "data/squad_tr_dev_bm25_berturk_selector_medium_hybrid05_top5_200_metrics.json"),
    ]
    out = [
        "## Retrieval And Selection",
        "",
        line(["Setting", "Examples", "R@1", "R@5", "R@10", "R@25", "R@50", "R@100", "MRR", "Avg ctx", "Saving"]),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for name, path in rows:
        metrics = load(path)
        if not metrics:
            continue
        out.append(
            line(
                [
                    name,
                    str(metrics.get("examples", "-")),
                    pct(metrics.get("recall@1")),
                    pct(metrics.get("recall@5")),
                    pct(metrics.get("recall@10")),
                    pct(metrics.get("recall@25")),
                    pct(metrics.get("recall@50")),
                    pct(metrics.get("recall@100")),
                    pct(metrics.get("mrr")),
                    num(metrics.get("avg_contexts")),
                    pct(metrics.get("context_saving_vs_100")),
                ]
            )
        )
    return out


def rewriting_table() -> list[str]:
    rows = [
        ("BM25 original", "data/squad_tr_dev_bm25_answerable_retrieval_metrics.json"),
        ("Rule rewrite: broad", "data/squad_tr_dev_bm25_rewritten_all_retrieval_metrics.json"),
        ("Rule rewrite: keyword", "data/squad_tr_dev_bm25_rewritten_keyword_retrieval_metrics.json"),
        ("Selective keyword rewrite", "data/squad_tr_dev_bm25_selective_normscore_tuned_retrieval_metrics.json"),
    ]
    out = [
        "## Question Rewriting",
        "",
        "These are retrieval-only full-dev checks; 200-example reader checks are reported in the Reader / QA table.",
        "",
        line(["Setting", "Examples", "R@1", "R@5", "R@10", "R@20", "R@25", "R@50", "R@100", "MRR"]),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for name, path in rows:
        metrics = load(path)
        if not metrics:
            continue
        out.append(
            line(
                [
                    name,
                    str(metrics.get("examples", "-")),
                    pct(metrics.get("recall@1")),
                    pct(metrics.get("recall@5")),
                    pct(metrics.get("recall@10")),
                    pct(metrics.get("recall@20")),
                    pct(metrics.get("recall@25")),
                    pct(metrics.get("recall@50")),
                    pct(metrics.get("recall@100")),
                    pct(metrics.get("mrr")),
                ]
            )
        )
    return out


def classifier_table() -> list[str]:
    rows = [
        ("Question-only adaptive classifier", "checkpoints/complexity_classifier_tfidf_lr/model.metrics.json"),
        ("Retrieval-aware adaptive classifier", "checkpoints/complexity_classifier_retrieval_aware_tfidf_lr/metrics.json"),
    ]
    out = [
        "## Adaptive Classifier",
        "",
        line(["Setting", "Accuracy", "Macro F1", "Weighted F1", "Easy F1", "Medium F1", "Hard F1"]),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for name, path in rows:
        metrics = load(path)
        if not metrics:
            continue
        out.append(
            line(
                [
                    name,
                    pct(metrics.get("accuracy")),
                    pct(metrics.get("macro avg", {}).get("f1-score")),
                    pct(metrics.get("weighted avg", {}).get("f1-score")),
                    pct(metrics.get("easy", {}).get("f1-score")),
                    pct(metrics.get("medium", {}).get("f1-score")),
                    pct(metrics.get("hard", {}).get("f1-score")),
                ]
            )
        )
    return out


def reader_table() -> list[str]:
    rows = [
        ("BM25 top-5", "checkpoints/fsmodqa_squad_tr_bm25_top5_answerable_200/expanded_metrics.json"),
        ("BM25 top-25", "checkpoints/fsmodqa_squad_tr_bm25_top25_answerable_200/expanded_metrics.json"),
        ("Keyword rewrite top-5", "checkpoints/fsmodqa_squad_tr_keyword_rewrite_top5_answerable_200/expanded_metrics.json"),
        ("Selective keyword rewrite top-5", "checkpoints/fsmodqa_squad_tr_selective_rewrite_top5_answerable_200/expanded_metrics.json"),
        ("Adaptive retrieval-aware 50/50/100", "checkpoints/fsmodqa_squad_tr_adaptive_retrieval_aware_50_50_100_200/expanded_metrics.json"),
        ("Oracle top-5", "checkpoints/fsmodqa_squad_tr_oracle_top5_answerable_200/expanded_metrics.json"),
        ("TF-IDF selector hybrid top-5", "checkpoints/fsmodqa_squad_tr_selector_hybrid05_top5_answerable_200/expanded_metrics.json"),
        ("BERTurk selector medium hybrid top-5", "checkpoints/fsmodqa_squad_tr_berturk_selector_medium_hybrid05_top5_200/expanded_metrics.json"),
    ]
    out = [
        "## Reader / QA",
        "",
        "These are 200-example FSMODQA reader checks, so they are smoke/evidence runs rather than final full-dev scores.",
        "",
        line(
            [
                "Reader Input",
                "Examples",
                "EM",
                "F1",
                "Correctness Recall",
                "Faithfulness K-Prec",
                "Evidence Contains Pred",
                "Refusal",
                "Contains Gold",
            ]
        ),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for name, path in rows:
        metrics = load(path)
        if not metrics:
            continue
        out.append(
            line(
                [
                    name,
                    str(metrics.get("num_examples", "-")),
                    pct(metrics.get("em")),
                    pct(metrics.get("f1")),
                    pct(metrics.get("correctness_recall", metrics.get("recall"))),
                    pct(metrics.get("faithfulness_k_precision", metrics.get("k_precision"))),
                    pct(metrics.get("evidence_contains_prediction")),
                    pct(metrics.get("refusal_rate")),
                    pct(metrics.get("contains_gold_answer")),
                ]
            )
        )
    return out


def selector_table() -> list[str]:
    metrics = load("checkpoints/knowledge_selector_tfidf_lr/metrics.json")
    if not metrics:
        return []
    rows = [
        ("BM25", metrics.get("retrieval_bm25") or metrics.get("bm25")),
        ("TF-IDF selector only", metrics.get("retrieval_selector") or metrics.get("selector")),
        ("TF-IDF selector/BM25 hybrid", metrics.get("retrieval_hybrid") or metrics.get("hybrid")),
    ]
    out = [
        "## Knowledge Selector Held-Out Split",
        "",
        line(["Setting", "R@1", "R@5", "R@10", "R@25", "R@50", "R@100", "MRR"]),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for name, row in rows:
        if not row:
            continue
        out.append(
            line(
                [
                    name,
                    pct(row.get("recall@1")),
                    pct(row.get("recall@5")),
                    pct(row.get("recall@10")),
                    pct(row.get("recall@25")),
                    pct(row.get("recall@50")),
                    pct(row.get("recall@100")),
                    pct(row.get("mrr")),
                ]
            )
        )
    return out


def neural_selector_table() -> list[str]:
    rows = [
        ("BM25 held-out", "bm25"),
        ("BERTurk selector only", "selector"),
        ("Hybrid weight 0.5", "hybrid_selector_weight_0.50"),
    ]
    metrics = load("checkpoints/knowledge_selector_berturk_medium/metrics.json")
    if not metrics:
        return []
    out = [
        "## Neural Knowledge Selector Medium Run",
        "",
        line(["Setting", "R@1", "R@5", "R@10", "R@25", "R@50", "R@100", "MRR"]),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for name, key in rows:
        row = metrics.get(key)
        if not row:
            continue
        out.append(
            line(
                [
                    name,
                    pct(row.get("recall@1")),
                    pct(row.get("recall@5")),
                    pct(row.get("recall@10")),
                    pct(row.get("recall@25")),
                    pct(row.get("recall@50")),
                    pct(row.get("recall@100")),
                    pct(row.get("mrr")),
                ]
            )
        )
    return out


def refusal_table() -> list[str]:
    rows = [
        ("BM25 top-5 mixed 50 answerable / 50 unanswerable", "checkpoints/fsmodqa_squad_tr_bm25_mixed_100_top5/expanded_metrics.json"),
    ]
    out = [
        "## Refusal / Unanswerable QA",
        "",
        line(
            [
                "Setting",
                "Examples",
                "Answerable",
                "Unanswerable",
                "Answerable F1",
                "Faithfulness K-Prec",
                "Refusal Rate",
                "Correct Refusal",
                "Answered Unanswerable",
            ]
        ),
        line(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for name, path in rows:
        metrics = load(path)
        if not metrics:
            continue
        out.append(
            line(
                [
                    name,
                    str(metrics.get("num_examples", "-")),
                    str(metrics.get("answerable_examples", "-")),
                    str(metrics.get("unanswerable_examples", "-")),
                    pct(metrics.get("f1")),
                    pct(metrics.get("faithfulness_k_precision")),
                    pct(metrics.get("refusal_rate")),
                    pct(metrics.get("correct_refusal_rate")),
                    pct(metrics.get("answered_unanswerable_rate")),
                ]
            )
        )
    return out


def main() -> None:
    output = ROOT / "evaluation_summary.md"
    sections = [
        "# Evaluation Summary",
        "",
        "This file is generated from the current JSON metric files by `scripts/summarize_evaluation.py`.",
        "",
        *final_ablation_table(),
        "",
        *retrieval_table(),
        "",
        *rewriting_table(),
        "",
        *classifier_table(),
        "",
        *reader_table(),
        "",
        *refusal_table(),
        "",
        *selector_table(),
        "",
        *neural_selector_table(),
        "",
        "## Current Reading",
        "",
        "- BM25 top-100 is still the main full-dev retrieval baseline.",
        "- Oracle adaptive retrieval shows the efficiency upper bound: same retrieval recall with fewer contexts.",
        "- Retrieval-aware adaptive classification improves classifier F1, but still loses some Recall@100 compared with fixed top-100.",
        "- Keyword question rewriting gives a small retrieval gain at higher K, but it hurts the 200-example reader F1 in the current top-5 setup.",
        "- Reader/QA F1 has been checked on 200-example smoke runs, including adaptive retrieval-aware 50/50/100.",
        "- The current FSMODQA reader does not refuse on the mixed unanswerable smoke set; it answers all unanswerable questions.",
        "- The medium BERTurk selector improves held-out selector MRR but did not improve 200-example reader F1 in its current top-5 setup.",
        "",
    ]
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
