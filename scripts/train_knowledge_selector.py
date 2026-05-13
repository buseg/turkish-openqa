#!/usr/bin/env python3
"""Train a lightweight supervised knowledge selector/reranker.

The selector learns from BM25 top-k passage labels. A passage is positive when
it contains a gold answer string. This is a practical supervised baseline for
the proposal's knowledge selector before attempting neural or RL training.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL from build_knowledge_selector_data.py.")
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ks", default="1,5,10,25,50,100")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("id"))


def row_rank(row: dict[str, Any]) -> int:
    return int(row.get("rank") or row.get("ctx_rank") or 10**9)


def row_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", row.get("ctx_score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def row_text(row: dict[str, Any]) -> str:
    question = row.get("question", "")
    title = row.get("title", row.get("ctx_title", ""))
    text = row.get("text", "")
    return f"soru: {question} baslik: {title} metin: {text}"


def split_qids(rows: list[dict[str, Any]], test_size: float, seed: int) -> tuple[set[str], set[str]]:
    qids = sorted({row_qid(row) for row in rows})
    random.Random(seed).shuffle(qids)
    n_eval = max(1, int(len(qids) * test_size))
    eval_qids = set(qids[:n_eval])
    train_qids = set(qids[n_eval:])
    return train_qids, eval_qids


def first_positive_rank(rows: list[dict[str, Any]], key: str) -> int | None:
    sorted_rows = sorted(rows, key=lambda row: row[key])
    for index, row in enumerate(sorted_rows, start=1):
        if int(row.get("label", 0)) == 1:
            return index
    return None


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row_qid(row), []).append(row)
    return groups


def retrieval_metrics(rows: list[dict[str, Any]], ks: list[int], rank_key: str) -> dict[str, Any]:
    ranks = [first_positive_rank(group, rank_key) for group in grouped(rows).values()]
    found = [rank for rank in ranks if rank is not None]
    metrics: dict[str, Any] = {
        "examples": len(ranks),
        "found_any": len(found),
        "mrr": mean((1.0 / rank if rank is not None else 0.0) for rank in ranks) if ranks else 0.0,
        "mean_first_hit_rank": mean(found) if found else None,
    }
    for k in ks:
        metrics[f"recall@{k}"] = mean((rank is not None and rank <= k) for rank in ranks) if ranks else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    ks = [int(item) for item in args.ks.split(",") if item]
    rows = read_jsonl(args.input)
    train_qids, eval_qids = split_qids(rows, args.test_size, args.seed)
    train_rows = [row for row in rows if row_qid(row) in train_qids]
    eval_rows = [row for row in rows if row_qid(row) in eval_qids]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise SystemExit("Install scikit-learn first: pip install scikit-learn") from exc

    model = make_pipeline(
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=args.min_df,
            max_features=args.max_features,
            sublinear_tf=True,
        ),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
    )
    x_train = [row_text(row) for row in train_rows]
    y_train = [int(row.get("label", 0)) for row in train_rows]
    x_eval = [row_text(row) for row in eval_rows]
    y_eval = [int(row.get("label", 0)) for row in eval_rows]
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_eval)[:, 1]
    predictions = [int(probability >= 0.5) for probability in probabilities]
    for row, probability in zip(eval_rows, probabilities):
        row["selector_score"] = float(probability)
        row["selector_rank_key"] = -float(probability)
        row["bm25_rank_key"] = row_rank(row)
        row["hybrid_rank_key"] = -(float(probability) * 0.8 + (1.0 / max(1, row_rank(row))) * 0.2)

    report = classification_report(y_eval, predictions, output_dict=True, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_eval, probabilities)
    except ValueError:
        roc_auc = None
    try:
        average_precision = average_precision_score(y_eval, probabilities)
    except ValueError:
        average_precision = None

    metrics = {
        "input": str(args.input),
        "train_questions": len(train_qids),
        "eval_questions": len(eval_qids),
        "train_passages": len(train_rows),
        "eval_passages": len(eval_rows),
        "positive_train_passages": sum(y_train),
        "positive_eval_passages": sum(y_eval),
        "classification_report": report,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "retrieval_bm25": retrieval_metrics(eval_rows, ks, "bm25_rank_key"),
        "retrieval_selector": retrieval_metrics(eval_rows, ks, "selector_rank_key"),
        "retrieval_hybrid": retrieval_metrics(eval_rows, ks, "hybrid_rank_key"),
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    with args.model_output.open("wb") as handle:
        pickle.dump(model, handle)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Train questions: {metrics['train_questions']}")
    print(f"Eval questions: {metrics['eval_questions']}")
    print(f"Train passages: {metrics['train_passages']}")
    print(f"Eval passages: {metrics['eval_passages']}")
    print(f"Positive eval passages: {metrics['positive_eval_passages']}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Average precision: {average_precision}")
    print("BM25 retrieval:", metrics["retrieval_bm25"])
    print("Selector retrieval:", metrics["retrieval_selector"])
    print("Hybrid retrieval:", metrics["retrieval_hybrid"])
    print(f"Model: {args.model_output}")
    print(f"Metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
