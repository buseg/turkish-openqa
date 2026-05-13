#!/usr/bin/env python3
"""Train a retrieval-aware adaptive retrieval classifier.

The question-only classifier sees only the user question. This model also sees
BM25 score-shape and token-overlap features from the retrieved top candidates,
which makes the easy/medium/hard decision more realistic.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import string
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieved", required=True, type=Path, help="Retrieved JSON with ctxs.")
    parser.add_argument("--labels", required=True, type=Path, help="Complexity labels JSONL.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def normalize(text: Any) -> str:
    text = str(text).casefold()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def tokens(text: Any) -> set[str]:
    return set(re.findall(r"[\wçğıöşü]+", normalize(text), flags=re.UNICODE))


def score(ctx: dict[str, Any]) -> float:
    try:
        return float(ctx.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def score_values(example: dict[str, Any], n: int = 10) -> list[float]:
    values = [score(ctx) for ctx in example.get("ctxs", [])[:n]]
    if not values:
        return [0.0]
    return values


def safe_ratio(num: float, den: float) -> float:
    return num / den if abs(den) > 1e-9 else 0.0


def numeric_features(example: dict[str, Any]) -> dict[str, float]:
    ctxs = example.get("ctxs", [])
    scores = score_values(example, 10)
    top1 = scores[0]
    top5 = scores[:5]
    top10 = scores[:10]
    q_tokens = tokens(example.get("question", ""))
    top_text = " ".join(f"{ctx.get('title', '')} {ctx.get('text', '')}" for ctx in ctxs[:1])
    top5_text = " ".join(f"{ctx.get('title', '')} {ctx.get('text', '')}" for ctx in ctxs[:5])
    top_tokens = tokens(top_text)
    top5_tokens = tokens(top5_text)
    overlap_top1 = len(q_tokens & top_tokens)
    overlap_top5 = len(q_tokens & top5_tokens)
    return {
        "question_len": float(len(q_tokens)),
        "top1_score": top1,
        "top5_mean_score": mean(top5) if top5 else 0.0,
        "top10_mean_score": mean(top10) if top10 else 0.0,
        "top1_top2_gap": top1 - scores[1] if len(scores) > 1 else top1,
        "top1_top5_gap": top1 - scores[4] if len(scores) > 4 else top1,
        "top1_top10_gap": top1 - scores[9] if len(scores) > 9 else top1,
        "top1_top5_ratio": safe_ratio(top1, mean(top5) if top5 else 0.0),
        "top1_top10_ratio": safe_ratio(top1, mean(top10) if top10 else 0.0),
        "score_std_top10": math.sqrt(mean((value - mean(top10)) ** 2 for value in top10)) if top10 else 0.0,
        "query_top1_overlap": float(overlap_top1),
        "query_top5_overlap": float(overlap_top5),
        "query_top1_overlap_ratio": safe_ratio(overlap_top1, len(q_tokens)),
        "query_top5_overlap_ratio": safe_ratio(overlap_top5, len(q_tokens)),
        "num_contexts": float(len(ctxs)),
    }


def question_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [row.get("question", "") for row in rows]


def numeric_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [numeric_features(row) for row in rows]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    return data


def load_labels(path: Path) -> dict[str, str]:
    labels = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                labels[str(row["id"])] = row.get("label") or row.get("complexity") or "hard"
    return labels


def main() -> None:
    args = parse_args()
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import FunctionTransformer
    except ImportError as exc:
        raise SystemExit("Install scikit-learn first.") from exc

    # ColumnTransformer is imported intentionally to make sklearn fail early if
    # the installed version is too old for the pipeline components we use.
    _ = ColumnTransformer

    labels = load_labels(args.labels)
    rows = [row for row in load_json(args.retrieved) if str(row.get("id")) in labels]
    y = [labels[str(row.get("id"))] for row in rows]

    x_train, x_eval, y_train, y_eval = train_test_split(
        rows,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "question_tfidf",
                            Pipeline(
                                [
                                    ("select_question", FunctionTransformer(question_texts, validate=False)),
                                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                                ]
                            ),
                        ),
                        (
                            "retrieval_features",
                            Pipeline(
                                [
                                    ("build_features", FunctionTransformer(numeric_feature_rows, validate=False)),
                                    ("dict", DictVectorizer()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    report = classification_report(y_eval, predictions, output_dict=True, zero_division=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(model, handle)

    metrics_path = args.metrics_output or args.output.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(classification_report(y_eval, predictions, zero_division=0))
    print(f"Model: {args.output}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
