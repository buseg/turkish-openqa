#!/usr/bin/env python3
"""Train a lightweight question complexity classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise SystemExit("Install scikit-learn first: pip install scikit-learn") from exc

    rows = load_jsonl(args.labels)
    questions = [row.get("question", "") for row in rows]
    labels = [row.get("label") or row.get("complexity") or "hard" for row in rows]
    x_train, x_eval, y_train, y_eval = train_test_split(
        questions, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    report = classification_report(y_eval, predictions, output_dict=True, zero_division=0)

    import pickle

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(model, handle)
    metrics_path = args.output.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(classification_report(y_eval, predictions, zero_division=0))
    print(f"Model: {args.output}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
