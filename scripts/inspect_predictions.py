#!/usr/bin/env python3
"""Write human-readable prediction inspection examples."""

from __future__ import annotations

import argparse
import json

from common_qa import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics", help="Optional per-example metrics JSON from src/evaluate.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    dataset = {str(ex["id"]): ex for ex in load_json(args.dataset)}
    predictions = load_json(args.predictions)
    metrics = load_json(args.metrics) if args.metrics else []
    if metrics:
        ordered_ids = [str(row["id"]) for row in sorted(metrics, key=lambda row: (row.get("em", 0), row.get("f1", 0)))]
    else:
        ordered_ids = list(predictions)

    lines = []
    for qid in ordered_ids[: args.limit]:
        ex = dataset.get(qid, {})
        oracle = ex.get("oracle_ctx") or ex.get("oracle_context") or {}
        lines.extend(
            [
                "=" * 80,
                f"ID: {qid}",
                f"Question: {ex.get('question', '')}",
                f"Gold: {ex.get('answers', [])}",
                f"Prediction: {predictions.get(qid, '')}",
                f"Oracle title: {oracle.get('title', '')}",
                f"Oracle text: {(oracle.get('text', '') or '')[:900]}",
                "",
            ]
        )
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
