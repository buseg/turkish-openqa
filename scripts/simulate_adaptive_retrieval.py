#!/usr/bin/env python3
"""Simulate adaptive retrieval context budgets from complexity labels."""

from __future__ import annotations

import argparse
import json
from statistics import mean


DEFAULT_BUDGETS = {"easy": 10, "medium": 50, "hard": 100, "no_retrieval": 0}


def row_label(row: dict) -> str:
    return row.get("label") or row.get("complexity") or "hard"


def row_rank(row: dict) -> int | None:
    return row.get("first_answer_rank") or row.get("first_hit_rank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--easy-k", type=int, default=10)
    parser.add_argument("--medium-k", type=int, default=50)
    parser.add_argument("--hard-k", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    budgets = {"easy": args.easy_k, "medium": args.medium_k, "hard": args.hard_k, "no_retrieval": 0}
    rows = [json.loads(line) for line in open(args.labels, encoding="utf-8") if line.strip()]
    hits = []
    used_contexts = []
    for row in rows:
        rank = row_rank(row)
        label = row_label(row)
        budget = budgets.get(label, DEFAULT_BUDGETS.get(label, 100))
        used_contexts.append(budget)
        hits.append(rank is not None and rank <= budget)
    metrics = {
        "examples": len(rows),
        "recall": mean(hits) if hits else 0.0,
        "avg_contexts": mean(used_contexts) if used_contexts else 0.0,
        "context_saving_vs_100": 1 - (mean(used_contexts) / 100) if used_contexts else 0.0,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
