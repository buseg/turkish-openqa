#!/usr/bin/env python3
"""Tune conservative adaptive retrieval policies from labelled ranks."""

from __future__ import annotations

import argparse
import json
from statistics import mean


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def row_label(row: dict) -> str:
    return row.get("label") or row.get("complexity") or "hard"


def row_rank(row: dict) -> int | None:
    return row.get("first_answer_rank") or row.get("first_hit_rank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--easy-options", default="10,25")
    parser.add_argument("--medium-options", default="25,50")
    parser.add_argument("--hard-options", default="100")
    args = parser.parse_args()

    rows = load_rows(args.labels)
    easy_options = [int(x) for x in args.easy_options.split(",") if x]
    medium_options = [int(x) for x in args.medium_options.split(",") if x]
    hard_options = [int(x) for x in args.hard_options.split(",") if x]

    results = []
    for easy in easy_options:
        for medium in medium_options:
            for hard in hard_options:
                budgets = {"easy": easy, "medium": medium, "hard": hard}
                hits, contexts = [], []
                for row in rows:
                    budget = budgets.get(row_label(row), hard)
                    rank = row_rank(row)
                    contexts.append(budget)
                    hits.append(rank is not None and rank <= budget)
                results.append(
                    {
                        "easy_k": easy,
                        "medium_k": medium,
                        "hard_k": hard,
                        "recall": mean(hits),
                        "avg_contexts": mean(contexts),
                        "saving_vs_100": 1 - mean(contexts) / 100,
                    }
                )
    results.sort(key=lambda row: (-row["recall"], row["avg_contexts"]))
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for row in results[:10]:
        print(row)


if __name__ == "__main__":
    main()
