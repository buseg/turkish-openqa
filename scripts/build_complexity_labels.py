#!/usr/bin/env python3
"""Build adaptive retrieval labels from first answer-containing rank."""

from __future__ import annotations

import argparse
import json

from common_qa import answers_from, contains_answer, get_context_text, load_json


def first_rank(example: dict) -> int | None:
    answers = answers_from(example.get("answers") or example.get("target"))
    for rank, ctx in enumerate(example.get("ctxs", []), start=1):
        if contains_answer(get_context_text(ctx), answers):
            return rank
    return None


def label_from_rank(rank: int | None) -> str:
    if rank is None or rank > 50:
        return "hard"
    if rank <= 10:
        return "easy"
    return "medium"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = load_json(args.input)
    counts = {"easy": 0, "medium": 0, "hard": 0}
    with open(args.output, "w", encoding="utf-8") as handle:
        for example in data:
            rank = first_rank(example)
            label = label_from_rank(rank)
            target_k = {"easy": 10, "medium": 50, "hard": 100}[label]
            counts[label] += 1
            handle.write(
                json.dumps(
                    {
                        "id": example.get("id"),
                        "question": example.get("question", ""),
                        "answers": answers_from(example.get("answers") or example.get("target")),
                        "first_hit_rank": rank,
                        "first_answer_rank": rank,
                        "label": label,
                        "complexity": label,
                        "target_k": target_k,
                        "hit_at_10": rank is not None and rank <= 10,
                        "hit_at_50": rank is not None and rank <= 50,
                        "hit_at_100": rank is not None and rank <= 100,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(counts)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
