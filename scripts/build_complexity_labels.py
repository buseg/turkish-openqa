#!/usr/bin/env python3
"""Build Adaptive-RAG-style complexity labels from retrieval ranks."""

import argparse
import json
from pathlib import Path

from evaluate_retrieval import first_hit_rank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved", required=True, help="Retrieved JSON with ctxs.")
    parser.add_argument("--output", required=True, help="Output JSONL labels.")
    parser.add_argument("--easy-k", type=int, default=10)
    parser.add_argument("--medium-k", type=int, default=50)
    parser.add_argument(
        "--include-unanswered",
        action="store_true",
        help="Include examples with empty answers as unanswerable.",
    )
    return parser.parse_args()


def complexity_label(rank, easy_k, medium_k):
    if rank is not None and rank <= easy_k:
        return "easy"
    if rank is not None and rank <= medium_k:
        return "medium"
    return "hard"


def target_k(label, easy_k, medium_k):
    if label == "easy":
        return easy_k
    if label == "medium":
        return medium_k
    return 100


def make_record(example, easy_k, medium_k):
    rank = first_hit_rank(example) if example.get("answers") else None
    if not example.get("answers"):
        label = "unanswerable"
        chosen_k = 0
    else:
        label = complexity_label(rank, easy_k, medium_k)
        chosen_k = target_k(label, easy_k, medium_k)

    return {
        "id": str(example["id"]),
        "question": example.get("question", ""),
        "answers": example.get("answers", []),
        "first_hit_rank": rank,
        "complexity": label,
        "target_k": chosen_k,
        "hit_at_10": rank is not None and rank <= 10,
        "hit_at_50": rank is not None and rank <= 50,
        "hit_at_100": rank is not None and rank <= 100,
    }


def main():
    args = parse_args()
    with Path(args.retrieved).open(encoding="utf-8") as fin:
        data = json.load(fin)

    records = []
    skipped = 0
    for example in data:
        if not args.include_unanswered and not example.get("answers"):
            skipped += 1
            continue
        records.append(make_record(example, args.easy_k, args.medium_k))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = {}
    for record in records:
        counts[record["complexity"]] = counts.get(record["complexity"], 0) + 1

    print(f"Wrote {len(records)} complexity labels to {output_path}")
    if skipped:
        print(f"Skipped unanswered examples: {skipped}")
    for label in ["easy", "medium", "hard", "unanswerable"]:
        if label in counts:
            print(f"{label}: {counts[label]}")


if __name__ == "__main__":
    main()
