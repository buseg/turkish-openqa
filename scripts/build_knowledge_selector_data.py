#!/usr/bin/env python3
"""Build passage-level labels for knowledge selector experiments."""

import argparse
import json
from pathlib import Path

from evaluate_retrieval import contains_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved", required=True, help="Retrieved JSON with ctxs.")
    parser.add_argument("--output", required=True, help="Output JSONL passage labels.")
    parser.add_argument("--n-context", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.retrieved).open(encoding="utf-8") as fin:
        data = json.load(fin)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = positives = negatives = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for example in data:
            answers = example.get("answers", [])
            if not answers:
                continue
            examples += 1
            for rank, ctx in enumerate(example.get("ctxs", [])[: args.n_context], start=1):
                label = int(contains_answer(ctx.get("text", ""), answers))
                positives += label
                negatives += int(not label)
                record = {
                    "id": str(example["id"]),
                    "question": example.get("question", ""),
                    "answers": answers,
                    "ctx_rank": rank,
                    "ctx_id": ctx.get("id"),
                    "ctx_title": ctx.get("title", ""),
                    "ctx_score": ctx.get("score"),
                    "label": label,
                    "text": ctx.get("text", ""),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote selector labels to {output_path}")
    print(f"Examples: {examples}")
    print(f"Positive passages: {positives}")
    print(f"Negative passages: {negatives}")


if __name__ == "__main__":
    main()
