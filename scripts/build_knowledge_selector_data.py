#!/usr/bin/env python3
"""Create passage-level labels for a supervised knowledge selector."""

from __future__ import annotations

import argparse
import json

from common_qa import answers_from, contains_answer, get_context_text, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Retrieved JSON with ctxs.")
    parser.add_argument("--output", required=True, help="Output JSONL.")
    parser.add_argument("--n-context", type=int, default=100)
    args = parser.parse_args()

    data = load_json(args.input)
    positives = negatives = 0
    with open(args.output, "w", encoding="utf-8") as handle:
        for example in data:
            answers = answers_from(example.get("answers") or example.get("target"))
            for rank, ctx in enumerate(example.get("ctxs", [])[: args.n_context], start=1):
                label = int(contains_answer(get_context_text(ctx), answers))
                positives += label
                negatives += 1 - label
                handle.write(
                    json.dumps(
                        {
                            "qid": example.get("id"),
                            "question": example.get("question", ""),
                            "pid": ctx.get("id"),
                            "rank": rank,
                            "score": ctx.get("score"),
                            "title": ctx.get("title", ""),
                            "text": ctx.get("text", ""),
                            "label": label,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"Positive passages: {positives}")
    print(f"Negative passages: {negatives}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
