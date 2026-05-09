#!/usr/bin/env python3
"""Select a smaller set of passages from retrieved top-k contexts."""

import argparse
import json
from pathlib import Path

from evaluate_retrieval import contains_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Retrieved JSON with ctxs.")
    parser.add_argument("--output", required=True, help="Output JSON with selected ctxs.")
    parser.add_argument("--n-context", type=int, default=5)
    parser.add_argument(
        "--strategy",
        default="bm25",
        choices=["bm25", "oracle-answer-first"],
        help="oracle-answer-first uses answer labels and is an upper bound.",
    )
    return parser.parse_args()


def has_answer(ctx, answers):
    return contains_answer(ctx.get("text", ""), answers)


def select_ctxs(ctxs, answers, n_context, strategy):
    annotated = []
    for ctx in ctxs:
        new_ctx = dict(ctx)
        new_ctx["hasanswer"] = has_answer(ctx, answers)
        annotated.append(new_ctx)

    if strategy == "bm25":
        return annotated[:n_context]

    positives = [ctx for ctx in annotated if ctx["hasanswer"]]
    negatives = [ctx for ctx in annotated if not ctx["hasanswer"]]
    return (positives + negatives)[:n_context]


def selected_hit(example):
    answers = example.get("answers", [])
    return any(has_answer(ctx, answers) for ctx in example.get("ctxs", []))


def original_hit_at(example, k):
    answers = example.get("answers", [])
    return any(has_answer(ctx, answers) for ctx in example.get("ctxs", [])[:k])


def main():
    args = parse_args()
    with Path(args.input).open(encoding="utf-8") as fin:
        data = json.load(fin)

    output = []
    answerable = original_topk_hits = original_top100_hits = selected_hits = 0
    selected_positive_passages = 0

    for example in data:
        new_example = dict(example)
        answers = example.get("answers", [])
        new_example["ctxs"] = select_ctxs(
            example.get("ctxs", []),
            answers,
            args.n_context,
            args.strategy,
        )
        new_example["selector_strategy"] = args.strategy
        output.append(new_example)

        if answers:
            answerable += 1
            original_topk_hits += int(original_hit_at(example, args.n_context))
            original_top100_hits += int(original_hit_at(example, 100))
            selected_hits += int(selected_hit(new_example))
            selected_positive_passages += sum(ctx.get("hasanswer", False) for ctx in new_example["ctxs"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(output, fout, ensure_ascii=False, indent=2)

    print(f"Wrote selected retrieval to {output_path}")
    print(f"Examples: {answerable}")
    if answerable:
        print(
            f"Original Recall@{args.n_context}: "
            f"{original_topk_hits / answerable:.4f} ({original_topk_hits}/{answerable})"
        )
        print(
            f"Original Recall@100: "
            f"{original_top100_hits / answerable:.4f} ({original_top100_hits}/{answerable})"
        )
        print(
            f"Selected Recall@{args.n_context}: "
            f"{selected_hits / answerable:.4f} ({selected_hits}/{answerable})"
        )
        print(f"Selected positive passages: {selected_positive_passages}")


if __name__ == "__main__":
    main()
