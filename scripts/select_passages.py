#!/usr/bin/env python3
"""Select or rerank retrieved passages with simple policies."""

from __future__ import annotations

import argparse

from common_qa import answers_from, contains_answer, get_context_text, load_json, write_json


def select_ctxs(example: dict, mode: str, n_context: int) -> list[dict]:
    ctxs = list(example.get("ctxs", []))
    answers = answers_from(example.get("answers") or example.get("target"))
    if mode == "bm25":
        return ctxs[:n_context]
    if mode == "oracle":
        positives = [ctx for ctx in ctxs if contains_answer(get_context_text(ctx), answers)]
        negatives = [ctx for ctx in ctxs if ctx not in positives]
        return (positives + negatives)[:n_context]
    if mode == "normscore":
        scores = [float(ctx.get("score", 0.0)) for ctx in ctxs] or [0.0]
        max_score = max(scores)
        min_score = min(scores)
        denom = max(max_score - min_score, 1e-9)
        for ctx in ctxs:
            ctx["selector_score"] = (float(ctx.get("score", 0.0)) - min_score) / denom
        return sorted(ctxs, key=lambda ctx: ctx["selector_score"], reverse=True)[:n_context]
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("bm25", "oracle", "normscore"), default="bm25")
    parser.add_argument("--n-context", type=int, default=5)
    args = parser.parse_args()

    data = load_json(args.input)
    for example in data:
        example["ctxs"] = select_ctxs(example, args.mode, args.n_context)
        example["selector_strategy"] = args.mode
    write_json(args.output, data)
    print(f"Wrote {len(data)} examples to {args.output}")


if __name__ == "__main__":
    main()
