#!/usr/bin/env python3
"""Print one retrieved QA example for manual inspection."""

from __future__ import annotations

import argparse
import random

from common_qa import answers_from, contains_answer, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    data = load_json(args.input)
    example = random.choice(data) if args.random else data[args.index]
    answers = answers_from(example.get("answers") or example.get("target"))

    lines = [
        f"File: {args.input}",
        f"ID: {example.get('id', '')}",
        f"Question: {example.get('question', '')}",
        f"Answers: {answers}",
        "",
        "Oracle Context",
    ]
    oracle = example.get("oracle_ctx") or example.get("oracle_context") or {}
    if oracle:
        oracle_text = oracle.get("text", "")
        lines += [
            f"Title: {oracle.get('title', '')}",
            f"Contains answer: {contains_answer(oracle_text, answers)}",
            oracle_text[:1000],
            "",
        ]

    lines.append(f"Top {args.top_k} Retrieved Contexts")
    for rank, ctx in enumerate(example.get("ctxs", [])[: args.top_k], start=1):
        text = ctx.get("text", "")
        lines += [
            "",
            f"#{rank} score={ctx.get('score', '')} has_answer={contains_answer(text, answers)} id={ctx.get('id', '')}",
            f"Title: {ctx.get('title', '')}",
            text[:700],
        ]

    output = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
            handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
