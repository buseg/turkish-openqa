#!/usr/bin/env python3
"""Print a readable QA/retrieval example."""

import argparse
import json
import random
import string
import unicodedata
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="QA/retrieved JSON file.")
    parser.add_argument("--index", type=int, default=None, help="Example index.")
    parser.add_argument("--id", default=None, help="Example id.")
    parser.add_argument("--random", action="store_true", help="Pick a random answerable example.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chars", type=int, default=700)
    return parser.parse_args()


def normalize(text):
    text = unicodedata.normalize("NFD", text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def contains_answer(text, answers):
    normalized_text = normalize(text)
    for answer in answers:
        normalized_answer = normalize(answer)
        if normalized_answer and normalized_answer in normalized_text:
            return True
    return False


def compact(text, limit):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def choose_example(data, args):
    if args.id is not None:
        for example in data:
            if str(example.get("id")) == str(args.id):
                return example
        raise SystemExit(f"No example found with id={args.id}")

    if args.index is not None:
        return data[args.index]

    answerable = [example for example in data if example.get("answers")]
    if args.random:
        return random.choice(answerable)

    return next(example for example in data if example.get("answers"))


def main():
    args = parse_args()
    with Path(args.input).open(encoding="utf-8") as fin:
        data = json.load(fin)

    example = choose_example(data, args)
    answers = example.get("answers", [])
    ctxs = example.get("ctxs", [])

    print(f"File: {args.input}")
    print(f"ID: {example.get('id')}")
    print(f"Question: {example.get('question')}")
    print(f"Answers: {answers}")

    if "oracle_ctx" in example:
        oracle = example["oracle_ctx"]
        oracle_has_answer = contains_answer(oracle.get("text", ""), answers)
        print("\nOracle Context")
        print(f"Title: {oracle.get('title')}")
        print(f"Contains answer: {oracle_has_answer}")
        print(compact(oracle.get("text", ""), args.chars))

    print(f"\nTop {min(args.top_k, len(ctxs))} Retrieved Contexts")
    for rank, ctx in enumerate(ctxs[: args.top_k], start=1):
        has_answer = contains_answer(ctx.get("text", ""), answers)
        print(f"\n#{rank} score={ctx.get('score')} has_answer={has_answer} id={ctx.get('id')}")
        print(f"Title: {ctx.get('title')}")
        print(compact(ctx.get("text", ""), args.chars))


if __name__ == "__main__":
    main()
