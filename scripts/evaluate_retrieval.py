#!/usr/bin/env python3
"""Compute answer-string Recall@K for retrieved FiD/DPR-style data."""

import argparse
import json
import string
import unicodedata
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="JSON file with question, answers, and retrieved ctxs.",
    )
    parser.add_argument(
        "--ks",
        default="1,5,10,20,50,100",
        help="Comma-separated K values.",
    )
    parser.add_argument(
        "--include-unanswered",
        action="store_true",
        help="Include examples with empty answers in the denominator.",
    )
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


def first_hit_rank(example):
    answers = example.get("answers", [])
    for rank, ctx in enumerate(example.get("ctxs", []), start=1):
        if contains_answer(ctx.get("text", ""), answers):
            return rank
    return None


def main():
    args = parse_args()
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    with Path(args.input).open(encoding="utf-8") as fin:
        data = json.load(fin)
    if not args.include_unanswered:
        skipped = sum(not example.get("answers") for example in data)
        data = [example for example in data if example.get("answers")]
        if skipped:
            print(f"Skipped unanswered examples: {skipped}")

    ranks = [first_hit_rank(example) for example in data]
    total = len(ranks)
    print(f"Examples: {total}")
    for k in ks:
        hits = sum(rank is not None and rank <= k for rank in ranks)
        print(f"Recall@{k}: {hits / total:.4f} ({hits}/{total})")


if __name__ == "__main__":
    main()
