#!/usr/bin/env python3
"""Evaluate retrieved contexts with Recall@K and MRR."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from statistics import mean
from typing import Any


def normalize_text(text: Any) -> str:
    text = str(text).casefold()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def answers_from(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("text", [])
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        answers = []
        for answer in raw:
            if isinstance(answer, dict):
                answer = answer.get("text", "")
            answer = str(answer).strip() if answer is not None else ""
            if answer:
                answers.append(answer)
        return answers
    answer = str(raw).strip()
    return [answer] if answer else []


def contains_answer(ctx: dict[str, Any], answers: list[str]) -> bool:
    text = normalize_text(f"{ctx.get('title', '')} {ctx.get('text', '')}")
    return any(normalize_text(answer) in text for answer in answers if normalize_text(answer))


def same_oracle(ctx: dict[str, Any], oracle_ctx: dict[str, Any] | None) -> bool:
    if not oracle_ctx:
        return False
    return (
        normalize_text(ctx.get("title", "")) == normalize_text(oracle_ctx.get("title", ""))
        and normalize_text(ctx.get("text", "")) == normalize_text(oracle_ctx.get("text", ""))
    )


def first_hit_rank(example: dict[str, Any], match_mode: str) -> int | None:
    answers = answers_from(example.get("answers") or example.get("target"))
    oracle_ctx = example.get("oracle_ctx") or example.get("oracle_context")
    for rank, ctx in enumerate(example.get("ctxs", []), start=1):
        answer_hit = contains_answer(ctx, answers)
        oracle_hit = same_oracle(ctx, oracle_ctx)
        if match_mode == "answer" and answer_hit:
            return rank
        if match_mode == "oracle" and oracle_hit:
            return rank
        if match_mode == "either" and (answer_hit or oracle_hit):
            return rank
    return None


def parse_ks(raw: str) -> list[int]:
    values = []
    for item in re.split(r"[, ]+", raw.strip()):
        if item:
            values.append(int(item))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Retrieved JSON file with ctxs.")
    parser.add_argument("--ks", default="1,5,10,20,25,50,100")
    parser.add_argument("--match-mode", choices=("answer", "oracle", "either"), default="answer")
    parser.add_argument("--output", type=Path, help="Optional JSON metrics output.")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected {args.input} to contain a JSON list.")

    ks = parse_ks(args.ks)
    ranks = [first_hit_rank(example, args.match_mode) for example in data]
    found_ranks = [rank for rank in ranks if rank is not None]

    metrics: dict[str, Any] = {
        "input": str(args.input),
        "match_mode": args.match_mode,
        "examples": len(data),
        "found_any": len(found_ranks),
        "missing_any": len(data) - len(found_ranks),
        "mrr": mean((1.0 / rank if rank is not None else 0.0) for rank in ranks) if ranks else 0.0,
        "mean_first_hit_rank": mean(found_ranks) if found_ranks else None,
    }
    for k in ks:
        metrics[f"recall@{k}"] = mean((rank is not None and rank <= k) for rank in ranks) if ranks else 0.0

    print(f"Input: {args.input}")
    print(f"Examples: {metrics['examples']}")
    print(f"Match mode: {args.match_mode}")
    print(f"Found any: {metrics['found_any']}")
    print(f"Missing any: {metrics['missing_any']}")
    for k in ks:
        print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    if metrics["mean_first_hit_rank"] is not None:
        print(f"Mean first hit rank: {metrics['mean_first_hit_rank']:.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
