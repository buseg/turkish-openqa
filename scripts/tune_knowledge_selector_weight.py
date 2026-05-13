#!/usr/bin/env python3
"""Sweep selector/BM25 hybrid weights for a trained knowledge selector."""

from __future__ import annotations

import argparse
import json
import pickle
import string
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Retrieved JSON with ctxs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--ks", default="1,5,10,25,50,100")
    return parser.parse_args()


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


def ctx_text(question: str, ctx: dict[str, Any]) -> str:
    return f"soru: {question} baslik: {ctx.get('title', '')} metin: {ctx.get('text', '')}"


def bm25_score(ctx: dict[str, Any]) -> float:
    try:
        return float(ctx.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    denom = max(hi - lo, 1e-9)
    return [(value - lo) / denom for value in values]


def first_hit_rank(scored_ctxs: list[tuple[float, dict[str, Any]]], answers: list[str]) -> int | None:
    for rank, (_, ctx) in enumerate(sorted(scored_ctxs, key=lambda item: item[0], reverse=True), start=1):
        if contains_answer(ctx, answers):
            return rank
    return None


def main() -> None:
    args = parse_args()
    weights = [float(item) for item in args.weights.split(",") if item]
    ks = [int(item) for item in args.ks.split(",") if item]
    with args.model.open("rb") as handle:
        model = pickle.load(handle)
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)

    prepared = []
    for example in data:
        ctxs = list(example.get("ctxs", []))
        if not ctxs:
            prepared.append(([], answers_from(example.get("answers") or example.get("target"))))
            continue
        question = example.get("question", "")
        selector_scores = [float(score) for score in model.predict_proba([ctx_text(question, ctx) for ctx in ctxs])[:, 1]]
        bm25_scores = minmax([bm25_score(ctx) for ctx in ctxs])
        prepared.append(
            (
                list(zip(selector_scores, bm25_scores, ctxs)),
                answers_from(example.get("answers") or example.get("target")),
            )
        )

    rows = []
    for weight in weights:
        ranks = []
        for scored, answers in prepared:
            hybrid = [
                (weight * selector_score + (1.0 - weight) * bm25_norm_score, ctx)
                for selector_score, bm25_norm_score, ctx in scored
            ]
            ranks.append(first_hit_rank(hybrid, answers))
        row: dict[str, Any] = {
            "selector_weight": weight,
            "mrr": mean((1.0 / rank if rank is not None else 0.0) for rank in ranks) if ranks else 0.0,
        }
        for k in ks:
            row[f"recall@{k}"] = mean((rank is not None and rank <= k) for rank in ranks) if ranks else 0.0
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for row in rows:
        print(row)
    best = max(rows, key=lambda row: (row.get("recall@5", 0.0), row.get("recall@25", 0.0), row["mrr"]))
    print(f"Best by Recall@5: {best}")


if __name__ == "__main__":
    main()
