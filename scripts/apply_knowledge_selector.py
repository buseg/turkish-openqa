#!/usr/bin/env python3
"""Apply a trained knowledge selector to rerank retrieved contexts."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Retrieved JSON with ctxs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-context", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("selector", "hybrid"),
        default="hybrid",
        help="Pure selector probability or a selector/BM25 hybrid score.",
    )
    parser.add_argument("--selector-weight", type=float, default=0.8)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    with args.model.open("rb") as handle:
        model = pickle.load(handle)
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)

    for example in data:
        ctxs = list(example.get("ctxs", []))
        question = example.get("question", "")
        if not ctxs:
            continue
        texts = [ctx_text(question, ctx) for ctx in ctxs]
        selector_scores = [float(score) for score in model.predict_proba(texts)[:, 1]]
        bm25_scores = minmax([bm25_score(ctx) for ctx in ctxs])
        for rank, (ctx, selector_score, bm25_norm) in enumerate(zip(ctxs, selector_scores, bm25_scores), start=1):
            ctx["original_rank"] = rank
            ctx["selector_score"] = selector_score
            ctx["bm25_norm_score"] = bm25_norm
            if args.mode == "selector":
                ctx["rerank_score"] = selector_score
            else:
                ctx["rerank_score"] = args.selector_weight * selector_score + (1.0 - args.selector_weight) * bm25_norm
        example["ctxs"] = sorted(ctxs, key=lambda ctx: ctx["rerank_score"], reverse=True)[: args.n_context]
        example["selector_mode"] = args.mode
        example["selector_weight"] = args.selector_weight

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(data)} reranked examples to {args.output}")


if __name__ == "__main__":
    main()
