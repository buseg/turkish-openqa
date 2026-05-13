#!/usr/bin/env python3
"""Tune BM25/cross-encoder interpolation weights on retrieved examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from apply_cross_encoder_reranker import bm25_score, ctx_pair, minmax, score_contexts
from evaluate_retrieval import first_hit_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Retrieved JSON with ctxs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--ks", default="1,5,10,25,50,100")
    parser.add_argument("--match-mode", choices=("answer", "oracle", "either"), default="answer")
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_floats(raw: str) -> list[float]:
    return [float(item) for item in raw.replace(" ", ",").split(",") if item]


def parse_ints(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(" ", ",").split(",") if item]


def rank_examples(data: list[dict[str, Any]], weight: float) -> list[dict[str, Any]]:
    ranked = []
    for example in data:
        copy = dict(example)
        ctxs = []
        for ctx in example.get("ctxs", []):
            ctx_copy = dict(ctx)
            ctx_copy["rerank_score"] = (
                weight * float(ctx_copy.get("cross_encoder_score", 0.0))
                + (1.0 - weight) * float(ctx_copy.get("bm25_norm_score", 0.0))
            )
            ctxs.append(ctx_copy)
        copy["ctxs"] = sorted(ctxs, key=lambda ctx: ctx["rerank_score"], reverse=True)
        ranked.append(copy)
    return ranked


def metrics_for(data: list[dict[str, Any]], ks: list[int], match_mode: str) -> dict[str, Any]:
    ranks = [first_hit_rank(example, match_mode) for example in data]
    found = [rank for rank in ranks if rank is not None]
    metrics: dict[str, Any] = {
        "examples": len(data),
        "found_any": len(found),
        "mrr": mean((1.0 / rank if rank is not None else 0.0) for rank in ranks) if ranks else 0.0,
        "mean_first_hit_rank": mean(found) if found else None,
    }
    for k in ks:
        metrics[f"recall@{k}"] = mean((rank is not None and rank <= k) for rank in ranks) if ranks else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers first.") from exc

    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected {args.input} to contain a JSON list.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = choose_device()
    model.to(device)

    scored = []
    for index, example in enumerate(data, start=1):
        ctxs = list(example.get("ctxs", []))
        if not ctxs:
            scored.append(example)
            continue
        question = example.get("question", "")
        ce_scores = score_contexts(model, tokenizer, question, ctxs, args.max_length, args.batch_size, device)
        bm25_scores = minmax([bm25_score(ctx) for ctx in ctxs])
        example_copy = dict(example)
        scored_ctxs = []
        for rank, (ctx, ce_score, bm25_norm) in enumerate(zip(ctxs, ce_scores, bm25_scores), start=1):
            ctx_copy = dict(ctx)
            ctx_copy["original_rank"] = rank
            ctx_copy["cross_encoder_score"] = ce_score
            ctx_copy["bm25_norm_score"] = bm25_norm
            scored_ctxs.append(ctx_copy)
        example_copy["ctxs"] = scored_ctxs
        scored.append(example_copy)
        if index % 50 == 0:
            print(f"scored {index}/{len(data)} examples")

    weights = parse_floats(args.weights)
    ks = parse_ints(args.ks)
    results = []
    for weight in weights:
        ranked = rank_examples(scored, weight)
        metrics = metrics_for(ranked, ks, args.match_mode)
        results.append({"cross_encoder_weight": weight, **metrics})

    best_by_mrr = max(results, key=lambda item: item["mrr"]) if results else None
    best_by_r5 = max(results, key=lambda item: item.get("recall@5", 0.0)) if results else None
    output = {
        "model": str(args.model),
        "input": str(args.input),
        "match_mode": args.match_mode,
        "device": str(device),
        "results": results,
        "best_by_mrr": best_by_mrr,
        "best_by_recall@5": best_by_r5,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
