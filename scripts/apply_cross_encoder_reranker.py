#!/usr/bin/env python3
"""Apply a trained cross-encoder reranker to retrieved contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Retrieved JSON with ctxs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-context", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mode", choices=("cross-encoder", "hybrid"), default="hybrid")
    parser.add_argument("--cross-encoder-weight", type=float, default=0.7)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ctx_pair(question: str, ctx: dict[str, Any]) -> tuple[str, str]:
    return question, f"{ctx.get('title', '')}. {ctx.get('text', '')}".strip()


def bm25_score(ctx: dict[str, Any]) -> float:
    try:
        return float(ctx.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    denom = max(high - low, 1e-9)
    return [(value - low) / denom for value in values]


def score_contexts(model: Any, tokenizer: Any, question: str, ctxs: list[dict[str, Any]], max_length: int, batch_size: int, device: torch.device) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ctxs), batch_size):
            batch_ctxs = ctxs[start : start + batch_size]
            questions, passages = zip(*(ctx_pair(question, ctx) for ctx in batch_ctxs))
            encoded = tokenizer(
                list(questions),
                list(passages),
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            if logits.shape[-1] == 1:
                probabilities = torch.sigmoid(logits.squeeze(-1))
            else:
                probabilities = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(float(score) for score in probabilities.cpu())
    return scores


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = choose_device()
    model.to(device)

    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)

    for example in data:
        ctxs = list(example.get("ctxs", []))
        question = example.get("question", "")
        if not ctxs:
            continue
        ce_scores = score_contexts(model, tokenizer, question, ctxs, args.max_length, args.batch_size, device)
        bm25_scores = minmax([bm25_score(ctx) for ctx in ctxs])
        for rank, (ctx, ce_score, bm25_norm) in enumerate(zip(ctxs, ce_scores, bm25_scores), start=1):
            ctx["original_rank"] = rank
            ctx["cross_encoder_score"] = ce_score
            ctx["bm25_norm_score"] = bm25_norm
            if args.mode == "cross-encoder":
                ctx["rerank_score"] = ce_score
            else:
                ctx["rerank_score"] = args.cross_encoder_weight * ce_score + (1.0 - args.cross_encoder_weight) * bm25_norm
        example["ctxs"] = sorted(ctxs, key=lambda ctx: ctx["rerank_score"], reverse=True)[: args.n_context]
        example["reranker"] = args.mode
        example["cross_encoder_weight"] = args.cross_encoder_weight

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(data)} reranked examples to {args.output}")


if __name__ == "__main__":
    main()
