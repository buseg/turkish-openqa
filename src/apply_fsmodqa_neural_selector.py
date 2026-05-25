#!/usr/bin/env python3
"""Apply a saved neural FSModQA knowledge selector to top-k retrieval output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import train_fsmodqa_neural_selector as selector


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Saved selector model/tokenizer directory.")
    parser.add_argument("--ranking", type=Path, required=True, help="Input top-k ranking JSONL.")
    parser.add_argument("--queries", type=Path, default=ROOT / "odqa_data/fsmodqa_retrieval/test.query.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "odqa_data/fsmodqa_retrieval/corpus.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output reranked JSONL path.")
    parser.add_argument("--cache-dir", default=".hf_cache")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--n-context", type=int, default=100)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--hybrid-weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument(
        "--selector-weight",
        type=float,
        default=None,
        help="Hybrid selector weight. Defaults to the validation-best weight saved in model-dir/metrics.json.",
    )
    return parser.parse_args()


def saved_selector_weight(model_dir: Path, hybrid_weights: list[float]) -> float:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        selector.log(f"No saved metrics at {metrics_path}; using selector_weight=1.00")
        return 1.0

    with metrics_path.open(encoding="utf-8") as handle:
        metrics: dict[str, Any] = json.load(handle)

    if "written_reranked_selector_weight" in metrics:
        return float(metrics["written_reranked_selector_weight"])

    scored_weights = []
    for weight in hybrid_weights:
        values = metrics.get(f"hybrid_selector_weight_{weight:.2f}")
        if values:
            scored_weights.append(
                (
                    values.get("recall@10", 0.0),
                    values.get("recall@25", 0.0),
                    values.get("mrr", 0.0),
                    weight,
                )
            )

    if not scored_weights:
        selector.log(f"No hybrid metrics in {metrics_path}; using selector_weight=1.00")
        return 1.0
    return float(max(scored_weights)[-1])


def main() -> None:
    args = parse_args()
    hybrid_weights = [float(item) for item in args.hybrid_weights.split(",") if item]

    selector.log("Importing torch and transformers")
    try:
        import torch as torch_module
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install torch and transformers first.") from exc

    selector.torch = torch_module

    selector.log(f"Loading corpus: {args.corpus}")
    corpus = selector.load_corpus(args.corpus)
    queries = selector.load_queries(args.queries)
    rows = selector.build_rows(
        args.ranking,
        queries,
        corpus,
        n_context=args.n_context,
        max_questions=args.max_questions,
    )
    if not rows:
        raise SystemExit("No selector pairs were built. Check ranking/query/corpus paths.")

    selector.log(f"Loading saved selector from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=1, cache_dir=args.cache_dir)
    device = selector.choose_device()
    model.to(device)
    selector.log(f"Using device: {device}")

    selector.score_rows(
        model,
        tokenizer,
        rows,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )

    selector_weight = args.selector_weight
    if selector_weight is None:
        selector_weight = saved_selector_weight(args.model_dir, hybrid_weights)
    selector.log(f"Writing reranked output with selector_weight={selector_weight:.2f}")
    selector.write_reranked(args.output, rows, selector_weight=selector_weight)


if __name__ == "__main__":
    main()
