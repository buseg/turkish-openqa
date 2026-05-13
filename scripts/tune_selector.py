#!/usr/bin/env python3
"""Tune simple threshold rules for choosing original vs rewritten retrieval."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


FEATURES = [
    "top_score_delta",
    "top_score_ratio",
    "norm_top_score_delta",
    "norm_top_score_ratio",
    "score_gap_delta",
    "query_len_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL from build_selector_data.py.")
    parser.add_argument("--output", default=None, help="Optional JSON summary output.")
    parser.add_argument("--target-k", type=int, default=20, help="Recall@K to optimize.")
    parser.add_argument("--ks", default="1,5,10,20,50,100", help="Recall@K values to report.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-thresholds", type=int, default=200)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank_value(record: dict, source: str) -> int | None:
    return record.get(f"{source}_first_hit_rank")


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def choose_rewritten(record: dict, rule: dict) -> bool:
    kind = rule["kind"]
    if kind == "original":
        return False
    if kind == "rewritten":
        return True
    if kind == "oracle":
        original = rank_value(record, "original") or 10**9
        rewritten = rank_value(record, "rewritten") or 10**9
        return rewritten < original

    value = record["features"][rule["feature"]]
    if rule["direction"] == "ge":
        return value >= rule["threshold"]
    return value <= rule["threshold"]


def selected_rank(record: dict, rule: dict) -> int | None:
    source = "rewritten" if choose_rewritten(record, rule) else "original"
    return rank_value(record, source)


def evaluate(records: list[dict], rule: dict, ks: list[int]) -> dict:
    total = len(records)
    ranks = [selected_rank(record, rule) for record in records]
    selected_rewritten = sum(choose_rewritten(record, rule) for record in records)
    metrics = {f"recall@{k}": sum(hit_at(rank, k) for rank in ranks) / total for k in ks}
    label_counts = {"rewritten_better": 0, "original_better": 0, "same": 0}
    for record in records:
        if choose_rewritten(record, rule):
            label_counts[record["label"]] += 1
    return {
        "n": total,
        "selected_rewritten": selected_rewritten,
        "metrics": metrics,
        "selected_rewritten_labels": label_counts,
    }


def finite_values(records: list[dict], feature: str) -> list[float]:
    values = []
    for record in records:
        value = record["features"].get(feature)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return sorted(set(values))


def thresholds(records: list[dict], feature: str, max_thresholds: int) -> list[float]:
    values = finite_values(records, feature)
    if not values:
        return []
    if len(values) <= max_thresholds:
        return values
    return [values[round(i * (len(values) - 1) / (max_thresholds - 1))] for i in range(max_thresholds)]


def candidate_rules(train_records: list[dict], max_thresholds: int):
    yield {"kind": "original", "name": "all_original"}
    yield {"kind": "rewritten", "name": "all_rewritten"}
    for feature in FEATURES:
        for threshold in thresholds(train_records, feature, max_thresholds):
            for direction in ["ge", "le"]:
                symbol = ">=" if direction == "ge" else "<="
                yield {
                    "kind": "threshold",
                    "name": f"{feature} {symbol} {threshold:.6g}",
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                }


def split_records(records: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def format_metrics(result: dict, ks: list[int]) -> str:
    return " ".join(f"R@{k}={result['metrics'][f'recall@{k}']:.4f}" for k in ks)


def main() -> None:
    args = parse_args()
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    target_metric = f"recall@{args.target_k}"
    records = load_jsonl(args.input)
    train_records, eval_records = split_records(records, args.train_ratio, args.seed)

    scored = []
    for rule in candidate_rules(train_records, args.max_thresholds):
        train_result = evaluate(train_records, rule, ks)
        eval_result = evaluate(eval_records, rule, ks) if eval_records else train_result
        scored.append({"rule": rule, "train": train_result, "eval": eval_result})

    scored.sort(
        key=lambda item: (
            item["train"]["metrics"][target_metric],
            item["train"]["metrics"].get("recall@100", 0.0),
            -item["train"]["selected_rewritten"],
        ),
        reverse=True,
    )
    best = scored[0]
    baseline_rules = [
        {"kind": "original", "name": "all_original"},
        {"kind": "rewritten", "name": "all_rewritten"},
        {"kind": "oracle", "name": "oracle_upper_bound"},
    ]
    baselines = [
        {
            "rule": rule,
            "train": evaluate(train_records, rule, ks),
            "eval": evaluate(eval_records, rule, ks) if eval_records else evaluate(train_records, rule, ks),
        }
        for rule in baseline_rules
    ]

    summary = {
        "input": args.input,
        "target_metric": target_metric,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "best": best,
        "baselines": baselines,
        "top_rules": scored[:10],
    }

    print(f"Examples: {len(records)}")
    print(f"Train/Eval: {len(train_records)}/{len(eval_records)}")
    print(f"Optimized metric: {target_metric}")
    print()
    print("Baselines on eval:")
    for item in baselines:
        print(
            f"- {item['rule']['name']}: "
            f"selected={item['eval']['selected_rewritten']}/{item['eval']['n']} "
            f"{format_metrics(item['eval'], ks)}"
        )
    print()
    print("Best threshold rule:")
    print(f"- {best['rule']['name']}")
    print(
        f"  train selected={best['train']['selected_rewritten']}/{best['train']['n']} "
        f"{format_metrics(best['train'], ks)}"
    )
    print(
        f"  eval  selected={best['eval']['selected_rewritten']}/{best['eval']['n']} "
        f"{format_metrics(best['eval'], ks)}"
    )
    print("  eval selected labels:", best["eval"]["selected_rewritten_labels"])

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"\nWrote tuning summary to {output_path}")


if __name__ == "__main__":
    main()
