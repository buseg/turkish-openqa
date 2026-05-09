#!/usr/bin/env python3
"""Tune simple threshold rules for choosing original vs rewritten retrieval."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL from build_selector_data.py.")
    parser.add_argument("--output", default=None, help="Optional JSON summary output.")
    parser.add_argument("--target-k", type=int, default=20, help="Recall@K to optimize.")
    parser.add_argument("--ks", default="1,5,10,20,50,100", help="Recall@K values to report.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-thresholds", type=int, default=200)
    return parser.parse_args()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def rank_value(record, source):
    key = f"{source}_first_hit_rank"
    return record.get(key)


def hit_at(rank, k):
    return rank is not None and rank <= k


def selected_rank(record, rule):
    source = "rewritten" if choose_rewritten(record, rule) else "original"
    return rank_value(record, source)


def choose_rewritten(record, rule):
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


def evaluate(records, rule, ks):
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


def finite_values(records, feature):
    values = []
    for record in records:
        value = record["features"].get(feature)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return sorted(set(values))


def thresholds(records, feature, max_thresholds):
    values = finite_values(records, feature)
    if not values:
        return []
    if len(values) <= max_thresholds:
        return values
    return [
        values[round(i * (len(values) - 1) / (max_thresholds - 1))]
        for i in range(max_thresholds)
    ]


def candidate_rules(train_records, max_thresholds):
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


def split_records(records, train_ratio, seed):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def format_metrics(result, ks):
    return " ".join(f"R@{k}={result['metrics'][f'recall@{k}']:.4f}" for k in ks)


def main():
    args = parse_args()
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    target_metric = f"recall@{args.target_k}"
    records = load_jsonl(args.input)
    train_records, eval_records = split_records(records, args.train_ratio, args.seed)

    scored = []
    for rule in candidate_rules(train_records, args.max_thresholds):
        train_result = evaluate(train_records, rule, ks)
        eval_result = evaluate(eval_records, rule, ks) if eval_records else train_result
        scored.append(
            {
                "rule": rule,
                "train": train_result,
                "eval": eval_result,
            }
        )

    scored.sort(
        key=lambda item: (
            item["train"]["metrics"][target_metric],
            item["train"]["metrics"].get("recall@100", 0.0),
            -item["train"]["selected_rewritten"],
        ),
        reverse=True,
    )
    best = scored[0]
    baselines = [
        {
            "rule": rule,
            "train": evaluate(train_records, rule, ks),
            "eval": evaluate(eval_records, rule, ks) if eval_records else evaluate(train_records, rule, ks),
        }
        for rule in [
            {"kind": "original", "name": "all_original"},
            {"kind": "rewritten", "name": "all_rewritten"},
            {"kind": "oracle", "name": "oracle_upper_bound"},
        ]
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
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(summary, fout, ensure_ascii=False, indent=2)
        print(f"\nWrote tuning summary to {output_path}")


if __name__ == "__main__":
    main()
