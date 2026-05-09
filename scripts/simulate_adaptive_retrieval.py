#!/usr/bin/env python3
"""Simulate adaptive top-k retrieval from complexity labels."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="JSONL from build_complexity_labels.py.")
    parser.add_argument("--ks", default="10,50,100", help="Fixed top-k baselines to report.")
    return parser.parse_args()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def hit_at(record, k):
    rank = record.get("first_hit_rank")
    return rank is not None and rank <= k


def fixed_metrics(records, ks):
    total = len(records)
    return {
        k: {
            "recall": sum(hit_at(record, k) for record in records) / total,
            "avg_contexts": k,
            "total_contexts": total * k,
        }
        for k in ks
    }


def adaptive_metrics(records):
    total = len(records)
    total_contexts = sum(record["target_k"] for record in records)
    hits = sum(hit_at(record, record["target_k"]) for record in records)
    counts = {}
    for record in records:
        counts[record["complexity"]] = counts.get(record["complexity"], 0) + 1
    return {
        "recall": hits / total,
        "avg_contexts": total_contexts / total,
        "total_contexts": total_contexts,
        "counts": counts,
    }


def main():
    args = parse_args()
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    records = [
        record
        for record in load_jsonl(args.labels)
        if record.get("complexity") != "unanswerable"
    ]
    if not records:
        raise SystemExit("No answerable records found.")

    fixed = fixed_metrics(records, ks)
    adaptive = adaptive_metrics(records)

    print(f"Examples: {len(records)}")
    print("Label distribution:")
    for label in ["easy", "medium", "hard"]:
        print(f"{label}: {adaptive['counts'].get(label, 0)}")

    print("\nFixed retrieval baselines:")
    for k in ks:
        item = fixed[k]
        print(
            f"top-{k}: recall={item['recall']:.4f} "
            f"avg_contexts={item['avg_contexts']:.1f} "
            f"total_contexts={item['total_contexts']}"
        )

    fixed_100_contexts = fixed.get(100, {"total_contexts": len(records) * 100})[
        "total_contexts"
    ]
    saved = 1 - adaptive["total_contexts"] / fixed_100_contexts
    print("\nOracle adaptive retrieval:")
    print(
        f"recall={adaptive['recall']:.4f} "
        f"avg_contexts={adaptive['avg_contexts']:.1f} "
        f"total_contexts={adaptive['total_contexts']} "
        f"context_saving_vs_top100={saved:.2%}"
    )


if __name__ == "__main__":
    main()
