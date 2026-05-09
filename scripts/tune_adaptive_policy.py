#!/usr/bin/env python3
"""Tune conservative threshold policies for adaptive retrieval budgets."""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


FEATURES = ["top_score", "norm_top_score", "score_gap", "norm_score_gap"]
TARGET_K = {"easy": 10, "medium": 50, "hard": 100}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="JSONL from build_complexity_labels.py.")
    parser.add_argument("--retrieved", required=True, help="Retrieved JSON with BM25 scores.")
    parser.add_argument("--output", default=None, help="Optional JSON summary output.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-recall-drop",
        type=float,
        default=0.01,
        help="Allowed train recall drop compared with fixed top-100.",
    )
    parser.add_argument("--max-thresholds", type=int, default=80)
    return parser.parse_args()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def load_retrieved(path):
    with Path(path).open(encoding="utf-8") as fin:
        return {str(example["id"]): example for example in json.load(fin)}


def tokenize(text):
    normalized = unicodedata.normalize("NFKC", text or "").lower().replace("ı", "i")
    return re.findall(r"[\wçğıöşü]+", normalized, flags=re.UNICODE)


def safe_score(example, index):
    ctxs = example.get("ctxs", []) if example else []
    if index >= len(ctxs):
        return 0.0
    try:
        return float(ctxs[index].get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def feature_values(record, retrieved_by_id):
    retrieved = retrieved_by_id.get(str(record["id"]))
    top1 = safe_score(retrieved, 0)
    top2 = safe_score(retrieved, 1)
    gap = top1 - top2
    qlen = max(1, len(tokenize(record.get("question", ""))))
    return {
        "top_score": top1,
        "norm_top_score": top1 / qlen,
        "score_gap": gap,
        "norm_score_gap": gap / qlen,
    }


def stratified_split(records, train_ratio, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for record in records:
        by_label[record["complexity"]].append(record)
    train = []
    eval_records = []
    for label_records in by_label.values():
        rng.shuffle(label_records)
        split = int(len(label_records) * train_ratio)
        train.extend(label_records[:split])
        eval_records.extend(label_records[split:])
    rng.shuffle(train)
    rng.shuffle(eval_records)
    return train, eval_records


def hit_at(record, k):
    rank = record.get("first_hit_rank")
    return rank is not None and rank <= k


def fixed_recall(records, k=100):
    return sum(hit_at(record, k) for record in records) / len(records)


def predict_from_values(values, rule):
    value = values[rule["feature"]]
    if value >= rule["easy_threshold"]:
        return "easy"
    if value >= rule["medium_threshold"]:
        return "medium"
    return "hard"


def evaluate(records, values_by_id, rule):
    predictions = [predict_from_values(values_by_id[str(record["id"])], rule) for record in records]
    total_contexts = sum(TARGET_K[label] for label in predictions)
    hits = sum(hit_at(record, TARGET_K[label]) for record, label in zip(records, predictions))
    accuracy = sum(record["complexity"] == label for record, label in zip(records, predictions))
    return {
        "recall": hits / len(records),
        "avg_contexts": total_contexts / len(records),
        "total_contexts": total_contexts,
        "context_saving_vs_top100": 1 - total_contexts / (len(records) * 100),
        "label_accuracy": accuracy / len(records),
        "predicted_counts": dict(Counter(predictions)),
    }


def threshold_candidates(records, values_by_id, feature, max_thresholds):
    values = sorted({values_by_id[str(record["id"])][feature] for record in records})
    if len(values) <= max_thresholds:
        return values
    return [
        values[round(i * (len(values) - 1) / (max_thresholds - 1))]
        for i in range(max_thresholds)
    ]


def candidate_rules(train_records, values_by_id, max_thresholds):
    for feature in FEATURES:
        thresholds = threshold_candidates(train_records, values_by_id, feature, max_thresholds)
        for easy_threshold in thresholds:
            for medium_threshold in thresholds:
                if medium_threshold > easy_threshold:
                    continue
                yield {
                    "feature": feature,
                    "easy_threshold": easy_threshold,
                    "medium_threshold": medium_threshold,
                }


def rule_name(rule):
    return (
        f"{rule['feature']}: easy>={rule['easy_threshold']:.6g}, "
        f"medium>={rule['medium_threshold']:.6g}"
    )


def main():
    args = parse_args()
    records = [
        record
        for record in load_jsonl(args.labels)
        if record.get("complexity") in TARGET_K
    ]
    retrieved_by_id = load_retrieved(args.retrieved)
    values_by_id = {
        str(record["id"]): feature_values(record, retrieved_by_id)
        for record in records
    }
    train_records, eval_records = stratified_split(records, args.train_ratio, args.seed)

    train_top100 = fixed_recall(train_records, 100)
    eval_top100 = fixed_recall(eval_records, 100)
    best = None
    for rule in candidate_rules(train_records, values_by_id, args.max_thresholds):
        train_result = evaluate(train_records, values_by_id, rule)
        if train_result["recall"] < train_top100 - args.max_recall_drop:
            continue
        candidate = (
            train_result["avg_contexts"],
            -train_result["recall"],
            rule,
            train_result,
        )
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise SystemExit("No policy satisfied the recall-drop constraint.")

    _, _, best_rule, train_result = best
    eval_result = evaluate(eval_records, values_by_id, best_rule)
    oracle_rule = {"feature": "top_score", "easy_threshold": float("inf"), "medium_threshold": float("inf")}
    all_hard = {
        "recall": eval_top100,
        "avg_contexts": 100.0,
        "total_contexts": len(eval_records) * 100,
        "context_saving_vs_top100": 0.0,
        "label_accuracy": sum(record["complexity"] == "hard" for record in eval_records) / len(eval_records),
        "predicted_counts": {"hard": len(eval_records)},
    }
    oracle_predictions = [record["complexity"] for record in eval_records]
    oracle_contexts = sum(TARGET_K[label] for label in oracle_predictions)
    oracle_hits = sum(
        hit_at(record, TARGET_K[label])
        for record, label in zip(eval_records, oracle_predictions)
    )
    oracle = {
        "recall": oracle_hits / len(eval_records),
        "avg_contexts": oracle_contexts / len(eval_records),
        "total_contexts": oracle_contexts,
        "context_saving_vs_top100": 1 - oracle_contexts / (len(eval_records) * 100),
        "label_accuracy": 1.0,
        "predicted_counts": dict(Counter(oracle_predictions)),
    }

    summary = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "max_recall_drop": args.max_recall_drop,
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "train_top100_recall": train_top100,
        "eval_top100_recall": eval_top100,
        "best_rule": best_rule,
        "best_rule_name": rule_name(best_rule),
        "train": train_result,
        "eval": eval_result,
        "all_hard": all_hard,
        "oracle": oracle,
    }

    print(f"Train/Eval: {len(train_records)}/{len(eval_records)}")
    print(f"Train top-100 recall: {train_top100:.4f}")
    print(f"Eval top-100 recall: {eval_top100:.4f}")
    print(f"Max train recall drop: {args.max_recall_drop:.4f}")
    print(f"Best policy: {rule_name(best_rule)}")
    print(
        "Train policy: "
        f"recall={train_result['recall']:.4f} "
        f"avg_contexts={train_result['avg_contexts']:.1f} "
        f"saving={train_result['context_saving_vs_top100']:.2%} "
        f"counts={train_result['predicted_counts']}"
    )
    print(
        "Eval all-hard: "
        f"recall={all_hard['recall']:.4f} "
        f"avg_contexts={all_hard['avg_contexts']:.1f}"
    )
    print(
        "Eval policy: "
        f"recall={eval_result['recall']:.4f} "
        f"avg_contexts={eval_result['avg_contexts']:.1f} "
        f"saving={eval_result['context_saving_vs_top100']:.2%} "
        f"label_acc={eval_result['label_accuracy']:.4f} "
        f"counts={eval_result['predicted_counts']}"
    )
    print(
        "Eval oracle: "
        f"recall={oracle['recall']:.4f} "
        f"avg_contexts={oracle['avg_contexts']:.1f} "
        f"saving={oracle['context_saving_vs_top100']:.2%}"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(summary, fout, ensure_ascii=False, indent=2)
        print(f"\nWrote policy summary to {output_path}")


if __name__ == "__main__":
    main()
