#!/usr/bin/env python3
"""Train a lightweight question complexity classifier for adaptive retrieval."""

import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ["easy", "medium", "hard"]
TARGET_K = {"easy": 10, "medium": 50, "hard": 100}
QUESTION_TYPES = ["who", "when", "where", "how_many", "why", "how", "what", "which", "other"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="JSONL from build_complexity_labels.py.")
    parser.add_argument("--retrieved", default=None, help="Optional retrieved JSON for score features.")
    parser.add_argument(
        "--feature-set",
        default="question",
        choices=["question", "retrieval"],
        help="question uses only question text; retrieval also uses BM25 score features.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON summary output.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def load_retrieved(path):
    if not path:
        return {}
    with Path(path).open(encoding="utf-8") as fin:
        return {str(example["id"]): example for example in json.load(fin)}


def normalize(text):
    return unicodedata.normalize("NFKC", text or "").lower().replace("ı", "i")


def tokenize(text):
    return re.findall(r"[\wçğıöşü]+", normalize(text), flags=re.UNICODE)


def question_type(question):
    text = normalize(question)
    rules = [
        ("when", [r"\bne zaman\b", r"\bhangi yil\b", r"\bhangi tarihte\b", r"\byüzyil\b"]),
        ("where", [r"\bnerede\b", r"\bhangi ülke\b", r"\bhangi şehir\b", r"\bhangi bölg"]),
        ("who", [r"\bkim\b", r"\bkimin\b", r"\bkime\b", r"\bkimdir\b"]),
        ("how_many", [r"\bkaç\b", r"\bkac\b", r"\bne kadar\b"]),
        ("why", [r"\bneden\b", r"\bniçin\b"]),
        ("how", [r"\bnasil\b", r"\bnasıl\b"]),
        ("what", [r"\bne\b", r"\bnedir\b"]),
        ("which", [r"\bhangi\b"]),
    ]
    for label, patterns in rules:
        if any(re.search(pattern, text) for pattern in patterns):
            return label
    return "other"


def safe_score(example, index):
    if not example:
        return 0.0
    ctxs = example.get("ctxs", [])
    if index >= len(ctxs):
        return 0.0
    try:
        return float(ctxs[index].get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def feature_vector(record, retrieved_by_id, feature_set):
    question = record.get("question", "")
    tokens = tokenize(question)
    token_count = max(1, len(tokens))
    char_count = max(1, len(question))
    features = [
        float(token_count),
        float(char_count),
        sum(1 for token in tokens if token[:1].isupper()),
        1.0 if "?" in question else 0.0,
    ]
    qtype = question_type(question)
    features.extend(1.0 if qtype == item else 0.0 for item in QUESTION_TYPES)

    if feature_set == "retrieval":
        retrieved = retrieved_by_id.get(str(record["id"]))
        top1 = safe_score(retrieved, 0)
        top2 = safe_score(retrieved, 1)
        gap = top1 - top2
        features.extend(
            [
                top1,
                top2,
                gap,
                top1 / token_count,
                gap / token_count,
                top2 / top1 if top1 else 0.0,
            ]
        )
    return features


def stratified_split(records, train_ratio, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for record in records:
        by_label[record["complexity"]].append(record)

    train = []
    eval_ = []
    for label_records in by_label.values():
        rng.shuffle(label_records)
        split = int(len(label_records) * train_ratio)
        train.extend(label_records[:split])
        eval_.extend(label_records[split:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def mean_std(vectors):
    width = len(vectors[0])
    means = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(width)]
    stds = []
    for i, mean in enumerate(means):
        variance = sum((vector[i] - mean) ** 2 for vector in vectors) / len(vectors)
        stds.append(math.sqrt(variance) or 1.0)
    return means, stds


def scale(vector, means, stds):
    return [(value - mean) / std for value, mean, std in zip(vector, means, stds)]


def train_centroids(records, retrieved_by_id, feature_set):
    raw_vectors = [feature_vector(record, retrieved_by_id, feature_set) for record in records]
    means, stds = mean_std(raw_vectors)
    grouped = defaultdict(list)
    for record, vector in zip(records, raw_vectors):
        grouped[record["complexity"]].append(scale(vector, means, stds))

    centroids = {}
    for label, vectors in grouped.items():
        width = len(vectors[0])
        centroids[label] = [
            sum(vector[i] for vector in vectors) / len(vectors)
            for i in range(width)
        ]
    majority = Counter(record["complexity"] for record in records).most_common(1)[0][0]
    return {"means": means, "stds": stds, "centroids": centroids, "majority": majority}


def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def predict(record, model, retrieved_by_id, feature_set):
    vector = feature_vector(record, retrieved_by_id, feature_set)
    vector = scale(vector, model["means"], model["stds"])
    return min(
        model["centroids"],
        key=lambda label: distance(vector, model["centroids"][label]),
    )


def macro_f1(gold, pred):
    scores = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def hit_at(record, k):
    rank = record.get("first_hit_rank")
    return rank is not None and rank <= k


def retrieval_metrics(records, predicted_labels):
    total_contexts = sum(TARGET_K[label] for label in predicted_labels)
    hits = sum(
        hit_at(record, TARGET_K[label])
        for record, label in zip(records, predicted_labels)
    )
    return {
        "recall": hits / len(records),
        "avg_contexts": total_contexts / len(records),
        "total_contexts": total_contexts,
        "context_saving_vs_top100": 1 - total_contexts / (len(records) * 100),
    }


def evaluate(records, predicted_labels):
    gold = [record["complexity"] for record in records]
    accuracy = sum(g == p for g, p in zip(gold, predicted_labels)) / len(records)
    confusion = {
        gold_label: {pred_label: 0 for pred_label in LABELS}
        for gold_label in LABELS
    }
    for gold_label, pred_label in zip(gold, predicted_labels):
        confusion[gold_label][pred_label] += 1
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1(gold, predicted_labels),
        "label_counts": dict(Counter(predicted_labels)),
        "confusion": confusion,
        "retrieval": retrieval_metrics(records, predicted_labels),
    }


def fixed_predictions(records, label):
    return [label for _ in records]


def oracle_predictions(records):
    return [record["complexity"] for record in records]


def print_result(name, result):
    retrieval = result["retrieval"]
    print(
        f"{name}: acc={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
        f"recall={retrieval['recall']:.4f} avg_contexts={retrieval['avg_contexts']:.1f} "
        f"saving_vs_top100={retrieval['context_saving_vs_top100']:.2%} "
        f"pred_counts={result['label_counts']}"
    )


def main():
    args = parse_args()
    if args.feature_set == "retrieval" and not args.retrieved:
        raise SystemExit("--feature-set retrieval requires --retrieved")

    records = [
        record
        for record in load_jsonl(args.labels)
        if record.get("complexity") in LABELS
    ]
    retrieved_by_id = load_retrieved(args.retrieved)
    train_records, eval_records = stratified_split(records, args.train_ratio, args.seed)
    model = train_centroids(train_records, retrieved_by_id, args.feature_set)
    predictions = [
        predict(record, model, retrieved_by_id, args.feature_set)
        for record in eval_records
    ]

    results = {
        "feature_set": args.feature_set,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
        "train_distribution": dict(Counter(record["complexity"] for record in train_records)),
        "eval_distribution": dict(Counter(record["complexity"] for record in eval_records)),
        "classifier": evaluate(eval_records, predictions),
        "all_easy": evaluate(eval_records, fixed_predictions(eval_records, "easy")),
        "all_medium": evaluate(eval_records, fixed_predictions(eval_records, "medium")),
        "all_hard": evaluate(eval_records, fixed_predictions(eval_records, "hard")),
        "oracle": evaluate(eval_records, oracle_predictions(eval_records)),
    }

    print(f"Feature set: {args.feature_set}")
    print(f"Train/Eval: {len(train_records)}/{len(eval_records)}")
    print(f"Train distribution: {results['train_distribution']}")
    print(f"Eval distribution: {results['eval_distribution']}")
    print()
    print_result("all_easy", results["all_easy"])
    print_result("all_medium", results["all_medium"])
    print_result("all_hard", results["all_hard"])
    print_result("oracle", results["oracle"])
    print_result("classifier", results["classifier"])
    print("classifier confusion:", results["classifier"]["confusion"])

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=2)
        print(f"\nWrote classifier summary to {output_path}")


if __name__ == "__main__":
    main()
