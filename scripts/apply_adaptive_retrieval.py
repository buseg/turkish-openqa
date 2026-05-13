#!/usr/bin/env python3
"""Apply adaptive retrieval budgets to a retrieved JSON file."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_retrieval import first_hit_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Retrieved JSON with up to top-100 ctxs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strategy", choices=("oracle-labels", "classifier"), default="oracle-labels")
    parser.add_argument("--labels", type=Path, help="JSONL labels for oracle-labels strategy.")
    parser.add_argument("--model", type=Path, help="Pickled sklearn classifier for classifier strategy.")
    parser.add_argument("--easy-k", type=int, default=10)
    parser.add_argument("--medium-k", type=int, default=50)
    parser.add_argument("--hard-k", type=int, default=100)
    parser.add_argument("--ks", default="1,5,10,25,50,100")
    parser.add_argument("--match-mode", choices=("answer", "oracle", "either"), default="answer")
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    return data


def load_labels(path: Path) -> dict[str, str]:
    labels = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[str(row["id"])] = row.get("label") or row.get("complexity") or "hard"
    return labels


def parse_ks(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(" ", ",").split(",") if item]


def register_retrieval_aware_pickle_helpers() -> None:
    """Expose feature helpers for models pickled from the training script."""
    try:
        import train_retrieval_aware_complexity_classifier as helpers
    except ImportError:
        return
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return
    for name in ("question_texts", "numeric_feature_rows"):
        if hasattr(helpers, name):
            setattr(main_module, name, getattr(helpers, name))


def budget_for(label: str, easy_k: int, medium_k: int, hard_k: int) -> int:
    if label == "easy":
        return easy_k
    if label == "medium":
        return medium_k
    return hard_k


def retrieval_metrics(examples: list[dict[str, Any]], ks: list[int], match_mode: str) -> dict[str, Any]:
    ranks = [first_hit_rank(example, match_mode) for example in examples]
    found = [rank for rank in ranks if rank is not None]
    metrics: dict[str, Any] = {
        "examples": len(examples),
        "found_any": len(found),
        "missing_any": len(examples) - len(found),
        "mrr": mean((1.0 / rank if rank is not None else 0.0) for rank in ranks) if ranks else 0.0,
        "mean_first_hit_rank": mean(found) if found else None,
    }
    for k in ks:
        metrics[f"recall@{k}"] = mean((rank is not None and rank <= k) for rank in ranks) if ranks else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    budgets = []

    if args.strategy == "oracle-labels":
        if not args.labels:
            raise SystemExit("--labels is required for oracle-labels strategy.")
        labels_by_id = load_labels(args.labels)

        def predict_label(example: dict[str, Any]) -> str:
            return labels_by_id.get(str(example.get("id")), "hard")

    else:
        if not args.model:
            raise SystemExit("--model is required for classifier strategy.")
        register_retrieval_aware_pickle_helpers()
        with args.model.open("rb") as handle:
            model = pickle.load(handle)

        def predict_label(example: dict[str, Any]) -> str:
            try:
                return str(model.predict([example])[0])
            except Exception:
                return str(model.predict([example.get("question", "")])[0])

    selected = []
    label_counts = {"easy": 0, "medium": 0, "hard": 0}
    for example in data:
        label = predict_label(example)
        budget = budget_for(label, args.easy_k, args.medium_k, args.hard_k)
        label_counts[label] = label_counts.get(label, 0) + 1
        budgets.append(budget)
        copy = dict(example)
        copy["adaptive_label"] = label
        copy["adaptive_k"] = budget
        copy["adaptive_strategy"] = args.strategy
        copy["ctxs"] = list(example.get("ctxs", []))[:budget]
        selected.append(copy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    ks = parse_ks(args.ks)
    metrics = retrieval_metrics(selected, ks, args.match_mode)
    metrics.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "strategy": args.strategy,
            "label_counts": label_counts,
            "avg_contexts": mean(budgets) if budgets else 0.0,
            "context_saving_vs_100": 1 - (mean(budgets) / 100) if budgets else 0.0,
        }
    )
    metrics_path = args.output.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
