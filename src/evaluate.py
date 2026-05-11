#!/usr/bin/env python3
"""Evaluate QA predictions against a gold dataset.

Examples:
  python src/evaluate.py \
    --predictions checkpoint/fsmodqa_off_the_shelf/validation_reader_predictions.json \
    --dataset odqa_data/squad_tr_processed_validation

Metrics are printed as percentages, except counts.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable


METRIC_NAMES = [
    "em",
    "f1",
    "answer_precision",
    "recall",
    "k_precision",
    "jaccard",
    "contains_gold_answer",
]


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace."""
    text = str(text).casefold()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_answer(text), flags=re.UNICODE)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def exact_match(prediction: str, gold_answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def token_overlap(prediction: str, reference: str) -> tuple[int, int, int]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    return sum(common.values()), len(pred_tokens), len(ref_tokens)


def answer_precision(prediction: str, gold_answer: str) -> float:
    common, pred_len, _ = token_overlap(prediction, gold_answer)
    return safe_div(common, pred_len)


def answer_recall(prediction: str, gold_answer: str) -> float:
    common, _, gold_len = token_overlap(prediction, gold_answer)
    return safe_div(common, gold_len)


def f1_score(prediction: str, gold_answer: str) -> float:
    precision = answer_precision(prediction, gold_answer)
    recall = answer_recall(prediction, gold_answer)
    return safe_div(2 * precision * recall, precision + recall)


def jaccard_score(prediction: str, gold_answer: str) -> float:
    pred = set(tokenize(prediction))
    gold = set(tokenize(gold_answer))
    return safe_div(len(pred & gold), len(pred | gold))


def metric_max_over_answers(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    gold_answers: list[str],
) -> float:
    if not gold_answers:
        return 0.0
    return max(metric_fn(prediction, answer) for answer in gold_answers)


def k_precision(prediction: str, gold_passage: str) -> float:
    """Proportion of prediction tokens that appear in the gold passage K."""
    common, pred_len, _ = token_overlap(prediction, gold_passage)
    return safe_div(common, pred_len)


def contains_gold_answer(prediction: str, gold_answers: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    for answer in gold_answers:
        normalized_answer = normalize_answer(answer)
        if normalized_answer and normalized_answer in normalized_prediction:
            return 1.0
    return 0.0


def get_nested_value(row: Any, field_path: str, default: Any = None) -> Any:
    value = row
    for field in field_path.split("."):
        if isinstance(value, dict):
            value = value.get(field, default)
        else:
            return default
    return value


def answers_from(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("text", [])
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        answers: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("text", "")
            if item:
                answers.append(str(item))
        return answers
    return [str(raw)]


def load_json_or_jsonl(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def load_dataset(path: Path, split: str | None = None) -> Iterable[dict[str, Any]]:
    if path.suffix in {".json", ".jsonl"}:
        data = load_json_or_jsonl(path)
        if isinstance(data, dict) and split:
            data = data[split]
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError(f"Expected {path} to contain a list of examples.")
        return data

    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise SystemExit(
            "Install `datasets` or run this script in an environment that can read "
            "Hugging Face datasets saved with `load_from_disk`."
        ) from exc

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split is None:
            available = ", ".join(dataset.keys())
            raise ValueError(f"--split is required for DatasetDict inputs. Available splits: {available}")
        dataset = dataset[split]
    elif split is not None and hasattr(dataset, "keys") and split in dataset:
        dataset = dataset[split]
    return dataset


def load_predictions(path: Path) -> dict[str, str]:
    raw = load_json_or_jsonl(path)
    if isinstance(raw, dict):
        return {str(qid): "" if pred is None else str(pred) for qid, pred in raw.items()}
    if isinstance(raw, list):
        predictions: dict[str, str] = {}
        for item in raw:
            qid = item.get("id") or item.get("qid")
            prediction = item.get("prediction", item.get("answer", item.get("text", "")))
            if qid is not None:
                predictions[str(qid)] = "" if prediction is None else str(prediction)
        return predictions
    raise ValueError(f"Expected {path} to contain predictions as a JSON object or JSONL/list.")


def evaluate(
    dataset: Iterable[dict[str, Any]],
    predictions: dict[str, str],
    id_field: str,
    answers_field: str,
    passage_field: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    missing_predictions = 0

    for index, row in enumerate(dataset):
        qid = str(get_nested_value(row, id_field, index))
        prediction = predictions.get(qid, "")
        if qid not in predictions:
            missing_predictions += 1

        gold_answers = answers_from(get_nested_value(row, answers_field))
        gold_passage = str(get_nested_value(row, passage_field, "") or "")
        example = {
            "id": qid,
            "prediction": prediction,
            "answers": gold_answers,
            "em": metric_max_over_answers(exact_match, prediction, gold_answers),
            "f1": metric_max_over_answers(f1_score, prediction, gold_answers),
            "answer_precision": metric_max_over_answers(answer_precision, prediction, gold_answers),
            "recall": metric_max_over_answers(answer_recall, prediction, gold_answers),
            "k_precision": k_precision(prediction, gold_passage),
            "jaccard": metric_max_over_answers(jaccard_score, prediction, gold_answers),
            "contains_gold_answer": contains_gold_answer(prediction, gold_answers),
        }
        examples.append(example)

    summary = {
        "num_examples": len(examples),
        "num_predictions": len(predictions),
        "missing_predictions": missing_predictions,
    }
    for name in METRIC_NAMES:
        summary[name] = mean(example[name] for example in examples) if examples else 0.0
    return summary, examples


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSON/JSONL path.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Gold dataset path. Supports Hugging Face load_from_disk, JSON, or JSONL.",
    )
    parser.add_argument("--split", help="Dataset split name when --dataset is a DatasetDict or split JSON.")
    parser.add_argument("--id-field", default="id", help="Question id field. Supports dotted paths.")
    parser.add_argument("--answers-field", default="answers", help="Gold answers field. Supports dotted paths.")
    parser.add_argument(
        "--passage-field",
        default="context",
        help="Gold passage/context field used for K-Precision. Supports dotted paths.",
    )
    parser.add_argument("--output", type=Path, help="Optional path for summary metrics as JSON.")
    parser.add_argument("--per-example-output", type=Path, help="Optional path for per-example metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split)
    predictions = load_predictions(args.predictions)
    summary, examples = evaluate(
        dataset=dataset,
        predictions=predictions,
        id_field=args.id_field,
        answers_field=args.answers_field,
        passage_field=args.passage_field,
    )

    print(f"Examples: {summary['num_examples']}")
    print(f"Predictions: {summary['num_predictions']}")
    print(f"Missing predictions: {summary['missing_predictions']}")
    for name in METRIC_NAMES:
        print(f"{name}: {summary[name] * 100:.2f}")

    if args.output:
        write_json(args.output, summary)
    if args.per_example_output:
        write_json(args.per_example_output, examples)


if __name__ == "__main__":
    main()
