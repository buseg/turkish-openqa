#!/usr/bin/env python3
"""Evaluate QA predictions with standard and expanded QA metrics.

The standard ``src/evaluate.py`` script measures EM/F1 and a passage-based
K-Precision field. This script is tailored for our retrieved JSON files: it uses
``ctxs`` as the evidence set, reports correctness/faithfulness, and handles
unanswerable questions with refusal metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable


REFUSAL_PATTERNS = [
    "bilmiyorum",
    "cevap yok",
    "yanıt yok",
    "bilgi yok",
    "bulunamadı",
    "bulunamadi",
    "metinde yok",
    "bağlamda yok",
    "baglamda yok",
    "cevaplanamaz",
    "yanıtlanamaz",
    "answer not found",
    "no answer",
    "not enough information",
    "cannot answer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="Gold retrieved JSON/JSONL with answers and ctxs.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-example-output", type=Path)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--answers-field", default="answers")
    return parser.parse_args()


def normalize(text: Any) -> str:
    text = str(text).casefold()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def tokenize(text: Any) -> list[str]:
    return re.findall(r"\w+", normalize(text), flags=re.UNICODE)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def token_overlap(prediction: str, reference: str) -> tuple[int, int, int]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    return sum(common.values()), len(pred_tokens), len(ref_tokens)


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize(prediction) == normalize(answer))


def precision(prediction: str, answer: str) -> float:
    common, pred_len, _ = token_overlap(prediction, answer)
    return safe_div(common, pred_len)


def recall(prediction: str, answer: str) -> float:
    common, _, answer_len = token_overlap(prediction, answer)
    return safe_div(common, answer_len)


def f1(prediction: str, answer: str) -> float:
    p = precision(prediction, answer)
    r = recall(prediction, answer)
    return safe_div(2 * p * r, p + r)


def max_over_answers(metric: Callable[[str, str], float], prediction: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    return max(metric(prediction, answer) for answer in answers)


def contains_gold_answer(prediction: str, answers: list[str]) -> float:
    norm_prediction = normalize(prediction)
    return float(any(normalize(answer) and normalize(answer) in norm_prediction for answer in answers))


def evidence_text(row: dict[str, Any]) -> str:
    ctxs = row.get("ctxs", [])
    if isinstance(ctxs, list) and ctxs:
        return " ".join(f"{ctx.get('title', '')} {ctx.get('text', '')}" for ctx in ctxs if isinstance(ctx, dict))
    oracle = row.get("oracle_ctx") or row.get("oracle_context") or {}
    if isinstance(oracle, dict):
        return f"{oracle.get('title', '')} {oracle.get('text', '')}"
    return str(row.get("context", "") or "")


def k_precision(prediction: str, evidence: str) -> float:
    common, pred_len, _ = token_overlap(prediction, evidence)
    return safe_div(common, pred_len)


def evidence_contains_prediction(prediction: str, evidence: str) -> float:
    pred = normalize(prediction)
    ev = normalize(evidence)
    if not pred:
        return 0.0
    return float(pred in ev)


def answers_from(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("text", [])
    if isinstance(raw, str):
        answer = raw.strip()
        return [answer] if answer else []
    if isinstance(raw, list):
        answers = []
        for answer in raw:
            if isinstance(answer, dict):
                answer = answer.get("text", "")
            answer = str(answer).strip() if answer is not None else ""
            if answer:
                answers.append(answer)
        return answers
    answer = str(raw).strip()
    return [answer] if answer else []


def nested(row: dict[str, Any], field: str, default: Any = None) -> Any:
    value: Any = row
    for part in field.split("."):
        if isinstance(value, dict):
            value = value.get(part, default)
        else:
            return default
    return value


def is_refusal(prediction: str) -> bool:
    norm = normalize(prediction)
    if not norm:
        return True
    return any(pattern in norm for pattern in REFUSAL_PATTERNS)


def load_json_or_jsonl(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = load_json_or_jsonl(path)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    return data


def load_predictions(path: Path) -> dict[str, str]:
    raw = load_json_or_jsonl(path)
    if isinstance(raw, dict):
        return {str(key): "" if value is None else str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        preds = {}
        for item in raw:
            qid = item.get("id") or item.get("qid")
            prediction = item.get("prediction", item.get("answer", item.get("text", "")))
            if qid is not None:
                preds[str(qid)] = "" if prediction is None else str(prediction)
        return preds
    raise ValueError(f"Expected {path} to contain dict/list predictions.")


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = load_dataset(args.dataset)
    predictions = load_predictions(args.predictions)
    examples = []
    missing = 0

    for index, row in enumerate(dataset):
        qid = str(nested(row, args.id_field, index))
        prediction = predictions.get(qid, "")
        if qid not in predictions:
            missing += 1
        answers = answers_from(nested(row, args.answers_field))
        answerable = bool(answers)
        evidence = evidence_text(row)
        refused = is_refusal(prediction)
        example = {
            "id": qid,
            "answerable": answerable,
            "prediction": prediction,
            "answers": answers,
            "refused": refused,
            "em": max_over_answers(exact_match, prediction, answers) if answerable else 0.0,
            "f1": max_over_answers(f1, prediction, answers) if answerable else 0.0,
            "answer_precision": max_over_answers(precision, prediction, answers) if answerable else 0.0,
            "correctness_recall": max_over_answers(recall, prediction, answers) if answerable else 0.0,
            "contains_gold_answer": contains_gold_answer(prediction, answers) if answerable else 0.0,
            "faithfulness_k_precision": k_precision(prediction, evidence) if not refused else 0.0,
            "evidence_contains_prediction": evidence_contains_prediction(prediction, evidence) if not refused else 0.0,
            "correct_refusal": float((not answerable) and refused),
            "false_refusal": float(answerable and refused),
            "answered_unanswerable": float((not answerable) and (not refused)),
        }
        examples.append(example)

    answerable_rows = [row for row in examples if row["answerable"]]
    unanswerable_rows = [row for row in examples if not row["answerable"]]
    answered_rows = [row for row in examples if not row["refused"]]
    summary = {
        "num_examples": len(examples),
        "num_predictions": len(predictions),
        "missing_predictions": missing,
        "answerable_examples": len(answerable_rows),
        "unanswerable_examples": len(unanswerable_rows),
        "em": avg(answerable_rows, "em"),
        "f1": avg(answerable_rows, "f1"),
        "answer_precision": avg(answerable_rows, "answer_precision"),
        "correctness_recall": avg(answerable_rows, "correctness_recall"),
        "contains_gold_answer": avg(answerable_rows, "contains_gold_answer"),
        "faithfulness_k_precision": avg(answered_rows, "faithfulness_k_precision"),
        "evidence_contains_prediction": avg(answered_rows, "evidence_contains_prediction"),
        "refusal_rate": mean(float(row["refused"]) for row in examples) if examples else 0.0,
        "correct_refusal_rate": avg(unanswerable_rows, "correct_refusal"),
        "answered_unanswerable_rate": avg(unanswerable_rows, "answered_unanswerable"),
        "false_refusal_rate": avg(answerable_rows, "false_refusal"),
    }
    return summary, examples


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    summary, examples = evaluate(args)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value * 100:.2f}")
        else:
            print(f"{key}: {value}")
    if args.output:
        write_json(args.output, summary)
    if args.per_example_output:
        write_json(args.per_example_output, examples)


if __name__ == "__main__":
    main()
