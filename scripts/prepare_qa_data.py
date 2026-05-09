#!/usr/bin/env python3
"""Convert SQuAD-TR saved dataset to FiD/DPR-style QA JSON files."""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="odqa_data/squad_tr",
        help="Path created by download_data.ipynb with save_to_disk().",
    )
    parser.add_argument("--train-output", default="data/squad_tr_train.json")
    parser.add_argument("--dev-output", default="data/squad_tr_dev.json")
    parser.add_argument(
        "--include-oracle-ctx",
        action="store_true",
        help="Attach each example's gold SQuAD context as a ctx for reader smoke tests.",
    )
    parser.add_argument(
        "--keep-unanswered",
        action="store_true",
        help="Keep examples with empty answers. By default they are skipped.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split row limit for quick smoke tests.",
    )
    return parser.parse_args()


def answer_texts(example):
    answers = example.get("answers", {})
    if isinstance(answers, dict):
        return list(answers.get("text", []))
    return list(answers)


def convert_split(split, include_oracle_ctx=False, limit=None, keep_unanswered=False):
    records = []
    for i, example in enumerate(split):
        answers = answer_texts(example)
        if not answers and not keep_unanswered:
            continue

        record = {
            "id": str(example.get("id", i)),
            "question": example["question"],
            "answers": answers,
            "target": answers[0] if answers else "",
            "ctxs": [],
            "oracle_ctx": {
                "title": example.get("title", ""),
                "text": example.get("context", ""),
            },
        }
        if include_oracle_ctx:
            record["ctxs"] = [
                {
                    "id": f"oracle-{record['id']}",
                    "title": record["oracle_ctx"]["title"],
                    "text": record["oracle_ctx"]["text"],
                    "score": "1.0",
                    "hasanswer": True,
                }
            ]
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def write_json(path, records):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(records, fout, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} examples to {output_path}")


def main():
    args = parse_args()
    dataset = load_from_disk(args.input)

    write_json(
        args.train_output,
        convert_split(
            dataset["train"],
            args.include_oracle_ctx,
            args.limit,
            args.keep_unanswered,
        ),
    )
    write_json(
        args.dev_output,
        convert_split(
            dataset["validation"],
            args.include_oracle_ctx,
            args.limit,
            args.keep_unanswered,
        ),
    )


if __name__ == "__main__":
    main()
