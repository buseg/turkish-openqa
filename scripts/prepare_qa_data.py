#!/usr/bin/env python3
"""Convert SQuAD-style QA data into compact JSON files used by this project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_qa import answers_from, write_json


def load_dataset_split(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    if path.suffix in {".json", ".jsonl"}:
        import json

        with path.open(encoding="utf-8") as handle:
            if path.suffix == ".jsonl":
                data = [json.loads(line) for line in handle if line.strip()]
            else:
                data = json.load(handle)
        if isinstance(data, dict) and split:
            data = data[split]
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError(f"Expected list-like data in {path}")
        return data

    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise SystemExit("Install `datasets` to read Hugging Face datasets saved to disk.") from exc

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split is None:
            raise ValueError(f"--split is required for DatasetDict input: {path}")
        dataset = dataset[split]
    return [dict(row) for row in dataset]


def convert(rows: list[dict[str, Any]], *, answerable_only: bool, limit: int) -> list[dict[str, Any]]:
    examples = []
    for index, row in enumerate(rows):
        answers = answers_from(row.get("answers") or row.get("answer"))
        if answerable_only and not answers:
            continue
        qid = str(row.get("id") or row.get("qid") or index)
        context = row.get("context") or row.get("text") or row.get("passage") or ""
        title = row.get("title") or row.get("name") or ""
        examples.append(
            {
                "id": qid,
                "question": row.get("question") or row.get("query") or "",
                "answers": answers,
                "target": answers[0] if answers else "",
                "oracle_ctx": {"title": title, "text": context},
            }
        )
        if limit > 0 and len(examples) >= limit:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", help="Split name for DatasetDict inputs.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-unanswerable", action="store_true")
    args = parser.parse_args()

    rows = load_dataset_split(args.input, args.split)
    examples = convert(rows, answerable_only=not args.include_unanswerable, limit=args.limit)
    write_json(args.output, examples)
    print(f"Input rows: {len(rows)}")
    print(f"Wrote examples: {len(examples)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
