#!/usr/bin/env python3
"""Filter QA/retrieval JSON files down to examples with non-empty answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def clean_answers(raw_answers: Any) -> list[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, dict):
        raw_answers = raw_answers.get("text", [])
    if isinstance(raw_answers, str):
        answer = raw_answers.strip()
        return [answer] if answer else []
    if isinstance(raw_answers, list):
        answers = []
        for answer in raw_answers:
            if isinstance(answer, dict):
                answer = answer.get("text", "")
            answer = str(answer).strip() if answer is not None else ""
            if answer:
                answers.append(answer)
        return answers
    answer = str(raw_answers).strip()
    return [answer] if answer else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected {args.input} to contain a JSON list.")

    filtered = []
    for example in data:
        answers = clean_answers(example.get("answers", []))
        if not answers:
            continue
        example = dict(example)
        example["answers"] = answers
        filtered.append(example)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Input examples: {len(data)}")
    print(f"Answerable examples: {len(filtered)}")
    print(f"Removed empty-answer examples: {len(data) - len(filtered)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
