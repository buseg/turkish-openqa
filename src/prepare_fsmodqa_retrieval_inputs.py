#!/usr/bin/env python3
"""Prepare local knowledge and SQuAD splits for FSMODQA dense retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_dataset(path: Path) -> Any:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("Install `datasets` in the active environment before running this script.") from exc
    return load_from_disk(str(path))


def text_from(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return str(value)
    return ""


def answers_from(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("text", [])
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        answers = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("text", "")
            if item:
                answers.append(str(item))
        return answers
    return [str(raw)]


def write_corpus(dataset: Any, path: Path, limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(dataset):
            if limit > 0 and index >= limit:
                break
            title = text_from(row, "title", "name")
            text = text_from(row, "text", "context", "passage")
            if not text:
                continue
            docid = text_from(row, "id", "docid") or f"ks-{index}"
            out.write(
                json.dumps(
                    {"id": docid, "docid": docid, "title": title, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def write_queries(dataset: Any, split: str, path: Path, limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(dataset):
            if limit > 0 and index >= limit:
                break
            question = text_from(row, "question", "query")
            answers = answers_from(row.get("answers") or row.get("answer"))
            if not question:
                continue
            qid = text_from(row, "id", "qid") or f"{split}-{index}"
            out.write(
                json.dumps(
                    {
                        "id": qid,
                        "question": question,
                        "answers": answers or ["placeholder"],
                        "lang": "tr",
                        "cl_answers": {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knowledge-source-path",
        type=Path,
        default=ROOT / "odqa_data" / "final_knowledge_source_chunked_200",
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "odqa_data" / "squad_tr_processed_train")
    parser.add_argument("--validation-path", type=Path, default=ROOT / "odqa_data" / "squad_tr_processed_validation")
    parser.add_argument("--test-path", type=Path, default=ROOT / "odqa_data" / "squad_tr_processed_test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "odqa_data" / "fsmodqa_retrieval")
    parser.add_argument("--knowledge-limit", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    args = parser.parse_args()

    knowledge = load_dataset(args.knowledge_source_path)
    corpus_count = write_corpus(knowledge, args.output_dir / "corpus.jsonl", args.knowledge_limit)
    print(f"Wrote {corpus_count} corpus passages to {args.output_dir / 'corpus.jsonl'}")

    for split, split_path, limit in [
        ("train", args.train_path, args.max_train_examples),
        ("validation", args.validation_path, args.max_validation_examples),
        ("test", args.test_path, args.max_test_examples),
    ]:
        if not split_path.exists():
            print(f"Skipping missing {split} dataset: {split_path}")
            continue
        dataset = load_dataset(split_path)
        count = write_queries(dataset, split, args.output_dir / f"{split}.query.jsonl", limit)
        print(f"Wrote {count} {split} queries to {args.output_dir / f'{split}.query.jsonl'}")


if __name__ == "__main__":
    main()
