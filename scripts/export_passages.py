#!/usr/bin/env python3
"""Export a passage corpus to TSV: id, text, title."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def iter_rows(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        for row in data:
            yield row
        return

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("Install `datasets` to export a Hugging Face dataset.") from exc
    dataset = load_from_disk(str(path))
    for row in dataset:
        yield dict(row)


def text_from(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for index, row in enumerate(iter_rows(args.input)):
            if args.limit > 0 and count >= args.limit:
                break
            text = text_from(row, "text", "context", "passage")
            if not text:
                continue
            pid = text_from(row, "id", "docid") or str(index)
            title = text_from(row, "title", "name")
            writer.writerow([pid, text, title])
            count += 1
    print(f"Wrote {count} passages to {args.output}")


if __name__ == "__main__":
    main()
