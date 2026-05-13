#!/usr/bin/env python3
"""Slice a retrieved QA JSON file for smoke reader/evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--n-context", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        data: list[dict[str, Any]] = json.load(handle)

    data = data[: args.limit]
    if args.n_context is not None:
        for example in data:
            example["ctxs"] = list(example.get("ctxs", []))[: args.n_context]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {len(data)} examples to {args.output}")


if __name__ == "__main__":
    main()
