#!/usr/bin/env python3
"""Gather retrieval and reader metrics into tables.

Example:
  python scripts/gather_metric_tables.py \
    --run "Fine-tuned base=checkpoint/final_fsmodqa_squad_tr_full_deavg/results/results_base" \
    --run "Fine-tuned rewrite=checkpoint/final_fsmodqa_squad_tr_full_deavg/results/results_rewrite" \
    --run "Off-the-shelf=checkpoint/fsmodqa_off_the_shelf/results" \
    --run "Off-the-shelf root=checkpoint/fsmodqa_off_the_shelf" \
    --csv-prefix checkpoint/metric_tables
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PREFERRED_RETRIEVAL_KEYS = [
    "num_queries",
    "num_found",
    "missed",
    "mrr",
    "mean_rank_found",
    "hit@1",
    "hit@5",
    "hit@10",
    "hit@25",
    "hit@50",
    "hit@100",
]
PREFERRED_READER_KEYS = [
    "num_examples",
    "num_predictions",
    "missing_predictions",
    "em",
    "f1",
    "answer_precision",
    "recall",
    "k_precision",
    "jaccard",
    "contains_gold_answer",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return payload


def parse_run(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(value)
    return path.name, path


def topk_from_name(path: Path) -> int | None:
    match = re.search(r"top(\d+)", path.name)
    return int(match.group(1)) if match else None


def ordered_keys(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    keys = {key for row in rows for key in row["metrics"]}
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---:" if column not in {"Run", "File", "Metric type"} else "---" for column in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)



def csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([[csv_value(value) for value in row] for row in rows])


def collect_metrics(runs: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retrieval_rows = []
    reader_rows = []

    for run_name, directory in runs:
        for path in sorted(directory.glob("*.json")):
            if "per_example" in path.name:
                continue
            metrics = load_json(path)
            if "retrieval_metrics" in path.name:
                retrieval_rows.append({"run": run_name, "file": path.name, "metrics": metrics})
                continue
            if path.name.startswith("test_metrics"):
                topk = topk_from_name(path)
                reader_rows.append(
                    {
                        "run": run_name,
                        "top_k": topk if topk is not None else "",
                        "file": path.name,
                        "metrics": metrics,
                    }
                )

    reader_rows.sort(key=lambda row: (row["run"], row["top_k"] if isinstance(row["top_k"], int) else -1, row["file"]))
    retrieval_rows.sort(key=lambda row: (row["run"], row["file"]))
    return retrieval_rows, reader_rows


def build_tables(retrieval_rows: list[dict[str, Any]], reader_rows: list[dict[str, Any]]) -> str:
    parts = []

    if retrieval_rows:
        keys = ordered_keys(retrieval_rows, PREFERRED_RETRIEVAL_KEYS)
        rows = [[row["run"], row["file"], *[row["metrics"].get(key) for key in keys]] for row in retrieval_rows]
        parts.append("## Retrieval Metrics\n\n" + markdown_table(["Run", "File", *keys], rows))

    if reader_rows:
        keys = ordered_keys(reader_rows, PREFERRED_READER_KEYS)
        rows = [
            [row["run"], row["top_k"], row["file"], *[row["metrics"].get(key) for key in keys]]
            for row in reader_rows
        ]
        parts.append("## Reader Metrics\n\n" + markdown_table(["Run", "Top-k", "File", *keys], rows))

    return "\n\n".join(parts) + ("\n" if parts else "")



def build_table_data(
    retrieval_rows: list[dict[str, Any]],
    reader_rows: list[dict[str, Any]],
) -> tuple[tuple[list[str], list[list[Any]]] | None, tuple[list[str], list[list[Any]]] | None]:
    retrieval_table = None
    reader_table = None

    if retrieval_rows:
        keys = ordered_keys(retrieval_rows, PREFERRED_RETRIEVAL_KEYS)
        retrieval_table = (
            ["Run", "File", *keys],
            [[row["run"], row["file"], *[row["metrics"].get(key) for key in keys]] for row in retrieval_rows],
        )

    if reader_rows:
        keys = ordered_keys(reader_rows, PREFERRED_READER_KEYS)
        reader_table = (
            ["Run", "Top-k", "File", *keys],
            [
                [row["run"], row["top_k"], row["file"], *[row["metrics"].get(key) for key in keys]]
                for row in reader_rows
            ],
        )

    return retrieval_table, reader_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as NAME=DIR. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    parser.add_argument(
        "--csv-prefix",
        type=Path,
        help="Optional CSV prefix. Writes PREFIX_retrieval.csv and PREFIX_reader.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [parse_run(value) for value in args.run]
    retrieval_rows, reader_rows = collect_metrics(runs)
    tables = build_tables(retrieval_rows, reader_rows)
    print(tables, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tables, encoding="utf-8")
    if args.csv_prefix:
        retrieval_table, reader_table = build_table_data(retrieval_rows, reader_rows)
        if retrieval_table:
            write_csv(args.csv_prefix.with_name(args.csv_prefix.name + "_retrieval.csv"), *retrieval_table)
        if reader_table:
            write_csv(args.csv_prefix.with_name(args.csv_prefix.name + "_reader.csv"), *reader_table)


if __name__ == "__main__":
    main()
