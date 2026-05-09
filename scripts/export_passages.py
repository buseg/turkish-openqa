#!/usr/bin/env python3
"""Export the prepared knowledge source to FiD/DPR passage TSV format."""

import argparse
import csv
from pathlib import Path

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="odqa_data/final_knowledge_source_chunked",
        help="Path created by download_data.ipynb with save_to_disk().",
    )
    parser.add_argument(
        "--output",
        default="data/passages.tsv",
        help="Output TSV path: id, text, title.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_from_disk(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    with output_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["id", "text", "title"])
        for idx, row in enumerate(dataset.select(range(total))):
            writer.writerow([idx, row["text"], row["title"]])

    print(f"Wrote {total} passages to {output_path}")


if __name__ == "__main__":
    main()
