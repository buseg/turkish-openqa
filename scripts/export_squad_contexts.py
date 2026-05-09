#!/usr/bin/env python3
"""Export unique SQuAD-TR contexts as extra retrieval passages."""

import argparse
import csv
from pathlib import Path

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="odqa_data/squad_tr",
        help="Path created by download_data.ipynb with save_to_disk().",
    )
    parser.add_argument(
        "--output",
        default="data/squad_tr_contexts.tsv",
        help="Output TSV path: id, text, title.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=["train", "validation"],
        help="SQuAD-TR splits to export.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional context limit for quick smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_from_disk(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["id", "text", "title"])

        for split_name in args.splits:
            for row in dataset[split_name]:
                key = (row.get("title", ""), row.get("context", ""))
                if key in seen:
                    continue
                seen.add(key)

                writer.writerow(
                    [
                        f"squad-{split_name}-{count}",
                        row["context"],
                        row.get("title", ""),
                    ]
                )
                count += 1

                if args.limit is not None and count >= args.limit:
                    print(f"Wrote {count} SQuAD-TR contexts to {output_path}")
                    return

    print(f"Wrote {count} SQuAD-TR contexts to {output_path}")


if __name__ == "__main__":
    main()
