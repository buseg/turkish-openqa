#!/usr/bin/env python3
"""Build FiD reader JSON from QA data and retrieved passages."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa",
        required=True,
        help="QA JSON from scripts/prepare_qa_data.py.",
    )
    parser.add_argument(
        "--retrieved",
        default=None,
        help="Retrieved JSON from external/fid/passage_retrieval.py. If omitted, oracle ctxs are used.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-context", type=int, default=100)
    return parser.parse_args()


def load_json(path):
    with Path(path).open(encoding="utf-8") as fin:
        return json.load(fin)


def by_id(records):
    return {str(record["id"]): record for record in records}


def main():
    args = parse_args()
    qa_data = load_json(args.qa)
    retrieved_by_id = by_id(load_json(args.retrieved)) if args.retrieved else {}

    reader_data = []
    for qa_example in qa_data:
        example_id = str(qa_example["id"])
        source = retrieved_by_id.get(example_id, qa_example)
        ctxs = source.get("ctxs", [])[: args.n_context]

        if not ctxs and "oracle_ctx" in qa_example:
            ctxs = [
                {
                    "id": f"oracle-{example_id}",
                    "title": qa_example["oracle_ctx"]["title"],
                    "text": qa_example["oracle_ctx"]["text"],
                    "score": "1.0",
                    "hasanswer": True,
                }
            ]

        reader_data.append(
            {
                "id": example_id,
                "question": qa_example["question"],
                "answers": qa_example.get("answers", []),
                "target": qa_example.get("target", ""),
                "ctxs": ctxs,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(reader_data, fout, ensure_ascii=False, indent=2)

    print(f"Wrote {len(reader_data)} reader examples to {output_path}")


if __name__ == "__main__":
    main()
