#!/usr/bin/env python3
"""Build generic reader JSON by combining QA examples with retrieved contexts."""

from __future__ import annotations

import argparse

from common_qa import load_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--retrieved", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-context", type=int, default=100)
    args = parser.parse_args()

    qa_by_id = {str(ex["id"]): ex for ex in load_json(args.qa)}
    retrieved = load_json(args.retrieved)
    rows = []
    for item in retrieved:
        base = dict(qa_by_id.get(str(item.get("id")), item))
        base["ctxs"] = item.get("ctxs", [])[: args.n_context]
        rows.append(base)
    write_json(args.output, rows)
    print(f"Wrote {len(rows)} reader examples to {args.output}")


if __name__ == "__main__":
    main()
