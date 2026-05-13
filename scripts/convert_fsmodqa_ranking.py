#!/usr/bin/env python3
"""Convert FSMODQA dense rankings into the project's retrieved-JSON format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", required=True, type=Path, help="JSONL with qid/pids.")
    parser.add_argument("--corpus", required=True, type=Path, help="FSMODQA corpus.jsonl.")
    parser.add_argument("--queries", required=True, type=Path, help="FSMODQA queries.jsonl.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-context", type=int, default=100)
    parser.add_argument(
        "--gold",
        type=Path,
        help="Optional original retrieved JSON file; preserves oracle_ctx when available.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    return {str(row["id"]): row for row in rows}


def main() -> None:
    args = parse_args()
    corpus = {str(row["id"]): row for row in read_jsonl(args.corpus)}
    queries = {str(row["id"]): row for row in read_jsonl(args.queries)}
    gold = load_gold(args.gold)

    output_rows = []
    for ranking in read_jsonl(args.ranking):
        qid = str(ranking["qid"])
        query = queries.get(qid)
        if query is None:
            raise KeyError(f"Ranking references query id not found in queries file: {qid}")

        scores = ranking.get("scores", [])
        ctxs = []
        for rank, pid in enumerate(ranking.get("pids", [])[: args.n_context], start=1):
            pid = str(pid)
            passage = corpus.get(pid)
            if passage is None:
                raise KeyError(f"Ranking references passage id not found in corpus file: {pid}")
            score = scores[rank - 1] if rank - 1 < len(scores) else None
            ctxs.append(
                {
                    "id": pid,
                    "title": passage.get("title", ""),
                    "text": passage.get("text", ""),
                    "score": score,
                    "rank": rank,
                }
            )

        row = {
            "id": qid,
            "question": query.get("question", ""),
            "answers": query.get("answers", []),
            "target": query.get("answers", []),
            "ctxs": ctxs,
        }
        if qid in gold:
            oracle = gold[qid].get("oracle_ctx") or gold[qid].get("oracle_context")
            if oracle:
                row["oracle_ctx"] = oracle
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {len(output_rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
