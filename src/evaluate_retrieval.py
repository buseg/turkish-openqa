#!/usr/bin/env python3
"""Evaluate FSMODQA retrieval rankings.

The expected input is a JSONL ranking file like:

  {"qid": "...", "pids": ["squad-<qid>-...", "..."]}

For SQuAD-derived corpus passages, a query is counted as retrieved when one of
its ranked passage ids contains the qid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "qid" not in row or "pids" not in row:
                raise ValueError(f"{path}:{line_number} must contain 'qid' and 'pids'.")
            rows.append(row)
    return rows


def load_qid_aliases(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}

    with path.open(encoding="utf-8") as handle:
        raw_aliases = json.load(handle)

    aliases: dict[str, list[str]] = {}
    for qid, raw_value in raw_aliases.items():
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        aliases[str(qid)] = [str(value) for value in values]
    return aliases


def normalize_qid_alias(alias: str) -> str:
    if alias.startswith("squad-"):
        alias = alias.removeprefix("squad-")
    if alias.endswith("-"):
        alias = alias[:-1]
    return alias


def pid_matches_qid(pid: str, qid: str, match_mode: str) -> bool:
    if match_mode == "contains":
        return qid in pid
    if match_mode == "squad-prefix":
        return pid.startswith(f"squad-{qid}-")
    raise ValueError(f"Unknown match mode: {match_mode}")


def first_relevant_rank(
    pids: list[str],
    qid: str,
    match_mode: str,
    qid_aliases: dict[str, list[str]] | None = None,
) -> int | None:
    accepted_qids = [qid]
    if qid_aliases:
        accepted_qids.extend(normalize_qid_alias(alias) for alias in qid_aliases.get(qid, []))

    for index, pid in enumerate(pids, start=1):
        if any(pid_matches_qid(str(pid), accepted_qid, match_mode) for accepted_qid in accepted_qids):
            return index
    return None


def evaluate(
    rows: list[dict[str, Any]],
    ks: list[int],
    match_mode: str,
    qid_aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    ranks = [
        first_relevant_rank([str(pid) for pid in row["pids"]], str(row["qid"]), match_mode, qid_aliases)
        for row in rows
    ]
    found_ranks = [rank for rank in ranks if rank is not None]

    metrics: dict[str, Any] = {
        "num_queries": len(rows),
        "num_found": len(found_ranks),
        "missed": len(rows) - len(found_ranks),
        "mrr": mean((1.0 / rank) if rank is not None else 0.0 for rank in ranks) if ranks else 0.0,
        "mean_rank_found": mean(found_ranks) if found_ranks else 0.0,
    }
    for k in ks:
        metrics[f"hit@{k}"] = (
            mean(1.0 if rank is not None and rank <= k else 0.0 for rank in ranks)
            if ranks
            else 0.0
        )
    return metrics


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True, help="Path to test_top100.jsonl.")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25, 50, 100],
        help="Cutoffs for hit@k.",
    )
    parser.add_argument(
        "--first-n-examples",
        type=int,
        default=0,
        help="Only evaluate the first N examples. 0 means no limit.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("contains", "squad-prefix"),
        default="squad-prefix",
        help="How to decide whether a pid is relevant for a qid.",
    )
    parser.add_argument(
        "--qid-aliases",
        type=Path,
        help="Optional JSON mapping qid -> equivalent retained qid from duplicate passage deduplication.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.rankings)
    if args.first_n_examples > 0:
        rows = rows[:args.first_n_examples]
    qid_aliases = load_qid_aliases(args.qid_aliases)
    metrics = evaluate(rows, sorted(set(args.k)), args.match_mode, qid_aliases)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    if args.output:
        write_json(args.output, metrics)


if __name__ == "__main__":
    main()
