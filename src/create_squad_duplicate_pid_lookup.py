#!/usr/bin/env python3
"""Create a lookup from dropped duplicate SQuAD qids to retained corpus qids.

The SQuAD passages in the retrieval corpus were deduplicated by exact context
text before chunking. When several questions share the same context, only the
first qid-backed passage remains in corpus.jsonl. This script reconstructs that
relationship so retrieval evaluation can treat the retained duplicate passage
as gold for the dropped qid too.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SQUAD_PID_RE = re.compile(r"^squad-(.+)-\d+-\d+$")


def chunked_squad_text(context: str, chunk_size: int) -> str | None:
    """Mirror the notebook's SQuAD chunking for short contexts."""
    if not context:
        return None

    processed_text = context
    for punc in string.punctuation:
        processed_text = processed_text.replace(punc, f" {punc} ")

    tokens = processed_text.split()
    if not tokens or len(tokens) > chunk_size:
        return None

    chunk_str = " ".join(tokens[:chunk_size])
    return chunk_str.replace(" .", ".").replace(" ,", ",").replace(" ' ", "'")


def qid_from_squad_pid(pid: str) -> str | None:
    match = SQUAD_PID_RE.match(pid)
    return match.group(1) if match else None


def load_squad_corpus_text_lookup(corpus_path: Path) -> dict[str, str]:
    text_to_qid: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row.get("id") or row.get("docid") or "")
            qid = qid_from_squad_pid(pid)
            if qid is None:
                continue
            text = str(row.get("text") or "")
            text_to_qid.setdefault(text, qid)
    return text_to_qid


def iter_dataset_rows(dataset_paths: list[Path]):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("Install `datasets`, or run this with the odqa_env environment.") from exc

    for dataset_path in dataset_paths:
        dataset = load_from_disk(str(dataset_path))
        for row in dataset:
            yield dataset_path.name, row


def build_lookup(dataset_paths: list[Path], corpus_path: Path, chunk_size: int) -> dict[str, str]:
    text_to_retained_qid = load_squad_corpus_text_lookup(corpus_path)
    duplicate_lookup: dict[str, str] = {}

    for _split_name, row in iter_dataset_rows(dataset_paths):
        qid = str(row.get("id") or "")
        context = str(row.get("context") or row.get("text") or "")
        text = chunked_squad_text(context, chunk_size)
        if not qid or text is None:
            continue

        retained_qid = text_to_retained_qid.get(text)
        if retained_qid and retained_qid != qid:
            duplicate_lookup[qid] = retained_qid

    return duplicate_lookup


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-paths",
        type=Path,
        nargs="+",
        default=[
            ROOT / "odqa_data" / "squad_tr_processed_train",
            ROOT / "odqa_data" / "squad_tr_processed_validation",
            ROOT / "odqa_data" / "squad_tr_processed_test",
        ],
        help="Processed SQuAD dataset directories to scan.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "odqa_data" / "fsmodqa_retrieval" / "corpus.jsonl",
        help="Retrieval corpus JSONL containing deduplicated SQuAD passages.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "odqa_data" / "fsmodqa_retrieval" / "squad_duplicate_qid_lookup.json",
        help="Output JSON mapping dropped qid -> retained qid.",
    )
    parser.add_argument("--chunk-size", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lookup = build_lookup(args.dataset_paths, args.corpus, args.chunk_size)
    write_json(args.output, lookup)
    print(f"Wrote {len(lookup)} duplicate qid mappings to {args.output}")


if __name__ == "__main__":
    main()
