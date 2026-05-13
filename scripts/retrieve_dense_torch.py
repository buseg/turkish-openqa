#!/usr/bin/env python3
"""Run dense retrieval over FSMODQA embedding files without FAISS.

The FSMODQA encoder saves ``(embeddings, ids)`` tuples with ``torch.save``.
This script loads query and passage embeddings, computes inner-product scores,
and writes a JSONL ranking compatible with ``external/FSMODQA/test_reader.py``:

    {"qid": "...", "pids": ["...", "..."], "scores": [12.3, 11.9]}
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-embeddings", required=True, type=Path)
    parser.add_argument(
        "--passage-embeddings",
        required=True,
        help="Path or glob pattern for one or more FSMODQA passage embedding files.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument(
        "--pooling",
        choices=("auto", "mean", "first", "flatten"),
        default="auto",
        help="How to reduce 3D token-level embeddings before inner-product search.",
    )
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--passage-batch-size", type=int, default=50000)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before scoring.")
    return parser.parse_args()


def embedding_files(pattern: str) -> list[Path]:
    if any(char in pattern for char in "*?[]"):
        files = [Path(path) for path in sorted(glob.glob(pattern))]
    else:
        files = [Path(pattern)]
    if not files:
        raise FileNotFoundError(f"No passage embedding files matched: {pattern}")
    return files


def pool_embeddings(embeddings: torch.Tensor, pooling: str) -> torch.Tensor:
    if embeddings.ndim == 2:
        return embeddings
    if embeddings.ndim != 3:
        raise ValueError(f"Expected 2D or 3D embeddings, got shape {tuple(embeddings.shape)}")
    if pooling == "auto":
        pooling = "mean"
    if pooling == "mean":
        return embeddings.mean(dim=1)
    if pooling == "first":
        return embeddings[:, 0, :]
    if pooling == "flatten":
        return embeddings.flatten(start_dim=1)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def load_embeddings(path: Path, pooling: str) -> tuple[torch.Tensor, list[str]]:
    embeddings, ids = torch.load(path, map_location="cpu")
    if not isinstance(ids, list):
        ids = list(ids)
    return pool_embeddings(embeddings.float(), pooling), [str(item) for item in ids]


def batched_range(total: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def merge_topk(
    current_scores: torch.Tensor | None,
    current_ids: list[list[str]] | None,
    new_scores: torch.Tensor,
    new_ids: list[list[str]],
    depth: int,
) -> tuple[torch.Tensor, list[list[str]]]:
    if current_scores is None or current_ids is None:
        return new_scores, new_ids

    merged_scores = torch.cat([current_scores, new_scores], dim=1)
    merged_ids = [left + right for left, right in zip(current_ids, new_ids)]
    keep_scores, keep_indices = torch.topk(merged_scores, k=min(depth, merged_scores.shape[1]), dim=1)
    keep_ids = [
        [ids[index] for index in row_indices.tolist()]
        for ids, row_indices in zip(merged_ids, keep_indices)
    ]
    return keep_scores, keep_ids


def main() -> None:
    args = parse_args()
    query_embeddings, query_ids = load_embeddings(args.query_embeddings, args.pooling)
    if args.normalize:
        query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)
    depth = min(args.depth, args.passage_batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for q_start, q_end in batched_range(len(query_ids), args.query_batch_size):
            q_batch = query_embeddings[q_start:q_end]
            best_scores: torch.Tensor | None = None
            best_ids: list[list[str]] | None = None

            for passage_file in embedding_files(args.passage_embeddings):
                passage_embeddings, passage_ids = load_embeddings(passage_file, args.pooling)
                if args.normalize:
                    passage_embeddings = torch.nn.functional.normalize(passage_embeddings, dim=1)
                for p_start, p_end in batched_range(len(passage_ids), args.passage_batch_size):
                    p_batch = passage_embeddings[p_start:p_end]
                    scores = q_batch @ p_batch.T
                    local_depth = min(depth, scores.shape[1])
                    top_scores, top_indices = torch.topk(scores, k=local_depth, dim=1)
                    top_ids = [
                        [passage_ids[p_start + index] for index in row_indices.tolist()]
                        for row_indices in top_indices
                    ]
                    best_scores, best_ids = merge_topk(
                        best_scores,
                        best_ids,
                        top_scores,
                        top_ids,
                        args.depth,
                    )

            assert best_scores is not None and best_ids is not None
            for qid, scores, pids in zip(query_ids[q_start:q_end], best_scores, best_ids):
                output.write(
                    json.dumps(
                        {
                            "qid": qid,
                            "pids": pids[: args.depth],
                            "scores": [float(score) for score in scores[: args.depth]],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
