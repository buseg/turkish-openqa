#!/usr/bin/env python3
"""Build labeled data for learning when question rewriting helps retrieval."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from common_qa import answers_from, contains_answer, get_context_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="Retrieved JSON for original questions.")
    parser.add_argument("--rewritten", required=True, help="Retrieved JSON for rewritten questions.")
    parser.add_argument("--output", required=True, help="Output JSONL selector dataset.")
    return parser.parse_args()


def first_hit_rank(example: dict) -> int | None:
    answers = answers_from(example.get("answers") or example.get("target"))
    for rank, ctx in enumerate(example.get("ctxs", []), start=1):
        if contains_answer(get_context_text(ctx), answers):
            return rank
    return None


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ı", "i")
    return re.findall(r"[\wçğıöşü]+", normalized, flags=re.UNICODE)


def safe_score(example: dict, index: int) -> float:
    ctxs = example.get("ctxs", [])
    if index >= len(ctxs):
        return 0.0
    try:
        return float(ctxs[index].get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def top_score(example: dict) -> float:
    return safe_score(example, 0)


def score_gap(example: dict) -> float:
    return safe_score(example, 0) - safe_score(example, 1)


def query_len(example: dict) -> int:
    return max(1, len(tokenize(example.get("question", ""))))


def norm_score(example: dict) -> float:
    return top_score(example) / query_len(example)


def rank_value(rank: int | None) -> int:
    return rank if rank is not None else 10**9


def compact_context(example: dict, index: int = 0, chars: int = 220) -> dict:
    ctxs = example.get("ctxs", [])
    if index >= len(ctxs):
        return {"title": "", "text": ""}
    ctx = ctxs[index]
    text = " ".join(ctx.get("text", "").split())
    if len(text) > chars:
        text = text[:chars].rstrip() + "..."
    return {"title": ctx.get("title", ""), "text": text}


def make_record(original: dict, rewritten: dict) -> dict:
    original_rank = first_hit_rank(original)
    rewritten_rank = first_hit_rank(rewritten)
    original_rank_value = rank_value(original_rank)
    rewritten_rank_value = rank_value(rewritten_rank)
    original_question = original.get("original_question", original.get("question", ""))
    rewritten_question = rewritten.get("question", "")
    original_len = query_len(original)
    rewritten_len = query_len(rewritten)
    original_top = top_score(original)
    rewritten_top = top_score(rewritten)
    original_norm = norm_score(original)
    rewritten_norm = norm_score(rewritten)

    if rewritten_rank_value < original_rank_value:
        label = "rewritten_better"
    elif original_rank_value < rewritten_rank_value:
        label = "original_better"
    else:
        label = "same"

    return {
        "id": str(original["id"]),
        "original_question": original_question,
        "rewritten_question": rewritten_question,
        "rewrite_changed": original_question != rewritten_question,
        "answers": answers_from(original.get("answers")),
        "label": label,
        "label_rewritten_better": int(label == "rewritten_better"),
        "original_first_hit_rank": original_rank,
        "rewritten_first_hit_rank": rewritten_rank,
        "original_hit_at_100": original_rank is not None and original_rank <= 100,
        "rewritten_hit_at_100": rewritten_rank is not None and rewritten_rank <= 100,
        "features": {
            "original_query_len": original_len,
            "rewritten_query_len": rewritten_len,
            "query_len_delta": rewritten_len - original_len,
            "original_top_score": original_top,
            "rewritten_top_score": rewritten_top,
            "top_score_delta": rewritten_top - original_top,
            "top_score_ratio": rewritten_top / original_top if original_top else 0.0,
            "original_norm_top_score": original_norm,
            "rewritten_norm_top_score": rewritten_norm,
            "norm_top_score_delta": rewritten_norm - original_norm,
            "norm_top_score_ratio": rewritten_norm / original_norm if original_norm else 0.0,
            "original_score_gap_1_2": score_gap(original),
            "rewritten_score_gap_1_2": score_gap(rewritten),
            "score_gap_delta": score_gap(rewritten) - score_gap(original),
        },
        "top_contexts": {
            "original": compact_context(original),
            "rewritten": compact_context(rewritten),
        },
    }


def main() -> None:
    args = parse_args()
    with Path(args.original).open(encoding="utf-8") as handle:
        original_data = json.load(handle)
    with Path(args.rewritten).open(encoding="utf-8") as handle:
        rewritten_data = json.load(handle)

    rewritten_by_id = {str(example["id"]): example for example in rewritten_data}
    records = []
    missing = 0
    for original in original_data:
        if not answers_from(original.get("answers")):
            continue
        rewritten = rewritten_by_id.get(str(original["id"]))
        if rewritten is None:
            missing += 1
            continue
        records.append(make_record(original, rewritten))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = {}
    for record in records:
        counts[record["label"]] = counts.get(record["label"], 0) + 1

    print(f"Wrote {len(records)} selector records to {output_path}")
    if missing:
        print(f"Skipped examples missing rewritten retrieval: {missing}")
    for label in ["rewritten_better", "original_better", "same"]:
        print(f"{label}: {counts.get(label, 0)}")


if __name__ == "__main__":
    main()
