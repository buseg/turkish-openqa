#!/usr/bin/env python3
"""Select per-question retrieval results from original and rewritten queries."""

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
    parser.add_argument("--output", required=True, help="Output JSON with selected retrievals.")
    parser.add_argument(
        "--common-only",
        action="store_true",
        help="Keep only examples that exist in both original and rewritten files.",
    )
    parser.add_argument(
        "--strategy",
        default="normalized-top-score",
        choices=["top-score", "normalized-top-score", "oracle"],
        help="Selection strategy. Oracle uses answer labels and is only an analysis upper bound.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Require rewritten confidence to exceed original confidence by this ratio.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Additional absolute margin required after applying ratio.",
    )
    parser.add_argument("--ks", default="1,5,10,20,50,100", help="Recall@K values to print.")
    return parser.parse_args()


def first_hit_rank(example: dict) -> int | None:
    answers = answers_from(example.get("answers") or example.get("target"))
    for rank, ctx in enumerate(example.get("ctxs", []), start=1):
        if contains_answer(get_context_text(ctx), answers):
            return rank
    return None


def top_score(example: dict) -> float:
    ctxs = example.get("ctxs", [])
    if not ctxs:
        return 0.0
    try:
        return float(ctxs[0].get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def query_length(question: str) -> int:
    text = unicodedata.normalize("NFKC", question).casefold().replace("ı", "i")
    tokens = re.findall(r"[\wçğıöşü]+", text, flags=re.UNICODE)
    return max(1, len(tokens))


def confidence(example: dict, strategy: str) -> float:
    score = top_score(example)
    if strategy == "normalized-top-score":
        return score / query_length(example.get("question", ""))
    return score


def choose_rewritten(original: dict, rewritten: dict, strategy: str, ratio: float, margin: float) -> bool:
    if strategy == "oracle":
        original_rank = first_hit_rank(original) or 10**9
        rewritten_rank = first_hit_rank(rewritten) or 10**9
        return rewritten_rank < original_rank

    original_confidence = confidence(original, strategy)
    rewritten_confidence = confidence(rewritten, strategy)
    return rewritten_confidence > (original_confidence * ratio + margin)


def recall_at_k(examples: list[dict], ks: list[int]) -> dict[int, float]:
    ranks = [first_hit_rank(example) for example in examples]
    total = len(ranks)
    return {k: sum(rank is not None and rank <= k for rank in ranks) / total for k in ks}


def main() -> None:
    args = parse_args()
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]

    with Path(args.original).open(encoding="utf-8") as handle:
        original_data = json.load(handle)
    with Path(args.rewritten).open(encoding="utf-8") as handle:
        rewritten_data = json.load(handle)

    rewritten_by_id = {str(example["id"]): example for example in rewritten_data}
    selected = []
    rewritten_count = 0
    missing = 0

    for original in original_data:
        qid = str(original["id"])
        rewritten = rewritten_by_id.get(qid)
        if rewritten is None:
            missing += 1
            if args.common_only:
                continue
            chosen = dict(original)
            source = "original"
        elif choose_rewritten(original, rewritten, args.strategy, args.ratio, args.margin):
            chosen = dict(rewritten)
            source = "rewritten"
            rewritten_count += 1
        else:
            chosen = dict(original)
            source = "original"

        if rewritten is not None:
            chosen["original_question"] = original.get("original_question", original.get("question"))
            chosen["rewritten_question"] = rewritten.get("question")
            chosen["original_top_score"] = top_score(original)
            chosen["rewritten_top_score"] = top_score(rewritten)
        chosen["selected_retrieval"] = source
        chosen["selection_strategy"] = args.strategy
        selected.append(chosen)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    answerable = [example for example in selected if answers_from(example.get("answers"))]
    metrics = recall_at_k(answerable, ks)
    print(f"Wrote {len(selected)} selected examples to {output_path}")
    print(f"Selected rewritten retrieval: {rewritten_count}/{len(selected)}")
    if missing and args.common_only:
        print(f"Skipped original-only examples: {missing}")
    elif missing:
        print(f"Missing rewritten examples: {missing}")
    print(f"Examples: {len(answerable)}")
    for k in ks:
        print(f"Recall@{k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
