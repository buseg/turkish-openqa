#!/usr/bin/env python3
"""Select per-question retrieval results from original and rewritten queries."""

import argparse
import json
import re
import unicodedata
from pathlib import Path

from evaluate_retrieval import first_hit_rank


def parse_args():
    parser = argparse.ArgumentParser()
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
        help=(
            "Selection strategy. oracle uses answer labels and should be used only "
            "as an analysis upper bound."
        ),
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


def top_score(example):
    ctxs = example.get("ctxs", [])
    if not ctxs:
        return 0.0
    try:
        return float(ctxs[0].get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def query_length(question):
    text = unicodedata.normalize("NFKC", question).lower().replace("ı", "i")
    tokens = re.findall(r"[\wçğıöşü]+", text, flags=re.UNICODE)
    return max(1, len(tokens))


def confidence(example, strategy):
    score = top_score(example)
    if strategy == "normalized-top-score":
        return score / query_length(example.get("question", ""))
    return score


def choose_rewritten(original, rewritten, strategy, ratio, margin):
    if strategy == "oracle":
        original_rank = first_hit_rank(original) or 10**9
        rewritten_rank = first_hit_rank(rewritten) or 10**9
        return rewritten_rank < original_rank

    original_confidence = confidence(original, strategy)
    rewritten_confidence = confidence(rewritten, strategy)
    return rewritten_confidence > (original_confidence * ratio + margin)


def recall_at_k(examples, ks):
    ranks = [first_hit_rank(example) for example in examples]
    total = len(ranks)
    return {
        k: sum(rank is not None and rank <= k for rank in ranks) / total
        for k in ks
    }


def main():
    args = parse_args()
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]

    with Path(args.original).open(encoding="utf-8") as fin:
        original_data = json.load(fin)
    with Path(args.rewritten).open(encoding="utf-8") as fin:
        rewritten_data = json.load(fin)

    rewritten_by_id = {str(example["id"]): example for example in rewritten_data}
    selected = []
    rewritten_count = 0
    missing = 0

    for original in original_data:
        qid = str(original["id"])
        rewritten = rewritten_by_id.get(qid)
        if rewritten is None:
            if args.common_only:
                missing += 1
                continue
            missing += 1
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
            chosen["original_question"] = original.get(
                "original_question",
                original.get("question"),
            )
            chosen["rewritten_question"] = rewritten.get("question")
            chosen["original_top_score"] = top_score(original)
            chosen["rewritten_top_score"] = top_score(rewritten)
        chosen["selected_retrieval"] = source
        chosen["selection_strategy"] = args.strategy
        selected.append(chosen)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(selected, fout, ensure_ascii=False, indent=2)

    answerable = [example for example in selected if example.get("answers")]
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
