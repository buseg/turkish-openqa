#!/usr/bin/env python3
"""Apply lightweight rule-based Turkish question rewriting."""

from __future__ import annotations

import argparse
import re

from common_qa import load_json, write_json


PATTERNS = [
    (re.compile(r"\bne anlama geliyor\??$", re.IGNORECASE), " açılımı nedir?"),
    (re.compile(r"\bkimdir\??$", re.IGNORECASE), " kimdir?"),
    (re.compile(r"\bnedir\??$", re.IGNORECASE), " nedir?"),
]


def rewrite_question(question: str, mode: str) -> tuple[str, bool]:
    q = " ".join(str(question).split())
    if not q.endswith("?"):
        q = q + "?"
    if mode == "none":
        return q, q != question
    if mode == "keyword":
        replacements = {
            "hangi ülkede": "hangi ülkede bulunur",
            "ne zaman": "hangi tarihte",
            "kaç": "kaç adet",
        }
        for old, new in replacements.items():
            if old in q.casefold() and new not in q.casefold():
                return re.sub(old, new, q, flags=re.IGNORECASE), True
        return q, q != question
    if mode == "all":
        for pattern, replacement in PATTERNS:
            if pattern.search(q):
                return pattern.sub(replacement, q), True
        return q, q != question
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("none", "keyword", "all"), default="keyword")
    args = parser.parse_args()

    data = load_json(args.input)
    changed = 0
    for example in data:
        new_question, did_change = rewrite_question(example.get("question", ""), args.mode)
        example["original_question"] = example.get("question", "")
        example["question"] = new_question
        example["question_rewritten"] = did_change
        changed += int(did_change)
    write_json(args.output, data)
    print(f"Examples: {len(data)}")
    print(f"Rewritten: {changed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
