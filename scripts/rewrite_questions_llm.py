#!/usr/bin/env python3
"""Rewrite Turkish questions with an Ollama-hosted LLM."""

from __future__ import annotations

import argparse
import json
import subprocess

from common_qa import load_json, write_json


PROMPT = """Türkçe açık alan soru cevaplama için aşağıdaki soruyu daha açık ve arama motoruna uygun hale getir.
Sadece yeniden yazılmış soruyu döndür. Cevap verme.

Soru: {question}
Yeniden yazılmış soru:"""


def ollama_rewrite(question: str, model: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=PROMPT.format(question=question),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    rewritten = result.stdout.strip().splitlines()[0].strip()
    return rewritten if rewritten else question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemma2:2b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = load_json(args.input)
    changed = 0
    for index, example in enumerate(data):
        if args.limit > 0 and index >= args.limit:
            break
        original = example.get("question", "")
        rewritten = original if args.dry_run else ollama_rewrite(original, args.model)
        example["original_question"] = original
        example["question"] = rewritten
        example["question_rewritten"] = rewritten != original
        changed += int(rewritten != original)
        print(json.dumps({"id": example.get("id"), "original": original, "rewritten": rewritten}, ensure_ascii=False))
    write_json(args.output, data[: args.limit] if args.limit > 0 else data)
    print(f"Rewritten: {changed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
