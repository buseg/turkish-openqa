#!/usr/bin/env python3
"""Run BM25 retrieval over a TSV passage corpus."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from common_qa import answers_from, contains_answer, load_json, write_json


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE)


def load_passages(path: Path, limit: int = 0) -> tuple[list[str], list[str], list[str]]:
    ids, texts, titles = [], [], []
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "id":
                continue
            pid = row[0]
            text = row[1] if len(row) > 1 else ""
            title = row[2] if len(row) > 2 else ""
            ids.append(pid)
            texts.append(text)
            titles.append(title)
            if limit > 0 and len(ids) >= limit:
                break
    return ids, texts, titles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--passages", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-docs", type=int, default=100)
    parser.add_argument("--passage-limit", type=int, default=0)
    args = parser.parse_args()

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise SystemExit("Install rank_bm25 first: pip install rank_bm25") from exc

    qa = load_json(args.qa)
    ids, texts, titles = load_passages(args.passages, args.passage_limit)
    print(f"Loaded {len(ids)} passages")
    tokenized_corpus = [tokenize(f"{title} {text}") for title, text in zip(titles, texts)]
    bm25 = BM25Okapi(tokenized_corpus)

    output = []
    for index, example in enumerate(qa, start=1):
        query_tokens = tokenize(example.get("question", ""))
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.n_docs]
        answers = answers_from(example.get("answers") or example.get("target"))
        ctxs = []
        for passage_index in ranked:
            text = texts[passage_index]
            title = titles[passage_index]
            ctxs.append(
                {
                    "id": ids[passage_index],
                    "title": title,
                    "text": text,
                    "score": float(scores[passage_index]),
                    "hasanswer": contains_answer(f"{title} {text}", answers),
                }
            )
        row = dict(example)
        row["ctxs"] = ctxs
        output.append(row)
        if index % 100 == 0:
            print(f"Retrieved {index}/{len(qa)}")

    write_json(args.output, output)
    print(f"Wrote {len(output)} retrieved examples to {args.output}")


if __name__ == "__main__":
    main()
