#!/usr/bin/env python3
"""Convert retrieved QA JSON into FSMODQA reader input files."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved", required=True, help="Retrieved JSON with question/answers/ctxs.")
    parser.add_argument("--output-dir", required=True, help="Directory for FSMODQA files.")
    parser.add_argument("--n-context", type=int, default=100)
    parser.add_argument("--prefix", default="", help="Optional prefix for generated passage ids.")
    parser.add_argument(
        "--keep-unanswerable",
        action="store_true",
        help="Keep empty-answer examples with a placeholder answer for reader generation.",
    )
    parser.add_argument("--placeholder-answer", default="no answer")
    return parser.parse_args()


def unique_pid(raw_id, title, index, prefix):
    base = str(raw_id) if raw_id is not None else f"ctx-{index}"
    if prefix:
        base = f"{prefix}-{base}"
    return base


def clean_answers(raw_answers):
    if raw_answers is None:
        return []
    if isinstance(raw_answers, dict):
        raw_answers = raw_answers.get("text", [])
    if isinstance(raw_answers, str):
        answer = raw_answers.strip()
        return [answer] if answer else []
    if isinstance(raw_answers, list):
        answers = []
        for answer in raw_answers:
            if isinstance(answer, dict):
                answer = answer.get("text", "")
            answer = str(answer).strip() if answer is not None else ""
            if answer:
                answers.append(answer)
        return answers
    answer = str(raw_answers).strip()
    return [answer] if answer else []


def main():
    args = parse_args()
    with Path(args.retrieved).open(encoding="utf-8") as fin:
        data = json.load(fin)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    query_path = output_dir / "queries.jsonl"
    reader_path = output_dir / "reader.jsonl"

    corpus = {}
    query_count = 0
    reader_count = 0
    missing_answers = 0
    short_context_examples = 0

    with query_path.open("w", encoding="utf-8") as qout, reader_path.open("w", encoding="utf-8") as rout:
        for example in data:
            qid = str(example["id"])
            answers = clean_answers(example.get("answers", []))
            if not answers:
                if not args.keep_unanswerable:
                    missing_answers += 1
                    continue
                answers = [args.placeholder_answer]

            qout.write(
                json.dumps(
                    {
                        "id": qid,
                        "question": example.get("question", ""),
                        "answers": answers,
                        "lang": "tr",
                        "cl_answers": {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            query_count += 1

            pids = []
            for index, ctx in enumerate(example.get("ctxs", [])[: args.n_context]):
                pid = unique_pid(ctx.get("id"), ctx.get("title", ""), index, args.prefix)
                if pid not in corpus:
                    corpus[pid] = {
                        "id": pid,
                        "docid": pid,
                        "title": ctx.get("title", ""),
                        "text": ctx.get("text", ""),
                    }
                pids.append(pid)

            if len(pids) < args.n_context:
                short_context_examples += 1

            rout.write(json.dumps({"qid": qid, "pids": pids}, ensure_ascii=False) + "\n")
            reader_count += 1

    with corpus_path.open("w", encoding="utf-8") as cout:
        for passage in corpus.values():
            cout.write(json.dumps(passage, ensure_ascii=False) + "\n")

    print(f"Wrote corpus: {corpus_path} ({len(corpus)} passages)")
    print(f"Wrote queries: {query_path} ({query_count} queries)")
    print(f"Wrote reader: {reader_path} ({reader_count} examples)")
    if missing_answers:
        print(f"Skipped unanswered examples: {missing_answers}")
    if short_context_examples:
        print(f"Examples with fewer than {args.n_context} contexts: {short_context_examples}")


if __name__ == "__main__":
    main()
