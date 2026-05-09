#!/usr/bin/env python3
"""Retrieve passages with a lightweight BM25 baseline."""

import argparse
import csv
import heapq
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


STOPWORDS = {
    "acaba",
    "ama",
    "ancak",
    "bir",
    "biri",
    "bu",
    "cok",
    "da",
    "de",
    "daha",
    "diye",
    "en",
    "gibi",
    "icin",
    "ile",
    "ise",
    "ki",
    "mi",
    "mu",
    "mü",
    "mı",
    "ne",
    "neden",
    "nerede",
    "nereden",
    "nereye",
    "nasil",
    "nasıl",
    "o",
    "olan",
    "olarak",
    "oldu",
    "olur",
    "ve",
    "veya",
    "ya",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", required=True, help="QA JSON from prepare_qa_data.py.")
    parser.add_argument(
        "--passages",
        required=True,
        nargs="+",
        help="One or more TSV files from export_passages.py/export_squad_contexts.py.",
    )
    parser.add_argument("--output", required=True, help="Retrieved FiD/DPR-style JSON.")
    parser.add_argument("--n-docs", type=int, default=100)
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="Optional passage limit for quick experiments.",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("ı", "i")
    return text


def tokenize(text):
    tokens = re.findall(r"[\wçğıöşü]+", normalize(text), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def load_passages(paths, max_passages=None):
    passages = []
    doc_lengths = []
    term_freqs = []
    inverted = defaultdict(list)

    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin, delimiter="\t")
            for row in reader:
                if max_passages is not None and len(passages) >= max_passages:
                    break

                passage_id = row["id"]
                text = row["text"]
                title = row["title"]
                tokens = tokenize(f"{title} {text}")
                counts = Counter(tokens)

                internal_id = len(passages)
                passages.append({"id": passage_id, "title": title, "text": text})
                doc_lengths.append(sum(counts.values()))
                term_freqs.append(counts)
                for token in counts:
                    inverted[token].append(internal_id)

                if len(passages) % 100000 == 0:
                    print(f"Indexed {len(passages)} passages")

        if max_passages is not None and len(passages) >= max_passages:
            break

    avgdl = sum(doc_lengths) / max(1, len(doc_lengths))
    return passages, doc_lengths, term_freqs, inverted, avgdl


def bm25_scores(query, passages, doc_lengths, term_freqs, inverted, avgdl, k1, b):
    query_terms = Counter(tokenize(query))
    scores = defaultdict(float)
    total_docs = len(passages)

    for term, query_tf in query_terms.items():
        postings = inverted.get(term)
        if not postings:
            continue
        df = len(postings)
        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
        for doc_id in postings:
            tf = term_freqs[doc_id][term]
            denom = tf + k1 * (1 - b + b * doc_lengths[doc_id] / avgdl)
            scores[doc_id] += query_tf * idf * (tf * (k1 + 1)) / denom

    return scores


def retrieve_one(example, passages, doc_lengths, term_freqs, inverted, avgdl, n_docs, k1, b):
    scores = bm25_scores(
        example["question"],
        passages,
        doc_lengths,
        term_freqs,
        inverted,
        avgdl,
        k1,
        b,
    )
    top = heapq.nlargest(n_docs, scores.items(), key=lambda item: item[1])
    ctxs = []
    for doc_id, score in top:
        passage = passages[doc_id]
        ctxs.append(
            {
                "id": passage["id"],
                "title": passage["title"],
                "text": passage["text"],
                "score": f"{score:.6f}",
            }
        )
    return ctxs


def main():
    args = parse_args()

    print(f"Loading passages from {', '.join(args.passages)}")
    passages, doc_lengths, term_freqs, inverted, avgdl = load_passages(
        args.passages,
        args.max_passages,
    )
    print(f"Indexed {len(passages)} passages with {len(inverted)} unique terms")

    with Path(args.qa).open(encoding="utf-8") as fin:
        qa_data = json.load(fin)

    output = []
    for idx, example in enumerate(qa_data):
        new_example = dict(example)
        new_example["ctxs"] = retrieve_one(
            example,
            passages,
            doc_lengths,
            term_freqs,
            inverted,
            avgdl,
            args.n_docs,
            args.k1,
            args.b,
        )
        output.append(new_example)
        if (idx + 1) % 100 == 0:
            print(f"Retrieved for {idx + 1} questions")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(output, fout, ensure_ascii=False, indent=2)

    print(f"Wrote retrieved data to {output_path}")


if __name__ == "__main__":
    main()
