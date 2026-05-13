#!/usr/bin/env python3
"""Fine-tune a transformer cross-encoder reranker on BM25 candidates.

Input is the JSONL produced by ``build_knowledge_selector_data.py``. The model
sees ``(question, passage)`` pairs and predicts whether the passage contains a
gold answer. This is stronger than the TF-IDF selector because self-attention
can model interactions between the question and the passage tokens.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL from build_knowledge_selector_data.py.")
    parser.add_argument("--model-name", default="distilbert-base-multilingual-cased")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", default=".hf_cache")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-negatives-per-question", type=int, default=3)
    parser.add_argument("--random-negatives-per-question", type=int, default=3)
    parser.add_argument("--max-train-questions", type=int, default=1000)
    parser.add_argument("--max-eval-questions", type=int, default=300)
    parser.add_argument("--ks", default="1,5,10,25,50,100")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_qid(row: dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("id"))


def row_rank(row: dict[str, Any]) -> int:
    return int(row.get("rank") or row.get("ctx_rank") or 10**9)


def row_text(row: dict[str, Any]) -> tuple[str, str]:
    question = row.get("question", "")
    title = row.get("title", row.get("ctx_title", ""))
    text = row.get("text", "")
    return question, f"{title}. {text}".strip()


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row_qid(row), []).append(row)
    return groups


def split_qids(groups: dict[str, list[dict[str, Any]]], test_size: float, seed: int) -> tuple[list[str], list[str]]:
    qids = sorted(groups)
    random.Random(seed).shuffle(qids)
    n_eval = max(1, int(len(qids) * test_size))
    return qids[n_eval:], qids[:n_eval]


def limit_qids(qids: list[str], limit: int) -> list[str]:
    if limit and limit > 0:
        return qids[:limit]
    return qids


def sample_training_rows(
    groups: dict[str, list[dict[str, Any]]],
    qids: list[str],
    *,
    top_negatives: int,
    random_negatives: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled = []
    for qid in qids:
        rows = sorted(groups[qid], key=row_rank)
        positives = [row for row in rows if int(row.get("label", 0)) == 1]
        negatives = [row for row in rows if int(row.get("label", 0)) == 0]
        selected_negatives = negatives[:top_negatives]
        tail = negatives[top_negatives:]
        if random_negatives > 0 and tail:
            selected_negatives.extend(rng.sample(tail, min(random_negatives, len(tail))))
        sampled.extend(positives + selected_negatives)
    rng.shuffle(sampled)
    return sampled


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        question, passage = row_text(row)
        encoded = self.tokenizer(
            question,
            passage,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(row.get("label", 0)), dtype=torch.long)
        return item


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def score_rows(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_length: int, batch_size: int, device: torch.device) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            questions, passages = zip(*(row_text(row) for row in batch_rows))
            encoded = tokenizer(
                list(questions),
                list(passages),
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(float(score) for score in probs.cpu())
    return scores


def first_positive_rank(rows: list[dict[str, Any]], key: str) -> int | None:
    sorted_rows = sorted(rows, key=lambda row: row[key])
    for index, row in enumerate(sorted_rows, start=1):
        if int(row.get("label", 0)) == 1:
            return index
    return None


def retrieval_metrics(rows: list[dict[str, Any]], ks: list[int], rank_key: str) -> dict[str, Any]:
    ranks = [first_positive_rank(group, rank_key) for group in grouped(rows).values()]
    found = [rank for rank in ranks if rank is not None]
    metrics: dict[str, Any] = {
        "examples": len(ranks),
        "found_any": len(found),
        "mrr": mean((1.0 / rank if rank is not None else 0.0) for rank in ranks) if ranks else 0.0,
        "mean_first_hit_rank": mean(found) if found else None,
    }
    for k in ks:
        metrics[f"recall@{k}"] = mean((rank is not None and rank <= k) for rank in ranks) if ranks else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ks = [int(item) for item in args.ks.split(",") if item]

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Install transformers first.") from exc

    rows = read_jsonl(args.input)
    groups = grouped(rows)
    train_qids, eval_qids = split_qids(groups, args.test_size, args.seed)
    train_qids = limit_qids(train_qids, args.max_train_questions)
    eval_qids = limit_qids(eval_qids, args.max_eval_questions)
    train_rows = sample_training_rows(
        groups,
        train_qids,
        top_negatives=args.top_negatives_per_question,
        random_negatives=args.random_negatives_per_question,
        seed=args.seed,
    )
    eval_rows = [row for qid in eval_qids for row in groups[qid]]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2, cache_dir=args.cache_dir)
    device = choose_device()
    model.to(device)

    loader = DataLoader(PairDataset(train_rows, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} train_loss={mean(losses):.4f}")

    eval_scores = score_rows(model, tokenizer, eval_rows, args.max_length, args.batch_size, device)
    for row, score in zip(eval_rows, eval_scores):
        row["cross_encoder_score"] = score
        row["cross_encoder_rank_key"] = -score
        row["bm25_rank_key"] = row_rank(row)

    metrics = {
        "model_name": args.model_name,
        "train_questions": len(train_qids),
        "eval_questions": len(eval_qids),
        "train_pairs": len(train_rows),
        "eval_pairs": len(eval_rows),
        "positive_train_pairs": sum(int(row.get("label", 0)) for row in train_rows),
        "positive_eval_pairs": sum(int(row.get("label", 0)) for row in eval_rows),
        "device": str(device),
        "bm25": retrieval_metrics(eval_rows, ks, "bm25_rank_key"),
        "cross_encoder": retrieval_metrics(eval_rows, ks, "cross_encoder_rank_key"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved cross-encoder reranker to {args.output_dir}")


if __name__ == "__main__":
    main()
