#!/usr/bin/env python3
"""Train a neural knowledge selector from FSMODQA top-k retrieval output.

This script trains a sequence classifier on ``(question, passage)``
pairs built from FSMODQA retrieval rankings such as
``final_fsmodqa_squad_tr_full_deavg/{split}_top100_with_scores.jsonl``.
Positive passages are labeled by answer containment in the retrieved passage.
At evaluation time it reports the original dense-retriever ranking, the neural
selector ranking, and selector/retriever-score hybrids.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
torch: Any = None


def log(message: str) -> None:
    print(f"[knowledge-selector] {message}", flush=True)


@dataclass(frozen=True)
class Query:
    qid: str
    question: str
    answers: list[str]


@dataclass(frozen=True)
class Passage:
    pid: str
    title: str
    text: str


@dataclass
class PairRow:
    qid: str
    pid: str
    question: str
    title: str
    text: str
    rank: int
    retriever_score: float
    label: int
    selector_score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-ranking", type=Path, required=True)
    parser.add_argument("--eval-ranking", type=Path)
    parser.add_argument("--train-queries", type=Path, default=ROOT / "odqa_data/fsmodqa_retrieval/train.query.jsonl")
    parser.add_argument("--eval-queries", type=Path, default=ROOT / "odqa_data/fsmodqa_retrieval/validation.query.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "odqa_data/fsmodqa_retrieval/corpus.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--cache-dir", default=".hf_cache")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-context", type=int, default=100)
    parser.add_argument("--top-negatives-per-question", type=int, default=8)
    parser.add_argument("--random-negatives-per-question", type=int, default=8)
    parser.add_argument("--max-train-questions", type=int, default=0)
    parser.add_argument("--max-eval-questions", type=int, default=0)
    parser.add_argument("--positive-weight", type=float, default=0.0, help="0 infers negatives/positives from sampled train pairs.")
    parser.add_argument("--hybrid-weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--ks", default="1,5,10,25,50,100")
    parser.add_argument("--write-eval-reranked", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(text: Any) -> str:
    folded = str(text).casefold()
    without_punctuation = "".join(" " if ch in string.punctuation else ch for ch in folded)
    return " ".join(without_punctuation.split())


def answers_from(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("text", [])
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        answers: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("text", "")
            answer = str(item).strip() if item is not None else ""
            if answer:
                answers.append(answer)
        return answers
    answer = str(raw).strip()
    return [answer] if answer else []


def contains_answer(passage: Passage, answers: list[str]) -> bool:
    normalized_passage = normalize_text(f"{passage.title} {passage.text}")
    return any(normalized and normalized in normalized_passage for normalized in (normalize_text(answer) for answer in answers))


def load_queries(path: Path) -> dict[str, Query]:
    log(f"Loading queries from {path}")
    queries = {}
    for row in load_jsonl(path):
        qid = str(row.get("id") or row.get("qid"))
        queries[qid] = Query(
            qid=qid,
            question=str(row.get("question") or row.get("query") or ""),
            answers=answers_from(row.get("answers") or row.get("answer") or row.get("target")),
        )
    log(f"Loaded {len(queries)} queries from {path}")
    return queries


def load_corpus(path: Path) -> dict[str, Passage]:
    log(f"Loading corpus passages from {path}")
    passages = {}
    for row in load_jsonl(path):
        pid = str(row.get("id") or row.get("docid"))
        passages[pid] = Passage(
            pid=pid,
            title=str(row.get("title") or row.get("name") or ""),
            text=str(row.get("text") or row.get("context") or row.get("passage") or ""),
        )
    log(f"Loaded {len(passages)} corpus passages from {path}")
    return passages


def ranking_items(row: dict[str, Any]) -> list[tuple[str, float]]:
    pids = row.get("pids") or row.get("ctxs") or []
    scores = row.get("scores") or []
    items: list[tuple[str, float]] = []
    for index, item in enumerate(pids):
        if isinstance(item, (list, tuple)) and item:
            score = item[1] if len(item) > 1 else 0.0
            items.append((str(item[0]), safe_float(score)))
        elif isinstance(item, dict):
            pid = item.get("id") or item.get("docid") or item.get("pid")
            score = item.get("score", scores[index] if index < len(scores) else 0.0)
            items.append((str(pid), safe_float(score)))
        else:
            score = scores[index] if index < len(scores) else 0.0
            items.append((str(item), safe_float(score)))
    return items


def safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_rows(
    ranking_path: Path,
    queries: dict[str, Query],
    corpus: dict[str, Passage],
    *,
    n_context: int,
    max_questions: int,
) -> list[PairRow]:
    log(f"Building selector pairs from {ranking_path}")
    rows: list[PairRow] = []
    missing_queries = 0
    missing_passages = 0
    ranking_records = load_jsonl(ranking_path)
    log(f"Loaded {len(ranking_records)} ranking records from {ranking_path}")
    if max_questions > 0:
        ranking_records = ranking_records[:max_questions]
        log(f"Limited ranking records to first {len(ranking_records)} questions")

    progress_every = max(1, len(ranking_records) // 10) if ranking_records else 1
    for record_index, record in enumerate(ranking_records, start=1):
        qid = str(record.get("qid") or record.get("id"))
        query = queries.get(qid)
        if query is None:
            missing_queries += 1
            continue
        for rank, (pid, score) in enumerate(ranking_items(record)[:n_context], start=1):
            passage = corpus.get(pid)
            if passage is None:
                missing_passages += 1
                continue
            rows.append(
                PairRow(
                    qid=qid,
                    pid=pid,
                    question=query.question,
                    title=passage.title,
                    text=passage.text,
                    rank=rank,
                    retriever_score=score,
                    label=int(contains_answer(passage, query.answers)),
                )
            )
        if record_index % progress_every == 0 or record_index == len(ranking_records):
            log(f"Processed {record_index}/{len(ranking_records)} ranking records; built {len(rows)} pairs so far")
    positives = sum(row.label for row in rows)
    log(
        f"Built {len(rows)} pairs from {ranking_path}; "
        f"positives={positives} negatives={len(rows) - positives} "
        f"missing_queries={missing_queries} missing_passages={missing_passages}"
    )
    return rows


def group_by_qid(rows: Iterable[PairRow]) -> dict[str, list[PairRow]]:
    groups: dict[str, list[PairRow]] = {}
    for row in rows:
        groups.setdefault(row.qid, []).append(row)
    return groups


def sample_train_rows(
    rows: list[PairRow],
    *,
    top_negatives: int,
    random_negatives: int,
    seed: int,
) -> list[PairRow]:
    log("Sampling train pairs")
    rng = random.Random(seed)
    sampled: list[PairRow] = []
    groups = group_by_qid(rows)
    progress_every = max(1, len(groups) // 10) if groups else 1
    for group_index, group in enumerate(groups.values(), start=1):
        ranked = sorted(group, key=lambda row: row.rank)
        positives = [row for row in ranked if row.label == 1]
        negatives = [row for row in ranked if row.label == 0]
        selected_negatives = negatives[:top_negatives]
        tail = negatives[top_negatives:]
        if random_negatives > 0 and tail:
            selected_negatives.extend(rng.sample(tail, min(random_negatives, len(tail))))
        sampled.extend(positives + selected_negatives)
        if group_index % progress_every == 0 or group_index == len(groups):
            log(f"Sampled {group_index}/{len(groups)} question groups; selected {len(sampled)} pairs so far")
    rng.shuffle(sampled)
    positives = sum(row.label for row in sampled)
    log(f"Sampled {len(sampled)} train pairs: positives={positives} negatives={len(sampled) - positives}")
    return sampled


class PairDataset:
    def __init__(self, rows: list[PairRow], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row.question,
            passage_text(row),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(float(row.label), dtype=torch.float)
        return item


def passage_text(row: PairRow) -> str:
    return f"{row.title}. {row.text}".strip()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    denom = max(high - low, 1e-9)
    return [(value - low) / denom for value in values]


def score_rows(
    model: Any,
    tokenizer: Any,
    rows: list[PairRow],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> None:
    log(f"Scoring {len(rows)} eval pairs with batch_size={batch_size}")
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            encoded = tokenizer(
                [row.question for row in batch_rows],
                [passage_text(row) for row in batch_rows],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            for row, score in zip(batch_rows, scores):
                row.selector_score = float(score)
            done = min(start + batch_size, len(rows))
            if done == len(rows) or done % max(batch_size, 1000) == 0:
                log(f"Scored {done}/{len(rows)} eval pairs")


def first_positive_rank(rows: list[PairRow], scores: list[float], *, reverse: bool = True) -> int | None:
    ranked = sorted(zip(scores, rows), key=lambda item: item[0], reverse=reverse)
    for rank, (_, row) in enumerate(ranked, start=1):
        if row.label == 1:
            return rank
    return None


def retrieval_metrics(rows: list[PairRow], ks: list[int], scorer: Any) -> dict[str, Any]:
    ranks = []
    for group in group_by_qid(rows).values():
        scores = [float(scorer(row, group)) for row in group]
        ranks.append(first_positive_rank(group, scores))
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


def retriever_norm_lookup(group: list[PairRow]) -> dict[str, float]:
    return {row.pid: score for row, score in zip(group, minmax([row.retriever_score for row in group]))}


def write_reranked(path: Path, rows: list[PairRow], selector_weight: float) -> None:
    log(f"Writing reranked eval rankings to {path} with selector_weight={selector_weight:.2f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = group_by_qid(rows)
    with path.open("w", encoding="utf-8") as handle:
        for index, (qid, group) in enumerate(groups.items(), start=1):
            score_lookup = retriever_norm_lookup(group)
            ranked = sorted(
                group,
                key=lambda row: selector_weight * row.selector_score + (1.0 - selector_weight) * score_lookup[row.pid],
                reverse=True,
            )
            handle.write(
                json.dumps(
                    {
                        "qid": qid,
                        "pids": [row.pid for row in ranked],
                        "scores": [
                            selector_weight * row.selector_score + (1.0 - selector_weight) * score_lookup[row.pid]
                            for row in ranked
                        ],
                        "selector_scores": [row.selector_score for row in ranked],
                        "retriever_scores": [row.retriever_score for row in ranked],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if index % max(1, len(groups) // 10) == 0 or index == len(groups):
                log(f"Wrote reranked records for {index}/{len(groups)} questions")
    log(f"Finished writing reranked rankings to {path}")


def main() -> None:
    started_at = time.time()
    args = parse_args()
    log("Starting neural FSModQA knowledge selector training")
    log(f"Train ranking: {args.train_ranking}")
    log(f"Eval ranking: {args.eval_ranking or args.train_ranking}")
    log(f"Corpus: {args.corpus}")
    log(f"Output dir: {args.output_dir}")
    random.seed(args.seed)
    ks = [int(item) for item in args.ks.split(",") if item]
    hybrid_weights = [float(item) for item in args.hybrid_weights.split(",") if item]
    log(f"Metrics K values: {ks}")
    log(f"Hybrid selector weights: {hybrid_weights}")

    log("Importing torch and transformers")
    try:
        import torch as torch_module
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Install torch and transformers first.") from exc

    globals()["torch"] = torch_module
    torch.manual_seed(args.seed)
    log(f"Random seed set to {args.seed}")

    log("Step 1/8: loading corpus and query files")
    corpus = load_corpus(args.corpus)
    train_queries = load_queries(args.train_queries)
    eval_queries = load_queries(args.eval_queries)

    log("Step 2/8: building train and eval pairs")
    train_rows_all = build_rows(
        args.train_ranking,
        train_queries,
        corpus,
        n_context=args.n_context,
        max_questions=args.max_train_questions,
    )
    eval_rows = build_rows(
        args.eval_ranking or args.train_ranking,
        eval_queries if args.eval_ranking else train_queries,
        corpus,
        n_context=args.n_context,
        max_questions=args.max_eval_questions,
    )

    log("Step 3/8: sampling negatives for training")
    train_rows = sample_train_rows(
        train_rows_all,
        top_negatives=args.top_negatives_per_question,
        random_negatives=args.random_negatives_per_question,
        seed=args.seed,
    )
    if not train_rows:
        raise SystemExit("No train pairs were built. Check ranking/query/corpus paths.")

    log("Step 4/8: loading tokenizer and sequence classification model")
    log(f"Model name: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1, cache_dir=args.cache_dir)
    device = choose_device()
    model.to(device)
    log(f"Using device: {device}")

    positives = sum(row.label for row in train_rows)
    negatives = len(train_rows) - positives
    positive_weight = args.positive_weight if args.positive_weight > 0 else negatives / max(positives, 1)
    log(
        "Training class balance: "
        f"positives={positives} negatives={negatives} positive_weight={positive_weight:.4f}"
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    loader = DataLoader(PairDataset(train_rows, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(loader) * args.epochs)
    log(
        "Training setup: "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} "
        f"max_length={args.max_length} steps_per_epoch={len(loader)} total_steps={total_steps}"
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    log("Step 5/8: training neural selector")
    for epoch in range(args.epochs):
        epoch_started_at = time.time()
        model.train()
        losses: list[float] = []
        log(f"Starting epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                log(f"epoch={epoch + 1} step={step}/{len(loader)} recent_loss={mean(losses[-100:]):.4f}")
        log(f"Finished epoch {epoch + 1}/{args.epochs}: train_loss={mean(losses):.4f} elapsed={time.time() - epoch_started_at:.1f}s")

    log("Step 6/8: scoring eval pairs")
    score_rows(
        model,
        tokenizer,
        eval_rows,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )

    log("Step 7/8: computing retrieval metrics")
    metrics: dict[str, Any] = {
        "model_name": args.model_name,
        "train_ranking": str(args.train_ranking),
        "eval_ranking": str(args.eval_ranking or args.train_ranking),
        "train_questions": len(group_by_qid(train_rows_all)),
        "eval_questions": len(group_by_qid(eval_rows)),
        "train_pairs_before_sampling": len(train_rows_all),
        "train_pairs": len(train_rows),
        "eval_pairs": len(eval_rows),
        "positive_train_pairs": positives,
        "negative_train_pairs": negatives,
        "positive_eval_pairs": sum(row.label for row in eval_rows),
        "positive_weight": positive_weight,
        "device": str(device),
        "retriever": retrieval_metrics(eval_rows, ks, lambda row, _group: row.retriever_score),
        "selector": retrieval_metrics(eval_rows, ks, lambda row, _group: row.selector_score),
    }
    log("Computed retriever baseline metrics")
    log(f"Retriever metrics: {json.dumps(metrics['retriever'], ensure_ascii=False)}")
    log("Computed selector metrics")
    log(f"Selector metrics: {json.dumps(metrics['selector'], ensure_ascii=False)}")
    for weight in hybrid_weights:
        metrics[f"hybrid_selector_weight_{weight:.2f}"] = retrieval_metrics(
            eval_rows,
            ks,
            lambda row, group, weight=weight: weight * row.selector_score
            + (1.0 - weight) * retriever_norm_lookup(group)[row.pid],
        )
        log(
            f"Hybrid weight {weight:.2f} metrics: "
            f"{json.dumps(metrics[f'hybrid_selector_weight_{weight:.2f}'], ensure_ascii=False)}"
        )

    log("Step 8/8: saving model, tokenizer, and metrics")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log(f"Saved model and tokenizer to {args.output_dir}")

    if args.write_eval_reranked:
        best_weight = max(
            hybrid_weights,
            key=lambda weight: (
                metrics[f"hybrid_selector_weight_{weight:.2f}"].get("recall@10", 0.0),
                metrics[f"hybrid_selector_weight_{weight:.2f}"].get("recall@25", 0.0),
                metrics[f"hybrid_selector_weight_{weight:.2f}"].get("mrr", 0.0),
            ),
        )
        log(f"Best hybrid weight for reranked output: {best_weight:.2f}")
        write_reranked(args.write_eval_reranked, eval_rows, selector_weight=best_weight)
        metrics["written_reranked_eval"] = str(args.write_eval_reranked)
        metrics["written_reranked_selector_weight"] = best_weight

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    log(f"Saved metrics to {args.output_dir / 'metrics.json'}")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    log(f"Finished neural FSModQA knowledge selector training in {time.time() - started_at:.1f}s")
    print(f"Saved neural FSModQA knowledge selector to {args.output_dir}")


if __name__ == "__main__":
    main()
