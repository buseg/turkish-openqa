#!/usr/bin/env python3
"""Train the adaptive retrieval process end to end.

1. load top-k ranking JSONL, query JSONL, and corpus JSONL;
2. derive easy/medium/hard labels with evaluate_retrieval matching;
3. train a retrieval-aware complexity classifier from train data;
4. tune context budgets on validation data with a recall floor;
5. classify test questions and write per-class cropped rankings.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import re
import string
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_retrieval import first_relevant_rank, load_qid_aliases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "checkpoint" / "final_fsmodqa_squad_tr_full_deavg"
DEFAULT_DATA_DIR = ROOT / "odqa_data" / "fsmodqa_retrieval"


@dataclass(frozen=True)
class Policy:
    easy_k: int
    medium_k: int
    hard_k: int
    recall: float
    avg_contexts: float
    context_saving_vs_hard: float

    def asdict(self) -> dict[str, Any]:
        return {
            "easy_k": self.easy_k,
            "medium_k": self.medium_k,
            "hard_k": self.hard_k,
            "recall": self.recall,
            "avg_contexts": self.avg_contexts,
            "context_saving_vs_hard": self.context_saving_vs_hard,
        }


def log(message: str) -> None:
    print(f"[adaptive-retrieval] {message}", flush=True)


def parse_ints(raw: str) -> list[int]:
    values = [int(item) for item in raw.replace(" ", ",").split(",") if item]
    if not values:
        raise ValueError(f"Expected at least one integer in {raw!r}.")
    return sorted(set(values))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def load_queries(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in read_jsonl(path)}


def load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    corpus = {}
    for row in read_jsonl(path):
        pid = str(row.get("id") or row.get("docid"))
        corpus[pid] = row
    return corpus



def label_from_rank(rank: int | None, easy_k: int, medium_k: int) -> str:
    if rank is not None and rank <= easy_k:
        return "easy"
    if rank is not None and rank <= medium_k:
        return "medium"
    return "hard"


def safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def ranking_items(row: dict[str, Any]) -> list[tuple[str, float]]:
    pids = row.get("pids") or row.get("ctxs") or []
    scores = row.get("scores") or []
    items = []
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


def build_examples(
    rankings_path: Path,
    queries_path: Path,
    corpus: dict[str, dict[str, Any]],
    match_mode: str,
    qid_aliases: dict[str, list[str]],
    easy_label_k: int,
    medium_label_k: int,
) -> list[dict[str, Any]]:
    queries = load_queries(queries_path)
    examples = []
    for row in read_jsonl(rankings_path):
        qid = str(row.get("qid") or row.get("id"))
        query = queries.get(qid)
        if query is None:
            continue
        items = ranking_items(row)
        pids = [pid for pid, _score in items]
        scores = [score for _pid, score in items]
        ctxs = []
        for pid, score in zip(pids, scores):
            passage = corpus.get(pid, {})
            ctxs.append(
                {
                    "id": pid,
                    "title": passage.get("title", ""),
                    "text": passage.get("text", ""),
                    "score": score,
                }
            )
        rank = first_relevant_rank(pids, qid, match_mode, qid_aliases)
        label = label_from_rank(rank, easy_label_k, medium_label_k)
        examples.append(
            {
                "id": qid,
                "question": query.get("question", ""),
                "answers": query.get("answers", []),
                "pids": pids,
                "scores": scores,
                "ctxs": ctxs,
                "first_hit_rank": rank,
                "label": label,
            }
        )
    return examples


def normalize(text: Any) -> str:
    text = str(text).casefold()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def tokens(text: Any) -> set[str]:
    return set(re.findall(r"[\wçğıöşü]+", normalize(text), flags=re.UNICODE))


def score(ctx: dict[str, Any]) -> float:
    try:
        return float(ctx.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def safe_ratio(num: float, den: float) -> float:
    return num / den if abs(den) > 1e-9 else 0.0


def numeric_features(example: dict[str, Any]) -> dict[str, float]:
    ctxs = example.get("ctxs", [])
    scores = [score(ctx) for ctx in ctxs[:10]] or [0.0]
    top1 = scores[0]
    top5 = scores[:5]
    top10 = scores[:10]
    top5_mean = mean(top5) if top5 else 0.0
    top10_mean = mean(top10) if top10 else 0.0
    q_tokens = tokens(example.get("question", ""))
    top1_tokens = tokens(" ".join(f"{ctx.get('title', '')} {ctx.get('text', '')}" for ctx in ctxs[:1]))
    top5_tokens = tokens(" ".join(f"{ctx.get('title', '')} {ctx.get('text', '')}" for ctx in ctxs[:5]))
    overlap_top1 = len(q_tokens & top1_tokens)
    overlap_top5 = len(q_tokens & top5_tokens)
    return {
        "question_len": float(len(q_tokens)),
        "top1_score": top1,
        "top5_mean_score": top5_mean,
        "top10_mean_score": top10_mean,
        "top1_top2_gap": top1 - scores[1] if len(scores) > 1 else top1,
        "top1_top5_gap": top1 - scores[4] if len(scores) > 4 else top1,
        "top1_top10_gap": top1 - scores[9] if len(scores) > 9 else top1,
        "top1_top5_ratio": safe_ratio(top1, top5_mean),
        "top1_top10_ratio": safe_ratio(top1, top10_mean),
        "score_std_top10": math.sqrt(mean((value - top10_mean) ** 2 for value in top10)) if top10 else 0.0,
        "query_top1_overlap": float(overlap_top1),
        "query_top5_overlap": float(overlap_top5),
        "query_top1_overlap_ratio": safe_ratio(overlap_top1, len(q_tokens)),
        "query_top5_overlap_ratio": safe_ratio(overlap_top5, len(q_tokens)),
        "num_contexts": float(len(ctxs)),
    }


def question_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [row.get("question", "") for row in rows]


def numeric_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [numeric_features(row) for row in rows]


def make_pickle_importable() -> None:
    sys.modules.setdefault("train_adaptive_retrieval", sys.modules[__name__])
    question_texts.__module__ = "train_adaptive_retrieval"
    numeric_feature_rows.__module__ = "train_adaptive_retrieval"


def train_classifier(train_examples: list[dict[str, Any]]) -> Any:
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import FunctionTransformer
    except ImportError as exc:
        raise SystemExit("Install scikit-learn first.") from exc

    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "question_tfidf",
                            Pipeline(
                                [
                                    ("select_question", FunctionTransformer(question_texts, validate=False)),
                                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                                ]
                            ),
                        ),
                        (
                            "retrieval_features",
                            Pipeline(
                                [
                                    ("build_features", FunctionTransformer(numeric_feature_rows, validate=False)),
                                    ("dict", DictVectorizer()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    ).fit(train_examples, [row["label"] for row in train_examples])


def classification_metrics(model: Any, examples: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as exc:
        raise SystemExit("Install scikit-learn first.") from exc

    labels = sorted({"easy", "medium", "hard"} | {row["label"] for row in examples})
    gold = [row["label"] for row in examples]
    pred = [str(value) for value in model.predict(examples)]
    return {
        "report": classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
    }


def tune_policy(
    rows: list[dict[str, Any]],
    easy_options: list[int],
    medium_options: list[int],
    hard_options: list[int],
) -> list[Policy]:
    policies = []
    for easy_k in easy_options:
        for medium_k in medium_options:
            for hard_k in hard_options:
                budgets = {"easy": easy_k, "medium": medium_k, "hard": hard_k}
                hits = []
                contexts = []
                for row in rows:
                    budget = budgets.get(row["label"], hard_k)
                    rank = row.get("first_hit_rank")
                    contexts.append(budget)
                    hits.append(rank is not None and rank <= budget)
                avg_contexts = mean(contexts) if contexts else 0.0
                policies.append(
                    Policy(
                        easy_k=easy_k,
                        medium_k=medium_k,
                        hard_k=hard_k,
                        recall=mean(hits) if hits else 0.0,
                        avg_contexts=avg_contexts,
                        context_saving_vs_hard=1 - safe_ratio(avg_contexts, hard_k),
                    )
                )
    policies.sort(key=lambda item: (-item.recall, item.avg_contexts, item.easy_k, item.medium_k, item.hard_k))
    return policies


def select_policy(
    policies: list[Policy],
    baseline_recall: float,
    min_recall_ratio: float,
) -> tuple[Policy, dict[str, Any]]:
    min_required_recall = baseline_recall * min_recall_ratio
    eligible = [policy for policy in policies if policy.recall >= min_required_recall]
    if eligible:
        selected = sorted(
            eligible,
            key=lambda item: (item.avg_contexts, -item.recall, item.easy_k, item.medium_k, item.hard_k),
        )[0]
        strategy = "min_contexts_with_recall_floor"
    else:
        selected = policies[0]
        strategy = "fallback_highest_recall"
    return selected, {
        "strategy": strategy,
        "baseline_recall": baseline_recall,
        "min_recall_ratio": min_recall_ratio,
        "min_required_recall": min_required_recall,
        "eligible_policies": len(eligible),
        "fallback_used": not eligible,
    }


def budget_for(label: str, policy: Policy) -> int:
    if label == "easy":
        return policy.easy_k
    if label == "medium":
        return policy.medium_k
    return policy.hard_k


def apply_policy(model: Any, examples: list[dict[str, Any]], policy: Policy) -> list[dict[str, Any]]:
    labels = [str(value) for value in model.predict(examples)]
    selected = []
    for example, label in zip(examples, labels):
        budget = budget_for(label, policy)
        original_rank = example.get("first_hit_rank")
        copy = dict(example)
        copy["adaptive_label"] = label
        copy["adaptive_k"] = budget
        copy["original_first_hit_rank"] = original_rank
        copy["first_hit_rank"] = original_rank if original_rank is not None and original_rank <= budget else None
        copy["ctxs"] = list(example.get("ctxs", []))[:budget]
        copy["pids"] = list(example.get("pids", []))[:budget]
        copy["scores"] = list(example.get("scores", []))[:budget]
        selected.append(copy)
    return selected


def retrieval_metrics(examples: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    ranks = [row.get("first_hit_rank") for row in examples]
    found = [rank for rank in ranks if rank is not None]
    metrics: dict[str, Any] = {
        "examples": len(examples),
        "found_any": len(found),
        "missing_any": len(examples) - len(found),
        "mrr": mean((1.0 / rank if rank is not None else 0.0) for rank in ranks) if ranks else 0.0,
        "mean_first_hit_rank": mean(found) if found else None,
    }
    for k in ks:
        metrics[f"recall@{k}"] = mean((rank is not None and rank <= k) for rank in ranks) if ranks else 0.0
    return metrics


def adaptive_metrics(selected: list[dict[str, Any]], policy: Policy, ks: list[int]) -> dict[str, Any]:
    metrics = retrieval_metrics(selected, ks)
    budgets = [int(row.get("adaptive_k", policy.hard_k)) for row in selected]
    label_counts: dict[str, int] = {}
    for row in selected:
        label = str(row.get("adaptive_label", "hard"))
        label_counts[label] = label_counts.get(label, 0) + 1
    metrics.update(
        {
            "label_counts": label_counts,
            "avg_contexts": mean(budgets) if budgets else 0.0,
            "context_saving_vs_hard": 1 - safe_ratio(mean(budgets), policy.hard_k) if budgets else 0.0,
            "policy": policy.asdict(),
        }
    )
    return metrics


def label_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "label": row["label"],
            "first_hit_rank": row.get("first_hit_rank"),
        }
        for row in examples
    ]


def classification_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "predicted_label": row.get("adaptive_label", "hard"),
            "adaptive_k": row.get("adaptive_k"),
            "first_hit_rank_before_budget": row.get("original_first_hit_rank"),
            "first_hit_rank_after_budget": row.get("first_hit_rank"),
        }
        for row in examples
    ]


def ranking_row(example: dict[str, Any]) -> dict[str, Any]:
    row = {
        "qid": example["id"],
        "pids": list(example.get("pids", [])),
    }
    scores = list(example.get("scores", []))
    if scores:
        row["scores"] = scores
    return row


def write_test_rankings_by_class(output_dir: Path, examples: list[dict[str, Any]]) -> dict[str, str]:
    paths = {
        "easy": output_dir / "test_easy_rankings.jsonl",
        "medium": output_dir / "test_medium_rankings.jsonl",
        "hard": output_dir / "test_hard_rankings.jsonl",
    }
    grouped = {label: [] for label in paths}
    for example in examples:
        label = str(example.get("adaptive_label", "hard"))
        if label not in grouped:
            label = "hard"
        grouped[label].append(ranking_row(example))
    for label, rows in grouped.items():
        write_jsonl(paths[label], rows)
    return {label: str(path) for label, path in paths.items()}


def load_policy(path: Path) -> Policy:
    with path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    return Policy(
        easy_k=int(row["easy_k"]),
        medium_k=int(row["medium_k"]),
        hard_k=int(row["hard_k"]),
        recall=float(row.get("recall", 0.0)),
        avg_contexts=float(row.get("avg_contexts", 0.0)),
        context_saving_vs_hard=float(row.get("context_saving_vs_hard", row.get("context_saving_vs_100", 0.0))),
    )


def load_classifier(path: Path) -> Any:
    make_pickle_importable()
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_apply_outputs(
    output_dir: Path,
    model: Any,
    policy: Policy,
    test_examples: list[dict[str, Any]],
    metric_ks: list[int],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    selected_test = apply_policy(model, test_examples, policy)
    test_ranking_outputs = write_test_rankings_by_class(output_dir, selected_test)
    metrics = {
        **metadata,
        "test_examples": len(test_examples),
        "base_test_retrieval": retrieval_metrics(test_examples, metric_ks),
        "adaptive_test_retrieval": adaptive_metrics(selected_test, policy, metric_ks),
        "test_ranking_outputs": test_ranking_outputs,
        "best_policy": policy.asdict(),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "policy.json", policy.asdict())
    write_json(output_dir / "test_adaptive.json", selected_test)
    write_jsonl(output_dir / "test_classifications.jsonl", classification_rows(selected_test))
    write_json(output_dir / "test_metrics.json", metrics["adaptive_test_retrieval"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "apply"), default="train", help="train fits/tunes then applies; apply loads --model/--policy and only crops test rankings.")
    parser.add_argument("--model", type=Path, default=ROOT / "checkpoint" / "adaptive_retrieval" / "adaptive_retrieval_classifier.pkl")
    parser.add_argument("--policy", type=Path, default=ROOT / "checkpoint" / "adaptive_retrieval" / "policy.json")
    parser.add_argument(
        "--train-rankings",
        type=Path,
        default=DEFAULT_RUN_DIR / "train_top100_with_scores.jsonl",
        help="Ranking JSONL used to create supervised training data. Defaults to train.",
    )
    parser.add_argument(
        "--validation-rankings",
        type=Path,
        default=DEFAULT_RUN_DIR / "validation_top100_with_scores.jsonl",
        help="Ranking JSONL used to tune adaptive budget details. Defaults to validation.",
    )
    parser.add_argument(
        "--test-rankings",
        type=Path,
        default=DEFAULT_RUN_DIR / "test_top100_with_scores.jsonl",
        help="Ranking JSONL to classify and split after training. Defaults to test.",
    )
    parser.add_argument(
        "--train-queries",
        type=Path,
        default=DEFAULT_DATA_DIR / "train.query.jsonl",
        help="Queries for --train-rankings. Defaults to train queries.",
    )
    parser.add_argument("--validation-queries", type=Path, default=DEFAULT_DATA_DIR / "validation.query.jsonl")
    parser.add_argument("--test-queries", type=Path, default=DEFAULT_DATA_DIR / "test.query.jsonl")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_DATA_DIR / "corpus.jsonl")
    parser.add_argument("--qid-aliases", type=Path, default=DEFAULT_DATA_DIR / "squad_duplicate_qid_lookup.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoint" / "adaptive_retrieval")
    parser.add_argument("--easy-label-k", type=int, default=10)
    parser.add_argument("--medium-label-k", type=int, default=50)
    parser.add_argument("--easy-options", default="10,25")
    parser.add_argument("--medium-options", default="25,50")
    parser.add_argument("--hard-options", default="100")
    parser.add_argument("--metric-ks", default="1,5,10,25,50,100")
    parser.add_argument("--min-recall-ratio", type=float, default=0.95, help="Minimum fraction of validation top-hard recall to keep while minimizing average contexts. Try 0.90 for more savings.")
    parser.add_argument("--match-mode", choices=("contains", "squad-prefix"), default="squad-prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.min_recall_ratio <= 1.0:
        raise SystemExit("--min-recall-ratio must be in (0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading corpus: {args.corpus}")
    corpus = load_corpus(args.corpus)
    qid_aliases = load_qid_aliases(args.qid_aliases if args.qid_aliases.exists() else None)

    if args.mode == "apply":
        log(f"Loading adaptive classifier: {args.model}")
        model = load_classifier(args.model)
        log(f"Loading adaptive policy: {args.policy}")
        best_policy = load_policy(args.policy)
        log("Building test examples for apply-only adaptive retrieval")
        test_examples = build_examples(
            args.test_rankings,
            args.test_queries,
            corpus,
            args.match_mode,
            qid_aliases,
            args.easy_label_k,
            args.medium_label_k,
        )
        if not test_examples:
            raise SystemExit("No test examples were built. Check test ranking/query paths.")
        metrics = write_apply_outputs(
            args.output_dir,
            model,
            best_policy,
            test_examples,
            parse_ints(args.metric_ks),
            {
                "mode": "apply",
                "model": str(args.model),
                "policy": str(args.policy),
                "test_rankings": str(args.test_rankings),
                "label_thresholds": {"easy_k": args.easy_label_k, "medium_k": args.medium_label_k},
                "matching": f"evaluate_retrieval.first_relevant_rank/{args.match_mode}",
            },
        )
        log("Saved apply-only adaptive outputs to {}".format(args.output_dir))
        log("Adaptive test avg contexts: {:.2f}".format(metrics["adaptive_test_retrieval"].get("avg_contexts", 0.0)))
        return

    log("Building train examples from train rankings")
    train_examples = build_examples(
        args.train_rankings,
        args.train_queries,
        corpus,
        args.match_mode,
        qid_aliases,
        args.easy_label_k,
        args.medium_label_k,
    )
    if not train_examples:
        raise SystemExit("No train examples were built. Check ranking/query paths.")

    log("Building validation examples for policy tuning")
    validation_examples = build_examples(
        args.validation_rankings,
        args.validation_queries,
        corpus,
        args.match_mode,
        qid_aliases,
        args.easy_label_k,
        args.medium_label_k,
    )
    if not validation_examples:
        raise SystemExit("No validation examples were built. Check ranking/query paths.")

    write_jsonl(args.output_dir / "train_labels.jsonl", label_rows(train_examples))
    write_jsonl(args.output_dir / "validation_labels.jsonl", label_rows(validation_examples))

    log(f"Training classifier on {len(train_examples)} train examples")
    make_pickle_importable()
    model = train_classifier(train_examples)
    model_path = args.output_dir / "adaptive_retrieval_classifier.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    log("Tuning policy on validation examples")
    class_metrics = classification_metrics(model, validation_examples)
    validation_predictions = [str(pred) for pred in model.predict(validation_examples)]
    policy_rows = [dict(row, label=pred) for row, pred in zip(validation_examples, validation_predictions)]
    hard_options = parse_ints(args.hard_options)
    policies = tune_policy(
        policy_rows,
        parse_ints(args.easy_options),
        parse_ints(args.medium_options),
        hard_options,
    )
    metric_ks = parse_ints(args.metric_ks)
    baseline_k = max(hard_options)
    baseline_validation = retrieval_metrics(validation_examples, [baseline_k])
    baseline_recall = float(baseline_validation.get(f"recall@{baseline_k}", 0.0))
    best_policy, policy_selection = select_policy(policies, baseline_recall, args.min_recall_ratio)
    log(
        "Selected policy easy={} medium={} hard={} with validation recall {:.4f}, avg contexts {:.2f}".format(
            best_policy.easy_k,
            best_policy.medium_k,
            best_policy.hard_k,
            best_policy.recall,
            best_policy.avg_contexts,
        )
    )
    selected_validation = apply_policy(model, validation_examples, best_policy)

    log("Classifying test questions")
    test_examples = build_examples(
        args.test_rankings,
        args.test_queries,
        corpus,
        args.match_mode,
        qid_aliases,
        args.easy_label_k,
        args.medium_label_k,
    )
    if not test_examples:
        raise SystemExit("No test examples were built. Check test ranking/query paths.")
    selected_test = apply_policy(model, test_examples, best_policy)
    test_ranking_outputs = write_test_rankings_by_class(args.output_dir, selected_test)

    metrics = {
        "model": str(model_path),
        "train_rankings": str(args.train_rankings),
        "validation_rankings": str(args.validation_rankings),
        "test_rankings": str(args.test_rankings),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "label_thresholds": {"easy_k": args.easy_label_k, "medium_k": args.medium_label_k},
        "matching": f"evaluate_retrieval.first_relevant_rank/{args.match_mode}",
        "classifier_validation": class_metrics,
        "base_train_retrieval": retrieval_metrics(train_examples, metric_ks),
        "base_validation_retrieval": retrieval_metrics(validation_examples, metric_ks),
        "adaptive_validation_retrieval": adaptive_metrics(selected_validation, best_policy, metric_ks),
        "base_test_retrieval": retrieval_metrics(test_examples, metric_ks),
        "adaptive_test_retrieval": adaptive_metrics(selected_test, best_policy, metric_ks),
        "test_ranking_outputs": test_ranking_outputs,
        "policy_selection": policy_selection,
        "policy_candidates": [policy.asdict() for policy in policies],
        "best_policy": best_policy.asdict(),
    }

    write_json(args.output_dir / "policy.json", best_policy.asdict())
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "validation_adaptive.json", selected_validation)
    write_json(args.output_dir / "test_adaptive.json", selected_test)
    write_jsonl(args.output_dir / "test_classifications.jsonl", classification_rows(selected_test))
    write_json(args.output_dir / "test_metrics.json", metrics["adaptive_test_retrieval"])

    log(f"Saved model: {model_path}")
    log("Saved policy: {}".format(args.output_dir / "policy.json"))
    log("Saved metrics: {}".format(args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
