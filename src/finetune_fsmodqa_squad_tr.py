#!/usr/bin/env python3
"""Fine-tune FSMODQA on SQuAD-TR using FSMODQA's retrieval/training flow.

This script uses the prepared FSMODQA retrieval files and the existing
off-the-shelf embeddings:

* odqa_data/fsmodqa_retrieval/corpus.jsonl
* odqa_data/fsmodqa_retrieval/{train,validation,test}.query.jsonl
* checkpoint/fsmodqa_off_the_shelf/encoding/*_query_embedding.pt
* checkpoint/fsmodqa_off_the_shelf/encoding/passage_embedding_split*.pt

It creates native FSMODQA ranking files, ``{split}.jsonl`` with ``qid`` and
``pids``, by calling external/FSMODQA/retriever.py. Then it calls
external/FSMODQA/train.py. Use ``--mode reader`` for reader-only fine-tuning or
``--mode full`` for reader plus retriever/distillation fine-tuning.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FSMODQA_ROOT = ROOT / "external" / "FSMODQA"


def _native_resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _native_run(command: list[str], cwd: Path = ROOT) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _native_require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _native_ranking_path(data_dir: Path, split: str) -> Path:
    return data_dir / f"{split}.jsonl"


def _native_link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        import shutil

        shutil.copy2(source, destination)


def _native_materialize_work_dir(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    _native_link_or_copy(args.data_dir / args.corpus_file, args.work_dir / args.corpus_file)
    for split in args.splits:
        _native_link_or_copy(args.data_dir / f"{split}.query.jsonl", args.work_dir / f"{split}.query.jsonl")


def _native_make_ranking(args: argparse.Namespace, split: str) -> None:
    query_embeddings = args.embedding_dir / f"{split}_query_embedding.pt"
    query_file = args.work_dir / f"{split}.query.jsonl"
    output_path = _native_ranking_path(args.work_dir, split)
    passage_pattern = str(args.embedding_dir / "passage_embedding_split*.pt")

    _native_require_file(query_embeddings, f"{split} query embeddings")
    _native_require_file(query_file, f"{split} query JSONL")
    if not list(args.embedding_dir.glob("passage_embedding_split*.pt")):
        raise FileNotFoundError(f"No passage embedding shards under {args.embedding_dir}")

    if output_path.exists() and not args.overwrite_rankings:
        print(f"Keeping existing ranking: {output_path}")
        return

    command = [
        sys.executable,
        str(FSMODQA_ROOT / "retriever.py"),
        "--query_embeddings",
        str(query_embeddings),
        "--passage_embeddings",
        passage_pattern,
        "--train_dir",
        str(args.work_dir),
        "--qas_file",
        str(query_file),
        "--corpus_file",
        args.corpus_file,
        "--save_ranking_to",
        str(output_path),
        "--save_jsonl",
        "--depth",
        str(args.retrieval_depth),
        "--batch_size",
        str(args.retrieval_batch_size),
    ]
    if args.search_then_merge:
        command.append("--search_then_merge")
    if args.use_gpu_retrieval:
        command.append("--use_gpu")

    print(f"Generating {split} pids with external/FSMODQA/retriever.py")
    _native_run(command, cwd=FSMODQA_ROOT)


def _native_prepare_rankings(args: argparse.Namespace) -> None:
    _native_require_file(args.data_dir / args.corpus_file, "FSMODQA corpus")
    _native_materialize_work_dir(args)

    for split in args.splits:
        if split == "validation" and args.reuse_existing_validation_top100:
            existing = args.off_the_shelf_checkpoint / "validation_top100.jsonl"
            destination = _native_ranking_path(args.work_dir, "validation")
            if existing.exists() and (args.overwrite_rankings or not destination.exists()):
                import shutil

                print(f"Copying existing validation ranking: {existing} -> {destination}")
                shutil.copy2(existing, destination)
                continue
        _native_make_ranking(args, split)


def _native_train(args: argparse.Namespace) -> None:
    _native_materialize_work_dir(args)
    _native_require_file(_native_ranking_path(args.work_dir, "train"), "train pids JSONL")
    _native_require_file(args.work_dir / "train.query.jsonl", "train query JSONL")
    _native_require_file(args.work_dir / args.corpus_file, "corpus JSONL")

    command = [
        sys.executable,
        str(FSMODQA_ROOT / "train.py"),
        "--output_dir",
        str(args.output_dir),
        "--overwrite_output_dir",
        "--model_name_or_path",
        args.model_name_or_path,
        "--task",
        "XOR-Retrieve",
        "--train_dir",
        str(args.work_dir),
        "--train_path",
        "train.jsonl",
        "--corpus_file",
        args.corpus_file,
        "--query_file",
        "train.query.jsonl",
        "--eval_query_file",
        "validation.query.jsonl",
        "--load_corpus",
        "True",
        "--train_n_passages",
        str(args.train_n_passages),
        "--max_query_length",
        str(args.max_query_length),
        "--max_passage_length",
        str(args.max_passage_length),
        "--max_query_passage_length",
        str(args.max_query_passage_length),
        "--max_answer_length",
        str(args.max_answer_length),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--max_steps",
        str(args.max_steps),
        "--distillation_start_steps",
        str(args.distillation_start_steps),
        "--save_steps",
        str(args.save_steps),
        "--print_steps",
        str(args.print_steps),
        "--tensorboard_log_dir",
        str(args.tensorboard_log_dir or args.output_dir / "runs"),
        "--tb_log_examples",
        str(args.tb_log_examples),
        "--tb_metric_examples",
        str(args.tb_metric_examples),
        "--tb_log_generation_steps",
        str(args.tb_log_generation_steps),
        "--multi_task",
    ]
    if args.separate_joint_encoding:
        command.append("--separate_joint_encoding")
    if args.mode == "reader":
        command.extend(["--only_reader", "True"])
    if args.mode == "full" and args.refresh_passages:
        command.append("--refresh_passages")
        command.extend(["--refresh_intervals", str(args.refresh_intervals)])
    if args.add_positive_passage:
        command.extend(["--add_positive_passage", "True"])
    if args.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if args.fp16:
        command.extend(["--fp16", "True"])
    if args.bf16:
        command.extend(["--bf16", "True"])
    if args.tf32:
        command.extend(["--tf32", "True"])

    print(f"Fine-tuning FSMODQA in {args.mode!r} mode")
    _native_run(command, cwd=FSMODQA_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "odqa_data" / "fsmodqa_retrieval")
    parser.add_argument("--corpus-file", default="corpus.jsonl")
    parser.add_argument("--off-the-shelf-checkpoint", type=Path, default=ROOT / "checkpoint" / "fsmodqa_off_the_shelf")
    parser.add_argument("--embedding-dir", type=Path, default=ROOT / "checkpoint" / "fsmodqa_off_the_shelf" / "encoding")
    parser.add_argument("--model-name-or-path", default="fanjiang98/FSMODQA-100k")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoint" / "fsmodqa_squad_tr")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--mode", choices=("reader", "full"), default="reader")

    parser.add_argument("--skip-rankings", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--overwrite-rankings", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=["train", "validation", "test"])
    parser.add_argument("--reuse-existing-validation-top100", action="store_true")
    parser.add_argument("--retrieval-depth", type=int, default=100)
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--search-then-merge", action="store_true")
    parser.add_argument("--use-gpu-retrieval", action="store_true")
    parser.add_argument("--refresh-passages", action="store_true")
    parser.add_argument("--refresh-intervals", type=int, default=3000)

    parser.add_argument("--train-n-passages", type=int, default=100)
    parser.add_argument("--max-query-length", type=int, default=50)
    parser.add_argument("--max-passage-length", type=int, default=200)
    parser.add_argument("--max-query-passage-length", type=int, default=250)
    parser.add_argument("--max-answer-length", type=int, default=50)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--distillation-start-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--print-steps", type=int, default=20)
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    parser.add_argument("--tb-log-examples", type=int, default=0)
    parser.add_argument("--tb-metric-examples", type=int, default=0)
    parser.add_argument("--tb-log-generation-steps", type=int, default=0)
    parser.add_argument("--separate-joint-encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-positive-passage", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = _native_resolve(args.data_dir)
    args.off_the_shelf_checkpoint = _native_resolve(args.off_the_shelf_checkpoint)
    args.embedding_dir = _native_resolve(args.embedding_dir)
    args.output_dir = _native_resolve(args.output_dir)
    if args.tensorboard_log_dir is not None:
        args.tensorboard_log_dir = _native_resolve(args.tensorboard_log_dir)
    if args.work_dir is None:
        args.work_dir = args.output_dir / "reader_finetuning_data"
    else:
        args.work_dir = _native_resolve(args.work_dir)
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Using CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
    if not args.skip_rankings:
        _native_prepare_rankings(args)
    if not args.skip_train:
        _native_train(args)


if __name__ == "__main__":
    main()
