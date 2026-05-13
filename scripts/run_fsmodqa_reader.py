#!/usr/bin/env python3
"""Convenience wrapper for external/FSMODQA/test_reader.py."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FSMODQA_ROOT = ROOT / "external" / "FSMODQA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="busegi/FSMODQA-SQUAD-TR")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--train-path", default="reader.jsonl")
    parser.add_argument("--corpus-file", default="corpus.jsonl")
    parser.add_argument("--query-file", default="queries.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-path", default="predictions.json")
    parser.add_argument("--n-passages", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--conda-env", default="")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--add-lang-token", action="store_true")
    parser.add_argument("--de-avg-pooling", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        "test_reader.py",
        "--model_name_or_path",
        args.model,
        "--train_dir",
        str(Path(args.train_dir).resolve()),
        "--train_path",
        args.train_path,
        "--corpus_file",
        args.corpus_file,
        "--query_file",
        args.query_file,
        "--output_path",
        args.output_path,
        "--train_n_passages",
        str(args.n_passages),
        "--max_query_length",
        "50",
        "--max_passage_length",
        "200",
        "--max_query_passage_length",
        "250",
        "--max_answer_length",
        "50",
        "--per_device_eval_batch_size",
        str(args.batch_size),
        "--output_dir",
        str(Path(args.output_dir).resolve()),
        "--separate_joint_encoding",
        "true",
    ]
    if not args.use_cuda:
        command.extend(["--no_cuda", "true"])
    if args.add_lang_token:
        command.extend(["--add_lang_token", "true"])
    if args.de_avg_pooling:
        command.extend(["--de_avg_pooling", "true"])

    if args.conda_env:
        command[0] = "python"
        command = ["conda", "run", "-n", args.conda_env] + command

    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf_cache"))
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    print(" ".join(command))
    subprocess.run(command, cwd=FSMODQA_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
