#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FSMODQA_DIR="${ROOT_DIR}/external/FSMODQA"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
TOPKS=(${TOPKS:-5 10})

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoint/final_fsmodqa_squad_tr_full_deavg}"
MODEL_DIR="${MODEL_DIR:-${CHECKPOINT_DIR}/checkpoint-best}"
TRAIN_DIR="${TRAIN_DIR:-${ROOT_DIR}/odqa_data/fsmodqa_retrieval}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/odqa_data/squad_tr_processed_test}"
RESULTS_DIR="${RESULTS_DIR:-${CHECKPOINT_DIR}/results/results_rewrite}"
RANKING_FILE="${RANKING_FILE:-${CHECKPOINT_DIR}/results/results_rewrite/test_top10_rw_full.jsonl}"

DETAILED_RESULTS_DIR="${RESULTS_DIR:-${CHECKPOINT_DIR}/results/results_rewrite/detailed}"

mkdir -p "${DETAILED_RESULTS_DIR}"

for topk in "${TOPKS[@]}"; do
  predictions="${RESULTS_DIR}/test_reader_predictions_top${topk}.json"
  metrics="${DETAILED_RESULTS_DIR}/test_metrics_top${topk}.json"
  per_example_metrics="${DETAILED_RESULTS_DIR}/test_per_example_metrics_top${topk}.json"

  echo "==> Running reader with top ${topk} passages"
  pushd "${FSMODQA_DIR}" >/dev/null
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" test_reader.py \
    --output_dir "${CHECKPOINT_DIR}" \
    --model_name_or_path "${MODEL_DIR}" \
    --output_path "${predictions}" \
    --train_dir "${TRAIN_DIR}" \
    --train_path "${RANKING_FILE}" \
    --corpus_file corpus.jsonl \
    --query_file test.query.jsonl \
    --per_device_eval_batch_size 1 \
    --train_n_passages "${topk}" \
    --max_query_length 50 \
    --max_passage_length 200 \
    --max_query_passage_length 250 \
    --max_answer_length 50 \
    --separate_joint_encoding \
    --de_avg_pooling \
    --add_lang_token \
    --bf16 False \
    --tf32 True
  popd >/dev/null

  echo "==> Evaluating top ${topk} predictions"
  "${PYTHON_BIN}" "${ROOT_DIR}/src/evaluate.py" \
    --predictions "${predictions}" \
    --dataset "${DATASET_DIR}" \
    --output "${metrics}" \
    --per-example-output "${per_example_metrics}"

  echo "==> Wrote ${metrics}"
done
