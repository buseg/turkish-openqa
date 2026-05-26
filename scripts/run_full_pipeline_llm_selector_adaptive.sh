#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GEMMA_PYTHON_BIN="${GEMMA_PYTHON_BIN:-/cta/users/buse/miniconda3/envs/fsmodqa_gemma/bin/python}"
FSMODQA_PYTHON_BIN="${FSMODQA_PYTHON_BIN:-/cta/users/buse/miniconda3/envs/fsmodqa_env/bin/python}"

if [[ ! -x "${GEMMA_PYTHON_BIN}" ]]; then
  echo "Missing GEMMA_PYTHON_BIN: ${GEMMA_PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${FSMODQA_PYTHON_BIN}" ]]; then
  echo "Missing FSMODQA_PYTHON_BIN: ${FSMODQA_PYTHON_BIN}" >&2
  exit 1
fi

CUDA_DEVICE="${CUDA_DEVICE:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoint/final_fsmodqa_squad_tr_full_deavg}"
TRAIN_DIR="${TRAIN_DIR:-${ROOT_DIR}/odqa_data/fsmodqa_retrieval}"
CORPUS_FILE="${CORPUS_FILE:-${TRAIN_DIR}/corpus.jsonl}"
TEST_QUERY_FILE="${TEST_QUERY_FILE:-${TRAIN_DIR}/test.query.jsonl}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/odqa_data/squad_tr_processed_test}"
QID_ALIASES="${QID_ALIASES:-${TRAIN_DIR}/squad_duplicate_qid_lookup.json}"

PIPELINE_DIR="${PIPELINE_DIR:-${CHECKPOINT_DIR}/results/results_full_pipeline}"
LLM_WORK_DIR="${LLM_WORK_DIR:-${PIPELINE_DIR}/llm_expand_work}"
LLM_RANKING_FILE="${LLM_RANKING_FILE:-${PIPELINE_DIR}/test_top100_rw_full.jsonl}"
LLM_DEBUG_OUTPUT="${LLM_DEBUG_OUTPUT:-${LLM_WORK_DIR}/expansion_debug_rw_full.jsonl}"

SELECTOR_MODEL_DIR="${SELECTOR_MODEL_DIR:-${CHECKPOINT_DIR}/neural_knowledge_selector}"
SELECTOR_RANKING_FILE="${SELECTOR_RANKING_FILE:-${PIPELINE_DIR}/test_top100_rw_full_selector_reranked.jsonl}"

ADAPTIVE_OUTPUT_DIR="${ADAPTIVE_OUTPUT_DIR:-${PIPELINE_DIR}/adaptive_retrieval}"
ADAPTIVE_MODEL="${ADAPTIVE_MODEL:-${ADAPTIVE_OUTPUT_DIR}/adaptive_retrieval_classifier.pkl}"
ADAPTIVE_POLICY="${ADAPTIVE_POLICY:-${ADAPTIVE_OUTPUT_DIR}/policy.json}"
ADAPTIVE_MODE="${ADAPTIVE_MODE:-train}"
COMBINED_RANKING_FILE="${COMBINED_RANKING_FILE:-${PIPELINE_DIR}/test_adaptive_combined_rankings.jsonl}"
RETRIEVAL_METRICS="${RETRIEVAL_METRICS:-${PIPELINE_DIR}/test_adaptive_combined_retrieval_metrics.json}"
READER_RESULTS_DIR="${READER_RESULTS_DIR:-${PIPELINE_DIR}/reader}"

ADAPTIVE_BASE_TRAIN_RANKINGS="${ADAPTIVE_BASE_TRAIN_RANKINGS:-${CHECKPOINT_DIR}/train_top100_with_scores.jsonl}"
ADAPTIVE_BASE_VALIDATION_RANKINGS="${ADAPTIVE_BASE_VALIDATION_RANKINGS:-${CHECKPOINT_DIR}/validation_top100_with_scores.jsonl}"
ADAPTIVE_TRAIN_RANKINGS="${ADAPTIVE_TRAIN_RANKINGS:-${PIPELINE_DIR}/train_top100_selector_reranked.jsonl}"
ADAPTIVE_VALIDATION_RANKINGS="${ADAPTIVE_VALIDATION_RANKINGS:-${PIPELINE_DIR}/validation_top100_selector_reranked.jsonl}"
ADAPTIVE_TRAIN_QUERIES="${ADAPTIVE_TRAIN_QUERIES:-${TRAIN_DIR}/train.query.jsonl}"
ADAPTIVE_VALIDATION_QUERIES="${ADAPTIVE_VALIDATION_QUERIES:-${TRAIN_DIR}/validation.query.jsonl}"

REWRITE_MODEL="${REWRITE_MODEL:-google/gemma-2-2b-it}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${CHECKPOINT_DIR}/checkpoint-best}"
PASSAGE_EMBEDDINGS="${PASSAGE_EMBEDDINGS:-${CHECKPOINT_DIR}/encoding/passage_embedding_split*.pt}"
TOP_N="${TOP_N:-100}"
RETRIEVE_DEPTH="${RETRIEVE_DEPTH:-100}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-128}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-1}"
MAX_QUERY_LENGTH="${MAX_QUERY_LENGTH:-50}"
MAX_PASSAGE_LENGTH="${MAX_PASSAGE_LENGTH:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

SELECTOR_N_CONTEXT="${SELECTOR_N_CONTEXT:-100}"
SELECTOR_EVAL_BATCH_SIZE="${SELECTOR_EVAL_BATCH_SIZE:-32}"
SELECTOR_MAX_LENGTH="${SELECTOR_MAX_LENGTH:-256}"
SELECTOR_WEIGHT="${SELECTOR_WEIGHT:-}"

EASY_OPTIONS="${EASY_OPTIONS:-5,10,15,20,25}"
MEDIUM_OPTIONS="${MEDIUM_OPTIONS:-25,35,50,75}"
HARD_OPTIONS="${HARD_OPTIONS:-100}"
MIN_RECALL_RATIO="${MIN_RECALL_RATIO:-0.95}"
MATCH_MODE="${MATCH_MODE:-squad-prefix}"

SKIP_LLM="${SKIP_LLM:-0}"
SKIP_SELECTOR="${SKIP_SELECTOR:-0}"
SKIP_ADAPTIVE_TRAINVAL_SELECTOR="${SKIP_ADAPTIVE_TRAINVAL_SELECTOR:-0}"
SKIP_ADAPTIVE="${SKIP_ADAPTIVE:-0}"
SKIP_RETRIEVAL_EVAL="${SKIP_RETRIEVAL_EVAL:-0}"
SKIP_READER="${SKIP_READER:-0}"
USE_QUERY_REWRITE="${USE_QUERY_REWRITE:-1}"

mkdir -p "${PIPELINE_DIR}" "${ADAPTIVE_OUTPUT_DIR}" "${READER_RESULTS_DIR}"

if [[ "${SKIP_LLM}" != "1" ]]; then
  echo "==> Step 1/5: LLM expansion + retrieval"
  llm_cmd=(
    "${GEMMA_PYTHON_BIN}" "${ROOT_DIR}/src/llm_expand_retrieve.py"
    --input-query-file "${TEST_QUERY_FILE}"
    --output-ranking "${LLM_RANKING_FILE}"
    --work-dir "${LLM_WORK_DIR}"
    --debug-output "${LLM_DEBUG_OUTPUT}"
    --rewrite-model "${REWRITE_MODEL}"
    --retriever-model "${RETRIEVER_MODEL}"
    --train-dir "${TRAIN_DIR}"
    --corpus-file "$(basename "${CORPUS_FILE}")"
    --passage-embeddings "${PASSAGE_EMBEDDINGS}"
    --retrieve-depth "${RETRIEVE_DEPTH}"
    --top-n "${TOP_N}"
    --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE}"
    --encode-batch-size "${ENCODE_BATCH_SIZE}"
    --max-query-length "${MAX_QUERY_LENGTH}"
    --max-passage-length "${MAX_PASSAGE_LENGTH}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --separate-joint-encoding
    --de-avg-pooling
    --add-lang-token
    --tf32
  )
  if [[ "${USE_QUERY_REWRITE:-0}" == "1" ]]; then
    llm_cmd+=(--use-query-rewrite)
  fi
  if [[ "${USE_GPU_RETRIEVAL:-0}" == "1" ]]; then
    llm_cmd+=(--use-gpu-retrieval)
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${llm_cmd[@]}"
else
  echo "==> Skipping LLM expansion; using ${LLM_RANKING_FILE}"
fi

if [[ "${SKIP_SELECTOR}" != "1" ]]; then
  echo "==> Step 2/5: knowledge selector reranking"
  selector_cmd=(
    "${FSMODQA_PYTHON_BIN}" "${ROOT_DIR}/src/apply_fsmodqa_neural_selector.py"
    --model-dir "${SELECTOR_MODEL_DIR}"
    --ranking "${LLM_RANKING_FILE}"
    --queries "${TEST_QUERY_FILE}"
    --corpus "${CORPUS_FILE}"
    --output "${SELECTOR_RANKING_FILE}"
    --n-context "${SELECTOR_N_CONTEXT}"
    --eval-batch-size "${SELECTOR_EVAL_BATCH_SIZE}"
    --max-length "${SELECTOR_MAX_LENGTH}"
  )
  if [[ -n "${SELECTOR_WEIGHT}" ]]; then
    selector_cmd+=(--selector-weight "${SELECTOR_WEIGHT}")
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${selector_cmd[@]}"
else
  echo "==> Skipping selector; using ${SELECTOR_RANKING_FILE}"
fi

if [[ "${SKIP_ADAPTIVE}" != "1" && "${ADAPTIVE_MODE}" == "train" && "${SKIP_ADAPTIVE_TRAINVAL_SELECTOR}" != "1" ]]; then
  echo "==> Preparing adaptive train/validation rankings with saved knowledge selector"
  adaptive_train_selector_cmd=(
    "${FSMODQA_PYTHON_BIN}" "${ROOT_DIR}/src/apply_fsmodqa_neural_selector.py"
    --model-dir "${SELECTOR_MODEL_DIR}"
    --ranking "${ADAPTIVE_BASE_TRAIN_RANKINGS}"
    --queries "${ADAPTIVE_TRAIN_QUERIES}"
    --corpus "${CORPUS_FILE}"
    --output "${ADAPTIVE_TRAIN_RANKINGS}"
    --n-context "${SELECTOR_N_CONTEXT}"
    --eval-batch-size "${SELECTOR_EVAL_BATCH_SIZE}"
    --max-length "${SELECTOR_MAX_LENGTH}"
  )
  if [[ -n "${SELECTOR_WEIGHT}" ]]; then
    adaptive_train_selector_cmd+=(--selector-weight "${SELECTOR_WEIGHT}")
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${adaptive_train_selector_cmd[@]}"

  adaptive_validation_selector_cmd=(
    "${FSMODQA_PYTHON_BIN}" "${ROOT_DIR}/src/apply_fsmodqa_neural_selector.py"
    --model-dir "${SELECTOR_MODEL_DIR}"
    --ranking "${ADAPTIVE_BASE_VALIDATION_RANKINGS}"
    --queries "${ADAPTIVE_VALIDATION_QUERIES}"
    --corpus "${CORPUS_FILE}"
    --output "${ADAPTIVE_VALIDATION_RANKINGS}"
    --n-context "${SELECTOR_N_CONTEXT}"
    --eval-batch-size "${SELECTOR_EVAL_BATCH_SIZE}"
    --max-length "${SELECTOR_MAX_LENGTH}"
  )
  if [[ -n "${SELECTOR_WEIGHT}" ]]; then
    adaptive_validation_selector_cmd+=(--selector-weight "${SELECTOR_WEIGHT}")
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${adaptive_validation_selector_cmd[@]}"
else
  echo "==> Skipping adaptive train/validation selector preparation"
fi

if [[ "${SKIP_ADAPTIVE}" != "1" ]]; then
  echo "==> Step 3/5: adaptive retrieval training/tuning + test cropping"
  "${GEMMA_PYTHON_BIN}" "${ROOT_DIR}/src/train_adaptive_retrieval.py" \
    --mode "${ADAPTIVE_MODE}" \
    --model "${ADAPTIVE_MODEL}" \
    --policy "${ADAPTIVE_POLICY}" \
    --train-rankings "${ADAPTIVE_TRAIN_RANKINGS}" \
    --validation-rankings "${ADAPTIVE_VALIDATION_RANKINGS}" \
    --test-rankings "${SELECTOR_RANKING_FILE}" \
    --train-queries "${ADAPTIVE_TRAIN_QUERIES}" \
    --validation-queries "${ADAPTIVE_VALIDATION_QUERIES}" \
    --test-queries "${TEST_QUERY_FILE}" \
    --corpus "${CORPUS_FILE}" \
    --qid-aliases "${QID_ALIASES}" \
    --output-dir "${ADAPTIVE_OUTPUT_DIR}" \
    --easy-options "${EASY_OPTIONS}" \
    --medium-options "${MEDIUM_OPTIONS}" \
    --hard-options "${HARD_OPTIONS}" \
    --min-recall-ratio "${MIN_RECALL_RATIO}" \
    --match-mode "${MATCH_MODE}"
else
  echo "==> Skipping adaptive retrieval; using ${ADAPTIVE_OUTPUT_DIR}"
fi

if [[ "${SKIP_RETRIEVAL_EVAL}" != "1" ]]; then
  echo "==> Step 4/5: merge adaptive rankings + retrieval evaluation"
  cat \
    "${ADAPTIVE_OUTPUT_DIR}/test_easy_rankings.jsonl" \
    "${ADAPTIVE_OUTPUT_DIR}/test_medium_rankings.jsonl" \
    "${ADAPTIVE_OUTPUT_DIR}/test_hard_rankings.jsonl" \
    > "${COMBINED_RANKING_FILE}"

  "${FSMODQA_PYTHON_BIN}" "${ROOT_DIR}/src/evaluate_retrieval.py" \
    --rankings "${COMBINED_RANKING_FILE}" \
    --match-mode "${MATCH_MODE}" \
    --qid-aliases "${QID_ALIASES}" \
    --output "${RETRIEVAL_METRICS}"
else
  echo "==> Skipping retrieval evaluation"
fi

if [[ "${SKIP_READER}" != "1" ]]; then
  echo "==> Step 5/5: reader evaluation on adaptive class files"
  PYTHON_BIN="${FSMODQA_PYTHON_BIN}" \
  CUDA_DEVICE="${CUDA_DEVICE}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  TRAIN_DIR="${TRAIN_DIR}" \
  DATASET_DIR="${DATASET_DIR}" \
  ADAPTIVE_DIR="${ADAPTIVE_OUTPUT_DIR}" \
  RESULTS_DIR="${READER_RESULTS_DIR}" \
  COMBINED_RANKING_FILE="${COMBINED_RANKING_FILE}" \
  bash "${ROOT_DIR}/scripts/run_test_reader_adaptive_retrieval.sh"
else
  echo "==> Skipping reader evaluation"
fi

echo "==> Full pipeline complete"
echo "Gemma Python: ${GEMMA_PYTHON_BIN}"
echo "FSMODQA Python: ${FSMODQA_PYTHON_BIN}"
echo "LLM ranking: ${LLM_RANKING_FILE}"
echo "Selector ranking: ${SELECTOR_RANKING_FILE}"
echo "Adaptive dir: ${ADAPTIVE_OUTPUT_DIR}"
echo "Adaptive mode: ${ADAPTIVE_MODE}"
echo "Adaptive base train ranking: ${ADAPTIVE_BASE_TRAIN_RANKINGS}"
echo "Adaptive base validation ranking: ${ADAPTIVE_BASE_VALIDATION_RANKINGS}"
echo "Adaptive train ranking: ${ADAPTIVE_TRAIN_RANKINGS}"
echo "Adaptive validation ranking: ${ADAPTIVE_VALIDATION_RANKINGS}"
echo "Adaptive model: ${ADAPTIVE_MODEL}"
echo "Adaptive policy: ${ADAPTIVE_POLICY}"
echo "Combined ranking: ${COMBINED_RANKING_FILE}"
echo "Retrieval metrics: ${RETRIEVAL_METRICS}"
echo "Reader results: ${READER_RESULTS_DIR}"
