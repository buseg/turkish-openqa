#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FSMODQA_DIR="${ROOT_DIR}/external/FSMODQA"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoint/final_fsmodqa_squad_tr_full_deavg}"
MODEL_DIR="${MODEL_DIR:-${CHECKPOINT_DIR}/checkpoint-best}"
TRAIN_DIR="${TRAIN_DIR:-${ROOT_DIR}/odqa_data/fsmodqa_retrieval}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/odqa_data/squad_tr_processed_test}"
ADAPTIVE_DIR="${ADAPTIVE_DIR:-${ROOT_DIR}/checkpoint/adaptive_retrieval}"
RESULTS_DIR="${RESULTS_DIR:-${CHECKPOINT_DIR}/results/results_adaptive_retrieval}"
DETAILED_RESULTS_DIR="${DETAILED_RESULTS_DIR:-${RESULTS_DIR}/detailed}"

EASY_RANKING_FILE="${EASY_RANKING_FILE:-${ADAPTIVE_DIR}/test_easy_rankings.jsonl}"
MEDIUM_RANKING_FILE="${MEDIUM_RANKING_FILE:-${ADAPTIVE_DIR}/test_medium_rankings.jsonl}"
HARD_RANKING_FILE="${HARD_RANKING_FILE:-${ADAPTIVE_DIR}/test_hard_rankings.jsonl}"
COMBINED_RANKING_FILE="${COMBINED_RANKING_FILE:-${RESULTS_DIR}/test_adaptive_combined_rankings.jsonl}"
COMBINED_PREDICTIONS="${COMBINED_PREDICTIONS:-${RESULTS_DIR}/test_reader_predictions_adaptive.json}"
METRICS="${METRICS:-${DETAILED_RESULTS_DIR}/test_metrics_adaptive.json}"
PER_EXAMPLE_METRICS="${PER_EXAMPLE_METRICS:-${DETAILED_RESULTS_DIR}/test_per_example_metrics_adaptive.json}"

mkdir -p "${RESULTS_DIR}" "${DETAILED_RESULTS_DIR}"

passage_count() {
  local ranking_file="$1"
  "${PYTHON_BIN}" - "$ranking_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
counts = []
if path.exists():
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                counts.append(len(json.loads(line).get("pids", [])))
if not counts:
    print(0)
else:
    min_count = min(counts)
    max_count = max(counts)
    if min_count != max_count:
        print(f"warning: {path} has variable passage counts min={min_count} max={max_count}; using min={min_count}", file=sys.stderr)
    print(min_count)
PY
}

run_reader_for_class() {
  local label="$1"
  local ranking_file="$2"
  local predictions="$3"
  local n_passages

  if [[ ! -s "${ranking_file}" ]]; then
    echo "==> Skipping ${label}: ${ranking_file} is missing or empty"
    "${PYTHON_BIN}" - "$predictions" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8") as handle:
    json.dump({}, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
    return
  fi

  n_passages="$(passage_count "${ranking_file}")"
  if [[ "${n_passages}" -le 0 ]]; then
    echo "==> Skipping ${label}: no pids in ${ranking_file}"
    return
  fi

  echo "==> Running reader for ${label} with ${n_passages} passages"
  pushd "${FSMODQA_DIR}" >/dev/null
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" test_reader.py \
    --output_dir "${CHECKPOINT_DIR}" \
    --model_name_or_path "${MODEL_DIR}" \
    --output_path "${predictions}" \
    --train_dir "${TRAIN_DIR}" \
    --train_path "${ranking_file}" \
    --corpus_file corpus.jsonl \
    --query_file test.query.jsonl \
    --per_device_eval_batch_size 1 \
    --train_n_passages "${n_passages}" \
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
}

EASY_PREDICTIONS="${RESULTS_DIR}/test_reader_predictions_easy.json"
MEDIUM_PREDICTIONS="${RESULTS_DIR}/test_reader_predictions_medium.json"
HARD_PREDICTIONS="${RESULTS_DIR}/test_reader_predictions_hard.json"

run_reader_for_class easy "${EASY_RANKING_FILE}" "${EASY_PREDICTIONS}"
run_reader_for_class medium "${MEDIUM_RANKING_FILE}" "${MEDIUM_PREDICTIONS}"
run_reader_for_class hard "${HARD_RANKING_FILE}" "${HARD_PREDICTIONS}"

echo "==> Merging adaptive ranking files"
cat "${EASY_RANKING_FILE}" "${MEDIUM_RANKING_FILE}" "${HARD_RANKING_FILE}" > "${COMBINED_RANKING_FILE}"

echo "==> Merging reader predictions"
"${PYTHON_BIN}" - "${COMBINED_PREDICTIONS}" "${EASY_PREDICTIONS}" "${MEDIUM_PREDICTIONS}" "${HARD_PREDICTIONS}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
merged = {}
for raw_path in sys.argv[2:]:
    path = Path(raw_path)
    if not path.exists():
        continue
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        merged.update({str(key): value for key, value in data.items()})
    elif isinstance(data, list):
        for item in data:
            qid = item.get("id") or item.get("qid")
            prediction = item.get("prediction", item.get("answer", item.get("text", "")))
            if qid is not None:
                merged[str(qid)] = prediction
    else:
        raise TypeError(f"Unsupported prediction format in {path}: {type(data)!r}")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    json.dump(merged, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(f"merged_predictions={len(merged)}")
PY

echo "==> Evaluating merged adaptive reader predictions"
"${PYTHON_BIN}" "${ROOT_DIR}/src/evaluate.py" \
  --predictions "${COMBINED_PREDICTIONS}" \
  --dataset "${DATASET_DIR}" \
  --output "${METRICS}" \
  --per-example-output "${PER_EXAMPLE_METRICS}"

echo "==> Wrote ${METRICS}"
echo "==> Combined rankings: ${COMBINED_RANKING_FILE}"
echo "==> Combined predictions: ${COMBINED_PREDICTIONS}"
