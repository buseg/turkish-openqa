# Turkish Open-Domain Question Answering

This repository contains our Turkish open-domain question answering (ODQA) pipeline and experiments for SQuAD-TR and a Turkish Wikipedia passage corpus.

The project follows a retrieve-then-read setup:

1. Retrieve candidate passages from the Turkish passage corpus.
2. Pass the retrieved contexts to an FSMODQA/FiD-style reader.
3. Evaluate retrieval quality, answer quality, faithfulness, and refusal behavior.

The main reader checkpoint used in the final experiments is:

```text
busegi/FSMODQA-SQUAD-TR
```

Generated data, checkpoints, and model caches are ignored by git. The most important generated folders are `data/`, `checkpoints/`, `.hf_cache/`, and `odqa_data/`.

## Final Result

The strongest deployable variant in our experiments is:

```text
Retrieval-aware adaptive retrieval + FSMODQA reader
```

It gives the best reader F1 among the tested deployable settings while reducing the average number of retrieved contexts compared with a fixed top-100 policy.

All metric values in the table are percentages. Reader metrics are 200-example answerable FSMODQA smoke/evidence checks; retrieval metrics are full-dev unless noted.

| Ablation | System | Retrieval Ref. (R@K) | Contexts | EM | F1 | Correctness | Faithfulness | Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | BM25 top-5 + FSMODQA | 15.67 | 5 | 9.00 | 15.17 | 15.32 | 72.84 | Main fixed-retrieval baseline. |
| More Contexts | BM25 top-25 + FSMODQA | 27.49 | 25 | 10.00 | 14.80 | 15.16 | 84.74 | More evidence improves EM, but not F1. |
| Adaptive Retrieval | Retrieval-aware 50/50/100 + FSMODQA | 34.81 | 76.86 avg | 10.00 | 15.42 | 15.95 | 81.74 | Best tested reader F1 with fewer contexts than fixed top-100. |
| Question Rewriting | Keyword rewrite top-5 + FSMODQA | 15.74 | 5 | 7.00 | 10.55 | 11.33 | 78.65 | Small retrieval gain does not transfer to reader quality. |
| Selective Rewriting | Selective keyword rewrite top-5 + FSMODQA | 15.67 | 5 | 7.00 | 10.91 | 11.45 | 79.39 | Still below baseline. |
| Knowledge Selector | TF-IDF selector hybrid top-5 + FSMODQA | 16.53 | 5 | 9.00 | 14.85 | 15.24 | 72.11 | Close to baseline, but no reader gain. |
| Neural Selector | BERTurk selector hybrid top-5 + FSMODQA | 23.50 | 5 | 8.50 | 13.95 | 14.32 | 75.35 | Improves selector MRR, but not reader F1 yet. |
| Oracle Upper Bound | Oracle top-5 + FSMODQA | - | 5 | 12.50 | 19.31 | 20.16 | 73.83 | Not deployable; shows retrieval is still the bottleneck. |

For the full generated metric summary, see [`evaluation_summary.md`](evaluation_summary.md). For the longer command and experiment log, see [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

Regenerate the metric summary with:

```bash
python scripts/summarize_evaluation.py
```

## Experiment Scope

### Core System

These components form the core ODQA system:

- SQuAD-TR and Turkish Wikipedia/passage corpus preparation.
- BM25 retrieval over the Turkish passage corpus.
- FSMODQA/FiD-style reader input construction.
- FSMODQA reader inference using `busegi/FSMODQA-SQUAD-TR`.
- Retrieval evaluation with Recall@K and MRR.
- Reader evaluation with EM and token-level F1.
- Expanded evaluation with correctness recall, faithfulness/k-precision proxy, and refusal metrics.

### Main Experiments

These experiments are included in the final ablation table:

- Fixed retrieval baseline: BM25 top-5 and BM25 top-25.
- Adaptive retrieval: retrieval-aware 50/50/100 policy.
- Question rewriting: keyword and selective keyword rewriting before retrieval.
- Knowledge selection: TF-IDF and BERTurk selectors over BM25 candidates.
- Oracle top-5: answer-aware upper bound for retrieval quality.

### Supporting Probes

These were useful for engineering and analysis, but they are not the main final claims:

- Base FiD smoke training to validate the reader training pipeline.
- FSMODQA dense retriever probe over BM25 top-100 candidates.
- Cross-encoder reranker pilot.
- Mixed answerable/unanswerable refusal check.

### Optional / Future Work

These proposal-aligned extensions remain future work:

- DPR/ConvBERTurk retriever training over Turkish Wikipedia.
- RL-based knowledge selector training.
- Document rewriting after retrieval.
- Synthetic Turkish QA data augmentation.
- Refusal-aware reader training or prompting.
- Larger full-dev reader runs for the strongest settings.

## Repository Layout

```text
download_data.ipynb           Data download/preparation notebook
scripts/                      Project scripts for retrieval, rewriting, selection, and evaluation
external/FSMODQA/             FSMODQA reader/retriever code
external/fid/                 Legacy FiD smoke-training code
evaluation_summary.md         Generated compact metric summary
EXPERIMENT_LOG.md             Longer command log and intermediate experiment notes
data/                         Generated datasets and retrieval files, ignored by git
checkpoints/                  Generated predictions, metrics, and model outputs, ignored by git
```

Important scripts:

| Script | Purpose |
| --- | --- |
| `scripts/prepare_qa_data.py` | Convert SQuAD-style data to compact QA JSON. |
| `scripts/export_passages.py` | Export passage corpus to TSV. |
| `scripts/retrieve_bm25.py` | Run BM25 retrieval. |
| `scripts/evaluate_retrieval.py` | Compute Recall@K and MRR. |
| `scripts/build_fsmodqa_reader_inputs.py` | Convert retrieved JSON to FSMODQA reader files. |
| `scripts/run_fsmodqa_reader.py` | Run FSMODQA reader inference. |
| `scripts/evaluate_qa_expanded.py` | Compute EM, F1, correctness, faithfulness, and refusal metrics. |
| `scripts/rewrite_questions.py` | Rule-based Turkish question rewriting. |
| `scripts/rewrite_questions_llm.py` | LLM-based question rewriting interface. |
| `scripts/select_retrieval.py` | Select original vs rewritten retrieval results. |
| `scripts/build_complexity_labels.py` | Build easy/medium/hard adaptive retrieval labels. |
| `scripts/train_retrieval_aware_complexity_classifier.py` | Train retrieval-aware adaptive classifier. |
| `scripts/apply_adaptive_retrieval.py` | Apply adaptive retrieval budgets. |
| `scripts/build_knowledge_selector_data.py` | Build passage selector data. |
| `scripts/train_knowledge_selector.py` | Train TF-IDF/logistic-regression selector. |
| `scripts/train_neural_knowledge_selector.py` | Train BERT-style neural selector. |
| `scripts/apply_cross_encoder_reranker.py` | Apply neural selector/cross-encoder reranking. |
| `scripts/summarize_evaluation.py` | Generate `evaluation_summary.md`. |

## Setup

Create the project environments:

```bash
conda env create -f odqa_environment.yml
conda env create -f fsmodqa_environment.yml
```

Most final FSMODQA commands were run with:

```bash
conda run -n turkish-odqa-fid python ...
```

If the Hugging Face model is not already cached locally, log in once:

```bash
conda run -n turkish-odqa-fid huggingface-cli login
```

## Reproducing The Pipeline

### 1. Prepare Data

Run `download_data.ipynb` to prepare SQuAD-TR and the Turkish passage corpus. Then export compact QA files and passages:

```bash
python scripts/prepare_qa_data.py \
  --input odqa_data/squad_tr \
  --split train \
  --output data/squad_tr_train_answerable.json

python scripts/prepare_qa_data.py \
  --input odqa_data/squad_tr \
  --split validation \
  --output data/squad_tr_dev_answerable.json

python scripts/export_passages.py \
  --input odqa_data/wiki_20230901_tr_chunked \
  --output data/passages.tsv
```

To keep unanswerable examples for refusal evaluation, add `--include-unanswerable` when preparing QA data.

### 2. Run BM25 Retrieval

```bash
python scripts/retrieve_bm25.py \
  --qa data/squad_tr_dev_answerable.json \
  --passages data/passages.tsv \
  --output data/squad_tr_dev_bm25_answerable.json \
  --n-docs 100

python scripts/evaluate_retrieval.py \
  --input data/squad_tr_dev_bm25_answerable.json \
  --ks 1,5,10,20,25,50,100 \
  --match-mode answer \
  --output data/squad_tr_dev_bm25_answerable_retrieval_metrics.json
```

Observed BM25 full-dev answerable retrieval:

| Metric | Value |
| --- | ---: |
| Examples | 2910 |
| Recall@1 | 7.80 |
| Recall@5 | 15.67 |
| Recall@10 | 19.90 |
| Recall@25 | 27.49 |
| Recall@50 | 31.79 |
| Recall@100 | 37.29 |
| MRR | 11.94 |

### 3. Build FSMODQA Reader Inputs

```bash
python scripts/build_fsmodqa_reader_inputs.py \
  --retrieved data/squad_tr_dev_bm25_answerable.json \
  --output-dir data/fsmodqa_dev_bm25_top5_answerable \
  --n-context 5 \
  --prefix bm25top5
```

For 200-example smoke/evidence checks:

```bash
sed -n '1,200p' \
  data/fsmodqa_dev_bm25_top5_answerable/reader.jsonl \
  > data/fsmodqa_dev_bm25_top5_answerable/reader_200.jsonl
```

### 4. Run FSMODQA Reader

```bash
python scripts/run_fsmodqa_reader.py \
  --model busegi/FSMODQA-SQUAD-TR \
  --train-dir data/fsmodqa_dev_bm25_top5_answerable \
  --train-path reader_200.jsonl \
  --output-dir checkpoints/fsmodqa_squad_tr_bm25_top5_answerable_200 \
  --output-path predictions_200.json \
  --n-passages 5 \
  --batch-size 1 \
  --conda-env turkish-odqa-fid \
  --offline
```

### 5. Evaluate QA

```bash
python scripts/slice_retrieved.py \
  --input data/squad_tr_dev_bm25_answerable.json \
  --output data/squad_tr_dev_bm25_top5_answerable_200.json \
  --limit 200 \
  --n-context 5

python scripts/evaluate_qa_expanded.py \
  --predictions checkpoints/fsmodqa_squad_tr_bm25_top5_answerable_200/predictions_200.json \
  --dataset data/squad_tr_dev_bm25_top5_answerable_200.json \
  --output checkpoints/fsmodqa_squad_tr_bm25_top5_answerable_200/expanded_metrics.json \
  --per-example-output checkpoints/fsmodqa_squad_tr_bm25_top5_answerable_200/expanded_per_example_metrics.json
```

### 6. Regenerate The Summary

```bash
python scripts/summarize_evaluation.py
```

## Main Experiment Commands

The exact generated files used in the final ablation table are already listed in `scripts/summarize_evaluation.py`. The commands below show the main experimental entry points.

Adaptive retrieval:

```bash
python scripts/build_complexity_labels.py \
  --input data/squad_tr_train_5k_bm25.json \
  --output data/squad_tr_train_5k_complexity_labels.jsonl

python scripts/build_complexity_labels.py \
  --input data/squad_tr_dev_bm25_answerable.json \
  --output data/squad_tr_dev_bm25_answerable_complexity_labels.jsonl

python scripts/train_retrieval_aware_complexity_classifier.py \
  --retrieved data/squad_tr_train_5k_bm25.json \
  --labels data/squad_tr_train_5k_complexity_labels.jsonl \
  --output checkpoints/complexity_classifier_retrieval_aware_tfidf_lr/model.pkl \
  --metrics-output checkpoints/complexity_classifier_retrieval_aware_tfidf_lr/metrics.json

python scripts/apply_adaptive_retrieval.py \
  --input data/squad_tr_dev_bm25_answerable.json \
  --output data/squad_tr_dev_bm25_adaptive_retrieval_aware_50_50_100.json \
  --strategy classifier \
  --model checkpoints/complexity_classifier_retrieval_aware_tfidf_lr/model.pkl \
  --easy-k 50 \
  --medium-k 50 \
  --hard-k 100 \
  --ks 1,5,10,25,50,100
```

Question rewriting:

```bash
python scripts/rewrite_questions.py \
  --input data/squad_tr_dev_answerable.json \
  --output data/squad_tr_dev_answerable_rewritten_keyword.json \
  --mode keyword

python scripts/retrieve_bm25.py \
  --qa data/squad_tr_dev_answerable_rewritten_keyword.json \
  --passages data/passages.tsv \
  --output data/squad_tr_dev_bm25_rewritten_keyword.json \
  --n-docs 100

python scripts/select_retrieval.py \
  --original data/squad_tr_dev_bm25_answerable.json \
  --rewritten data/squad_tr_dev_bm25_rewritten_keyword.json \
  --output data/squad_tr_dev_bm25_selective_normscore_tuned.json \
  --strategy normalized-top-score \
  --ratio 1.125 \
  --common-only \
  --ks 1,5,10,20,50,100
```

Knowledge selector:

```bash
python scripts/build_knowledge_selector_data.py \
  --input data/squad_tr_train_5k_bm25.json \
  --output data/squad_tr_train_5k_knowledge_selector_top100.jsonl \
  --n-context 100

python scripts/train_knowledge_selector.py \
  --input data/squad_tr_train_5k_knowledge_selector_top100.jsonl \
  --model-output checkpoints/knowledge_selector_tfidf_lr/model.pkl \
  --metrics-output checkpoints/knowledge_selector_tfidf_lr/metrics.json \
  --ks 1,5,10,25,50,100

python scripts/apply_knowledge_selector.py \
  --model checkpoints/knowledge_selector_tfidf_lr/model.pkl \
  --input data/squad_tr_dev_bm25_answerable.json \
  --output data/squad_tr_dev_bm25_selector_hybrid05_top5.json \
  --n-context 5 \
  --mode hybrid \
  --selector-weight 0.5
```

## Proposal Coverage

| Proposal Item | Status | Final Use |
| --- | --- | --- |
| SQuAD-TR + Turkish Wikipedia corpus | Completed | Core system |
| FiD/FSMODQA reader | Completed | Core system and ablation table |
| Retriever baseline | Completed with BM25 | Core system |
| DPR/ConvBERTurk retriever | Future work | Not used in final claims |
| Recall@K and MRR | Completed | Retrieval evaluation |
| EM and token F1 | Completed | Reader evaluation |
| Correctness and faithfulness metrics | Completed | Expanded QA evaluation |
| Refusal metrics | Completed as evaluation support | Mixed unanswerable smoke set |
| Adaptive retrieval | Completed | Main experiment |
| Question rewriting | Completed | Main experiment with limited/negative reader result |
| Knowledge selector | Completed as supervised selector/reranker | Main experiment with no reader gain yet |
| Cross-encoder reranker | Completed as pilot | Supporting probe |
| FSMODQA dense retriever probe | Completed as pilot | Supporting probe |
| Document rewriting | Future work | Not implemented |
| Synthetic data augmentation | Future work | Not implemented |
| RL-based selector training | Future work | Not implemented |

## Key Interpretation

BM25 remains the most reliable retriever baseline in our local experiments. Adaptive retrieval is the most useful extension because it gives the best reader F1 among tested deployable systems while reducing the average number of contexts. Question rewriting and knowledge selector variants are implemented and evaluated, but in the current setup their retrieval-side improvements do not translate into better reader F1.

The oracle top-5 result is important: it improves reader F1 from 15.17 to 19.31. This means the reader can benefit from better evidence, and retrieval quality remains the main bottleneck.

## Limitations

- Reader ablations are 200-example answerable smoke/evidence checks, not full-dev reader inference.
- The uploaded FSMODQA checkpoint did not work as a strong dense retriever in our small probe.
- The current FSMODQA reader does not naturally refuse unanswerable questions.
- DPR/ConvBERTurk retriever training, document rewriting, synthetic augmentation, and RL selector training are left as future work.
