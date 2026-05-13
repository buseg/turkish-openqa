# Experiment Log

This file keeps the longer command log and intermediate results for the Turkish ODQA project. The main `README.md` is the final submission summary; this file is the reproducibility and experiment notebook in Markdown form.

Generated files referenced here live under `data/`, `checkpoints/`, `.hf_cache/`, and `odqa_data/`. These folders are ignored by git.

## 1. Data Preparation

The notebook `download_data.ipynb` prepares SQuAD-TR and the Turkish Wikipedia/passage corpus. After running the notebook, we convert the QA data and export passages.

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

For mixed answerable/unanswerable evaluation:

```bash
python scripts/prepare_qa_data.py \
  --input odqa_data/squad_tr \
  --split validation \
  --include-unanswerable \
  --output data/squad_tr_dev.json
```

## 2. BM25 Retrieval Baseline

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

Observed full-dev answerable BM25 result:

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

Earlier baseline run recorded during development:

```text
Examples: 2910
Recall@1:   0.0763
Recall@5:   0.1540
Recall@10:  0.1945
Recall@20:  0.2512
Recall@50:  0.3124
Recall@100: 0.3656
```

## 3. FSMODQA Reader Baseline

Build FSMODQA reader inputs:

```bash
python scripts/build_fsmodqa_reader_inputs.py \
  --retrieved data/squad_tr_dev_bm25_answerable.json \
  --output-dir data/fsmodqa_dev_bm25_top5_answerable \
  --n-context 5 \
  --prefix bm25top5

sed -n '1,200p' \
  data/fsmodqa_dev_bm25_top5_answerable/reader.jsonl \
  > data/fsmodqa_dev_bm25_top5_answerable/reader_200.jsonl
```

Run the reader:

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

Evaluate:

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

Reader results:

| Reader Input | EM | F1 | Correctness | Faithfulness | Contains Gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 top-5 | 9.00 | 15.17 | 15.32 | 72.84 | 11.50 |
| BM25 top-25 | 10.00 | 14.80 | 15.16 | 84.74 | 12.00 |
| Oracle top-5 | 12.50 | 19.31 | 20.16 | 73.83 | 17.00 |

## 4. Base FiD Smoke Training

This was an early pipeline validation step, not the final model.

```bash
cd external/fid

WORLD_SIZE=1 \
GLOBAL_RANK=0 \
TRANSFORMERS_CACHE=../../.hf_cache \
conda run -n turkish-odqa-fid python train_reader.py \
  --train_data ../../data/squad_tr_train_5k_reader_bm25.json \
  --eval_data ../../data/squad_tr_dev_reader_bm25_answerable.json \
  --model_size small \
  --per_gpu_batch_size 1 \
  --n_context 5 \
  --total_steps 100 \
  --eval_freq 25 \
  --save_freq 100 \
  --answer_maxlength 32 \
  --name fid_bm25_5k_100steps \
  --checkpoint_dir ../../checkpoints
```

Observed training log:

```text
25 / 100  | train: 3.977 | evaluation: 0.03 EM
50 / 100  | train: 3.763 | evaluation: 0.21 EM
75 / 100  | train: 3.582 | evaluation: 0.21 EM
100 / 100 | train: 3.558 | evaluation: 0.24 EM
```

## 5. Question Rewriting

Broad rule-based rewriting:

```bash
python scripts/rewrite_questions.py \
  --input data/squad_tr_dev_answerable.json \
  --output data/squad_tr_dev_answerable_rewritten_all.json \
  --mode all

python scripts/retrieve_bm25.py \
  --qa data/squad_tr_dev_answerable_rewritten_all.json \
  --passages data/passages.tsv \
  --output data/squad_tr_dev_bm25_rewritten_all.json \
  --n-docs 100

python scripts/evaluate_retrieval.py \
  --input data/squad_tr_dev_bm25_rewritten_all.json \
  --ks 1,5,10,20,25,50,100 \
  --match-mode answer \
  --output data/squad_tr_dev_bm25_rewritten_all_retrieval_metrics.json
```

Keyword-focused rewriting:

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

python scripts/evaluate_retrieval.py \
  --input data/squad_tr_dev_bm25_rewritten_keyword.json \
  --ks 1,5,10,20,25,50,100 \
  --match-mode answer \
  --output data/squad_tr_dev_bm25_rewritten_keyword_retrieval_metrics.json
```

Selective rewriting:

```bash
python scripts/select_retrieval.py \
  --original data/squad_tr_dev_bm25_answerable.json \
  --rewritten data/squad_tr_dev_bm25_rewritten_keyword.json \
  --output data/squad_tr_dev_bm25_selective_normscore_tuned.json \
  --strategy normalized-top-score \
  --ratio 1.125 \
  --common-only \
  --ks 1,5,10,20,25,50,100
```

Retrieval results:

| Setting | R@1 | R@5 | R@10 | R@20 | R@25 | R@50 | R@100 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 original | 7.80 | 15.67 | 19.90 | - | 27.49 | 31.79 | 37.29 | 11.94 |
| Broad rewrite | 7.11 | 14.54 | 18.49 | 23.47 | 25.12 | 30.38 | 36.15 | 10.95 |
| Keyword rewrite | 7.63 | 15.74 | 20.14 | 25.67 | 27.42 | 31.99 | 37.66 | 11.87 |
| Selective keyword rewrite | 7.84 | 15.67 | 20.03 | 25.77 | 27.73 | 31.89 | 37.46 | 11.98 |

Reader result:

| Reader Input | EM | F1 | Correctness | Faithfulness | Contains Gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 top-5 | 9.00 | 15.17 | 15.32 | 72.84 | 11.50 |
| Keyword rewrite top-5 | 7.00 | 10.55 | 11.33 | 78.65 | 10.00 |
| Selective keyword rewrite top-5 | 7.00 | 10.91 | 11.45 | 79.39 | 9.50 |

Interpretation: keyword rewriting gives a small retrieval gain, but it hurts reader F1 in the current setup.

## 6. LLM-Based Question Rewriting Pilot

The LLM rewriting interface supports local Ollama calls and dry runs.

Dry run:

```bash
python scripts/rewrite_questions_llm.py \
  --input data/squad_tr_dev_answerable.json \
  --output data/squad_tr_dev_answerable_rewritten_llm.json \
  --provider ollama \
  --model gemma2:2b \
  --limit 2 \
  --dry-run
```

Local pilot:

```bash
ollama pull gemma2:2b

python scripts/rewrite_questions_llm.py \
  --input data/squad_tr_dev_answerable.json \
  --output data/squad_tr_dev_answerable_rewritten_llm_5.json \
  --provider ollama \
  --model gemma2:2b \
  --limit 5 \
  --resume \
  --save-every 1
```

Example rewrites:

```text
Normandiya hangi ülkede bulunur?
-> Normandiya hangi ülkeye aittir?

Norman'lar ne zaman Normandiya'daydılar?
-> Normanlar hangi tarihte Normandiya'da bulunuyordu?
```

The 100-question strict-prompt pilot did not improve retrieval, so LLM rewriting was kept as an implemented pilot rather than a final system improvement.

## 7. Adaptive Retrieval

Build complexity labels:

```bash
python scripts/build_complexity_labels.py \
  --input data/squad_tr_train_5k_bm25.json \
  --output data/squad_tr_train_5k_complexity_labels.jsonl

python scripts/build_complexity_labels.py \
  --input data/squad_tr_dev_bm25_answerable.json \
  --output data/squad_tr_dev_bm25_answerable_complexity_labels.jsonl
```

Train retrieval-aware classifier:

```bash
python scripts/train_retrieval_aware_complexity_classifier.py \
  --retrieved data/squad_tr_train_5k_bm25.json \
  --labels data/squad_tr_train_5k_complexity_labels.jsonl \
  --output checkpoints/complexity_classifier_retrieval_aware_tfidf_lr/model.pkl \
  --metrics-output checkpoints/complexity_classifier_retrieval_aware_tfidf_lr/metrics.json
```

Apply adaptive budgets:

```bash
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

Adaptive retrieval results:

| Setting | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | Avg Contexts | Saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 top-100 | 7.80 | 15.67 | 19.90 | 27.49 | 31.79 | 37.29 | 100.00 | 0.00 |
| Oracle 10/50/100 | 7.80 | 15.67 | 19.90 | 27.49 | 31.79 | 37.29 | 76.15 | 23.85 |
| Retrieval-aware 50/50/100 | 7.80 | 15.67 | 19.90 | 27.49 | 31.79 | 34.81 | 76.86 | 23.14 |

Reader result:

| Reader Input | EM | F1 | Correctness | Faithfulness | Contains Gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 top-5 | 9.00 | 15.17 | 15.32 | 72.84 | 11.50 |
| Adaptive retrieval-aware 50/50/100 | 10.00 | 15.42 | 15.95 | 81.74 | 13.50 |

Qualitative adaptive retrieval examples:

```text
Example 1: Recovered by retrieval-aware classifier
Question: Norman kelimesinin Latince versiyonu ilk ne zaman kaydedildi?
Answer: 9. yüzyıl
BM25 answer rank: 53
Question-only policy: easy, top-25, answer removed
Retrieval-aware policy: hard, top-100, answer kept

Example 2: Borderline example recovered by safer budget
Question: Jersey ve Guernsey nerede?
Answer: Channel Adaları
BM25 answer rank: 26
Question-only policy: easy, top-25, answer removed by one rank
Retrieval-aware policy: easy, top-50, answer kept

Example 3: Remaining failure case
Question: Durum karmaşıklığı olasılıkları, hangi genel ölçüye ilişkin değişken olasılıklar sağlar?
Answer: karmaşıklık ölçüsü / karmaşıklık
BM25 answer rank: 64
Retrieval-aware policy: easy, top-50, answer removed
```

## 8. Expanded QA And Refusal Evaluation

Expanded answerable evaluation:

```bash
python scripts/evaluate_qa_expanded.py \
  --predictions checkpoints/fsmodqa_squad_tr_adaptive_retrieval_aware_50_50_100_200/predictions.json \
  --dataset data/squad_tr_dev_bm25_adaptive_retrieval_aware_50_50_100_200.json \
  --output checkpoints/fsmodqa_squad_tr_adaptive_retrieval_aware_50_50_100_200/expanded_metrics.json \
  --per-example-output checkpoints/fsmodqa_squad_tr_adaptive_retrieval_aware_50_50_100_200/expanded_per_example_metrics.json
```

Mixed answerable/unanswerable smoke set:

| Setting | Examples | Answerable | Unanswerable | Answerable F1 | Refusal Rate | Answered Unanswerable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 top-5 mixed smoke | 100 | 50 | 50 | 24.56 | 0.00 | 100.00 |

Interpretation: the current FSMODQA reader does not naturally refuse unanswerable questions.

## 9. Knowledge Selector

Build selector data:

```bash
python scripts/build_knowledge_selector_data.py \
  --input data/squad_tr_train_5k_bm25.json \
  --output data/squad_tr_train_5k_knowledge_selector_top100.jsonl \
  --n-context 100
```

Train TF-IDF selector:

```bash
python scripts/train_knowledge_selector.py \
  --input data/squad_tr_train_5k_knowledge_selector_top100.jsonl \
  --model-output checkpoints/knowledge_selector_tfidf_lr/model.pkl \
  --metrics-output checkpoints/knowledge_selector_tfidf_lr/metrics.json \
  --test-size 0.2 \
  --seed 13 \
  --ks 1,5,10,25,50,100
```

Apply tuned hybrid selector:

```bash
python scripts/apply_knowledge_selector.py \
  --model checkpoints/knowledge_selector_tfidf_lr/model.pkl \
  --input data/squad_tr_dev_bm25_answerable.json \
  --output data/squad_tr_dev_bm25_selector_hybrid05_top5.json \
  --n-context 5 \
  --mode hybrid \
  --selector-weight 0.5
```

Held-out selector split:

| Setting | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 34.40 | 54.10 | 59.50 | 66.50 | 72.70 | 78.70 | 43.31 |
| TF-IDF selector only | 27.70 | 55.70 | 65.10 | 73.50 | 77.50 | 78.70 | 40.46 |
| TF-IDF selector/BM25 hybrid | 40.70 | 61.20 | 67.90 | 75.10 | 77.30 | 78.70 | 50.10 |

Reader result:

| Reader Input | EM | F1 | Correctness | Faithfulness | Contains Gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 top-5 | 9.00 | 15.17 | 15.32 | 72.84 | 11.50 |
| TF-IDF selector hybrid top-5 | 9.00 | 14.85 | 15.24 | 72.11 | 11.50 |

## 10. Neural Knowledge Selector

Train BERTurk selector:

```bash
TRANSFORMERS_CACHE=.hf_cache \
conda run -n turkish-odqa-fid python scripts/train_neural_knowledge_selector.py \
  --input data/squad_tr_train_5k_knowledge_selector_top100.jsonl \
  --model-name dbmdz/bert-base-turkish-cased \
  --output-dir checkpoints/knowledge_selector_berturk_medium \
  --cache-dir .hf_cache \
  --epochs 1 \
  --batch-size 8 \
  --max-length 128 \
  --max-train-questions 300 \
  --max-eval-questions 100 \
  --top-negatives-per-question 6 \
  --random-negatives-per-question 6 \
  --ks 1,5,10,25,50,100
```

Apply to 200-example dev slice:

```bash
TRANSFORMERS_CACHE=.hf_cache \
conda run -n turkish-odqa-fid python scripts/apply_cross_encoder_reranker.py \
  --model checkpoints/knowledge_selector_berturk_medium \
  --input data/squad_tr_dev_bm25_answerable_200_for_ce.json \
  --output data/squad_tr_dev_bm25_berturk_selector_medium_hybrid05_top5_200.json \
  --n-context 5 \
  --batch-size 8 \
  --max-length 128 \
  --mode hybrid \
  --cross-encoder-weight 0.5
```

Medium held-out split:

| Setting | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 held-out | 47.00 | 63.00 | 69.00 | 70.00 | 77.00 | 80.00 | 53.61 |
| BERTurk selector only | 33.00 | 64.00 | 67.00 | 75.00 | 76.00 | 80.00 | 44.55 |
| Hybrid weight 0.5 | 51.00 | 65.00 | 71.00 | 74.00 | 77.00 | 80.00 | 57.30 |

200-example reader result:

| Reader Input | EM | F1 | Correctness | Faithfulness | Contains Gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERTurk selector medium hybrid top-5 | 8.50 | 13.95 | 14.32 | 75.35 | 10.00 |

## 11. Cross-Encoder Reranker Pilot

```bash
TRANSFORMERS_CACHE=.hf_cache \
conda run -n turkish-odqa-fid python scripts/train_cross_encoder_reranker.py \
  --input data/squad_tr_train_5k_knowledge_selector_top100.jsonl \
  --model-name distilbert-base-multilingual-cased \
  --output-dir checkpoints/cross_encoder_distilmbert_pilot \
  --cache-dir .hf_cache \
  --epochs 1 \
  --batch-size 8 \
  --max-length 256 \
  --max-train-questions 500 \
  --max-eval-questions 200 \
  --top-negatives-per-question 3 \
  --random-negatives-per-question 3 \
  --ks 1,5,10,25,50,100
```

Pilot split:

| Setting | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 39.50 | 58.50 | 64.50 | 70.50 | 76.50 | 80.50 | 47.89 |
| Cross-encoder only | 17.50 | 45.50 | 56.50 | 68.00 | 76.50 | 80.50 | 29.51 |

Interpretation: this pilot did not beat BM25. The reusable infrastructure remains useful.

## 12. FSMODQA Dense Retriever Probe

We tested whether `busegi/FSMODQA-SQUAD-TR` could act as a dense reranker over BM25 top-100 candidates.

```bash
python scripts/build_fsmodqa_reader_inputs.py \
  --retrieved data/squad_tr_dev_bm25_answerable_50.json \
  --output-dir data/fsmodqa_dev_bm25_top100_answerable_50 \
  --n-context 100 \
  --prefix dense50
```

Encode passages and queries with `external/FSMODQA/encode.py`, then retrieve with:

```bash
conda run -n turkish-odqa-fid python scripts/retrieve_dense_torch.py \
  --query-embeddings checkpoints/fsmodqa_dense_rerank_50_head/query_embedding.pt \
  --passage-embeddings checkpoints/fsmodqa_dense_rerank_50_head/passage_embedding.pt \
  --output checkpoints/fsmodqa_dense_rerank_50_head/dense_top100_mean_norm.jsonl \
  --depth 100 \
  --pooling mean \
  --normalize

python scripts/convert_fsmodqa_ranking.py \
  --ranking checkpoints/fsmodqa_dense_rerank_50_head/dense_top100_mean_norm.jsonl \
  --corpus data/fsmodqa_dev_bm25_top100_answerable_50/corpus.jsonl \
  --queries data/fsmodqa_dev_bm25_top100_answerable_50/queries.jsonl \
  --gold data/squad_tr_dev_bm25_answerable_50.json \
  --output data/squad_tr_dev_dense_fsmodqa_rerank_50_head_mean_norm.json \
  --n-context 100
```

Observed 50-example comparison:

| Retriever | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 candidate order | 14.00 | 28.00 | 30.00 | 38.00 | 42.00 | 50.00 | 19.98 |
| FSMODQA dense rerank probe | 0.00 | 4.00 | 8.00 | 8.00 | 16.00 | 26.00 | 2.93 |

Interpretation: the uploaded checkpoint should not replace BM25 as the retriever unless a separately trained retriever checkpoint is available.

## 13. Regenerate Final Tables

```bash
python scripts/summarize_evaluation.py
```

This writes `evaluation_summary.md`.
