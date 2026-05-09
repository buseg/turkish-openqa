# turkish-openqa

Turkish open-domain question answering pipeline with:

- SQuAD-TR question-answer data
- Turkish Wikipedia passage corpus
- BM25 retrieval baseline
- FiD/DPR-compatible data formats

## Current Data

The data preparation notebook writes Hugging Face datasets under `odqa_data/`.

Important outputs:

- `odqa_data/squad_tr`: SQuAD-TR train/validation QA data
- `odqa_data/final_knowledge_source_chunked`: final passage corpus for retrieval

## Environment

On this machine, the working Python is the Anaconda interpreter:

```bash
/opt/anaconda3/bin/python
```

The system `python3` may not have `datasets` installed.

## Prepare FiD/DPR Files

Small smoke-test files:

```bash
/opt/anaconda3/bin/python scripts/export_passages.py \
  --limit 1000 \
  --output data/passages_smoke.tsv

/opt/anaconda3/bin/python scripts/prepare_qa_data.py \
  --limit 20 \
  --train-output data/train_smoke.json \
  --dev-output data/dev_smoke.json
```

Full files:

```bash
/opt/anaconda3/bin/python scripts/export_passages.py
/opt/anaconda3/bin/python scripts/prepare_qa_data.py
```

This creates:

- `data/passages.tsv`
- `data/squad_tr_train.json`
- `data/squad_tr_dev.json`

By default, `prepare_qa_data.py` skips examples with empty answers. The
answerable-only full files can be written explicitly as:

```bash
/opt/anaconda3/bin/python scripts/prepare_qa_data.py \
  --train-output data/squad_tr_train_answerable.json \
  --dev-output data/squad_tr_dev_answerable.json
```

## BM25 Baseline

Smoke retrieval:

```bash
/opt/anaconda3/bin/python scripts/retrieve_bm25.py \
  --qa data/dev_smoke.json \
  --passages data/passages_smoke.tsv \
  --output data/dev_smoke_bm25.json \
  --n-docs 20
```

Evaluate retrieval:

```bash
/opt/anaconda3/bin/python scripts/evaluate_retrieval.py \
  --input data/dev_smoke_bm25.json \
  --ks 1,5,10,20
```

Full dev retrieval:

```bash
/opt/anaconda3/bin/python scripts/retrieve_bm25.py \
  --qa data/squad_tr_dev_answerable.json \
  --passages data/passages.tsv \
  --output data/squad_tr_dev_bm25.json \
  --n-docs 100

/opt/anaconda3/bin/python scripts/evaluate_retrieval.py \
  --input data/squad_tr_dev_bm25.json \
  --ks 1,5,10,20,50,100
```

Current BM25 baseline over the full passage corpus:

```text
Examples: 2910
Recall@1:   0.0763
Recall@5:   0.1540
Recall@10:  0.1945
Recall@20:  0.2512
Recall@50:  0.3124
Recall@100: 0.3656
```

Sanity check using only SQuAD-TR validation gold contexts:

```text
Examples: 2910
Recall@1:   0.5863
Recall@5:   0.7323
Recall@10:  0.7674
Recall@20:  0.7969
Recall@50:  0.8165
Recall@100: 0.8247
```

Inspect a retrieved example:

```bash
/opt/anaconda3/bin/python scripts/show_example.py \
  --input data/squad_tr_dev_bm25.json \
  --random \
  --top-k 5
```

## Reader Data

After retrieval, build FiD reader input:

```bash
/opt/anaconda3/bin/python scripts/build_reader_data.py \
  --qa data/squad_tr_dev_answerable.json \
  --retrieved data/squad_tr_dev_bm25.json \
  --output data/squad_tr_dev_reader_bm25_answerable.json \
  --n-context 100
```

## FiD Reader Smoke Baseline

The legacy FiD code under `external/fid` now runs on CPU with:

- `torch 2.5.1`
- `transformers 4.32.1`
- `tokenizers 0.13.3`

Use the conda environment:

```bash
conda activate turkish-odqa-fid
```

Train a small 5k-reader baseline:

```bash
cd external/fid

WORLD_SIZE=1 \
GLOBAL_RANK=0 \
TRANSFORMERS_CACHE=/Users/aslieren/CMPE58T/turkish-openqa/.hf_cache \
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

Training log:

```text
25 / 100  | train: 3.977 | evaluation: 0.03 EM
50 / 100  | train: 3.763 | evaluation: 0.21 EM
75 / 100  | train: 3.582 | evaluation: 0.21 EM
100 / 100 | train: 3.948 | evaluation: 0.45 EM
```

Evaluate the best checkpoint:

```bash
cd external/fid

WORLD_SIZE=1 \
GLOBAL_RANK=0 \
TRANSFORMERS_CACHE=/Users/aslieren/CMPE58T/turkish-openqa/.hf_cache \
conda run -n turkish-odqa-fid python test_reader.py \
  --model_path ../../checkpoints/fid_bm25_5k_100steps/checkpoint/best_dev \
  --eval_data ../../data/squad_tr_dev_reader_bm25_answerable.json \
  --per_gpu_batch_size 1 \
  --n_context 5 \
  --name fid_bm25_5k_100steps_eval \
  --checkpoint_dir ../../checkpoints \
  --write_results
```

Current reader smoke baseline on the answerable dev set:

```text
Examples: 2910
EM: 0.45
F1: 0.78
```

## Question Rewriting Experiment

Rule-based rewriting script:

```bash
/opt/anaconda3/bin/python scripts/rewrite_questions.py \
  --input data/squad_tr_dev_answerable.json \
  --output data/squad_tr_dev_answerable_rewritten_all.json \
  --mode all
```

This rewrites or expands answer-type cues for 1574 / 2910 answerable dev
questions. Examples:

```text
Normandiya hangi ülkede bulunur?
-> Normandiya hangi ülkede bulunur yer ülke şehir bölge?

Norman'lar ne zaman Normandiya'daydılar?
-> Norman'lar ne zaman Normandiya'daydılar tarih yıl zaman?
```

Run retrieval:

```bash
/opt/anaconda3/bin/python scripts/retrieve_bm25.py \
  --qa data/squad_tr_dev_answerable_rewritten_all.json \
  --passages data/passages.tsv \
  --output data/squad_tr_dev_bm25_rewritten_all.json \
  --n-docs 100
```

Evaluate:

```bash
/opt/anaconda3/bin/python scripts/evaluate_retrieval.py \
  --input data/squad_tr_dev_bm25_rewritten_all.json \
  --ks 1,5,10,20,50,100
```

Result compared with original BM25:

```text
Metric      Original    Rewritten    Delta
Recall@1    0.0763      0.0694      -0.0069
Recall@5    0.1540      0.1430      -0.0110
Recall@10   0.1945      0.1804      -0.0141
Recall@20   0.2512      0.2313      -0.0199
Recall@50   0.3124      0.2983      -0.0141
Recall@100  0.3656      0.3557      -0.0100
```

On rewritten questions only:

```text
n = 1574
Recall@100 original = 0.3977
Recall@100 rewritten = 0.3793
Rank movement: 99 improved, 378 worsened, 1097 unchanged
```

Interpretation: naive answer-type query expansion hurts BM25 slightly. This is a
useful negative baseline and motivates a more selective rewriting strategy.

Keyword-focused rewriting:

```bash
/opt/anaconda3/bin/python scripts/rewrite_questions.py \
  --input data/squad_tr_dev_answerable.json \
  --output data/squad_tr_dev_answerable_rewritten_keyword.json \
  --mode keyword

/opt/anaconda3/bin/python scripts/retrieve_bm25.py \
  --qa data/squad_tr_dev_answerable_rewritten_keyword.json \
  --passages data/passages.tsv \
  --output data/squad_tr_dev_bm25_rewritten_keyword.json \
  --n-docs 100

/opt/anaconda3/bin/python scripts/evaluate_retrieval.py \
  --input data/squad_tr_dev_bm25_rewritten_keyword.json \
  --ks 1,5,10,20,50,100
```

This mode rewrites 2891 / 2910 answerable dev questions by removing common
Turkish question/stop words and adding a short answer-type hint when available.

```text
Normandiya hangi ülkede bulunur?
-> Normandiya ülkede bulunur yer

Norman'lar ne zaman Normandiya'daydılar?
-> Norman'lar zaman Normandiya'daydılar tarih
```

Result compared with original BM25:

```text
Metric      Original    Keyword      Delta
Recall@1    0.0763      0.0749      -0.0014
Recall@5    0.1540      0.1550      +0.0010
Recall@10   0.1945      0.1962      +0.0017
Recall@20   0.2512      0.2522      +0.0010
Recall@50   0.3124      0.3141      +0.0017
Recall@100  0.3656      0.3694      +0.0038
```

On rewritten questions only:

```text
n = 2891
Recall@100 original = 0.3673
Recall@100 keyword = 0.3712
Rank movement: 213 improved, 204 worsened, 2474 unchanged
```

Interpretation: broad keyword rewriting changes almost all questions and gives a
very small retrieval gain at larger K values, while slightly hurting Recall@1.
The next useful step is selective rewriting: keep the rewritten query only when
it improves retrieval confidence, or train/evaluate a neural reranker on top of
BM25 candidates.

LLM-based question rewriting:

```bash
/opt/anaconda3/bin/python scripts/rewrite_questions_llm.py \
  --input data/squad_tr_dev_answerable.json \
  --output data/squad_tr_dev_answerable_rewritten_llm_100.json \
  --provider ollama \
  --model gemma2:2b \
  --limit 100 \
  --resume
```

The LLM prompt asks the model to rewrite the Turkish question for lexical
retrieval without answering it, preserving named entities, dates, and numbers.
The script supports:

- `--provider ollama` for a local Ollama server at `http://localhost:11434`
- `--provider chat-completions` for local OpenAI-compatible servers such as LM Studio
- `--resume` to continue a partially completed rewrite run
- `--dry-run` to inspect prompts without calling a model

Dry-run example:

```bash
/opt/anaconda3/bin/python scripts/rewrite_questions_llm.py \
  --input data/squad_tr_dev_answerable.json \
  --output data/squad_tr_dev_answerable_rewritten_llm.json \
  --provider ollama \
  --model gemma2:2b \
  --limit 2 \
  --dry-run
```

After generating LLM rewrites, use the same retrieval/evaluation pipeline:

```bash
/opt/anaconda3/bin/python scripts/retrieve_bm25.py \
  --qa data/squad_tr_dev_answerable_rewritten_llm_100.json \
  --passages data/passages.tsv \
  --output data/squad_tr_dev_bm25_rewritten_llm_100.json \
  --n-docs 100

/opt/anaconda3/bin/python scripts/evaluate_retrieval.py \
  --input data/squad_tr_dev_bm25_rewritten_llm_100.json \
  --ks 1,5,10,20,50,100
```

Start with `--limit 100` before running all 2910 dev questions, because local
LLM rewriting can be slow and the rewrite quality depends strongly on the model.

Local Gemma pilot:

```bash
brew install ollama
brew services start ollama
ollama pull gemma2:2b

/opt/anaconda3/bin/python scripts/rewrite_questions_llm.py \
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

İskandinav lideri kimdi?
-> İskandinavya liderleri kimdir
```

Pilot retrieval result:

```text
Examples: 5
Recall@1:   0.0000
Recall@5:   0.4000
Recall@10:  0.4000
Recall@20:  0.4000
Recall@50:  0.4000
Recall@100: 0.4000
```

Interpretation: the local LLM pipeline works end-to-end, but some rewrites may
shift meaning. For example, singular "leader" became plural "leaders" in one
pilot example. Larger LLM rewrite experiments should therefore use a stricter
prompt and inspect a small sample before running the full dev set.

Strict-prompt 100-question pilot:

```bash
/opt/anaconda3/bin/python scripts/rewrite_questions_llm.py \
  --input data/squad_tr_dev_answerable.json \
  --output data/squad_tr_dev_answerable_rewritten_llm_100_strict.json \
  --provider ollama \
  --model gemma2:2b \
  --limit 100 \
  --resume \
  --save-every 10

/opt/anaconda3/bin/python scripts/retrieve_bm25.py \
  --qa data/squad_tr_dev_answerable_rewritten_llm_100_strict.json \
  --passages data/passages.tsv \
  --output data/squad_tr_dev_bm25_rewritten_llm_100_strict.json \
  --n-docs 100
```

The stricter prompt rewrote 95 / 100 questions. Compared on the same 100
examples:

```text
Metric      Original    LLM rewrite   Delta
Recall@1    0.1200      0.1000       -0.0200
Recall@5    0.3000      0.2500       -0.0500
Recall@10   0.3300      0.2900       -0.0400
Recall@20   0.4000      0.3500       -0.0500
Recall@50   0.4800      0.4500       -0.0300
Recall@100  0.5300      0.5000       -0.0300

Rank movement: 17 improved, 25 worsened, 58 unchanged
```

Selective LLM rewrite on the same 100 examples:

```bash
/opt/anaconda3/bin/python scripts/select_retrieval.py \
  --original data/squad_tr_dev_bm25.json \
  --rewritten data/squad_tr_dev_bm25_rewritten_llm_100_strict.json \
  --output data/squad_tr_dev_bm25_selective_llm_100_normscore.json \
  --strategy normalized-top-score \
  --ratio 1.0 \
  --common-only \
  --ks 1,5,10,20,50,100
```

```text
Selected rewritten retrieval: 42 / 100
Recall@100: 0.4900
```

Oracle upper bound:

```text
Selected rewritten retrieval: 17 / 100
Recall@100: 0.5700
```

Interpretation: Gemma-2B LLM rewriting is a real implemented experiment, but in
this pilot it hurts BM25 retrieval on average. The oracle result shows that some
rewrites are useful, but a stronger selector or stricter rewrite-quality filter
is needed before using LLM rewrites broadly.

## Selective Rewriting Experiment

Select between original-query BM25 and keyword-rewritten BM25 per question:

```bash
/opt/anaconda3/bin/python scripts/select_retrieval.py \
  --original data/squad_tr_dev_bm25.json \
  --rewritten data/squad_tr_dev_bm25_rewritten_keyword.json \
  --output data/squad_tr_dev_bm25_selective_normscore.json \
  --strategy normalized-top-score \
  --ratio 1.0 \
  --common-only \
  --ks 1,5,10,20,50,100
```

This uses a label-free confidence rule: compare the top BM25 score normalized by
query length, and choose the rewritten retrieval only when its normalized score
is higher.

```text
Selected rewritten retrieval: 2056 / 2910

Metric      Original    Keyword-all  Selective
Recall@1    0.0763      0.0749       0.0756
Recall@5    0.1540      0.1550       0.1550
Recall@10   0.1945      0.1962       0.1969
Recall@20   0.2512      0.2522       0.2546
Recall@50   0.3124      0.3141       0.3124
Recall@100  0.3656      0.3694       0.3677
```

Oracle upper bound, using answer labels only for analysis:

```bash
/opt/anaconda3/bin/python scripts/select_retrieval.py \
  --original data/squad_tr_dev_bm25.json \
  --rewritten data/squad_tr_dev_bm25_rewritten_keyword.json \
  --output data/squad_tr_dev_bm25_selective_oracle.json \
  --strategy oracle \
  --common-only \
  --ks 1,5,10,20,50,100
```

```text
Selected rewritten retrieval: 213 / 2910
Recall@100: 0.3759
```

Interpretation: the simple confidence rule improves mid-rank recall, especially
Recall@20, but does not beat keyword-all at Recall@100. The oracle result shows
there is still useful headroom if the system learns when rewriting actually
helps.

Build a labeled selector dataset for later classifier/selector experiments:

```bash
/opt/anaconda3/bin/python scripts/build_selector_data.py \
  --original data/squad_tr_dev_bm25.json \
  --rewritten data/squad_tr_dev_bm25_rewritten_keyword.json \
  --output data/rewrite_selector_dev.jsonl
```

Current selector labels:

```text
Examples: 2910
rewritten_better: 213
original_better: 204
same: 2493
```

The labels are made by comparing the first answer-containing passage rank in
the original and rewritten retrieval outputs. These labels use answer strings,
so they are for analysis/training only, not for test-time selection.

Reader-ready dev files for downstream FiD evaluation:

```bash
/opt/anaconda3/bin/python scripts/build_reader_data.py \
  --qa data/squad_tr_dev_answerable.json \
  --retrieved data/squad_tr_dev_bm25_selective_normscore.json \
  --output data/squad_tr_dev_reader_selective_normscore.json \
  --n-context 100

/opt/anaconda3/bin/python scripts/build_reader_data.py \
  --qa data/squad_tr_dev_answerable.json \
  --retrieved data/squad_tr_dev_bm25_rewritten_keyword.json \
  --output data/squad_tr_dev_reader_keyword_bm25.json \
  --n-context 100
```

Tune simple threshold selectors on the labeled selector data:

```bash
/opt/anaconda3/bin/python scripts/tune_selector.py \
  --input data/rewrite_selector_dev.jsonl \
  --output data/rewrite_selector_tuning.json \
  --target-k 20 \
  --train-ratio 0.7 \
  --seed 13
```

Held-out eval split results for Recall@20 tuning:

```text
Train/Eval: 2036 / 874

Rule                 Selected rewritten    R@20    R@100
all_original         0 / 874               0.2529  0.3661
all_rewritten        874 / 874             0.2529  0.3707
oracle_upper_bound   63 / 874              0.2609  0.3787
best threshold       498 / 874             0.2517  0.3696
```

Best learned threshold:

```text
norm_top_score_ratio >= 1.125
```

Interpretation: threshold tuning does not generalize better than the simple
baselines on this split. This is a useful negative result: current BM25 score
features are not enough to reliably predict when rewriting helps. The oracle
upper bound still shows possible headroom for a stronger learned selector or
reranker.

Apply the tuned Recall@20 threshold on the full answerable dev set:

```bash
/opt/anaconda3/bin/python scripts/select_retrieval.py \
  --original data/squad_tr_dev_bm25.json \
  --rewritten data/squad_tr_dev_bm25_rewritten_keyword.json \
  --output data/squad_tr_dev_bm25_selective_normscore_tuned.json \
  --strategy normalized-top-score \
  --ratio 1.125 \
  --common-only \
  --ks 1,5,10,20,50,100
```

```text
Selected rewritten retrieval: 1553 / 2910
Recall@1:   0.0770
Recall@5:   0.1543
Recall@10:  0.1955
Recall@20:  0.2529
Recall@50:  0.3131
Recall@100: 0.3677
```

Reader-ready file:

```bash
/opt/anaconda3/bin/python scripts/build_reader_data.py \
  --qa data/squad_tr_dev_answerable.json \
  --retrieved data/squad_tr_dev_bm25_selective_normscore_tuned.json \
  --output data/squad_tr_dev_reader_selective_normscore_tuned.json \
  --n-context 100
```

## Adaptive Retrieval Complexity Labels

Build Adaptive-RAG-style labels from the rank of the first answer-containing
retrieved passage:

```bash
/opt/anaconda3/bin/python scripts/build_complexity_labels.py \
  --retrieved data/squad_tr_dev_bm25.json \
  --output data/squad_tr_dev_complexity_labels.jsonl \
  --easy-k 10 \
  --medium-k 50
```

Label definition:

```text
easy:   first answer-containing passage rank <= 10
medium: 10 < rank <= 50
hard:   rank > 50, or answer not found in top-100
```

Current dev distribution:

```text
Examples: 2910
easy: 566
medium: 343
hard: 2001
```

Simulate oracle adaptive retrieval:

```bash
/opt/anaconda3/bin/python scripts/simulate_adaptive_retrieval.py \
  --labels data/squad_tr_dev_complexity_labels.jsonl \
  --ks 10,50,100
```

Result:

```text
Fixed retrieval baselines:
top-10:  recall=0.1945 avg_contexts=10.0  total_contexts=29100
top-50:  recall=0.3124 avg_contexts=50.0  total_contexts=145500
top-100: recall=0.3656 avg_contexts=100.0 total_contexts=291000

Oracle adaptive retrieval:
recall=0.3656 avg_contexts=76.6 total_contexts=222910
context_saving_vs_top100=23.40%
```

Interpretation: if a question complexity classifier could perfectly predict
these labels, the system would preserve top-100 recall while sending about 23%
fewer passages downstream. This is an oracle upper bound; the next step is to
train or approximate a classifier that predicts `easy`, `medium`, and `hard`
without using answer labels at test time.

## Knowledge Selector Baseline

Build passage-level labels for a supervised selector:

```bash
/opt/anaconda3/bin/python scripts/build_knowledge_selector_data.py \
  --retrieved data/squad_tr_dev_bm25.json \
  --output data/squad_tr_dev_knowledge_selector_top100.jsonl \
  --n-context 100
```

Current passage-label distribution:

```text
Examples: 2910
Positive passages: 8310
Negative passages: 282299
```

Compare top-5 passage selection strategies:

```bash
/opt/anaconda3/bin/python scripts/select_passages.py \
  --input data/squad_tr_dev_bm25.json \
  --output data/squad_tr_dev_bm25_top5.json \
  --n-context 5 \
  --strategy bm25

/opt/anaconda3/bin/python scripts/select_passages.py \
  --input data/squad_tr_dev_bm25.json \
  --output data/squad_tr_dev_bm25_oracle_top5.json \
  --n-context 5 \
  --strategy oracle-answer-first
```

Results:

```text
BM25 top-5:
Selected Recall@5: 0.1540 (448 / 2910)

Oracle answer-first top-5:
Selected Recall@5: 0.3656 (1064 / 2910)
```

Interpretation: plain BM25 top-5 loses a lot of the evidence available in the
top-100 list. A perfect knowledge selector could reduce the reader input from
100 passages to 5 while preserving the full top-100 retrieval recall. This is an
upper bound and motivates training a real passage selector.

Reader-ready files:

```bash
/opt/anaconda3/bin/python scripts/build_reader_data.py \
  --qa data/squad_tr_dev_answerable.json \
  --retrieved data/squad_tr_dev_bm25_top5.json \
  --output data/squad_tr_dev_reader_bm25_top5.json \
  --n-context 5

/opt/anaconda3/bin/python scripts/build_reader_data.py \
  --qa data/squad_tr_dev_answerable.json \
  --retrieved data/squad_tr_dev_bm25_oracle_top5.json \
  --output data/squad_tr_dev_reader_bm25_oracle_top5.json \
  --n-context 5
```

## Scripts

```
scripts/
├── export_passages.py              # Export passages from knowledge source
├── prepare_qa_data.py              # Prepare SQuAD-TR QA data
├── retrieve_bm25.py                # BM25 retrieval
├── evaluate_retrieval.py           # Evaluate retrieval metrics
├── show_example.py                 # Inspect retrieved examples
├── build_reader_data.py            # Build FiD reader input
├── rewrite_questions.py            # Rule-based question rewriting
├── rewrite_questions_llm.py        # LLM-based question rewriting
├── select_retrieval.py             # Selective retrieval strategy
├── select_passages.py              # Passage selection strategies
├── build_selector_data.py          # Build labeled selector dataset
├── tune_selector.py                # Tune selector thresholds
├── build_complexity_labels.py      # Build adaptive retrieval complexity labels
├── simulate_adaptive_retrieval.py  # Simulate oracle adaptive retrieval
├── build_knowledge_selector_data.py # Build passage-level selector labels
├── train_complexity_classifier.py  # Train complexity classifier
├── tune_adaptive_policy.py         # Tune adaptive policy
└── export_squad_contexts.py        # Export SQuAD contexts
```
