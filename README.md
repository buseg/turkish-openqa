# turkish-openqa

## Overview

This repository contains the steps to prepare data, encode the corpus and queries, run retrieval and reader inference, evaluate results, and fine-tune FSMODQA on the Turkish SQuAD dataset.

## Setup

1. Create the knowledge source and preprocess the Squad TR. Use the odqa environment for the kernel and run the Jupyter Notebook `download_data.ipynb`:

```bash
conda env create -f fsmodqa_environment.yml
```

2. Prepare retriever inputs:

```bash
python src/prepare_fsmodqa_retrieval_inputs.py
python src/create_squad_duplicate_pid_lookup.py
```

## Encode the corpus

Set up the fsmodqa_env
```bash
mkdir -p checkpoint/fsmodqa_off_the_shelf/encoding
cd external/FSMODQA
conda env create -f fsmodqa_environment.yml
```


Encode the corpus in 8 shards:
```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=0 python encode.py \
    --model_name_or_path fanjiang98/FSMODQA-100k \
    --output_dir ../../checkpoint/fsmodqa_off_the_shelf \
    --train_dir ../../odqa_data/fsmodqa_retrieval \
    --corpus_file corpus.jsonl \
    --query_file validation.query.jsonl \
    --separate_joint_encoding \
    --max_passage_length 200 \
    --per_device_eval_batch_size 512 \
    --encode_shard_index $i \
    --encode_num_shard 8 \
    --dataloader_num_workers 4 \
    --tf32 True \
    --encoded_save_path ../../checkpoint/fsmodqa_off_the_shelf/encoding/passage_embedding_split${i}.pt

done
```

Encode SQUAD dataset queries:

```bash
CUDA_VISIBLE_DEVICES=1 python encode.py \
  --model_name_or_path fanjiang98/FSMODQA-100k \
  --output_dir ../../checkpoint/fsmodqa_off_the_shelf \
  --train_dir ../../odqa_data/fsmodqa_retrieval \
  --corpus_file corpus.jsonl \
  --query_file train.query.jsonl \
  --encode_is_qry \
  --normalize_text \
  --separate_joint_encoding \
  --max_query_length 50 \
  --per_device_eval_batch_size 512 \
  --dataloader_num_workers 4 \
  --tf32 True \
  --encoded_save_path ../../checkpoint/fsmodqa_off_the_shelf/encoding/train_query_embedding.pt

CUDA_VISIBLE_DEVICES=1 python encode.py \
  --model_name_or_path fanjiang98/FSMODQA-100k \
  --output_dir ../../checkpoint/fsmodqa_off_the_shelf \
  --train_dir ../../odqa_data/fsmodqa_retrieval \
  --corpus_file corpus.jsonl \
  --query_file validation.query.jsonl \
  --encode_is_qry \
  --normalize_text \
  --separate_joint_encoding \
  --max_query_length 50 \
  --per_device_eval_batch_size 512 \
  --dataloader_num_workers 4 \
  --tf32 True \
  --encoded_save_path ../../checkpoint/fsmodqa_off_the_shelf/encoding/validation_query_embedding.pt

CUDA_VISIBLE_DEVICES=1 python encode.py \
  --model_name_or_path fanjiang98/FSMODQA-100k \
  --output_dir ../../checkpoint/fsmodqa_off_the_shelf \
  --train_dir ../../odqa_data/fsmodqa_retrieval \
  --corpus_file corpus.jsonl \
  --query_file validation.query.jsonl \
  --encode_is_qry \
  --normalize_text \
  --separate_joint_encoding \
  --max_query_length 50 \
  --per_device_eval_batch_size 512 \
  --dataloader_num_workers 4 \
  --tf32 True \
  --encoded_save_path ../../checkpoint/fsmodqa_off_the_shelf/encoding/validation_query_embedding.pt
```

## Evaluating the Model
Run FAISS GPU retrieval top-100:

```bash
cd external/FSMODQA

CUDA_VISIBLE_DEVICES=0 python retriever.py \
  --query_embeddings ../../checkpoint/fsmodqa_off_the_shelf/encoding/validation_query_embedding.pt \
  --passage_embeddings '../../checkpoint/fsmodqa_off_the_shelf/encoding/passage_embedding_split*.pt' \
  --depth 100 \
  --batch_size 5000 \
  --search_then_merge \
  --save_jsonl \
  --use_gpu \
  --save_ranking_to ../../checkpoint/fsmodqa_off_the_shelf/validation_top100.jsonl
```

Evaluate the retriever:

```bash
python src/evaluate_retrieval.py \
  --rankings checkpoint/final_fsmodqa_squad_tr_full_deavg/test_top100.jsonl \
  --qid-aliases odqa_data/fsmodqa_retrieval/squad_duplicate_qid_lookup.json
```

Run the reader part with:

```bash
cd external/FSMODQA

python test_reader.py \
  --output_dir /cta/users/buse/repos/turkish-openqa/checkpoint/fsmodqa_off_the_shelf \
  --model_name_or_path fanjiang98/FSMODQA-100k \
  --output_path /cta/users/buse/repos/turkish-openqa/checkpoint/fsmodqa_off_the_shelf/validation_reader_predictions.json \
  --train_dir /cta/users/buse/repos/turkish-openqa/odqa_data/fsmodqa_retrieval \
  --train_path /cta/users/buse/repos/turkish-openqa/checkpoint/fsmodqa_off_the_shelf/test_top100.jsonl \
  --corpus_file corpus.jsonl \
  --query_file test.query.jsonl \
  --per_device_eval_batch_size 1 \
  --train_n_passages 100 \
  --max_query_length 50 \
  --max_passage_length 200 \
  --max_query_passage_length 250 \
  --max_answer_length 50 \
  --separate_joint_encoding \
  --de_avg_pooling \
  --add_lang_token \
  --bf16 False \
  --tf32 True
```

Evaluate the results:

```
python src/evaluate.py \
  --predictions checkpoint/fsmodqa_off_the_shelf/validation_reader_predictions.json \
  --dataset odqa_data/squad_tr_processed_validation \
  --output checkpoint/fsmodqa_off_the_shelf/validation_metrics.json \
  --per-example-output checkpoint/fsmodqa_off_the_shelf/validation_per_example_metrics.json
```

For a full evaluation you can edit the script:

```
bash scripts/run_test_reader_topk.sh
```

## Finetuning


To finetune the retriever and reader from base model:

```bash
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
  src/finetune_fsmodqa_squad_tr.py \
  --mode full \
  --skip-rankings \
  --local-retriever-eval \
  --local-retriever-eval-metric mrr \
  --output-dir checkpoint/fsmodqa_squad_tr_full_deavg \
  --work-dir checkpoint/fsmodqa_squad_tr_full/finetuning_data \
  --model-name-or-path fanjiang98/FSMODQA-100k \
  --task finetuning_data \
  --train-n-passages 25 \
  --per-device-eval-batch-size 512 \
  --max-steps 36000 \
  --save-steps 2000 \
  --print-steps 100 \
  --tb-metric-examples 500 \
  --tb-log-examples 10 \
  --tb-log-generation-steps 500 \
  --tf32 \
  --gradient-checkpointing \
  --de-avg-pooling \
  --refresh-passages \
  --refresh-intervals 2000
```


## Question Rewrite

```
CUDA_VISIBLE_DEVICES=2 python src/llm_expand_retrieve.py \
  --input-query-file odqa_data/fsmodqa_retrieval/test.query.jsonl \
  --output-ranking checkpoint/llm_expand_smoke/test_top10_rw_full.jsonl \
  --work-dir checkpoint/llm_expand_smoke/work_one \
  --debug-output checkpoint/llm_expand_smoke/work_one/expansion_debug_rw_full.jsonl \
  --rewrite-model google/gemma-2-2b-it \
  --retriever-model checkpoint/final_fsmodqa_squad_tr_full_deavg/checkpoint-best \
  --passage-embeddings 'checkpoint/final_fsmodqa_squad_tr_full_deavg/encoding/passage_embedding_split*.pt' \
  --max-new-tokens 128 \
  --retrieve-depth 10 \
  --top-n 10 \
  --encode-batch-size 1 \
  --de-avg-pooling \
  --add-lang-token \
  --use-query-rewrite
```

## Knowledge Selector

```
python src/knowledge_selector/train_fsmodqa_neural_selector.py \
  --train-ranking checkpoint/final_fsmodqa_squad_tr_full_deavg/train_top100_with_scores.jsonl \
  --eval-ranking checkpoint/final_fsmodqa_squad_tr_full_deavg/validation_top100_with_scores.jsonl \
  --output-dir checkpoint/final_fsmodqa_squad_tr_full_deavg/neural_knowledge_selector \
  --write-eval-reranked checkpoint/final_fsmodqa_squad_tr_full_deavg/validation_top100_selector_reranked.jsonl
```