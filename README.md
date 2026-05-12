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
  --query_file test.query.jsonl \
  --encode_is_qry \
  --normalize_text \
  --separate_joint_encoding \
  --max_query_length 50 \
  --per_device_eval_batch_size 512 \
  --dataloader_num_workers 4 \
  --tf32 True \
  --encoded_save_path ../../checkpoint/fsmodqa_off_the_shelf/encoding/test_query_embedding.pt
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
  --save_ranking_to ../../checkpoint/fsmodqa_off_the_shelf/validation_top100
```

Run the reader part with:

```bash
cd external/FSMODQA

python test_reader.py \
  --output_dir /cta/users/buse/repos/turkish-openqa/checkpoint/fsmodqa_off_the_shelf \
  --model_name_or_path fanjiang98/FSMODQA-100k \
  --output_path /cta/users/buse/repos/turkish-openqa/checkpoint/fsmodqa_off_the_shelf/validation_reader_predictions.json \
  --train_dir /cta/users/buse/repos/turkish-openqa/odqa_data/fsmodqa_retrieval \
  --train_path /cta/users/buse/repos/turkish-openqa/checkpoint/fsmodqa_off_the_shelf/validation_top100.jsonl \
  --corpus_file corpus.jsonl \
  --query_file validation.query.jsonl \
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
  --dataset odqa_data/squad_tr_processed_validation
```

## Finetuning

To finetune the model from base model:

  CUDA_VISIBLE_DEVICES=1 /cta/users/buse/miniconda3/envs/fsmodqa_env/bin/torchrun --nproc_per_node=1 \
  src/finetune_fsmodqa_squad_tr.py \
  --mode reader \
  --skip-rankings \
  --output-dir checkpoint/fsmodqa_squad_tr_reader \
  --work-dir checkpoint/fsmodqa_squad_tr_reader/reader_finetuning_data \
  --print-steps 100 \
  --save-steps 500 \
  --tb-metric-examples 500 \
  --tb-log-examples 10 \
  --tb-log-generation-steps 500 \
  --tf32 \
  --gradient-checkpointing \
  --train-n-passages 25
  --num-train-epochs 1.0