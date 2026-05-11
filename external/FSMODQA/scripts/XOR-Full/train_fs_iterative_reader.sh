#!/bin/bash
#
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --mem=800G
#SBATCH --partition=feit-gpu-a100
#SBATCH -A punim2015
#SBATCH --gres=gpu:A100:4
#SBATCH -q feit
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

unset SLURM_MEM_PER_NODE
distributed_port=$(( (RANDOM % 5 + 1) * 10000 + RANDOM % 10000 ))

srun python train.py --distributed_port ${distributed_port} \
  --output_dir ./checkpoints/FSMODQA-100k \
  --model_name_or_path fanjiang98/FSMODQA-EN \
  --save_steps 500 \
  --task XOR-Full \
  --train_dir data/XOR-Full/ \
  --train_path fs-qa.cl+il.llm-qa.100k.jsonl \
  --corpus_file all_w100.tsv \
  --query_file fs-qa.cl+il.llm-qa.100k.jsonl \
  --eval_query_file xor_dev_full_v1_1.original.jsonl \
  --tf32 True \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --negatives_x_device \
  --grad_cache \
  --refresh_passages \
  --refresh_intervals 1000 \
  --separate_joint_encoding \
  --de_avg_pooling \
  --wikidata \
  --add_lang_token \
  --e2e_training \
  --gradient_checkpointing \
  --gc_chunk_size 8 \
  --retriever_weight 8 \
  --multi_task \
  --ddp_find_unused_parameters False \
  --train_n_passages 100 \
  --max_query_length 50 \
  --max_passage_length 200 \
  --max_query_passage_length 250 \
  --max_answer_length 50 \
  --learning_rate 5e-5 \
  --max_steps 6000 \
  --num_train_epochs 1 \
  --distillation_start_steps 0 \
  --weight_decay 0.01 \
  --dataloader_num_workers 2 \
  --print_steps 20
