#!/bin/bash
#
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=168:00:00
#SBATCH --mem=900G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:A100:4
#SBATCH -q gpgpudeeplearn
#SBATCH --constraint=dlg5
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err


srun python train.py --distributed_port 23333 \
  --output_dir checkpoints/fsmlqa-wikidata-unsupervised \
  --model_name_or_path google/mt5-large \
  --save_steps 2000 \
  --task XOR-Retrieve \
  --train_dir data/XOR-Full \
  --train_path wikidata.train.jsonl \
  --corpus_file all_w100.tsv \
  --query_file wikidata.query.jsonl \
  --eval_query_file xor_dev_retrieve_eng_span_v1_1.jsonl \
  --tf32 True \
  --per_device_train_batch_size 50 \
  --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --negatives_x_device False \
  --grad_cache \
  --separate_joint_encoding \
  --de_avg_pooling \
  --wikidata \
  --add_lang_token \
  --load_partial True \
  --gradient_checkpointing \
  --retriever_weight 8 \
  --multi_task \
  --ddp_find_unused_parameters False \
  --train_n_passages 1 \
  --max_query_length 50 \
  --max_passage_length 200 \
  --max_query_passage_length 250 \
  --max_answer_length 50 \
  --learning_rate 5e-5 \
  --max_steps 100000 \
  --num_train_epochs 1 \
  --distillation_start_steps 0 \
  --weight_decay 0.01 \
  --dataloader_num_workers 2 \
  --print_steps 20
