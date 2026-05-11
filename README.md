# turkish-openqa

First create the knowledge source and preprocess the Squad TR dataset with download_data.ipynb

Then prepare retriever inputs with python src/prepare_fsmodqa_retrieval_inputs.py

mkdir -p checkpoint/fsmodqa_off_the_shelf/encoding
cd external/FSMODQA
conda activate fsmodqa_env

Encode the corpus in 8 shards:
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


Encode SQUAD dataset queries:

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


Then run FAISS GPU retrieval top-100:

CUDA_VISIBLE_DEVICES=0 python retriever.py \
  --query_embeddings ../../checkpoint/fsmodqa_off_the_shelf/encoding/validation_query_embedding.pt \
  --passage_embeddings '../../checkpoint/fsmodqa_off_the_shelf/encoding/passage_embedding_split*.pt' \
  --depth 100 \
  --batch_size 5000 \
  --search_then_merge \
  --save_jsonl \
  --use_gpu \
  --save_ranking_to ../../checkpoint/fsmodqa_off_the_shelf/validation_top100

Now reader will run:

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

Evaluate the results:

python src/evaluate.py \
  --predictions checkpoint/fsmodqa_off_the_shelf/validation_reader_predictions.json \
  --dataset odqa_data/squad_tr_processed_validation