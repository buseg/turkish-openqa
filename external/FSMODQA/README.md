## Few-Shot Multilingual Open-Domain QA from 5 Examples
The source code for our TACL 2025 Paper Few-Shot Multilingual Open-Domain QA from 5 Examples.

## Install environment
```shell
pip install -r requirements.txt
```

## Evaluation
### Models
- [fanjiang98/FSMODQA-EN](https://huggingface.co/fanjiang98/FSMODQA-EN): model pre-trained on MLWIKIQA with fine-tuning on English NQ data.
- [fanjiang98/FSMODQA-100k](https://huggingface.co/fanjiang98/FSMODQA-100k): ```FSMODQA-EN``` fine-tuned on FSMLQA with 100k samples.
- [fanjiang98/FSMODQA-600k](https://huggingface.co/fanjiang98/FSMODQA-600k): ```FSMODQA-EN``` fine-tuned on FSMLQA with 600k samples.
### XOR-TYDI-QA
#### Download Dataset
```shell
mkdir -p data/XOR-Retrieve
cd data/XOR-Retrieve
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_retrieve_eng_span.jsonl
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_retrieve_eng_span_v1_1.jsonl
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/models/enwiki_20190201_w100.tsv -O psgs_w100.tsv
cd ../../

mkdir -p data/XOR-Full
cd data/XOR-Full
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_full.jsonl
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv
wget https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv
cd ../../
```

#### XOR-Retrieve
#### Generate Embeddings
Encode Query
```shell
bash scripts/XOR-Retrieve/encode_query.sh
```
Encode Corpus
```shell
bash scripts/XOR-Retrieve/encode_corpus.sh
```
Note that ```MODEL_PATH``` should be ```fanjiang98/FSMODQA-100k```.
#### Retrieve
```shell
bash scripts/XOR-Retrieve/retrieve_hn.sh
```
Note that ```MODEL_PATH``` should be ```fanjiang98/FSMODQA-100k```
We use the official scripts provided by XOR-TYDI-QA for evaluation:
```shell
python3 evals/eval_xor_retrieve.py \
    --data_file <path_to_input_data> \
    --pred_file <path_to_predictions>
```

#### XOR-Full
#### Retrieve
It is the same as in XOR-Retrieve. Please find corresponding scripts under ```scripts/XOR-Full``` and replace ```MODEL_PATH``` with ```fanjiang98/FSMODQA-600k```.

#### Answer Generation
```shell
bash scripts/XOR-Full/eval_reader.sh
```
```MODEL_PATH``` should be ```fanjiang98/FSMODQA-600k```. We use the official scripts provided by XOR-TYDI-QA for evaluation:
```shell
python3 evals/eval_xor_full.py \
    --data_file <path_to_input_data> \
    --pred_file <path_to_predictions>
```

### Training
Please download the training data from [OneDrive](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/jifj_student_unimelb_edu_au/EgGeygVqFI1EjBhwVBR8n2oBAErRDDzVH7rjU6OPuPAgnQ?e=DSN57e) and put them on corresponding directories under `data`.

1. Pre-training (Optionally):
```shell
bash scripts/train_wikidata_reader.sh
```
Using our released pre-trained model is recommended.

2. Fine-tuning on FSMLQA synthetic training data (i.e., our released ```FSMODQA-100k``` model):
```shell
bash scripts/XOR-Full/train_fs_iterative_reader.sh
```
Switching to the dataset ```fs-qa.cl+il.llm-qa.600k.jsonl``` results in improved performance on XOR-Full for end-to-end ODQA.

We use slurm for training on 16 80G A100.

### Acknowledgement
Some of the code was adapted from https://github.com/jzbjyb/ReAtt.