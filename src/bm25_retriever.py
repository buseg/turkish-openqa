import re
import numpy as np
from rank_bm25 import BM25Okapi
from datasets import load_from_disk
from tqdm import tqdm
import pickle
import os
import gc
import time
import json
import concurrent.futures
import argparse

_token_pattern = re.compile(r"\w+", re.UNICODE)


class TurkishBM25Retriever:
    def __init__(self, passages=None, titles=None, ids=None, load_path=None):
        self.passages = passages
        self.titles = titles
        self.ids = ids
        self.bm25 = None

        if load_path and os.path.exists(load_path):
            self.load_index(load_path)
        elif passages is not None and titles is not None:
            self._build_index()

    def _tokenize(self, text):
        return _token_pattern.findall(text.lower())

    def _build_index(self):
        print("Building BM25 index...")
        tokenized_passages = [
            self._tokenize(p)
            for p in tqdm(self.passages, desc="Tokenizing passages", unit="passage")
        ]
        self.bm25 = BM25Okapi(tokenized_passages)
        print("Index built successfully.")

    def save_index(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.bm25, f)
        print(f"Index saved to {path}")

    def load_index(self, path):
        with open(path, "rb") as f:
            self.bm25 = pickle.load(f)
        print(f"Index loaded from {path}")

    def retrieve_with_scores(self, query, k=100):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        if len(scores) > k:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
        else:
            top_indices = np.argsort(scores)[::-1]

        return [
            {
                "id": self.ids[idx],
                "title": self.titles[idx],
                "text": self.passages[idx],
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]

    def _process_example(self, example, k=100):
        retrieved_passages = self.retrieve_with_scores(example["question"], k=k)
        return {
            "id": example["id"],
            "question": example["question"],
            "answers": example["answers"]["text"],
            "real_ctx": {"title": example["title"], "text": example["context"]},
            "ctxs": [
                {
                    "title": passage["title"],
                    "text": passage["text"],
                    "id": passage["id"],
                    "score": passage["score"],
                }
                for passage in retrieved_passages
            ],
        }

    def generate_fid_dataset(
        self,
        squad_dataset_path="odqa_data/squad_tr",
        output_path="fid_train_data.jsonl",
        k=100,
        shard_id=0,
        num_shards=1,
    ):
        def chunks(dataset, batch_size):
            for start in range(0, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                yield dataset.select(range(start, end))

        print("Loading SQUAD dataset...")
        squad_data = load_from_disk(squad_dataset_path)
        train_data = squad_data["train"]

        if num_shards > 1:
            indices = list(range(shard_id, len(train_data), num_shards))
            train_data = train_data.select(indices)

        print(
            f"Shard {shard_id}/{num_shards}: processing {len(train_data)} examples",
            flush=True,
        )

        print(f"Processing {len(train_data)} questions...")
        start_time = time.perf_counter()

        batch_size = 200
        max_workers = 4

        with open(output_path, "w", encoding="utf-8") as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for batch in chunks(train_data, batch_size):
                    results = list(
                        tqdm(
                            executor.map(lambda ex: self._process_example(ex, k), batch),
                            total=len(batch),
                            desc="Retrieving examples",
                            unit="example",
                        )
                    )

                    for entry in results:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()

        elapsed = time.perf_counter() - start_time
        print(f"Retrieval completed in {elapsed:.2f} seconds.")
        print(f"FiD dataset generation complete. Saved to {output_path}.")

# Execution 
if __name__ == "__main__":
    DATA_PATH = "odqa_data/final_knowledge_source_chunked"
    INDEX_SAVE_PATH = "odqa_data/wikipedia_tr_bm25.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    print("Loading chunked knowledge source...")
    knowledge_source = load_from_disk(DATA_PATH)

    titles = knowledge_source["title"]
    passages = knowledge_source["text"]
    ids = knowledge_source["id"]

    if not os.path.exists(INDEX_SAVE_PATH):
        retriever = TurkishBM25Retriever(
            passages=passages,
            titles=titles,
            ids=ids,
        )

        retriever.save_index(INDEX_SAVE_PATH)
        print("Indexing complete.")

    else:
        print("Loading existing BM25 index...")
        retriever = TurkishBM25Retriever(
            passages=passages,
            titles=titles,
            ids=ids,
            load_path=INDEX_SAVE_PATH,
        )
        print("Index loaded successfully.")
    

    del knowledge_source
    gc.collect()

    retriever.generate_fid_dataset(
        squad_dataset_path="odqa_data/squad_tr",
        output_path=f"fid_train_data_{args.shard_id}_{args.num_shards}.jsonl",
        k=100,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )