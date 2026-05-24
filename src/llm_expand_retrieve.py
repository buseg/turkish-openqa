#!/usr/bin/env python3
"""LLM query expansion followed by FSMODQA dense retrieval.

This script:
1. Generates clarification queries for each input query.
2. Generates related queries plus keywords for each clarification query.
3. Samples 4-8 keywords and appends them to the clarification query.
4. Encodes expanded queries with external/FSMODQA/encode.py.
5. Retrieves from existing passage embeddings and aggregates duplicates by
   keeping each passage's highest score.

Example:
  python src/llm_expand_retrieve.py \
    --input-query-file odqa_data/fsmodqa_retrieval/test.query.jsonl \
    --output-ranking checkpoint/fsmodqa_llm_expanded/test_top100.jsonl \
    --work-dir checkpoint/fsmodqa_llm_expanded/work \
    --rewrite-model google/gemma-2-2b-it
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
FSMODQA_ROOT = ROOT / "external" / "FSMODQA"

QC_PROMPT = """Verilen soru belirsiz veya eksik olabilir. Bu soruyu daha net ve açık hale getirebilecek olası alternatif soruları madde madde listele.

Her madde:
- Orijinal sorudaki ana niyeti korumalıdır.
- Belirsiz yer, kategori, kapsam veya tercihleri netleştirmelidir.
- Kısa ve doğal Türkçe ile yazılmalıdır.

Örnek soru: "Bahçelievler'deki en iyi restoranlar hangileri?"

Örnek net sorular listesi:
* Bahçelievler, İstanbul'daki en iyi restoranlar hangileri?
* Bahçelievler, Ankara'daki en iyi restoranlar hangileri?
* Bahçelievler'deki en iyi uygun fiyatlı restoranlar hangileri?
* Bahçelievler'deki aileye uygun en iyi restoranlar hangileri?

Soru: {query}

Net sorular listesi:"""

QR_PROMPT = """Verilen net soruya dayanarak, bu sorunun sonuna eklenebilecek kısa anahtar kelime ifadelerini madde madde listele.

Her madde:
- Tek başına tam bir soru olmamalıdır.
- Orijinal sorunun sonuna eklendiğinde aramayı daha spesifik hale getirmelidir.
- Soru ile doğrudan ilişkili yer, kategori, özellik, tercih veya kapsam bilgisi içermelidir.
- Kısa olmalıdır; tercihen 1-4 kelime içermelidir.
- Gereksiz açıklama veya tam cümle içermemelidir.

Örnek soru: "Bahçelievler'deki en iyi restoranlar hangileri?"

Örnek anahtar kelimeler listesi:
* uygun fiyatlı
* aileye uygun
* kahvaltı
* İtalyan
* tavsiye
* yorumlar

Soru: {clarification_query}

Anahtar kelimeler listesi:"""

QUERY_REWRITE_PROMPT = """Verilen kullanıcı sorusunu, RAG sisteminde belge aramak için kullanılabilecek tek bir net arama sorgusuna dönüştür.

Amaç:
Kullanıcının orijinal niyetini koruyarak soruyu daha açık, spesifik ve belge aramaya uygun hale getirmek.

Kurallar:
- Yalnızca tek bir sorgu döndür.
- Madde listesi oluşturma.
- Açıklama yazma.
- Orijinal sorunun niyetinden uzaklaşma.
- Alakasız yeni konu, kategori veya varsayım ekleme.
- Belirsiz ifadeleri yalnızca sorudaki bağlama dayanarak netleştir.
- Gereksiz kelimeleri çıkar, önemli anahtar terimleri koru.
- Sorgu kısa, doğal ve arama motoruna yazılabilir olmalıdır.

Soru: {query}

Arama sorgusu:"""


def load_query_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rewrite_model(model_name_or_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    except Exception as exc:
        print(
            f"Fast tokenizer failed for {model_name_or_path}: {exc}\n"
            "Retrying with the slow tokenizer.",
            file=sys.stderr,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
        inputs = {key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask"}}
        prompt_len = inputs["input_ids"].shape[-1]
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    encoded = {key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    prompt_len = encoded["input_ids"].shape[-1]
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()


def parse_bullets(text: str, max_items: int) -> list[str]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^(?:[-*•]|\d+[.)])\s*(.+)$", line)
        if match:
            item = match.group(1).strip()
            if item:
                items.append(item)
    if not items:
        items = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []
    seen = set()
    for item in items:
        item = item.strip(" -\t")
        # Remove surrounding Markdown bold, e.g. **text** -> text
        item = re.sub(r"^\*\*(.*?)\*\*$", r"\1", item).strip()

        if item and item.casefold() not in seen:
            cleaned.append(item)
            seen.add(item.casefold())
    if len(cleaned) > max_items:
        cleaned = random.sample(cleaned, max_items)
    return cleaned


def extract_keywords_from_bullets(bullets: list[str]) -> list[str]:
    keywords = []

    for bullet in bullets:
        bullet = bullet.strip()

        if re.search(r"keywords?\s*:", bullet, flags=re.IGNORECASE):
            tail = re.split(
                r"keywords?\s*:",
                bullet,
                flags=re.IGNORECASE,
                maxsplit=1
            )[1]
            pieces = re.split(r"[,;|]", tail)

        elif "[" in bullet and "]" in bullet:
            tail = bullet[bullet.rfind("[") + 1 : bullet.rfind("]")]
            pieces = re.split(r"[,;|]", tail)

        else:
            # Keep the whole bullet as one keyword/phrase
            pieces = [bullet]

        for piece in pieces:
            keyword = piece.strip(" .:-–—\t\n\"'")

            # Remove Markdown bold, e.g. **machine learning** -> machine learning
            keyword = re.sub(r"\*\*(.*?)\*\*", r"\1", keyword).strip()

            if keyword:
                keywords.append(keyword)

    deduped = []
    seen = set()

    for keyword in keywords:
        key = keyword.casefold()
        if key not in seen:
            deduped.append(keyword)
            seen.add(key)

    return deduped


def build_expanded_queries(
    input_rows: list[dict[str, Any]],
    tokenizer,
    model,
    seed: int,
    max_clarifications: int,
    max_related: int,
    samples_per_clarification: int,
    min_keywords: int,
    max_keywords: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    expanded_rows = []
    debug_rows = []

    for row in tqdm(input_rows, desc="Generating query expansions"):
        qid = str(row["id"])
        query = str(row["question"])
        answers = row.get("answers") or ["placeholder"]
        lang = row.get("lang") or "tr"

        qc_text = generate_text(
            tokenizer,
            model,
            QC_PROMPT.format(query=query),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        clarifications = parse_bullets(qc_text, max_clarifications) or [query]

        q_debug = {
            "id": qid,
            "question": query,
            "clarification_generation": qc_text,
            "clarifications": [],
        }

        for clarification_index, clarification in enumerate(clarifications):
            qr_text = generate_text(
                tokenizer,
                model,
                QR_PROMPT.format(clarification_query=clarification),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            related = parse_bullets(qr_text, max_related)
            keywords = extract_keywords_from_bullets(related)
            if not keywords:
                keywords = extract_keywords_from_bullets([clarification])

            generated_questions = []
            for sample_index in range(samples_per_clarification):
                sample_size = rng.randint(min_keywords, max_keywords)
                sampled = rng.sample(keywords, k=min(sample_size, len(keywords))) if keywords else []
                expanded_question = " ".join([clarification, *sampled]).strip()
                expanded_id = f"{qid}::c{clarification_index}::r{sample_index}"
                expanded_rows.append(
                    {
                        "id": expanded_id,
                        "question": expanded_question,
                        "answers": answers,
                        "lang": lang,
                        "cl_answers": {},
                        "original_id": qid,
                    }
                )
                generated_questions.append(
                    {
                        "id": expanded_id,
                        "question": expanded_question,
                        "sampled_keywords": sampled,
                    }
                )

            q_debug["clarifications"].append(
                {
                    "query": clarification,
                    "related_generation": qr_text,
                    "related_bullets": related,
                    "keywords": keywords,
                    "expanded_queries": generated_questions,
                }
            )

        debug_rows.append(q_debug)

    return expanded_rows, debug_rows



def build_rewritten_queries(
    input_rows: list[dict[str, Any]],
    tokenizer,
    model,
    max_new_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rewritten_rows = []
    debug_rows = []

    for row in tqdm(input_rows, desc="Generating query rewrites"):
        qid = str(row["id"])
        query = str(row["question"])
        answers = row.get("answers") or ["placeholder"]
        lang = row.get("lang") or "tr"

        rewrite_text = generate_text(
            tokenizer,
            model,
            QUERY_REWRITE_PROMPT.format(query=query),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        candidates = parse_bullets(rewrite_text, 1)
        rewritten_query = candidates[0] if candidates else rewrite_text.strip()
        rewritten_query = re.sub(r"^(?:Arama sorgusu|Sorgu|Soru)\s*:\s*", "", rewritten_query).strip()
        rewritten_query = rewritten_query.strip(" \t\n\"'") or query

        rewritten_rows.append(
            {
                "id": qid,
                "question": rewritten_query,
                "answers": answers,
                "lang": lang,
                "cl_answers": {},
                "original_id": qid,
            }
        )
        debug_rows.append(
            {
                "id": qid,
                "question": query,
                "rewrite_generation": rewrite_text,
                "rewritten_query": rewritten_query,
            }
        )

    return rewritten_rows, debug_rows


def run_encode(args: argparse.Namespace, expanded_query_file: Path, encoded_query_file: Path) -> None:
    encoded_query_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(FSMODQA_ROOT / "encode.py"),
        "--model_name_or_path",
        args.retriever_model,
        "--output_dir",
        str((args.work_dir / "encode_output").resolve()),
        "--train_dir",
        str(args.train_dir.resolve()),
        "--corpus_file",
        args.corpus_file,
        "--query_file",
        str(expanded_query_file.resolve()),
        "--encode_is_qry",
        "--per_device_eval_batch_size",
        str(args.encode_batch_size),
        "--max_query_length",
        str(args.max_query_length),
        "--max_passage_length",
        str(args.max_passage_length),
        "--encoded_save_path",
        str(encoded_query_file.resolve()),
    ]
    if args.separate_joint_encoding:
        command.append("--separate_joint_encoding")
    if args.de_avg_pooling:
        command.append("--de_avg_pooling")
    if args.add_lang_token:
        command.append("--add_lang_token")
    if args.bf16:
        command.extend(["--bf16", "True"])
    if args.fp16:
        command.extend(["--fp16", "True"])
    if args.tf32:
        command.extend(["--tf32", "True"])

    print("Encoding expanded queries with FSMODQA...")
    child_env = os.environ.copy()
    child_env.setdefault("MKL_THREADING_LAYER", "GNU")
    subprocess.run(command, cwd=FSMODQA_ROOT, check=True, env=child_env)


def load_passage_embeddings(pattern: str) -> tuple[np.ndarray, list[str]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No passage embedding files matched: {pattern}")

    reps = []
    lookup = []
    for path in tqdm(paths, desc="Loading passage embeddings"):
        shard_reps, shard_lookup = torch.load(path, map_location="cpu")
        reps.append(shard_reps.float().numpy())
        lookup.extend([str(pid) for pid in shard_lookup])
    return np.concatenate(reps, axis=0), lookup


def search_inner_product(
    query_embeddings_file: Path,
    passage_embedding_pattern: str,
    depth: int,
    batch_size: int,
    use_gpu: bool,
) -> tuple[np.ndarray, list[list[str]], list[str]]:
    import faiss

    q_reps, q_lookup = torch.load(query_embeddings_file, map_location="cpu")
    q_reps = q_reps.float().numpy()
    q_lookup = [str(qid) for qid in q_lookup]

    p_reps, p_lookup = load_passage_embeddings(passage_embedding_pattern)
    index = faiss.IndexFlatIP(p_reps.shape[1])
    if use_gpu:
        ngpus = faiss.get_num_gpus()
        if ngpus <= 0:
            raise RuntimeError("--use-gpu-retrieval was set, but FAISS found no GPUs.")
        config = faiss.GpuMultipleClonerOptions()
        config.shard = True
        config.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=config)
    index.add(p_reps)

    all_scores = []
    all_indices = []
    for start in tqdm(range(0, q_reps.shape[0], batch_size), desc="Retrieving"):
        scores, indices = index.search(q_reps[start : start + batch_size], depth)
        all_scores.append(scores)
        all_indices.append(indices)

    scores = np.concatenate(all_scores, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    pids = [[p_lookup[index] for index in row] for row in indices]
    return scores, pids, q_lookup


def aggregate_rankings(
    scores: np.ndarray,
    pids: list[list[str]],
    expanded_qids: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    by_original: dict[str, dict[str, float]] = defaultdict(dict)
    for expanded_qid, row_scores, row_pids in zip(expanded_qids, scores, pids):
        original_qid = expanded_qid.split("::", 1)[0]
        best_scores = by_original[original_qid]
        for score, pid in zip(row_scores, row_pids):
            score = float(score)
            if pid not in best_scores or score > best_scores[pid]:
                best_scores[pid] = score

    output = []
    for original_qid, best_scores in by_original.items():
        ranked = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
        output.append(
            {
                "qid": original_qid,
                "pids": [pid for pid, _score in ranked],
                "scores": [score for _pid, score in ranked],
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-query-file", type=Path, required=True)
    parser.add_argument("--output-ranking", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--debug-output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for smoke tests.")
    parser.add_argument("--use-query-rewrite", action="store_true", help="Use QUERY_REWRITE_PROMPT to rewrite each input query once before retrieval.")

    parser.add_argument("--rewrite-model", default="google/gemma-2-2b-it")
    parser.add_argument("--retriever-model", default="fanjiang98/FSMODQA-100k")
    parser.add_argument("--train-dir", type=Path, default=ROOT / "odqa_data" / "fsmodqa_retrieval")
    parser.add_argument("--corpus-file", default="corpus.jsonl")
    parser.add_argument(
        "--passage-embeddings",
        default=str(ROOT / "checkpoint" / "fsmodqa_off_the_shelf" / "encoding" / "passage_embedding_split*.pt"),
    )

    parser.add_argument("--max-clarifications", type=int, default=4)
    parser.add_argument("--max-related", type=int, default=8)
    parser.add_argument("--samples-per-clarification", type=int, default=3)
    parser.add_argument("--min-keywords", type=int, default=4)
    parser.add_argument("--max-keywords", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--retrieve-depth", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--retrieval-batch-size", type=int, default=128)
    parser.add_argument("--use-gpu-retrieval", action="store_true")

    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--max-query-length", type=int, default=50)
    parser.add_argument("--max-passage-length", type=int, default=200)
    parser.add_argument("--separate-joint-encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--de-avg-pooling", action="store_true")
    parser.add_argument("--add-lang-token", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_query_file = args.input_query_file.resolve()
    args.output_ranking = args.output_ranking.resolve()
    args.work_dir = args.work_dir.resolve()
    args.train_dir = args.train_dir.resolve()
    if args.debug_output is not None:
        args.debug_output = args.debug_output.resolve()
    retriever_model_path = Path(args.retriever_model)
    if retriever_model_path.exists():
        args.retriever_model = str(retriever_model_path.resolve())
    if "*" not in args.passage_embeddings:
        args.passage_embeddings = str(Path(args.passage_embeddings).resolve())
    elif not Path(args.passage_embeddings).is_absolute():
        args.passage_embeddings = str((ROOT / args.passage_embeddings).resolve())
    args.work_dir.mkdir(parents=True, exist_ok=True)

    input_rows = load_query_jsonl(args.input_query_file, args.limit)
    tokenizer, model = load_rewrite_model(args.rewrite_model)
    if args.use_query_rewrite:
        expanded_rows, debug_rows = build_rewritten_queries(
            input_rows=input_rows,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        expanded_rows, debug_rows = build_expanded_queries(
            input_rows=input_rows,
            tokenizer=tokenizer,
            model=model,
            seed=args.seed,
            max_clarifications=args.max_clarifications,
            max_related=args.max_related,
            samples_per_clarification=args.samples_per_clarification,
            min_keywords=args.min_keywords,
            max_keywords=args.max_keywords,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    expanded_query_file = args.work_dir / "expanded.query.jsonl"
    debug_output = args.debug_output or args.work_dir / "expansion_debug.jsonl"
    encoded_query_file = args.work_dir / "expanded_query_embedding.pt"
    write_jsonl(expanded_query_file, expanded_rows)
    write_jsonl(debug_output, debug_rows)

    run_encode(args, expanded_query_file, encoded_query_file)
    scores, pids, expanded_qids = search_inner_product(
        query_embeddings_file=encoded_query_file,
        passage_embedding_pattern=args.passage_embeddings,
        depth=args.retrieve_depth,
        batch_size=args.retrieval_batch_size,
        use_gpu=args.use_gpu_retrieval,
    )
    ranking_rows = aggregate_rankings(scores, pids, expanded_qids, args.top_n)
    write_jsonl(args.output_ranking, ranking_rows)
    print(f"Wrote {len(ranking_rows)} aggregated rankings to {args.output_ranking}")
    print(f"Wrote expansion debug data to {debug_output}")


if __name__ == "__main__":
    main()
