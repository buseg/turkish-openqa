#!/usr/bin/env python3
"""Rewrite Turkish QA questions with a local or OpenAI-compatible LLM."""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="QA JSON file.")
    parser.add_argument("--output", required=True, help="Rewritten QA JSON file.")
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "chat-completions"],
        help="LLM backend. chat-completions works with local OpenAI-compatible servers.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults to OLLAMA_MODEL or gemma2:2b for Ollama.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Provider URL. Defaults to http://localhost:11434/api/generate for "
            "Ollama or http://localhost:1234/v1/chat/completions for chat-completions."
        ),
    )
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--resume", action="store_true", help="Reuse existing output examples by id.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling the LLM.")
    return parser.parse_args()


def clean_question(question):
    return " ".join((question or "").strip().split())


def default_model(provider):
    if provider == "ollama":
        return os.environ.get("OLLAMA_MODEL", "gemma2:2b")
    return os.environ.get("LLM_MODEL", "local-model")


def default_base_url(provider):
    if provider == "ollama":
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    return os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1/chat/completions")


def build_prompt(question):
    return f"""You rewrite Turkish open-domain QA questions for lexical passage retrieval.

Rules:
- Do not answer the question.
- Preserve named entities, dates, numbers, and the original meaning.
- Preserve singular/plural meaning and the expected answer type.
- Do not replace a person/group/adjective with a related place or concept unless the original question says so.
- Make implicit retrieval terms explicit only when useful.
- If a safe rewrite is not obvious, return the original question unchanged.
- Keep the rewrite in Turkish.
- Keep it short, preferably one query-like sentence.
- Return only valid JSON with this schema: {{"rewrite": "..."}}

Examples:
Question: vBNS ne anlama geliyor
JSON: {{"rewrite": "vBNS açılımı nedir"}}

Question: Norman'lar ne zaman Normandiya'daydılar?
JSON: {{"rewrite": "Normanlar Normandiya'da hangi tarihteydi"}}

Question: Normandiya hangi ülkede bulunur?
JSON: {{"rewrite": "Normandiya hangi ülkede yer alır"}}

Question: İskandinav lideri kimdi?
JSON: {{"rewrite": "İskandinav lideri kimdir"}}

Question: {question}
JSON:"""


def request_json(url, payload, timeout, headers=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_ollama(prompt, model, base_url, temperature, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = request_json(base_url, payload, timeout)
    return response.get("response", "")


def call_chat_completions(prompt, model, base_url, temperature, timeout, api_key_env):
    headers = {}
    api_key = os.environ.get(api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "You rewrite Turkish QA questions for retrieval. Return only JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    response = request_json(base_url, payload, timeout, headers=headers)
    return response["choices"][0]["message"]["content"]


def strip_code_fence(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_rewrite(raw_text):
    text = strip_code_fence(raw_text)
    try:
        parsed = json.loads(text)
        return clean_question(parsed.get("rewrite", ""))
    except json.JSONDecodeError:
        pass

    match = re.search(r'"rewrite"\s*:\s*"([^"]+)"', text)
    if match:
        return clean_question(match.group(1))

    first_line = text.splitlines()[0] if text.splitlines() else text
    first_line = re.sub(r"^(rewrite|json)\s*[:=-]\s*", "", first_line, flags=re.IGNORECASE)
    return clean_question(first_line.strip("\"' "))


def validate_rewrite(original, rewritten, max_words):
    rewritten = clean_question(rewritten)
    if not rewritten:
        return original
    if len(rewritten.split()) > max_words:
        return original
    if len(rewritten) > max(240, len(original) * 3):
        return original
    return rewritten


def call_llm(args, prompt, model, base_url):
    if args.provider == "ollama":
        return call_ollama(prompt, model, base_url, args.temperature, args.timeout)
    return call_chat_completions(
        prompt,
        model,
        base_url,
        args.temperature,
        args.timeout,
        args.api_key_env,
    )


def load_existing(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fin:
        return {str(example["id"]): example for example in json.load(fin)}


def save_output(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(examples, fout, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    model = args.model or default_model(args.provider)
    base_url = args.base_url or default_base_url(args.provider)

    with Path(args.input).open(encoding="utf-8") as fin:
        data = json.load(fin)

    selected = data[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    if args.dry_run:
        print(f"Provider: {args.provider}")
        print(f"Model: {model}")
        print(f"Base URL: {base_url}")
        for example in selected[: min(3, len(selected))]:
            question = clean_question(example["question"])
            print("\n--- prompt ---")
            print(build_prompt(question))
        return

    output_path = Path(args.output)
    existing = load_existing(output_path) if args.resume else {}
    output = []
    changed = 0
    reused = 0

    for idx, example in enumerate(selected, start=1):
        example_id = str(example["id"])
        if example_id in existing:
            new_example = existing[example_id]
            reused += 1
        else:
            original_question = clean_question(example["question"])
            prompt = build_prompt(original_question)
            try:
                raw = call_llm(args, prompt, model, base_url)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"LLM request failed for id={example_id}: {exc}\n"
                    f"Provider={args.provider} model={model} url={base_url}"
                ) from exc

            rewritten_question = validate_rewrite(
                original_question,
                extract_rewrite(raw),
                args.max_words,
            )
            new_example = dict(example)
            new_example["original_question"] = original_question
            new_example["question"] = rewritten_question
            new_example["rewrite_changed"] = rewritten_question != original_question
            new_example["rewrite_provider"] = args.provider
            new_example["rewrite_model"] = model
            new_example["llm_rewrite_raw"] = raw
            if args.sleep:
                time.sleep(args.sleep)

        changed += int(new_example.get("rewrite_changed", False))
        output.append(new_example)

        if idx % args.save_every == 0:
            save_output(output_path, output)
            print(f"Processed {idx}/{len(selected)} examples")

    save_output(output_path, output)
    print(f"Wrote {len(output)} examples to {output_path}")
    print(f"Rewritten questions: {changed}/{len(output)}")
    if reused:
        print(f"Reused from existing output: {reused}")


if __name__ == "__main__":
    main()
