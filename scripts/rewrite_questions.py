#!/usr/bin/env python3
"""Apply lightweight rule-based question rewriting for retrieval experiments."""

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="QA JSON file.")
    parser.add_argument("--output", required=True, help="Rewritten QA JSON file.")
    parser.add_argument(
        "--mode",
        default="definition",
        choices=["definition", "all", "keyword"],
        help="Rewrite strategy to apply.",
    )
    return parser.parse_args()


def clean_question(question):
    return " ".join(question.strip().split())


def looks_like_acronym(text):
    uppercase_count = sum(1 for char in text if char.isupper())
    return uppercase_count >= 2


STOPWORDS = {
    "acaba",
    "adı",
    "adıdır",
    "adlı",
    "ama",
    "bir",
    "biri",
    "bu",
    "da",
    "de",
    "daha",
    "değil",
    "diye",
    "en",
    "gibi",
    "hangi",
    "hangisi",
    "için",
    "ile",
    "ise",
    "kaç",
    "kac",
    "ki",
    "kim",
    "kime",
    "kimin",
    "mı",
    "mi",
    "mu",
    "mü",
    "nasıl",
    "nasil",
    "ne",
    "neden",
    "nerede",
    "nerededir",
    "nereden",
    "nereye",
    "olarak",
    "olan",
    "oldu",
    "olmuştur",
    "olur",
    "şu",
    "ve",
    "veya",
}


def rewrite_definition_question(question):
    q = clean_question(question)
    patterns = [
        r"^(?P<x>.+?)\s+ne\s+anlama\s+geliyor\??$",
        r"^(?P<x>.+?)\s+ne\s+demek\??$",
        r"^(?P<x>.+?)\s+neyi\s+ifade\s+eder\??$",
    ]
    for pattern in patterns:
        match = re.match(pattern, q, flags=re.IGNORECASE)
        if match:
            entity = match.group("x").strip(" ?")
            if entity:
                if "kısalt" in entity.lower() or looks_like_acronym(entity):
                    return f"{entity} açılımı nedir?"
                return f"{entity} anlamı nedir?"
    return q


def rewrite_question(question, mode):
    if mode == "keyword":
        return rewrite_keyword_question(question)
    if mode in {"definition", "all"}:
        rewritten = rewrite_definition_question(question)
        if rewritten != clean_question(question) or mode == "definition":
            return rewritten
    if mode == "all":
        return rewrite_with_answer_type(question)
    return clean_question(question)


def answer_type_tokens(question):
    lowered = question.lower()
    rules = [
        (r"\bne zaman\b|\bhangi yıl\b|\bhangi yüzyıl\b|\bhangi yy\b|\bhangi tarihte\b", ["tarih"]),
        (r"\bnerede\b|\bnerededir\b|\bhangi ülke\b|\bhangi ülkede\b|\bhangi şehir\b|\bhangi şehirde\b|\bhangi bölge\b|\bhangi bölgede\b", ["yer"]),
        (r"\bkim\b|\bkimin\b|\bkime\b|\bkimdir\b", ["kişi"]),
        (r"\bkaç\b|\bkac\b|\bne kadar\b", ["sayı"]),
        (r"\bneden\b|\bniçin\b", ["sebep"]),
        (r"\bnasıl\b|\bnasil\b", ["yöntem"]),
        (r"\bne anlama geliyor\b|\bne demek\b|\bneyi ifade eder\b", ["anlam"]),
    ]
    for pattern, tokens in rules:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return tokens
    return []


def rewrite_keyword_question(question):
    q = clean_question(question)
    definition_rewrite = rewrite_definition_question(q)
    if definition_rewrite != q:
        return definition_rewrite

    tokens = re.findall(r"[\wÇĞİÖŞÜçğıöşü'’.-]+", q, flags=re.UNICODE)
    kept = []
    for token in tokens:
        normalized = token.strip("'’.-").lower()
        if not normalized or normalized in STOPWORDS:
            continue
        if len(normalized) == 1 and not token.isupper():
            continue
        kept.append(token.strip("?"))

    hints = answer_type_tokens(q)
    for hint in hints:
        if hint not in [token.lower() for token in kept]:
            kept.append(hint)

    if not kept:
        return q
    return " ".join(kept)


def append_hint(question, hint):
    q = clean_question(question).rstrip("?")
    if hint.lower() in q.lower():
        return clean_question(question)
    return f"{q} {hint}?"


def rewrite_with_answer_type(question):
    q = clean_question(question)
    lowered = q.lower()
    rules = [
        (r"\bne zaman\b|\bhangi yıl\b|\bhangi yüzyıl\b|\bhangi yy\b|\bhangi tarihte\b", "tarih yıl zaman"),
        (r"\bnerede\b|\bnerededir\b|\bhangi ülke\b|\bhangi ülkede\b|\bhangi şehir\b|\bhangi şehirde\b|\bhangi bölge\b|\bhangi bölgede\b", "yer ülke şehir bölge"),
        (r"\bhangi nehir\b|\bhangi deniz\b|\bhangi dağ\b", "yer coğrafya"),
        (r"\bkim\b|\bkimin\b|\bkime\b|\bkimdir\b", "kişi ad isim"),
        (r"\bkaç\b|\bkac\b|\bne kadar\b", "sayı miktar"),
        (r"\bneden\b|\bniçin\b", "neden sebep"),
        (r"\bnasıl\b|\bnasil\b", "yöntem şekil"),
        (r"\bad[ıi]\s+nedir\b|\bismi\s+nedir\b|\bisim\s+nedir\b", "ad isim"),
        (r"\bhangi\b", "tür kategori"),
    ]
    for pattern, hint in rules:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return append_hint(q, hint)
    return q


def main():
    args = parse_args()
    with Path(args.input).open(encoding="utf-8") as fin:
        data = json.load(fin)

    changed = 0
    output = []
    for example in data:
        new_example = dict(example)
        original_question = clean_question(example["question"])
        rewritten_question = rewrite_question(original_question, args.mode)
        new_example["original_question"] = original_question
        new_example["question"] = rewritten_question
        new_example["rewrite_changed"] = rewritten_question != original_question
        changed += int(new_example["rewrite_changed"])
        output.append(new_example)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(output, fout, ensure_ascii=False, indent=2)

    print(f"Wrote {len(output)} examples to {output_path}")
    print(f"Rewritten questions: {changed}/{len(output)}")


if __name__ == "__main__":
    main()
