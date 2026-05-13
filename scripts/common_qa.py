"""Shared helpers for Turkish OpenQA scripts."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any


def normalize_text(text: Any) -> str:
    text = str(text).casefold()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def answers_from(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("text", [])
    if isinstance(raw, str):
        answer = raw.strip()
        return [answer] if answer else []
    if isinstance(raw, list):
        answers = []
        for answer in raw:
            if isinstance(answer, dict):
                answer = answer.get("text", "")
            answer = str(answer).strip() if answer is not None else ""
            if answer:
                answers.append(answer)
        return answers
    answer = str(raw).strip()
    return [answer] if answer else []


def contains_answer(text: str, answers: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return any(normalize_text(answer) in normalized_text for answer in answers if normalize_text(answer))


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def get_context_text(ctx: dict[str, Any]) -> str:
    return f"{ctx.get('title', '')} {ctx.get('text', '')}"
