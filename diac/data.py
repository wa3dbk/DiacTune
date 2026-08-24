from __future__ import annotations
from pathlib import Path
from camel_tools.utils.dediac import dediac_ar


def load_sentences(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def make_pairs(sentences: list[str]) -> list[tuple[str, str]]:
    return [(dediac_ar(s), s) for s in sentences]
