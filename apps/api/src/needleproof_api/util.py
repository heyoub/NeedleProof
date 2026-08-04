from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex}"


def new_corpus_version() -> str:
    return f"corpus_{utc_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


_LINE_HYPHEN = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
_WHITESPACE = re.compile(r"\s+")
_METRIC_TOKEN = re.compile(
    r"(?P<initialism>"
    r"(?<![\w])(?:"
    r"(?:[^\W\d_]\.){2,}|"
    r"(?:[^\W\d_]\.)+[^\W\d_](?![\w])"
    r")"
    r")|(?P<word>[^\W_]+)"
)
MAX_METRIC_FTS_VARIANTS = 64


@dataclass(frozen=True, slots=True)
class MetricToken:
    """One canonical metric token with its exact source span."""

    value: str
    start: int
    end: int
    is_initialism: bool


def metric_tokens(value: str) -> tuple[MetricToken, ...]:
    """Tokenize metric text while treating dotted initialisms as one word."""

    tokens = []
    for match in _METRIC_TOKEN.finditer(value):
        raw = match.group()
        is_initialism = match.lastgroup == "initialism"
        canonical = raw.replace(".", "") if is_initialism else raw
        tokens.append(
            MetricToken(
                value=canonical.casefold(),
                start=match.start(),
                end=match.end(),
                is_initialism=is_initialism,
            )
        )
    return tuple(tokens)


def normalize_evidence_text(text: str) -> tuple[str, list[str]]:
    operations: list[str] = []
    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        operations.append("unicode_nfkc")
    dehyphenated = _LINE_HYPHEN.sub("", normalized)
    if dehyphenated != normalized:
        operations.append("pdf_linebreak_dehyphenation")
    folded = _WHITESPACE.sub(" ", dehyphenated).strip()
    if folded != dehyphenated:
        operations.append("whitespace_folding")
    return folded, operations


def canonical_metric_key(value: str) -> str:
    """Return one normalized complete-word key for claim/probe identity."""

    normalized, _ = normalize_evidence_text(value)
    return " ".join(token.value for token in metric_tokens(normalized))


def metric_fts_phrase_variants(
    value: str,
    *,
    max_variants: int = MAX_METRIC_FTS_VARIANTS,
) -> tuple[str, ...] | None:
    """Return conservative FTS phrases for compact and dotted initialisms.

    SQLite FTS tokenizes ``U.S.`` as ``u s`` while the application canonicalizes
    it to ``us``. The exhaustive absence scan queries both representations and
    still post-filters every result with the span-preserving metric tokenizer.
    """

    normalized, _ = normalize_evidence_text(value)
    alternatives: list[tuple[str, ...]] = []
    variant_count = 1
    for token in metric_tokens(normalized):
        raw = normalized[token.start : token.end]
        looks_compact_initialism = (
            2 <= len(token.value) <= 8 and raw.isalpha() and raw.upper() == raw
        )
        if token.is_initialism or looks_compact_initialism:
            alternatives.append((token.value, " ".join(token.value)))
            variant_count *= 2
        else:
            alternatives.append((token.value,))
    if not alternatives:
        return ()
    if variant_count > max_variants:
        return None
    return tuple(dict.fromkeys(" ".join(candidate) for candidate in product(*alternatives)))


def evidence_text_contains(needle: str, haystack: str) -> bool:
    """Apply the canonical evidence normalization and case-folding containment rule."""

    normalized_needle, _ = normalize_evidence_text(needle)
    if not normalized_needle:
        return False
    normalized_haystack, _ = normalize_evidence_text(haystack)
    return normalized_needle.casefold() in normalized_haystack.casefold()


def estimate_tokens(text: str) -> int:
    # Stable, dependency-free estimate suitable for deterministic chunk boundaries.
    words = len(re.findall(r"\S+", text))
    return max(1, round(words * 1.32))
