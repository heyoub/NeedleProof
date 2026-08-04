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
from enum import StrEnum
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


class MetricTokenKind(StrEnum):
    WORD = "word"
    INITIALISM = "initialism"


@dataclass(frozen=True, slots=True)
class MetricToken:
    """One canonical metric token with its exact source span."""

    value: str
    start: int
    end: int
    kind: MetricTokenKind

    @property
    def is_initialism(self) -> bool:
        return self.kind is MetricTokenKind.INITIALISM

    @property
    def identity(self) -> tuple[MetricTokenKind, str]:
        return self.kind, self.value


@dataclass(frozen=True, slots=True)
class MetricIdentity:
    """Kind-preserving metric identity; source spans never participate in equality."""

    lexemes: tuple[tuple[MetricTokenKind, str], ...]

    @property
    def canonical_key(self) -> str:
        # Uppercase is a stable wire representation of INITIALISM, not the
        # authority-bearing comparison law. Internal equality uses ``lexemes``.
        return " ".join(
            value.upper() if kind is MetricTokenKind.INITIALISM else value
            for kind, value in self.lexemes
        )


def metric_tokens(value: str) -> tuple[MetricToken, ...]:
    """Tokenize metric text without collapsing initialisms into ordinary words."""

    tokens = []
    for match in _METRIC_TOKEN.finditer(value):
        raw = match.group()
        dotted_initialism = match.lastgroup == "initialism"
        compact_initialism = (
            not dotted_initialism
            # Compact 2-3 letter forms cover the identity-sensitive corpus
            # metrics (IT, US, AUM) and their dotted equivalents. Longer
            # uppercase tokens are treated as ordinary typography so REVENUE,
            # TOTAL, and RATE remain case-insensitive words. Ambiguous short
            # forms still fail closed in exhaustive absence scans.
            and 2 <= len(raw) <= 3
            and raw.isalpha()
            and raw.upper() == raw
            and raw.lower() != raw
        )
        kind = (
            MetricTokenKind.INITIALISM
            if dotted_initialism or compact_initialism
            else MetricTokenKind.WORD
        )
        canonical = raw.replace(".", "") if dotted_initialism else raw
        tokens.append(
            MetricToken(
                value=canonical.casefold(),
                start=match.start(),
                end=match.end(),
                kind=kind,
            )
        )
    return tuple(tokens)


def metric_identity(value: str) -> MetricIdentity:
    """Return the kind-preserving identity shared by every authority boundary."""

    normalized, _ = normalize_evidence_text(value)
    return MetricIdentity(tuple(token.identity for token in metric_tokens(normalized)))


def punctuation_boundaries(
    text: str,
    *,
    punctuation: frozenset[str] = frozenset(".!?;"),
) -> tuple[tuple[int, int], ...]:
    """Return punctuation spans without splitting inside dotted initialisms."""

    initialism_periods = {
        offset
        for token in metric_tokens(text)
        if token.is_initialism
        for offset in range(token.start, token.end)
        if text[offset] == "."
    }
    boundaries = []
    for offset, character in enumerate(text):
        if character not in punctuation or offset in initialism_periods:
            continue
        if character == "." and offset + 1 < len(text):
            if not text[offset + 1].isspace():
                continue
            next_offset = offset + 1
            while next_offset < len(text) and text[next_offset].isspace():
                next_offset += 1
            previous_word = re.search(r"[^\W\d_]+$", text[:offset])
            next_character = text[next_offset : next_offset + 1]
            if (
                next_character.islower()
                or next_character.isdigit()
                or (
                    previous_word
                    and len(previous_word.group()) <= 4
                    and previous_word.group()[0].isupper()
                    and next_character.isupper()
                )
            ):
                # Ambiguous abbreviation punctuation stays open. Expanding the
                # proof context can cause review, never stronger authority.
                continue
        boundaries.append((offset, offset + 1))
    return tuple(boundaries)


def sentence_fragments(text: str) -> tuple[str, ...]:
    """Split terminal sentences under the same dotted-initialism boundary law."""

    boundaries = punctuation_boundaries(text, punctuation=frozenset(".!?"))
    if not boundaries:
        return (text,)
    fragments = []
    start = 0
    for _boundary_start, boundary_end in boundaries:
        fragments.append(text[start:boundary_end])
        start = boundary_end
    if start < len(text):
        fragments.append(text[start:])
    return tuple(fragment for fragment in fragments if fragment.strip())


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
    """Serialize the shared typed identity for receipts, signatures, and map keys."""

    return metric_identity(value).canonical_key


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
        if token.is_initialism:
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
