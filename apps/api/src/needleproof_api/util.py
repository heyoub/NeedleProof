from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
import uuid
from datetime import UTC, datetime
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


def estimate_tokens(text: str) -> int:
    # Stable, dependency-free estimate suitable for deterministic chunk boundaries.
    words = len(re.findall(r"\S+", text))
    return max(1, round(words * 1.32))
