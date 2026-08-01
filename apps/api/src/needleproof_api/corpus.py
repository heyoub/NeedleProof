from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from pypdf import PdfReader

from .chunk_ids import ChunkId, chunk_id_from_uint64
from .config import Settings
from .models import ChunkRecord, CorpusSummary
from .util import (
    atomic_write_text,
    canonical_json,
    estimate_tokens,
    normalize_evidence_text,
    sha256_file,
    sha256_text,
    utc_now_iso,
)
from .vector_index import TurboVecAdapter

CORPUS_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    publication_date TEXT,
    reporting_period_date TEXT,
    file_creation_date TEXT,
    metadata_json TEXT NOT NULL
);

CREATE TABLE pages (
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    physical_page_index INTEGER NOT NULL,
    printed_page_label TEXT,
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    normalization_operations_json TEXT NOT NULL,
    table_markdown TEXT,
    raw_sha256 TEXT NOT NULL,
    normalized_sha256 TEXT NOT NULL,
    table_sha256 TEXT,
    PRIMARY KEY (document_id, physical_page_index)
);

CREATE TABLE chunks (
    internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_external_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    document_name TEXT NOT NULL,
    physical_page_index INTEGER NOT NULL,
    printed_page_label TEXT,
    chunk_position INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    previous_chunk_id TEXT,
    next_chunk_id TEXT,
    sha256 TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    UNIQUE (document_id, physical_page_index, chunk_position)
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_external_id UNINDEXED,
    document_id UNINDEXED,
    text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE INDEX idx_chunks_document_page ON chunks(document_id, physical_page_index, chunk_position);
"""

CHUNK_DIGEST_COLUMNS = (
    "chunk_external_id",
    "document_id",
    "document_name",
    "physical_page_index",
    "printed_page_label",
    "chunk_position",
    "raw_text",
    "normalized_text",
    "previous_chunk_id",
    "next_chunk_id",
    "sha256",
    "token_estimate",
)


def chunk_records_digest(connection: sqlite3.Connection) -> str:
    columns = ", ".join(CHUNK_DIGEST_COLUMNS)
    rows = connection.execute(f"SELECT {columns} FROM chunks ORDER BY chunk_external_id").fetchall()
    records = [dict(zip(CHUNK_DIGEST_COLUMNS, row, strict=True)) for row in rows]
    return sha256_text(canonical_json(records))


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding response contained a zero-length vector")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def deterministic_chunk_id(document_id: str, page_index: int, position: int) -> ChunkId:
    digest = bytes.fromhex(sha256_text(f"{document_id}:{page_index}:{position}"))
    return chunk_id_from_uint64(int.from_bytes(digest[:8], byteorder="big", signed=False))


def _split_paragraphs(raw_text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", raw_text) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in raw_text.splitlines() if part.strip()]
    return paragraphs


def _split_oversized_paragraph(paragraph: str, *, target_max: int, overlap: int) -> list[str]:
    if estimate_tokens(paragraph) <= target_max:
        return [paragraph]
    words = paragraph.split()
    maximum_words = max(1, int(target_max / 1.32))
    overlap_words = min(maximum_words - 1, max(1, int(overlap / 1.32)))
    step = maximum_words - overlap_words
    segments: list[str] = []
    for start in range(0, len(words), step):
        segment = " ".join(words[start : start + maximum_words])
        if segment:
            segments.append(segment)
        if start + maximum_words >= len(words):
            break
    return segments


def chunk_page(
    raw_text: str, *, target_min: int = 500, target_max: int = 700, overlap: int = 80
) -> list[str]:
    paragraphs = [
        segment
        for paragraph in _split_paragraphs(raw_text)
        for segment in _split_oversized_paragraph(paragraph, target_max=target_max, overlap=overlap)
    ]
    if not paragraphs:
        return []

    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if current and current_tokens + paragraph_tokens > target_max:
            chunks.append(current)
            carry: list[str] = []
            carry_tokens = 0
            for prior in reversed(current):
                if carry and carry_tokens >= overlap:
                    break
                carry.insert(0, prior)
                carry_tokens += estimate_tokens(prior)
            current = carry
            current_tokens = carry_tokens
        current.append(paragraph)
        current_tokens += paragraph_tokens
        if current_tokens >= target_min and current_tokens >= target_max * 0.9:
            chunks.append(current)
            current = []
            current_tokens = 0
    if current:
        combined = "\n\n".join([*chunks[-1], *current]) if chunks else ""
        if (
            chunks
            and estimate_tokens("\n\n".join(current)) < overlap
            and estimate_tokens(combined) <= target_max
        ):
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return ["\n\n".join(parts) for parts in chunks]


def extract_table_markdown(raw_text: str) -> str | None:
    rows: list[list[str]] = []
    for line in raw_text.splitlines():
        cells = [cell.strip() for cell in re.split(r"\s{2,}", line.strip()) if cell.strip()]
        if len(cells) >= 2:
            rows.append(cells)
    if len(rows) < 2:
        return None
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    body = padded[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


class CorpusBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _embed(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        client = OpenAI()
        vectors: list[list[float]] = []
        calls: list[dict[str, Any]] = []
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            started_at = utc_now_iso()
            import time

            clock = time.perf_counter()
            response = client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
                dimensions=self.settings.embedding_dimensions,
                encoding_format="float",
            )
            ended_at = utc_now_iso()
            vectors.extend(item.embedding for item in response.data)
            usage = response.usage.model_dump() if response.usage else {}
            calls.append(
                {
                    "operation": "embedding",
                    "model": self.settings.embedding_model,
                    "response_id": getattr(response, "id", None),
                    "request_id": getattr(response, "_request_id", None),
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_ms": round((time.perf_counter() - clock) * 1000, 3),
                    "token_usage": usage,
                    "retry_count": 0,
                    "error": None,
                    "input_count": len(batch),
                }
            )
        return l2_normalize(np.asarray(vectors, dtype=np.float32)), calls

    def build(self) -> CorpusSummary:
        source_path = self.settings.corpus_source.resolve()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        self.settings.corpora_dir.mkdir(parents=True, exist_ok=True)
        build_dir = self.settings.corpora_dir / f".build-{uuid.uuid4().hex}"
        build_dir.mkdir(parents=True)
        documents_dir = build_dir / "documents"
        documents_dir.mkdir()

        db_path = build_dir / "corpus.sqlite3"
        connection = sqlite3.connect(db_path)
        connection.executescript(CORPUS_SCHEMA)

        all_chunks: list[ChunkRecord] = []
        manifest_documents: list[dict[str, Any]] = []
        seen_chunk_ids: set[ChunkId] = set()
        seen_document_ids: set[str] = set()

        try:
            for definition in source["documents"]:
                original_path = Path(definition["path"]).resolve()
                if not original_path.exists():
                    raise FileNotFoundError(f"Corpus source is missing: {original_path}")
                source_sha = sha256_file(original_path)
                included_pages = [int(page) for page in definition.get("include_pages", [])]
                selection_digest = sha256_text(
                    canonical_json({"source_sha256": source_sha, "include_pages": included_pages})
                )
                document_id = f"doc_{selection_digest[:24]}"
                if document_id in seen_document_ids:
                    raise ValueError(f"Duplicate corpus source selection resolves to {document_id}")
                seen_document_ids.add(document_id)
                display_name = definition.get("display_name") or original_path.name
                destination_name = f"{document_id}{original_path.suffix.lower()}"
                shutil.copy2(original_path, documents_dir / destination_name)

                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id, display_name, source_file, source_sha256,
                        publication_date, reporting_period_date, file_creation_date, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        display_name,
                        destination_name,
                        source_sha,
                        definition.get("publication_date"),
                        definition.get("reporting_period_date"),
                        definition.get("file_creation_date"),
                        canonical_json(definition),
                    ),
                )

                document_chunks: list[ChunkRecord] = []
                if original_path.suffix.lower() == ".pdf":
                    reader = PdfReader(original_path)
                    labels = reader.page_labels
                    selected = included_pages or list(range(1, len(reader.pages) + 1))
                    for physical_page in selected:
                        if physical_page < 1 or physical_page > len(reader.pages):
                            raise ValueError(
                                f"Page {physical_page} is outside {original_path.name} ({len(reader.pages)} pages)"
                            )
                        raw_text = reader.pages[physical_page - 1].extract_text() or ""
                        if not raw_text.strip():
                            raise ValueError(
                                f"No usable text extracted from {original_path.name} page {physical_page}; OCR is out of scope"
                            )
                        normalized_page, operations = normalize_evidence_text(raw_text)
                        table_markdown = extract_table_markdown(raw_text)
                        printed_label = (
                            labels[physical_page - 1] if physical_page <= len(labels) else None
                        )
                        connection.execute(
                            """
                            INSERT INTO pages (
                                document_id, physical_page_index, printed_page_label, raw_text,
                                normalized_text, normalization_operations_json, table_markdown,
                                raw_sha256, normalized_sha256, table_sha256
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                document_id,
                                physical_page,
                                printed_label,
                                raw_text,
                                normalized_page,
                                canonical_json(operations),
                                table_markdown,
                                sha256_text(raw_text),
                                sha256_text(normalized_page),
                                sha256_text(table_markdown) if table_markdown else None,
                            ),
                        )
                        for position, chunk_text in enumerate(chunk_page(raw_text)):
                            normalized_chunk, _ = normalize_evidence_text(chunk_text)
                            chunk_id = deterministic_chunk_id(document_id, physical_page, position)
                            if chunk_id in seen_chunk_ids:
                                raise ValueError(f"Deterministic chunk ID collision: {chunk_id}")
                            seen_chunk_ids.add(chunk_id)
                            document_chunks.append(
                                ChunkRecord(
                                    chunk_id=chunk_id,
                                    document_id=document_id,
                                    document_name=display_name,
                                    physical_page_index=physical_page,
                                    printed_page_label=printed_label,
                                    chunk_position=position,
                                    text=chunk_text,
                                    normalized_text=normalized_chunk,
                                    sha256=sha256_text(normalized_chunk),
                                    token_estimate=estimate_tokens(normalized_chunk),
                                )
                            )
                elif original_path.suffix.lower() == ".txt":
                    raw_text = original_path.read_text(encoding="utf-8")
                    if not raw_text.strip():
                        raise ValueError(f"No usable text found in {original_path.name}")
                    normalized_page, operations = normalize_evidence_text(raw_text)
                    connection.execute(
                        """
                        INSERT INTO pages VALUES (?, 1, '1', ?, ?, ?, NULL, ?, ?, NULL)
                        """,
                        (
                            document_id,
                            raw_text,
                            normalized_page,
                            canonical_json(operations),
                            sha256_text(raw_text),
                            sha256_text(normalized_page),
                        ),
                    )
                    for position, chunk_text in enumerate(chunk_page(raw_text)):
                        normalized_chunk, _ = normalize_evidence_text(chunk_text)
                        chunk_id = deterministic_chunk_id(document_id, 1, position)
                        if chunk_id in seen_chunk_ids:
                            raise ValueError(f"Deterministic chunk ID collision: {chunk_id}")
                        seen_chunk_ids.add(chunk_id)
                        document_chunks.append(
                            ChunkRecord(
                                chunk_id=chunk_id,
                                document_id=document_id,
                                document_name=display_name,
                                physical_page_index=1,
                                printed_page_label="1",
                                chunk_position=position,
                                text=chunk_text,
                                normalized_text=normalized_chunk,
                                sha256=sha256_text(normalized_chunk),
                                token_estimate=estimate_tokens(normalized_chunk),
                            )
                        )
                else:
                    raise ValueError(f"Unsupported corpus input: {original_path.suffix}")

                for index, chunk in enumerate(document_chunks):
                    chunk.previous_chunk_id = (
                        document_chunks[index - 1].chunk_id if index > 0 else None
                    )
                    chunk.next_chunk_id = (
                        document_chunks[index + 1].chunk_id
                        if index + 1 < len(document_chunks)
                        else None
                    )
                    connection.execute(
                        """
                        INSERT INTO chunks (
                            chunk_external_id, document_id, document_name, physical_page_index,
                            printed_page_label, chunk_position, raw_text, normalized_text,
                            previous_chunk_id, next_chunk_id, sha256, token_estimate
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.document_name,
                            chunk.physical_page_index,
                            chunk.printed_page_label,
                            chunk.chunk_position,
                            chunk.text,
                            chunk.normalized_text,
                            chunk.previous_chunk_id
                            if chunk.previous_chunk_id is not None
                            else None,
                            chunk.next_chunk_id,
                            chunk.sha256,
                            chunk.token_estimate,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO chunks_fts (chunk_external_id, document_id, text) VALUES (?, ?, ?)",
                        (chunk.chunk_id, chunk.document_id, chunk.normalized_text),
                    )
                all_chunks.extend(document_chunks)
                manifest_documents.append(
                    {
                        "document_id": document_id,
                        "display_name": display_name,
                        "source_sha256": source_sha,
                        "included_pages": included_pages,
                        "publication_date": definition.get("publication_date"),
                        "reporting_period_date": definition.get("reporting_period_date"),
                        "file_creation_date": definition.get("file_creation_date"),
                        "chunk_count": len(document_chunks),
                        "chunk_hashes": [chunk.sha256 for chunk in document_chunks],
                    }
                )
            connection.commit()

            if not all_chunks:
                raise ValueError("Corpus produced no chunks")
            embeddings, embedding_calls = self._embed(
                [chunk.normalized_text for chunk in all_chunks]
            )
            index = TurboVecAdapter.create(
                dimensions=self.settings.embedding_dimensions, bit_width=4
            )
            index.add(embeddings, [chunk.chunk_id for chunk in all_chunks])
            index.write(build_dir / "index.tvim")

            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.commit()
            sqlite_chunk_digest = chunk_records_digest(connection)
            connection.close()
            sqlite_sha = sha256_file(db_path)
            index_sha = sha256_file(build_dir / "index.tvim")

            manifest_without_digest = {
                "schema_version": "1.2",
                "corpus_id": source["corpus_id"],
                "display_name": source["display_name"],
                "description": source.get("description"),
                "parser": {"name": "pypdf", "version": package_version("pypdf")},
                "chunker": {
                    "version": "page-semantic-v3-bounded-paragraphs",
                    "target_tokens": [500, 700],
                    "overlap_tokens": 80,
                    "crosses_pages": False,
                },
                "normalization": [
                    "unicode_nfkc",
                    "whitespace_folding",
                    "pdf_linebreak_dehyphenation",
                ],
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "embedding_l2_normalized": True,
                "turbovec_version": package_version("turbovec"),
                "turbovec_bit_width": 4,
                "retrieval": ["dense", "sqlite_fts5", "rrf_hybrid"],
                "documents": manifest_documents,
                "document_count": len(manifest_documents),
                "chunk_count": len(all_chunks),
                "artifacts": {
                    "corpus_sqlite3_sha256": sqlite_sha,
                    "index_tvim_sha256": index_sha,
                    "ordered_chunk_records_sha256": sqlite_chunk_digest,
                },
            }
            manifest_sha = sha256_text(canonical_json(manifest_without_digest))
            corpus_version = f"v_{manifest_sha[:16]}"
            manifest = {
                **manifest_without_digest,
                "corpus_version": corpus_version,
                "manifest_sha256": manifest_sha,
                "built_at": utc_now_iso(),
                "embedding_calls": embedding_calls,
            }
            atomic_write_text(build_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")

            if len(index) != len(all_chunks):
                raise ValueError("TurboVec length does not match chunk count")
            with sqlite3.connect(db_path) as validation_connection:
                row_count = validation_connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[
                    0
                ]
                if row_count != len(all_chunks):
                    raise ValueError("SQLite chunk count does not match manifest")
                if chunk_records_digest(validation_connection) != sqlite_chunk_digest:
                    raise ValueError("SQLite chunk-record digest failed validation")
                persisted_ids = {
                    row[0]
                    for row in validation_connection.execute(
                        "SELECT chunk_external_id FROM chunks"
                    ).fetchall()
                }
            loaded = TurboVecAdapter.load(
                build_dir / "index.tvim",
                dimensions=self.settings.embedding_dimensions,
                bit_width=4,
            )
            if len(loaded) != len(all_chunks):
                raise ValueError("Persisted TurboVec index failed length validation")
            if persisted_ids != {chunk.chunk_id for chunk in all_chunks}:
                raise ValueError("SQLite chunk ID set does not match the build")
            if not all(loaded.contains(chunk_id) for chunk_id in persisted_ids):
                raise ValueError("TurboVec index ID set does not match SQLite")
        except BaseException:
            connection.close()
            shutil.rmtree(build_dir, ignore_errors=True)
            raise
        finally:
            connection.close()

        version_dir = self.settings.corpora_dir / corpus_version
        if version_dir.exists():
            existing_manifest = json.loads(
                (version_dir / "manifest.json").read_text(encoding="utf-8")
            )
            expected_artifacts = manifest["artifacts"]
            existing_artifacts = existing_manifest.get("artifacts") or {}
            existing_valid = (
                existing_manifest.get("manifest_sha256") == manifest_sha
                and existing_artifacts == expected_artifacts
                and sha256_file(version_dir / "corpus.sqlite3")
                == expected_artifacts["corpus_sqlite3_sha256"]
                and sha256_file(version_dir / "index.tvim")
                == expected_artifacts["index_tvim_sha256"]
            )
            if not existing_valid:
                shutil.rmtree(build_dir)
                raise ValueError(
                    f"Existing immutable corpus version {corpus_version} is corrupt or mismatched"
                )
            shutil.rmtree(build_dir)
        else:
            os.replace(build_dir, version_dir)
        pointer = {
            "corpus_id": source["corpus_id"],
            "corpus_version": corpus_version,
            "manifest_sha256": manifest_sha,
        }
        atomic_write_text(
            self.settings.corpora_dir / "current.json", canonical_json(pointer) + "\n"
        )
        return CorpusSummary(
            corpus_id=source["corpus_id"],
            display_name=source["display_name"],
            corpus_version=corpus_version,
            manifest_sha256=manifest_sha,
            document_count=len(manifest_documents),
            chunk_count=len(all_chunks),
            embedding_model=self.settings.embedding_model,
            embedding_dimensions=self.settings.embedding_dimensions,
            turbovec_version=package_version("turbovec"),
        )


def corpus_version_path(settings: Settings, corpus_version: str) -> Path:
    if not re.fullmatch(r"v_[0-9a-f]{16}", corpus_version):
        raise ValueError("Invalid corpus version")
    path = settings.corpora_dir / corpus_version
    if not path.is_dir():
        raise FileNotFoundError(f"Corpus version is missing: {corpus_version}")
    return path


def current_corpus_path(settings: Settings) -> Path:
    pointer_path = settings.corpora_dir / "current.json"
    if not pointer_path.exists():
        raise FileNotFoundError("No published corpus. Run `pnpm corpus:build`.")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    return corpus_version_path(settings, pointer["corpus_version"])


def load_corpus_manifest(settings: Settings, corpus_version: str | None = None) -> dict[str, Any]:
    corpus_path = (
        corpus_version_path(settings, corpus_version)
        if corpus_version
        else current_corpus_path(settings)
    )
    manifest = json.loads((corpus_path / "manifest.json").read_text(encoding="utf-8"))
    actual = sha256_text(
        canonical_json(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"manifest_sha256", "corpus_version", "built_at", "embedding_calls"}
            }
        )
    )
    if actual != manifest["manifest_sha256"]:
        raise ValueError("Corpus manifest digest verification failed")
    expected_version = corpus_version or corpus_path.name
    if manifest["corpus_version"] != expected_version:
        raise ValueError("Corpus manifest version does not match its directory")
    return manifest


def load_current_manifest(settings: Settings) -> dict[str, Any]:
    pointer = json.loads((settings.corpora_dir / "current.json").read_text(encoding="utf-8"))
    manifest = load_corpus_manifest(settings, pointer["corpus_version"])
    for key in ("corpus_id", "corpus_version", "manifest_sha256"):
        if pointer.get(key) != manifest.get(key):
            raise ValueError(f"Current corpus pointer does not match manifest field {key}")
    return manifest
