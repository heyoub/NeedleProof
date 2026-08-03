from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

import numpy as np
from openai import AsyncOpenAI

from .chunk_ids import ChunkId
from .config import Settings
from .corpus import (
    chunk_records_digest,
    corpus_version_path,
    current_corpus_path,
    l2_normalize,
    load_corpus_manifest,
    load_current_manifest,
)
from .models import ChunkRecord, SearchHit, SearchResult
from .util import sha256_file, utc_now_iso
from .vector_index import TurboVecAdapter

CallRecorder = Callable[[dict[str, Any]], Awaitable[None]]


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[\w.$%+-]+", query, flags=re.UNICODE)
    if not tokens:
        return '""'
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens[:24])


class CorpusStore:
    def __init__(self, settings: Settings, corpus_version: str | None = None):
        self.settings = settings
        self.path = (
            corpus_version_path(settings, corpus_version)
            if corpus_version
            else current_corpus_path(settings)
        )
        self.manifest = (
            load_corpus_manifest(settings, corpus_version)
            if corpus_version
            else load_current_manifest(settings)
        )
        self.db_path = self.path / "corpus.sqlite3"
        self._validate_artifacts()
        self.index = TurboVecAdapter.load(
            self.path / "index.tvim",
            dimensions=settings.embedding_dimensions,
            bit_width=int(self.manifest["turbovec_bit_width"]),
        )
        self._openai: AsyncOpenAI | None = None
        self._search_lock = asyncio.Lock()
        self._validate_index_ids()

    def _validate_artifacts(self) -> None:
        if self.manifest.get("schema_version") != "1.2":
            raise ValueError("Corpus manifest uses an unsupported schema version")
        if int(self.manifest["embedding_dimensions"]) != self.settings.embedding_dimensions:
            raise ValueError("Configured embedding dimensions do not match the corpus")
        artifacts = self.manifest.get("artifacts") or {}
        expected_db_sha = artifacts.get("corpus_sqlite3_sha256")
        expected_index_sha = artifacts.get("index_tvim_sha256")
        if sha256_file(self.db_path) != expected_db_sha:
            raise ValueError("Corpus SQLite artifact digest verification failed")
        index_path = self.path / "index.tvim"
        if sha256_file(index_path) != expected_index_sha:
            raise ValueError("TurboVec artifact digest verification failed")

        with closing(sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)) as connection:
            document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if document_count != int(self.manifest["document_count"]):
                raise ValueError("SQLite document count does not match the corpus manifest")
            if chunk_count != int(self.manifest["chunk_count"]):
                raise ValueError("SQLite chunk count does not match the corpus manifest")
            if chunk_records_digest(connection) != artifacts.get("ordered_chunk_records_sha256"):
                raise ValueError("SQLite chunk-record digest verification failed")
            documents = {
                row[0]: (row[1], row[2])
                for row in connection.execute(
                    "SELECT document_id, source_file, source_sha256 FROM documents"
                ).fetchall()
            }
        for document_id, (source_file, source_sha) in documents.items():
            if sha256_file(self.path / "documents" / source_file) != source_sha:
                raise ValueError(f"Source artifact digest failed for {document_id}")

    def _validate_index_ids(self) -> None:
        with closing(sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)) as connection:
            ids = [
                row[0]
                for row in connection.execute("SELECT chunk_external_id FROM chunks").fetchall()
            ]
        if len(self.index) != len(ids):
            raise ValueError("TurboVec length does not match SQLite")
        if not all(self.index.contains(chunk_id) for chunk_id in ids):
            raise ValueError("TurboVec ID set does not match SQLite")

    @property
    def corpus_id(self) -> str:
        return str(self.manifest["corpus_id"])

    @property
    def corpus_version(self) -> str:
        return str(self.manifest["corpus_version"])

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])

    async def embed_query(self, query: str, recorder: CallRecorder | None = None) -> np.ndarray:
        started_at = utc_now_iso()
        clock = time.perf_counter()
        try:
            if self._openai is None:
                self._openai = AsyncOpenAI()
            response = await self._openai.embeddings.create(
                model=self.settings.embedding_model,
                input=[query],
                dimensions=self.settings.embedding_dimensions,
                encoding_format="float",
            )
        except Exception as error:
            if recorder:
                await recorder(
                    {
                        "operation": "embedding",
                        "model": self.settings.embedding_model,
                        "response_id": None,
                        "request_id": getattr(error, "request_id", None),
                        "started_at": started_at,
                        "ended_at": utc_now_iso(),
                        "duration_ms": round((time.perf_counter() - clock) * 1000, 3),
                        "token_usage": {},
                        "retry_count": 0,
                        "error": {"type": type(error).__name__, "message": str(error)[:500]},
                    }
                )
            raise
        if recorder:
            await recorder(
                {
                    "operation": "embedding",
                    "model": self.settings.embedding_model,
                    "response_id": getattr(response, "id", None),
                    "request_id": getattr(response, "_request_id", None),
                    "started_at": started_at,
                    "ended_at": utc_now_iso(),
                    "duration_ms": round((time.perf_counter() - clock) * 1000, 3),
                    "token_usage": response.usage.model_dump() if response.usage else {},
                    "retry_count": 0,
                    "error": None,
                }
            )
        return l2_normalize(np.asarray([response.data[0].embedding], dtype=np.float32))

    def _candidate_ids(
        self,
        document_ids: list[str] | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[ChunkId] | None:
        if not document_ids and not date_from and not date_to:
            return None
        conditions: list[str] = []
        params: list[Any] = []
        if document_ids:
            conditions.append(f"c.document_id IN ({','.join('?' for _ in document_ids)})")
            params.extend(document_ids)
        date_expression = (
            "COALESCE(d.reporting_period_date, d.publication_date, d.file_creation_date)"
        )
        if date_from:
            conditions.append(f"{date_expression} >= ?")
            params.append(date_from)
        if date_to:
            conditions.append(f"{date_expression} <= ?")
            params.append(date_to)
        query = (
            "SELECT c.chunk_external_id FROM chunks c JOIN documents d USING(document_id) WHERE "
            + " AND ".join(conditions)
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute(query, params).fetchall()
        return [row[0] for row in rows]

    def _dense_search(
        self, query_vector: np.ndarray, k: int, allowlist: list[ChunkId] | None
    ) -> list[tuple[ChunkId, float]]:
        if allowlist is not None and len(allowlist) == 0:
            return []
        return self.index.search(query_vector, top_k=k, allowlist=allowlist)

    def _lexical_search(
        self,
        query: str,
        k: int,
        document_ids: list[str] | None,
        allowed: set[ChunkId] | None,
    ) -> list[tuple[ChunkId, float]]:
        fts = _fts_query(query)
        sql = """
            SELECT f.chunk_external_id, bm25(chunks_fts) AS rank
            FROM chunks_fts f
            WHERE chunks_fts MATCH ?
        """
        params: list[Any] = [fts]
        if document_ids:
            sql += f" AND f.document_id IN ({','.join('?' for _ in document_ids)})"
            params.extend(document_ids)
        sql += " ORDER BY rank LIMIT ?"
        params.append(max(k * 4, 32))
        try:
            with closing(sqlite3.connect(self.db_path)) as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        output: list[tuple[ChunkId, float]] = []
        for chunk_id, rank in rows:
            if allowed is not None and chunk_id not in allowed:
                continue
            output.append((chunk_id, float(rank)))
            if len(output) >= k:
                break
        return output

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        mode: Literal["dense", "lexical", "hybrid"] = "hybrid",
        document_ids: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        recorder: CallRecorder | None = None,
    ) -> SearchResult:
        top_k = max(1, min(top_k, self.settings.max_top_k))
        allowlist = self._candidate_ids(document_ids, date_from, date_to)
        allowed_set = set(allowlist) if allowlist is not None else None
        dense: list[tuple[ChunkId, float]] = []
        lexical: list[tuple[ChunkId, float]] = []
        if mode in {"dense", "hybrid"}:
            vector = await self.embed_query(query, recorder=recorder)
            async with self._search_lock:
                dense = await asyncio.to_thread(
                    self._dense_search,
                    vector,
                    max(top_k * 3, 24) if mode == "hybrid" else top_k,
                    allowlist,
                )
        if mode in {"lexical", "hybrid"}:
            lexical = await asyncio.to_thread(
                self._lexical_search,
                query,
                max(top_k * 3, 24) if mode == "hybrid" else top_k,
                document_ids,
                allowed_set,
            )

        dense_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense, 1)}
        lexical_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(lexical, 1)}
        dense_score = dict(dense)
        lexical_score = dict(lexical)
        if mode == "dense":
            ordered = [chunk_id for chunk_id, _ in dense[:top_k]]
            fused_scores = {chunk_id: dense_score[chunk_id] for chunk_id in ordered}
        elif mode == "lexical":
            ordered = [chunk_id for chunk_id, _ in lexical[:top_k]]
            fused_scores = {chunk_id: 1.0 / (60 + lexical_rank[chunk_id]) for chunk_id in ordered}
        else:
            all_ids = set(dense_rank) | set(lexical_rank)
            fused_scores = {
                chunk_id: (1.0 / (60 + dense_rank[chunk_id]) if chunk_id in dense_rank else 0.0)
                + (1.0 / (60 + lexical_rank[chunk_id]) if chunk_id in lexical_rank else 0.0)
                for chunk_id in all_ids
            }
            ordered = sorted(all_ids, key=lambda item: fused_scores[item], reverse=True)[:top_k]

        records = self.get_chunks(ordered)
        by_id = {record.chunk_id: record for record in records}
        hits: list[SearchHit] = []
        for chunk_id in ordered:
            record = by_id.get(chunk_id)
            if not record:
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    score=round(fused_scores[chunk_id], 7),
                    dense_score=(
                        round(dense_score[chunk_id], 7) if chunk_id in dense_score else None
                    ),
                    lexical_score=(
                        round(lexical_score[chunk_id], 7) if chunk_id in lexical_score else None
                    ),
                    dense_rank=dense_rank.get(chunk_id),
                    lexical_rank=lexical_rank.get(chunk_id),
                    retrieval_mode=mode,
                    document_id=record.document_id,
                    document_name=record.document_name,
                    physical_page_index=record.physical_page_index,
                    printed_page_label=record.printed_page_label,
                    preview=(record.normalized_text[:277] + "…")
                    if len(record.normalized_text) > 280
                    else record.normalized_text,
                    sha256=record.sha256,
                )
            )
        return SearchResult(
            query=query,
            mode=mode,
            results=hits,
            corpus_manifest_sha256=self.manifest_sha256,
        )

    def resolve_chunk_ids(
        self, chunk_ids: list[ChunkId], neighbor_radius: int = 0
    ) -> list[ChunkId]:
        if not chunk_ids:
            return []
        wanted: list[ChunkId] = []
        seen: set[ChunkId] = set()
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            for original_id in chunk_ids:
                frontier = [original_id]
                for _ in range(neighbor_radius):
                    next_frontier: list[ChunkId] = []
                    for current_id in frontier:
                        row = connection.execute(
                            "SELECT previous_chunk_id, next_chunk_id FROM chunks WHERE chunk_external_id = ?",
                            (str(current_id),),
                        ).fetchone()
                        if row:
                            for value in (row["previous_chunk_id"], row["next_chunk_id"]):
                                if value is not None:
                                    next_frontier.append(value)
                    frontier.extend(next_frontier)
                for value in frontier:
                    if value not in seen:
                        seen.add(value)
                        wanted.append(value)
        return wanted

    def get_chunks(self, chunk_ids: list[ChunkId], neighbor_radius: int = 0) -> list[ChunkRecord]:
        wanted = self.resolve_chunk_ids(chunk_ids, neighbor_radius)
        if not wanted:
            return []
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in wanted)
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE chunk_external_id IN ({placeholders})",
                wanted,
            ).fetchall()
        mapped = {
            row["chunk_external_id"]: ChunkRecord(
                internal_id=row["internal_id"],
                chunk_id=row["chunk_external_id"],
                document_id=row["document_id"],
                document_name=row["document_name"],
                physical_page_index=row["physical_page_index"],
                printed_page_label=row["printed_page_label"],
                chunk_position=row["chunk_position"],
                text=row["raw_text"],
                normalized_text=row["normalized_text"],
                previous_chunk_id=(row["previous_chunk_id"] if row["previous_chunk_id"] else None),
                next_chunk_id=row["next_chunk_id"] if row["next_chunk_id"] else None,
                sha256=row["sha256"],
                token_estimate=row["token_estimate"],
            )
            for row in rows
        }
        return [mapped[value] for value in wanted if value in mapped]

    def document_chunk_ids(self, document_id: str, page_from: int, page_to: int) -> list[ChunkId]:
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute(
                """
                SELECT chunk_external_id FROM chunks
                WHERE document_id = ? AND physical_page_index BETWEEN ? AND ?
                ORDER BY physical_page_index, chunk_position
                """,
                (document_id, page_from, page_to),
            ).fetchall()
        return [row[0] for row in rows]

    def inspect_document(
        self, document_id: str, page_from: int | None = None, page_to: int | None = None
    ) -> dict[str, Any]:
        clauses = ["p.document_id = ?"]
        params: list[Any] = [document_id]
        if page_from is not None:
            clauses.append("p.physical_page_index >= ?")
            params.append(page_from)
        if page_to is not None:
            clauses.append("p.physical_page_index <= ?")
            params.append(page_to)
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            document = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            if not document:
                raise KeyError(f"Unknown document_id: {document_id}")
            rows = connection.execute(
                "SELECT * FROM pages p WHERE "
                + " AND ".join(clauses)
                + " ORDER BY p.physical_page_index",
                params,
            ).fetchall()
        return {
            "document_id": document_id,
            "document_name": document["display_name"],
            "pages": [
                {
                    "physical_page_index": row["physical_page_index"],
                    "printed_page_label": row["printed_page_label"],
                    "text": row["raw_text"],
                    "table_markdown": row["table_markdown"],
                    "sha256": row["raw_sha256"],
                }
                for row in rows
            ],
        }

    def source_pdf(self, document_id: str) -> Path:
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT source_file FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        if not row:
            raise KeyError(document_id)
        return self.path / "documents" / row[0]
