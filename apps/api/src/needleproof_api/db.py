from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from .models import RunEnvelope, RunStatus
from .util import utc_now_iso

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT,
    status TEXT NOT NULL,
    question TEXT NOT NULL,
    corpus_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    corpus_manifest_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    answer TEXT,
    result_json TEXT,
    receipt_path TEXT,
    receipt_sha256 TEXT,
    error_json TEXT,
    rehearsal INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ledger_events (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS openai_calls (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS model_token_reservations (
    run_id TEXT PRIMARY KEY,
    reserved_tokens INTEGER NOT NULL CHECK (reserved_tokens > 0),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_ledger_run_sequence ON ledger_events(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_model_token_reservations_created_at
    ON model_token_reservations(created_at);
"""


class AppDatabase:
    def __init__(self, path: Path):
        self.path = path

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as connection:
            await connection.executescript(SCHEMA)
            cursor = await connection.execute("PRAGMA table_info(runs)")
            columns = {row[1] for row in await cursor.fetchall()}
            if "rehearsal" not in columns:
                await connection.execute(
                    "ALTER TABLE runs ADD COLUMN rehearsal INTEGER NOT NULL DEFAULT 0"
                )
            if "receipt_sha256" not in columns:
                await connection.execute("ALTER TABLE runs ADD COLUMN receipt_sha256 TEXT")
            await connection.commit()

    async def create_run(
        self,
        *,
        run_id: str,
        session_id: str,
        rehearsal: bool,
        question: str,
        corpus_id: str,
        corpus_version: str,
        manifest_sha256: str,
    ) -> None:
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                """
                INSERT INTO runs (
                    run_id, session_id, status, question, corpus_id, corpus_version,
                    corpus_manifest_sha256, created_at, rehearsal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    RunStatus.QUEUED.value,
                    question,
                    corpus_id,
                    corpus_version,
                    manifest_sha256,
                    utc_now_iso(),
                    int(rehearsal),
                ),
            )
            await connection.commit()

    async def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "status",
            "started_at",
            "completed_at",
            "answer",
            "result_json",
            "receipt_path",
            "receipt_sha256",
            "error_json",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported run fields: {sorted(unknown)}")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [
            value.value if isinstance(value, RunStatus) else value for value in fields.values()
        ]
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                f"UPDATE runs SET {assignments} WHERE run_id = ?",
                (*values, run_id),
            )
            await connection.commit()

    async def get_run_row(self, run_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_recoverable_runs(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                """
                SELECT * FROM runs
                WHERE status IN (?, ?, ?)
                ORDER BY created_at
                """,
                (
                    RunStatus.QUEUED.value,
                    RunStatus.RUNNING.value,
                    RunStatus.INTERRUPTED.value,
                ),
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def delete_runs_created_before(self, cutoff: str) -> list[str]:
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                "SELECT receipt_path FROM runs WHERE created_at < ? AND receipt_path IS NOT NULL",
                (cutoff,),
            )
            paths = [str(row[0]) for row in await cursor.fetchall()]
            await connection.execute(
                """
                DELETE FROM model_token_reservations
                WHERE run_id IN (SELECT run_id FROM runs WHERE created_at < ?)
                """,
                (cutoff,),
            )
            await connection.execute("DELETE FROM runs WHERE created_at < ?", (cutoff,))
            await connection.commit()
        return paths

    async def reserve_model_tokens(
        self,
        *,
        run_id: str,
        reserved_tokens: int,
        created_at: str,
        hourly_cutoff: str,
        daily_cutoff: str,
        hourly_limit: int,
        daily_limit: int,
    ) -> str | None:
        """Atomically check persisted usage plus active reservations and reserve capacity."""

        async with aiosqlite.connect(self.path) as connection:
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute("BEGIN IMMEDIATE")
            try:
                hourly_actual = await self._sum_model_tokens(connection, hourly_cutoff)
                daily_actual = await self._sum_model_tokens(connection, daily_cutoff)
                hourly_reserved = await self._sum_reserved_tokens(connection, hourly_cutoff)
                daily_reserved = await self._sum_reserved_tokens(connection, daily_cutoff)
                if hourly_actual + hourly_reserved + reserved_tokens > hourly_limit:
                    await connection.rollback()
                    return "hourly"
                if daily_actual + daily_reserved + reserved_tokens > daily_limit:
                    await connection.rollback()
                    return "daily"
                await connection.execute(
                    """
                    INSERT INTO model_token_reservations (run_id, reserved_tokens, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (run_id, reserved_tokens, created_at),
                )
                await connection.commit()
                return None
            except BaseException:
                await connection.rollback()
                raise

    @staticmethod
    async def _sum_model_tokens(connection: aiosqlite.Connection, started_at: str) -> int:
        cursor = await connection.execute(
            """
            SELECT c.record_json, r.created_at
            FROM openai_calls c
            JOIN runs r ON r.run_id = c.run_id
            """
        )
        total = 0
        for record_json, run_created_at in await cursor.fetchall():
            record = json.loads(record_json)
            call_started_at = str(record.get("started_at") or run_created_at)
            if record.get("operation") != "model" or call_started_at < started_at:
                continue
            usage = record.get("token_usage") or {}
            total += int(usage.get("total_tokens") or 0)
        return total

    @staticmethod
    async def _sum_reserved_tokens(connection: aiosqlite.Connection, started_at: str) -> int:
        cursor = await connection.execute(
            """
            SELECT COALESCE(SUM(reserved_tokens), 0)
            FROM model_token_reservations
            WHERE created_at >= ?
            """,
            (started_at,),
        )
        row = await cursor.fetchone()
        return int(row[0] or 0)

    async def release_model_token_reservation(self, run_id: str) -> None:
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                "DELETE FROM model_token_reservations WHERE run_id = ?", (run_id,)
            )
            await connection.commit()

    async def get_model_token_reservation(self, run_id: str) -> int | None:
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                "SELECT reserved_tokens FROM model_token_reservations WHERE run_id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()
            return int(row[0]) if row else None

    async def release_terminal_or_orphan_token_reservations(self) -> int:
        async with aiosqlite.connect(self.path) as connection:
            cursor = await connection.execute(
                """
                DELETE FROM model_token_reservations
                WHERE NOT EXISTS (
                    SELECT 1 FROM runs
                    WHERE runs.run_id = model_token_reservations.run_id
                    AND runs.status IN (?, ?)
                )
                """,
                (RunStatus.QUEUED.value, RunStatus.RUNNING.value),
            )
            await connection.commit()
            return max(cursor.rowcount, 0)

    async def sum_model_tokens_since(self, started_at: str) -> int:
        async with aiosqlite.connect(self.path) as connection:
            return await self._sum_model_tokens(connection, started_at)

    async def get_envelope(self, run_id: str) -> RunEnvelope | None:
        row = await self.get_run_row(run_id)
        if not row:
            return None
        if row["result_json"]:
            return RunEnvelope.model_validate_json(row["result_json"])
        return RunEnvelope(
            run_id=row["run_id"],
            status=RunStatus(row["status"]),
            question=row["question"],
            corpus_id=row["corpus_id"],
            corpus_version=row["corpus_version"],
            corpus_manifest_sha256=row["corpus_manifest_sha256"],
            claims=[],
            receipt_url=f"/api/runs/{run_id}/receipt",
            receipt_json_url=f"/api/runs/{run_id}/receipt.json",
        )

    async def append_event_row(
        self,
        *,
        run_id: str,
        sequence: int,
        event_type: str,
        occurred_at: str,
        payload: dict[str, Any],
        previous_hash: str,
        event_hash: str,
    ) -> None:
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                """
                INSERT INTO ledger_events (
                    run_id, sequence, event_type, occurred_at, payload_json,
                    previous_hash, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sequence,
                    event_type,
                    occurred_at,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    previous_hash,
                    event_hash,
                ),
            )
            await connection.commit()

    async def list_events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                """
                SELECT sequence, event_type, occurred_at, payload_json, previous_hash, event_hash
                FROM ledger_events WHERE run_id = ? AND sequence > ? ORDER BY sequence
                """,
                (run_id, after),
            )
            rows = await cursor.fetchall()
        return [
            {
                "sequence": row["sequence"],
                "type": row["event_type"],
                "occurred_at": row["occurred_at"],
                "payload": json.loads(row["payload_json"]),
                "previous_hash": row["previous_hash"],
                "event_hash": row["event_hash"],
            }
            for row in rows
        ]

    async def append_openai_call(self, run_id: str, sequence: int, record: dict[str, Any]) -> None:
        async with aiosqlite.connect(self.path) as connection:
            await connection.execute(
                "INSERT INTO openai_calls (run_id, sequence, record_json) VALUES (?, ?, ?)",
                (run_id, sequence, json.dumps(record, ensure_ascii=False, sort_keys=True)),
            )
            await connection.commit()

    async def list_openai_calls(self, run_id: str) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.path) as connection:
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute(
                "SELECT record_json FROM openai_calls WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            )
            rows = await cursor.fetchall()
        return [json.loads(row["record_json"]) for row in rows]
