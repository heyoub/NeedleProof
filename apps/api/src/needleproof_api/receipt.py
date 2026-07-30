from __future__ import annotations

import asyncio
import html
import json
import subprocess
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings
from .db import AppDatabase
from .models import LedgerEvent, RunEnvelope
from .util import atomic_write_text, canonical_json, sha256_file, sha256_text, utc_now_iso


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


class RunLedger:
    def __init__(self, run_id: str, database: AppDatabase, settings: Settings):
        self.run_id = run_id
        self.database = database
        self.settings = settings
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._previous_hash = "0" * 64
        self._openai_sequence = 0

    async def hydrate(self) -> None:
        events = await self.database.list_events(self.run_id)
        if events:
            self._sequence = int(events[-1]["sequence"])
            self._previous_hash = str(events[-1]["event_hash"])
        calls = await self.database.list_openai_calls(self.run_id)
        self._openai_sequence = len(calls)

    async def append(self, event_type: str, payload: dict[str, Any] | None = None) -> LedgerEvent:
        async with self._lock:
            self._sequence += 1
            occurred_at = utc_now_iso()
            event_body = {
                "run_id": self.run_id,
                "sequence": self._sequence,
                "type": event_type,
                "occurred_at": occurred_at,
                "payload": payload or {},
                "previous_hash": self._previous_hash,
            }
            event_hash = sha256_text(self._previous_hash + canonical_json(event_body))
            event = LedgerEvent(**event_body, event_hash=event_hash)
            await self.database.append_event_row(
                run_id=self.run_id,
                sequence=event.sequence,
                event_type=event.type,
                occurred_at=event.occurred_at,
                payload=event.payload,
                previous_hash=event.previous_hash,
                event_hash=event.event_hash,
            )
            self._previous_hash = event_hash
            return event

    async def record_openai_call(self, record: dict[str, Any]) -> None:
        async with self._lock:
            self._openai_sequence += 1
            payload = {"sequence": self._openai_sequence, **record}
            await self.database.append_openai_call(self.run_id, self._openai_sequence, payload)

    async def seal(
        self,
        envelope: RunEnvelope,
        *,
        instruction_hash: str,
        tool_schema_hash: str,
        verifier_version: str,
        trace_id: str | None,
        error: dict[str, Any] | None = None,
    ) -> Path:
        events = await self.database.list_events(self.run_id)
        openai_calls = await self.database.list_openai_calls(self.run_id)
        lock_digests = {}
        for lock_path in (Path("uv.lock"), Path("pnpm-lock.yaml")):
            if lock_path.exists():
                lock_digests[lock_path.name] = sha256_file(lock_path)
        receipt: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": envelope.run_id,
            "status": envelope.status.value,
            "question": envelope.question,
            "answer": envelope.answer,
            "corpus_id": envelope.corpus_id,
            "corpus_version": envelope.corpus_version,
            "corpus_manifest_sha256": envelope.corpus_manifest_sha256,
            "claims": [claim.model_dump(mode="json") for claim in envelope.claims],
            "events": events,
            "openai_calls": openai_calls,
            "configuration": {
                "model": self.settings.model,
                "reasoning_effort": self.settings.reasoning_effort,
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "parallel_tool_calls": False,
                "max_turns": self.settings.max_turns,
                "trace_include_sensitive_data": self.settings.trace_include_sensitive_data,
            },
            "provenance": {
                "application_version": __version__,
                "git_commit_sha": _git_sha(),
                "dependency_lock_digests": lock_digests,
                "agent_instruction_hash": instruction_hash,
                "tool_schema_hash": tool_schema_hash,
                "verifier_version": verifier_version,
                "trace_id": trace_id,
                "sealed_at": utc_now_iso(),
                "event_chain_head": events[-1]["event_hash"] if events else "0" * 64,
            },
            "error": error,
        }
        receipt_sha = sha256_text(canonical_json(receipt))
        receipt["receipt_sha256"] = receipt_sha
        path = self.settings.receipts_dir / f"{self.run_id}.json"
        atomic_write_text(path, json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        return path


def receipt_html(receipt: dict[str, Any]) -> str:
    def escaped(value: Any) -> str:
        return html.escape(str(value))

    claim_cards = []
    for claim in receipt.get("claims", []):
        evidence = "".join(
            f"<blockquote><p>{escaped(item.get('quote', ''))}</p>"
            f"<footer>{escaped(item.get('document_name', ''))}, page "
            f"{escaped(item.get('printed_page_label') or item.get('physical_page_index'))}</footer></blockquote>"
            for item in claim.get("evidence", [])
        )
        claim_cards.append(
            f"<article><span class='status'>{escaped(claim.get('status'))}</span>"
            f"<h2>{escaped(claim.get('statement'))}</h2>{evidence}</article>"
        )
    events = "".join(
        f"<li><code>{escaped(event.get('sequence'))}</code> {escaped(event.get('type'))}</li>"
        for event in receipt.get("events", [])
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>NeedleProof receipt {escaped(receipt.get("run_id"))}</title>
<style>body{{font:16px/1.55 system-ui;max-width:920px;margin:3rem auto;padding:0 1.25rem;background:#f4f0e8;color:#171714}}article,blockquote{{border:1px solid #b9b2a5;padding:1rem;margin:1rem 0;background:#fffdf8}}code,.status{{font:12px ui-monospace;color:#0b6b50}}h1{{font-family:Georgia,serif}}dt{{font-weight:700}}dd{{margin:0 0 1rem}}</style>
</head><body><p>NeedleProof / sealed evidence receipt</p><h1>{escaped(receipt.get("question"))}</h1>
<dl><dt>Run</dt><dd>{escaped(receipt.get("run_id"))}</dd><dt>Corpus digest</dt><dd><code>{escaped(receipt.get("corpus_manifest_sha256"))}</code></dd><dt>Receipt digest</dt><dd><code>{escaped(receipt.get("receipt_sha256"))}</code></dd></dl>
<h2>Authoritative answer</h2><p>{escaped(receipt.get("answer"))}</p>{"".join(claim_cards)}
<h2>Execution ledger</h2><ol>{events}</ol></body></html>"""
