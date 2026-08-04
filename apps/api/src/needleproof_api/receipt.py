from __future__ import annotations

import asyncio
import html
import json
import os
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import __version__
from .absence import ABSENCE_PROTOCOL_SHA256, ABSENCE_PROTOCOL_VERSION
from .binding import BINDING_CONTRACT_SHA256, NUMERIC_CONTRACT_SHA256
from .chunk_ids import ChunkId
from .config import Settings
from .corpus import load_current_manifest
from .db import AppDatabase
from .models import ClaimStatus, EvidenceRelation, LedgerEvent, RunEnvelope, VerifiedClaim
from .util import atomic_write_text, canonical_json, sha256_file, sha256_text, utc_now_iso


def _git_sha() -> str | None:
    if configured := os.getenv("NEEDLEPROOF_GIT_SHA"):
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


_SHA256_PATTERN = re.compile(r"[a-f0-9]{64}")
Sha256Digest: TypeAlias = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


def _optional_sha256_env(name: str) -> str | None:
    value = _optional_env(name)
    if value is None:
        return None
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


class ReceiptConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    reasoning_effort: str
    embedding_model: str
    embedding_dimensions: int
    embedding_l2_normalized: bool
    turbovec_version: str
    turbovec_bit_width: int
    retrieval_modes: list[str]
    parallel_tool_calls: bool
    max_turns: int
    max_tool_calls: int
    max_searches: int
    max_top_k: int
    max_opened_chunks: int
    max_opened_tokens: int
    max_document_inspections: int
    max_inspection_pages: int
    max_inspection_characters: int
    soft_timeout_seconds: float
    hard_timeout_seconds: float
    max_model_output_tokens_per_call: int = 8_000
    model_token_reservation_per_run: int = 100_000
    trace_include_sensitive_data: bool
    openai_client_max_retries: int
    tool_execution_concurrency: int


class _ReceiptProvenanceBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_version: str
    git_commit_sha: str | None
    dependency_lock_digests: dict[str, Sha256Digest]
    agent_instruction_hash: str
    tool_schema_hash: str
    verifier_version: str
    binding_contract_sha256: Sha256Digest
    numeric_contract_sha256: Sha256Digest
    absence_protocol_version: str
    absence_protocol_sha256: Sha256Digest
    image_revision: str | None
    trace_id: str | None
    sealed_at: str
    event_chain_head: Sha256Digest


class LiveReceiptProvenance(_ReceiptProvenanceBase):
    receipt_derivation: Literal["live"]
    source_receipt_sha256: None
    source_verifier_version: None


class MigratedReceiptProvenance(_ReceiptProvenanceBase):
    receipt_derivation: Literal["contract_migration"]
    source_receipt_sha256: Sha256Digest
    source_verifier_version: str = Field(min_length=1)


ReceiptProvenance: TypeAlias = Annotated[
    LiveReceiptProvenance | MigratedReceiptProvenance,
    Field(discriminator="receipt_derivation"),
]


class LegacyReceiptConfigurationV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    reasoning_effort: str
    embedding_model: str
    embedding_dimensions: int
    embedding_l2_normalized: bool
    turbovec_version: str
    turbovec_bit_width: int
    retrieval_modes: list[str]
    parallel_tool_calls: bool
    max_turns: int
    max_model_output_tokens_per_call: int | None = None
    model_token_reservation_per_run: int | None = None
    trace_include_sensitive_data: bool


class LegacyReceiptProvenanceV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_version: str
    git_commit_sha: str | None
    dependency_lock_digests: dict[str, Sha256Digest]
    agent_instruction_hash: str
    tool_schema_hash: str
    verifier_version: str
    trace_id: str | None
    sealed_at: str
    event_chain_head: Sha256Digest


class LegacyEvidenceReferenceV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: ChunkId
    metric_anchor: str = Field(min_length=1)
    exact_quote: str
    relation: EvidenceRelation


class LegacyReportedValueV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    temporal_anchor: str | None
    evidence: list[LegacyEvidenceReferenceV12] = Field(min_length=1)


class LegacyVerifiedEvidenceV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: ChunkId
    document_id: str
    document_name: str
    physical_page_index: int
    printed_page_label: str | None
    metric_anchor: str
    metric_anchor_found: bool
    temporal_anchors: list[str]
    temporal_anchors_found: bool
    quote: str
    normalized_quote: str
    normalization_operations: list[str]
    relation: EvidenceRelation
    quote_found: bool
    value_found: bool
    chunk_sha256: Sha256Digest
    source_url: str


class LegacyVerifiedClaimV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    metric: str
    status: ClaimStatus
    values: list[LegacyReportedValueV12]
    evidence: list[LegacyVerifiedEvidenceV12]
    verification_notes: list[str]


class ReceiptRehearsal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_run_id: str
    validated: Literal[True]
    reverified: Literal[True]
    source_verifier_version: str
    current_verifier_version: str
    version_drift: bool


class ReceiptEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    type: str
    occurred_at: str
    payload: dict[str, Any]
    previous_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReceiptOpenAICall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    operation: Literal["embedding", "model"]
    model: str
    response_id: str | None
    request_id: str | None
    started_at: str
    ended_at: str
    duration_ms: float = Field(ge=0)
    token_usage: dict[str, Any]
    retry_count: int = Field(ge=0)
    error: dict[str, Any] | None


class ReceiptContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.4"]
    run_id: str = Field(pattern=r"^run_[0-9a-f]{32}$")
    status: Literal["completed", "incomplete", "cancelled", "failed", "interrupted"]
    question: str
    answer: str | None
    corpus_id: str
    corpus_version: str = Field(pattern=r"^v_[0-9a-f]{16}$")
    corpus_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    claims: list[VerifiedClaim]
    events: list[ReceiptEvent]
    openai_calls: list[ReceiptOpenAICall]
    configuration: ReceiptConfiguration
    provenance: ReceiptProvenance
    error: dict[str, Any] | None
    rehearsal: ReceiptRehearsal | None
    receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class LegacyReceiptContractV12(BaseModel):
    """Exact read-only contract for receipts retained from the previous release."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.2"]
    run_id: str = Field(pattern=r"^run_[0-9a-f]{32}$")
    status: Literal["completed", "incomplete", "cancelled", "failed", "interrupted"]
    question: str
    answer: str | None
    corpus_id: str
    corpus_version: str = Field(pattern=r"^v_[0-9a-f]{16}$")
    corpus_manifest_sha256: Sha256Digest
    claims: list[LegacyVerifiedClaimV12]
    events: list[ReceiptEvent]
    openai_calls: list[ReceiptOpenAICall]
    configuration: LegacyReceiptConfigurationV12
    provenance: LegacyReceiptProvenanceV12
    error: dict[str, Any] | None
    rehearsal: ReceiptRehearsal | None
    receipt_sha256: Sha256Digest


def receipt_contract_schema() -> dict[str, Any]:
    schema = ReceiptContract.model_json_schema()
    schema["$id"] = "https://needleproof.local/contracts/receipt.schema.json"
    return schema


class RunLedger:
    def __init__(
        self,
        run_id: str,
        database: AppDatabase,
        settings: Settings,
        *,
        corpus_manifest: dict[str, Any] | None = None,
    ):
        self.run_id = run_id
        self.database = database
        self.settings = settings
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._previous_hash = "0" * 64
        self._openai_sequence = 0
        self._corpus_manifest = corpus_manifest

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

    async def reserve_model_call_capacity(self, call_token_ceiling: int) -> None:
        current = datetime.now(UTC)
        exceeded = await self.database.expand_model_token_reservation(
            run_id=self.run_id,
            call_token_ceiling=call_token_ceiling,
            hourly_cutoff=(current - timedelta(hours=1)).isoformat(),
            daily_cutoff=(current - timedelta(days=1)).isoformat(),
            hourly_limit=self.settings.max_model_tokens_per_hour,
            daily_limit=self.settings.max_model_tokens_per_day,
        )
        if exceeded:
            raise RuntimeError(
                f"The public {exceeded} model budget cannot safely fund another model call."
            )

    async def seal(
        self,
        envelope: RunEnvelope,
        *,
        instruction_hash: str,
        tool_schema_hash: str,
        verifier_version: str,
        trace_id: str | None,
        error: dict[str, Any] | None = None,
        rehearsal: dict[str, Any] | None = None,
    ) -> Path:
        events = await self.database.list_events(self.run_id)
        openai_calls = await self.database.list_openai_calls(self.run_id)
        corpus_manifest = self._corpus_manifest or load_current_manifest(self.settings)
        lock_digests = {}
        injected_locks = {
            "uv.lock": _optional_sha256_env("NEEDLEPROOF_UV_LOCK_SHA256"),
            "pnpm-lock.yaml": _optional_sha256_env("NEEDLEPROOF_PNPM_LOCK_SHA256"),
        }
        for lock_path in (Path("uv.lock"), Path("pnpm-lock.yaml")):
            if lock_path.exists():
                lock_digests[lock_path.name] = sha256_file(lock_path)
            elif digest := injected_locks[lock_path.name]:
                lock_digests[lock_path.name] = digest
        receipt: dict[str, Any] = {
            "schema_version": "1.4",
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
                "embedding_model": corpus_manifest["embedding_model"],
                "embedding_dimensions": corpus_manifest["embedding_dimensions"],
                "embedding_l2_normalized": corpus_manifest["embedding_l2_normalized"],
                "turbovec_version": corpus_manifest["turbovec_version"],
                "turbovec_bit_width": corpus_manifest["turbovec_bit_width"],
                "retrieval_modes": corpus_manifest["retrieval"],
                "parallel_tool_calls": False,
                "max_turns": self.settings.max_turns,
                "max_tool_calls": self.settings.max_tool_calls,
                "max_searches": self.settings.max_searches,
                "max_top_k": self.settings.max_top_k,
                "max_opened_chunks": self.settings.max_opened_chunks,
                "max_opened_tokens": self.settings.max_opened_tokens,
                "max_document_inspections": self.settings.max_document_inspections,
                "max_inspection_pages": self.settings.max_inspection_pages,
                "max_inspection_characters": self.settings.max_inspection_characters,
                "soft_timeout_seconds": self.settings.soft_timeout_seconds,
                "hard_timeout_seconds": self.settings.hard_timeout_seconds,
                "max_model_output_tokens_per_call": self.settings.max_model_output_tokens_per_call,
                "model_token_reservation_per_run": self.settings.model_token_reservation_per_run,
                "trace_include_sensitive_data": self.settings.trace_include_sensitive_data,
                "openai_client_max_retries": 0,
                "tool_execution_concurrency": 1,
            },
            "provenance": {
                "application_version": __version__,
                "git_commit_sha": _git_sha(),
                "dependency_lock_digests": lock_digests,
                "agent_instruction_hash": instruction_hash,
                "tool_schema_hash": tool_schema_hash,
                "verifier_version": verifier_version,
                "binding_contract_sha256": BINDING_CONTRACT_SHA256,
                "numeric_contract_sha256": NUMERIC_CONTRACT_SHA256,
                "absence_protocol_version": ABSENCE_PROTOCOL_VERSION,
                "absence_protocol_sha256": ABSENCE_PROTOCOL_SHA256,
                "receipt_derivation": "live",
                "source_receipt_sha256": None,
                "source_verifier_version": None,
                "image_revision": _optional_env("NEEDLEPROOF_IMAGE_REVISION"),
                "trace_id": trace_id,
                "sealed_at": utc_now_iso(),
                "event_chain_head": events[-1]["event_hash"] if events else "0" * 64,
            },
            "error": error,
            "rehearsal": rehearsal,
        }
        receipt_sha = sha256_text(canonical_json(receipt))
        receipt["receipt_sha256"] = receipt_sha
        path = self.settings.receipts_dir / f"{self.run_id}.json"
        if path.exists():
            raise RuntimeError(f"Receipt {self.run_id} is already sealed")
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
    answer_heading = (
        "Authoritative answer" if receipt.get("status") == "completed" else "Verified findings"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>NeedleProof receipt {escaped(receipt.get("run_id"))}</title>
<style>body{{font:16px/1.55 system-ui;max-width:920px;margin:3rem auto;padding:0 1.25rem;background:#f4f0e8;color:#171714}}article,blockquote{{border:1px solid #b9b2a5;padding:1rem;margin:1rem 0;background:#fffdf8}}code,.status{{font:12px ui-monospace;color:#0b6b50}}h1{{font-family:Georgia,serif}}dt{{font-weight:700}}dd{{margin:0 0 1rem}}</style>
</head><body><p>NeedleProof / sealed evidence receipt</p><h1>{escaped(receipt.get("question"))}</h1>
<dl><dt>Run</dt><dd>{escaped(receipt.get("run_id"))}</dd><dt>Corpus digest</dt><dd><code>{escaped(receipt.get("corpus_manifest_sha256"))}</code></dd><dt>Receipt digest</dt><dd><code>{escaped(receipt.get("receipt_sha256"))}</code></dd></dl>
<h2>{answer_heading}</h2><p>{escaped(receipt.get("answer"))}</p>{"".join(claim_cards)}
<h2>Execution ledger</h2><ol>{events}</ol></body></html>"""


def validate_receipt(receipt: object) -> list[str]:
    """Validate arbitrary decoded JSON without leaking shape exceptions."""

    if not isinstance(receipt, dict):
        return ["Receipt contract violation at root: input must be an object."]

    errors: list[str] = []
    schema_version = receipt.get("schema_version")
    contract: type[BaseModel]
    if schema_version == "1.4":
        contract = ReceiptContract
    elif schema_version == "1.2":
        contract = LegacyReceiptContractV12
    else:
        return [f"Receipt schema version {schema_version!r} is unsupported."]
    try:
        contract.model_validate(receipt)
    except ValidationError as exc:
        errors.extend(
            f"Receipt contract violation at {'.'.join(map(str, item['loc']))}: {item['msg']}"
            for item in exc.errors()
        )
    expected_digest = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    actual_digest = sha256_text(canonical_json(unsigned))
    if expected_digest != actual_digest:
        errors.append("Receipt digest does not match canonical content.")

    events = receipt.get("events")
    if not isinstance(events, list):
        errors.append("Receipt contract violation at events: input must be an array.")
        return errors

    previous_hash = "0" * 64
    for expected_sequence, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            errors.append("Receipt contract violation at events: every event must be an object.")
            break
        if event.get("sequence") != expected_sequence:
            errors.append("Receipt event sequences must be contiguous and start at 1.")
            break
        if event.get("previous_hash") != previous_hash:
            errors.append(f"Event {event.get('sequence')} has a broken previous-hash link.")
            break
        event_body = {
            "run_id": receipt.get("run_id"),
            "sequence": event.get("sequence"),
            "type": event.get("type"),
            "occurred_at": event.get("occurred_at"),
            "payload": event.get("payload", {}),
            "previous_hash": previous_hash,
        }
        event_hash = sha256_text(previous_hash + canonical_json(event_body))
        if event.get("event_hash") != event_hash:
            errors.append(f"Event {event.get('sequence')} digest does not match its content.")
            break
        previous_hash = event_hash

    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("Receipt contract violation at provenance: input must be an object.")
    elif provenance.get("event_chain_head") != previous_hash:
        errors.append("Provenance event-chain head does not match the final ledger event.")

    openai_calls = receipt.get("openai_calls")
    if isinstance(openai_calls, list) and any(
        not isinstance(call, dict) or call.get("sequence") != sequence
        for sequence, call in enumerate(openai_calls, start=1)
    ):
        errors.append("OpenAI call sequences must be contiguous and start at 1.")

    status = receipt.get("status")
    expected_terminal_events = {
        "completed": {"run.completed"},
        "incomplete": {"run.incomplete", "run.timeout"},
        "cancelled": {"run.cancelled"},
        "failed": {"run.failed"},
        "interrupted": {"run.interrupted"},
    }.get(status)
    final_event = events[-1] if events else None
    if expected_terminal_events and (
        not isinstance(final_event, dict) or final_event.get("type") not in expected_terminal_events
    ):
        errors.append("Receipt status does not agree with its final terminal event.")
    claims = receipt.get("claims")
    if status == "completed":
        answer = receipt.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            errors.append("A completed receipt requires a nonempty authoritative answer.")
        if (
            not isinstance(claims, list)
            or not claims
            or any(
                not isinstance(claim, dict)
                or claim.get("status") not in {"verified", "conflict", "date_variant", "not_found"}
                for claim in claims
            )
        ):
            errors.append("A completed receipt requires only authoritative claims.")
    corpus_version = receipt.get("corpus_version")
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_evidence = claim.get("evidence")
            if not isinstance(claim_evidence, list):
                continue
            for evidence in claim_evidence:
                if not isinstance(evidence, dict):
                    continue
                if not evidence.get("chunk_sha256"):
                    errors.append("Every receipt evidence item requires a chunk digest.")
                if (
                    schema_version == "1.4"
                    and corpus_version
                    and f"/api/corpora/{corpus_version}/" not in str(evidence.get("source_url", ""))
                ):
                    errors.append("Every evidence source URL must bind the receipt corpus version.")
    return errors
