from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .absence import probe_metric_absence
from .agent import (
    INSTRUCTION_HASH,
    TOOL_SCHEMA_HASH,
    InvestigationContext,
    investigate,
)
from .config import Settings
from .db import AppDatabase
from .legacy import adapt_legacy_run_envelope
from .models import (
    ClaimStatus,
    DraftClaim,
    DraftObservation,
    EvidenceReference,
    RunCreateRequest,
    RunCreateResponse,
    RunEnvelope,
    RunStatus,
    VerifiedClaim,
)
from .receipt import (
    RunLedger,
    SealedReceiptSnapshot,
    load_sealed_receipt_snapshot,
    validate_receipt,
)
from .retrieval import CorpusStore
from .security import PublicUsageLimiter
from .util import canonical_metric_key, new_run_id, utc_now_iso
from .verification import EvidenceVerifier

VERIFIER_VERSION = EvidenceVerifier.version
logger = logging.getLogger(__name__)
TERMINAL_EVENT_TYPES = {
    "run.completed",
    "run.incomplete",
    "run.cancelled",
    "run.failed",
    "run.timeout",
    "run.interrupted",
}


def _metric_key(value: str) -> str:
    return canonical_metric_key(value)


@dataclass(slots=True)
class TerminalState:
    envelope: RunEnvelope
    event_type: str
    event_payload: dict[str, Any]
    error: dict[str, Any] | None = None
    trace_id: str | None = None
    rehearsal_metadata: dict[str, Any] | None = None


class ReceiptRecoveryOutcome(StrEnum):
    RECOVERED = "recovered"
    RETRYABLE = "retryable"
    UNRECOVERABLE = "unrecoverable"
    NOT_PENDING = "not_pending"


def _run_envelope_from_receipt(run_id: str, receipt: dict[str, Any]) -> RunEnvelope:
    envelope_data = {
        "run_id": run_id,
        "status": receipt["status"],
        "question": receipt["question"],
        "answer": receipt.get("answer"),
        "corpus_id": receipt["corpus_id"],
        "corpus_version": receipt["corpus_version"],
        "corpus_manifest_sha256": receipt["corpus_manifest_sha256"],
        "claims": receipt.get("claims", []),
        "receipt_url": f"/api/runs/{run_id}/receipt",
        "receipt_json_url": f"/api/runs/{run_id}/receipt.json",
    }
    if receipt.get("schema_version") == "1.2":
        return adapt_legacy_run_envelope(envelope_data)
    return RunEnvelope.model_validate(envelope_data)


class RunCapacityError(RuntimeError):
    """Raised before persistence when all investigation slots are occupied."""


class SessionLiveRunError(RuntimeError):
    """Raised when a browser already owns an active live investigation."""


def compose_authoritative_answer(
    claims: list[VerifiedClaim], searches: int, corpus_version: str
) -> str | None:
    accepted = [
        claim
        for claim in claims
        if claim.status
        in {
            ClaimStatus.VERIFIED,
            ClaimStatus.CONFLICT,
            ClaimStatus.DATE_VARIANT,
            ClaimStatus.NOT_FOUND,
        }
    ]
    if not accepted:
        return None

    sentences: list[str] = []
    for claim in accepted:
        statement = claim.statement.strip().rstrip(".")
        if claim.status == ClaimStatus.NOT_FOUND:
            bounded_search_note = next(
                (
                    note.rstrip(".")
                    for note in claim.verification_notes
                    if note.startswith("Not found after ")
                ),
                f"Not found after {searches} searches across corpus version {corpus_version}",
            )
            sentences.append(f"{statement}. {bounded_search_note}.")
        elif claim.status == ClaimStatus.CONFLICT:
            sentences.append(f"{statement}. The verified sources contain a conflict.")
        elif claim.status == ClaimStatus.DATE_VARIANT:
            sentences.append(f"{statement}. These are date variants, not a conflict.")
        else:
            sentences.append(f"{statement}.")
    return " ".join(sentences)


class InvestigationService:
    def __init__(
        self,
        settings: Settings,
        database: AppDatabase,
        corpus: CorpusStore,
        usage_limiter: PublicUsageLimiter | None = None,
    ):
        self.settings = settings
        self.database = database
        self.corpus = corpus
        self.usage_limiter = usage_limiter
        self.verifier = EvidenceVerifier(corpus)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_runs)
        self._admission_lock = asyncio.Lock()
        self._reconciliation_lock = asyncio.Lock()
        self._admitted_run_ids: set[str] = set()
        self._live_session_by_run: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def create_run(
        self, request: RunCreateRequest, *, session_id: str, client_ip: str = "unknown"
    ) -> RunCreateResponse:
        run_id = new_run_id()
        run_persisted = False
        async with self._admission_lock:
            if len(self._admitted_run_ids) >= self.settings.max_concurrent_runs:
                raise RunCapacityError(
                    "NeedleProof is at investigation capacity. Retry in a moment."
                )
            if not request.rehearsal and session_id in self._live_session_by_run.values():
                raise SessionLiveRunError("This browser already has an active live investigation.")
            self._admitted_run_ids.add(run_id)
            if not request.rehearsal:
                self._live_session_by_run[run_id] = session_id
        try:
            if self.usage_limiter:
                await self.usage_limiter.admit(
                    run_id=run_id,
                    session_id=session_id,
                    client_ip=client_ip,
                    rehearsal=request.rehearsal,
                )
            await self.database.create_run(
                run_id=run_id,
                session_id=session_id,
                rehearsal=request.rehearsal,
                question=request.question.strip(),
                corpus_id=self.corpus.corpus_id,
                corpus_version=self.corpus.corpus_version,
                manifest_sha256=self.corpus.manifest_sha256,
            )
            run_persisted = True
            task = asyncio.create_task(self._run_guarded(run_id, request, session_id))
        except BaseException:
            if self.usage_limiter:
                with suppress(Exception):
                    if run_persisted:
                        await self.usage_limiter.release(run_id)
                    else:
                        await self.usage_limiter.rollback(run_id)
            async with self._admission_lock:
                self._admitted_run_ids.discard(run_id)
                self._live_session_by_run.pop(run_id, None)
            raise
        self._tasks[run_id] = task
        task.add_done_callback(lambda completed: self._finish_task(run_id, completed))
        # Let the guarded coroutine enter its try/finally before the run ID can be
        # returned and immediately cancelled by a client.
        await asyncio.sleep(0)
        return RunCreateResponse(
            run_id=run_id,
            status=RunStatus.QUEUED,
            events_url=f"/api/runs/{run_id}/events",
            run_url=f"/api/runs/{run_id}",
        )

    def _finish_task(self, run_id: str, task: asyncio.Task[None]) -> None:
        with suppress(asyncio.CancelledError):
            task.exception()
        self._tasks.pop(run_id, None)
        self._admitted_run_ids.discard(run_id)
        self._live_session_by_run.pop(run_id, None)

    async def cancel(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def shutdown(self, grace_seconds: float = 10.0) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=grace_seconds)
        for task in done:
            with suppress(asyncio.CancelledError):
                task.exception()
        for task in pending:
            # Finalization is cancellation-shielded and may finish after the grace period.
            task.add_done_callback(
                lambda completed: completed.exception() if not completed.cancelled() else None
            )

    async def _recover_sealed_receipt(self, row: dict[str, object]) -> ReceiptRecoveryOutcome:
        run_id = str(row["run_id"])
        receipt_path = self.settings.receipts_dir / f"{run_id}.json"
        if not receipt_path.exists():
            return ReceiptRecoveryOutcome.UNRECOVERABLE
        try:
            snapshot = load_sealed_receipt_snapshot(receipt_path)
        except OSError:
            logger.exception("Could not read sealed receipt for run %s", run_id)
            return ReceiptRecoveryOutcome.RETRYABLE
        except (AttributeError, json.JSONDecodeError, TypeError, UnicodeError, ValueError):
            receipt_path.unlink(missing_ok=True)
            return ReceiptRecoveryOutcome.UNRECOVERABLE
        try:
            receipt = snapshot.decode()
            expected_identity = {
                "run_id": run_id,
                "corpus_id": str(row["corpus_id"]),
                "corpus_version": str(row["corpus_version"]),
                "corpus_manifest_sha256": str(row["corpus_manifest_sha256"]),
            }
            if any(receipt.get(key) != value for key, value in expected_identity.items()):
                raise ValueError("Sealed receipt does not belong to this persisted run")
            expected_receipt_sha256 = row.get("receipt_sha256")
            if (
                expected_receipt_sha256 is not None
                and receipt.get("receipt_sha256") != expected_receipt_sha256
            ):
                raise ValueError("Sealed receipt digest does not match the persisted receipt")
        except (AttributeError, KeyError, TypeError, ValueError):
            receipt_path.unlink(missing_ok=True)
            return ReceiptRecoveryOutcome.UNRECOVERABLE
        try:
            await self._commit_terminal(snapshot)
        except Exception:
            logger.exception("Could not persist validated receipt recovery for run %s", run_id)
            return ReceiptRecoveryOutcome.RETRYABLE
        return ReceiptRecoveryOutcome.RECOVERED

    async def reconcile_pending_receipt(self, run_id: str) -> ReceiptRecoveryOutcome:
        async with self._reconciliation_lock:
            row = await self.database.get_run_row(run_id)
            if not row:
                return ReceiptRecoveryOutcome.UNRECOVERABLE
            if row.get("status") != RunStatus.INTERRUPTED.value:
                return ReceiptRecoveryOutcome.NOT_PENDING
            return await self._recover_sealed_receipt(row)

    async def reconcile_abandoned_runs(self) -> None:
        for row in await self.database.list_recoverable_runs():
            run_id = str(row["run_id"])
            receipt_recovery = await self._recover_sealed_receipt(row)
            if receipt_recovery in {
                ReceiptRecoveryOutcome.RECOVERED,
                ReceiptRecoveryOutcome.RETRYABLE,
            }:
                continue

            bound_corpus = (
                self.corpus
                if row["corpus_version"] == self.corpus.corpus_version
                else CorpusStore(self.settings, str(row["corpus_version"]))
            )
            ledger = RunLedger(
                run_id,
                self.database,
                self.settings,
                corpus_manifest=bound_corpus.manifest,
            )
            await ledger.hydrate()
            error = {
                "type": "interrupted",
                "message": "The server restarted before this investigation reached a terminal state.",
            }
            state = TerminalState(
                envelope=RunEnvelope(
                    run_id=run_id,
                    status=RunStatus.INTERRUPTED,
                    question=str(row["question"]),
                    answer=None,
                    corpus_id=str(row["corpus_id"]),
                    corpus_version=str(row["corpus_version"]),
                    corpus_manifest_sha256=str(row["corpus_manifest_sha256"]),
                    claims=[],
                    receipt_url=f"/api/runs/{run_id}/receipt",
                    receipt_json_url=f"/api/runs/{run_id}/receipt.json",
                ),
                event_type="run.interrupted",
                event_payload={**error, "authoritative": False, "recovered": True},
                error=error,
            )
            await self._finalize_run(ledger, state)
        await self.database.release_terminal_or_orphan_token_reservations()

    async def _soft_timeout(self, ledger: RunLedger) -> None:
        await asyncio.sleep(self.settings.soft_timeout_seconds)
        await ledger.append(
            "run.soft_timeout",
            {
                "seconds": self.settings.soft_timeout_seconds,
                "message": "Sealing soon with available evidence.",
            },
        )

    async def _run_guarded(self, run_id: str, request: RunCreateRequest, session_id: str) -> None:
        ledger = RunLedger(
            run_id,
            self.database,
            self.settings,
            corpus_manifest=self.corpus.manifest,
        )
        state: TerminalState | None = None
        soft_timer: asyncio.Task[None] | None = None
        try:
            await ledger.hydrate()
            async with self._semaphore:
                await self.database.update_run(
                    run_id, status=RunStatus.RUNNING, started_at=utc_now_iso()
                )
                await ledger.append(
                    "run.started",
                    {
                        "question": request.question,
                        "corpus_version": self.corpus.corpus_version,
                        "corpus_manifest_sha256": self.corpus.manifest_sha256,
                        "rehearsal": request.rehearsal,
                    },
                )
                if request.rehearsal:
                    state = await self._run_rehearsal(run_id, request, ledger)
                else:
                    soft_timer = asyncio.create_task(self._soft_timeout(ledger))
                    async with asyncio.timeout(self.settings.hard_timeout_seconds):
                        state = await self._run_live(run_id, request, session_id, ledger)
        except asyncio.CancelledError:
            error = {"type": "cancelled", "message": "The investigation was cancelled."}
            state = TerminalState(
                envelope=self._envelope(
                    run_id,
                    request.question,
                    status=RunStatus.CANCELLED,
                    answer=None,
                    claims=[],
                ),
                event_type="run.cancelled",
                event_payload={**error, "authoritative": False},
                error=error,
            )
        except TimeoutError:
            error = {
                "type": "timeout",
                "message": f"Investigation reached the {self.settings.hard_timeout_seconds:g}-second limit.",
            }
            state = TerminalState(
                envelope=self._envelope(
                    run_id,
                    request.question,
                    status=RunStatus.INCOMPLETE,
                    answer=None,
                    claims=[],
                ),
                event_type="run.timeout",
                event_payload={**error, "authoritative": False},
                error=error,
            )
        except BaseException as exc:  # noqa: BLE001 - finalization is the outer boundary
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            state = TerminalState(
                envelope=self._envelope(
                    run_id,
                    request.question,
                    status=RunStatus.FAILED,
                    answer=None,
                    claims=[],
                ),
                event_type="run.failed",
                event_payload={**error, "authoritative": False},
                error=error,
            )
        finally:
            if soft_timer:
                soft_timer.cancel()
                with suppress(asyncio.CancelledError):
                    await soft_timer
            if state is None:
                error = {
                    "type": "failed",
                    "message": "Investigation exited without a terminal state.",
                }
                state = TerminalState(
                    envelope=self._envelope(
                        run_id,
                        request.question,
                        status=RunStatus.FAILED,
                        answer=None,
                        claims=[],
                    ),
                    event_type="run.failed",
                    event_payload={**error, "authoritative": False},
                    error=error,
                )
            finalizer = asyncio.create_task(self._finalize_run(ledger, state))
            try:
                await asyncio.shield(finalizer)
            except asyncio.CancelledError:
                await finalizer

    async def _run_live(
        self,
        run_id: str,
        request: RunCreateRequest,
        session_id: str,
        ledger: RunLedger,
    ) -> TerminalState:
        context = InvestigationContext(
            run_id=run_id,
            corpus=self.corpus,
            verifier=self.verifier,
            ledger=ledger,
            settings=self.settings,
        )
        outcome = await investigate(
            request.question,
            context=context,
            session_id=f"{session_id}:{run_id}",
        )
        await ledger.append("verification.authoritative_started", {})
        absence_probes = {}
        for claim in outcome.draft.claims:
            if not claim.request_absence_probe:
                continue
            await ledger.append("absence_probe.started", {"metric": claim.metric})
            probe = await probe_metric_absence(
                self.corpus,
                claim.metric,
                top_k=8,
                recorder=ledger.record_openai_call,
            )
            absence_probes[_metric_key(claim.metric)] = probe
            await ledger.append(
                "absence_probe.completed",
                probe.model_dump(mode="json"),
            )
        verification = self.verifier.verify_claims(
            outcome.draft.claims,
            absence_probes=absence_probes,
        )
        claims = verification.claims
        for claim in claims:
            await ledger.append(
                f"claim.{claim.status.value}",
                {
                    "statement": claim.statement,
                    "evidence_count": len(claim.evidence),
                    "verification_notes": claim.verification_notes,
                },
            )
        answer = compose_authoritative_answer(claims, context.searches, self.corpus.corpus_version)
        status = (
            RunStatus.COMPLETED
            if answer and verification.all_claims_authoritative
            else RunStatus.INCOMPLETE
        )
        return TerminalState(
            envelope=self._envelope(
                run_id, request.question, status=status, answer=answer, claims=claims
            ),
            event_type=("run.completed" if status == RunStatus.COMPLETED else "run.incomplete"),
            event_payload={
                "status": status.value,
                "authoritative": status == RunStatus.COMPLETED,
                "verified_partial_answer_available": bool(answer),
            },
            trace_id=outcome.trace_id,
        )

    async def _run_rehearsal(
        self, run_id: str, request: RunCreateRequest, ledger: RunLedger
    ) -> TerminalState:
        source = self.settings.rehearsal_path
        if not source.exists():
            raise FileNotFoundError(f"Missing rehearsal receipt {source}")
        receipt = json.loads(source.read_text(encoding="utf-8"))
        validation_errors = validate_receipt(receipt)
        if validation_errors:
            raise ValueError(
                "Rehearsal receipt failed integrity validation: " + "; ".join(validation_errors)
            )
        if receipt.get("schema_version") != "1.4":
            raise ValueError("Rehearsal receipt uses an unsupported schema version")
        if receipt.get("status") != RunStatus.COMPLETED.value:
            raise ValueError("Rehearsal source receipt must be completed")
        expected_corpus = (
            self.corpus.corpus_id,
            self.corpus.corpus_version,
            self.corpus.manifest_sha256,
        )
        source_corpus = (
            receipt.get("corpus_id"),
            receipt.get("corpus_version"),
            receipt.get("corpus_manifest_sha256"),
        )
        if source_corpus != expected_corpus:
            raise ValueError("Rehearsal receipt does not match the loaded corpus snapshot")

        draft_claims: list[DraftClaim] = []
        source_statuses: list[str] = []
        authoritative_statuses = {
            ClaimStatus.VERIFIED.value,
            ClaimStatus.CONFLICT.value,
            ClaimStatus.DATE_VARIANT.value,
            ClaimStatus.NOT_FOUND.value,
        }
        for source_claim in receipt.get("claims", []):
            source_status = str(source_claim.get("status"))
            if source_status not in authoritative_statuses:
                raise ValueError(
                    f"Rehearsal source contains non-authoritative claim {source_status!r}"
                )
            source_observations = source_claim.get("observations")
            if not isinstance(source_observations, list):
                raise TypeError("Rehearsal source claim has no typed observations")
            observations = [
                DraftObservation.model_validate(observation) for observation in source_observations
            ]
            source_context_evidence = source_claim.get("context_evidence", [])
            if not isinstance(source_context_evidence, list):
                raise TypeError("Rehearsal source claim has invalid context evidence")
            context_evidence = [
                EvidenceReference.model_validate(reference) for reference in source_context_evidence
            ]
            draft_claims.append(
                DraftClaim(
                    metric=source_claim["metric"],
                    observations=observations,
                    context_evidence=context_evidence,
                    request_absence_probe=source_status == ClaimStatus.NOT_FOUND.value,
                )
            )
            source_statuses.append(source_status)

        absence_probes = {}
        for claim in draft_claims:
            if claim.request_absence_probe:
                await ledger.append(
                    "absence_probe.started",
                    {"metric": claim.metric, "rehearsal_reverification": True},
                )
                probe = await probe_metric_absence(
                    self.corpus,
                    claim.metric,
                    top_k=8,
                    recorder=ledger.record_openai_call,
                )
                absence_probes[_metric_key(claim.metric)] = probe
                await ledger.append(
                    "absence_probe.completed",
                    {
                        **probe.model_dump(mode="json"),
                        "rehearsal_reverification": True,
                    },
                )
        verification = self.verifier.verify_claims(
            draft_claims,
            absence_probes=absence_probes,
        )
        if not verification.all_claims_authoritative:
            raise ValueError("Rehearsal evidence failed current deterministic verification")
        if [claim.status.value for claim in verification.claims] != source_statuses:
            raise ValueError("Rehearsal claim classifications changed during reverification")

        source_verifier = receipt.get("provenance", {}).get("verifier_version")
        version_drift = source_verifier != VERIFIER_VERSION
        await ledger.append(
            "rehearsal.validated",
            {
                "source_run_id": receipt.get("run_id"),
                "source_receipt_sha256": receipt.get("receipt_sha256"),
                "validated": True,
                "reverified": True,
                "verifier_version_drift": version_drift,
            },
        )
        for event in receipt.get("events", []):
            if event.get("type") in {"run.started", "run.completed"}:
                continue
            await ledger.append(event["type"], {**event.get("payload", {}), "replayed": True})
        answer = compose_authoritative_answer(
            verification.claims,
            sum(
                event.get("type") == "tool.search.completed" for event in receipt.get("events", [])
            ),
            self.corpus.corpus_version,
        )
        envelope = self._envelope(
            run_id,
            receipt.get("question", request.question),
            status=RunStatus.COMPLETED,
            answer=answer,
            claims=verification.claims,
        )
        return TerminalState(
            envelope=envelope,
            event_type="run.completed",
            event_payload={
                "status": "completed",
                "rehearsal": True,
                "source_receipt_sha256": receipt.get("receipt_sha256"),
            },
            trace_id=receipt.get("provenance", {}).get("trace_id"),
            rehearsal_metadata={
                "source_receipt_sha256": receipt.get("receipt_sha256"),
                "source_run_id": receipt.get("run_id"),
                "validated": True,
                "reverified": True,
                "source_verifier_version": source_verifier,
                "current_verifier_version": VERIFIER_VERSION,
                "version_drift": version_drift,
            },
        )

    async def _finalize_run(self, ledger: RunLedger, state: TerminalState) -> None:
        try:
            receipt_path = self.settings.receipts_dir / f"{ledger.run_id}.json"
            snapshot: SealedReceiptSnapshot | None = None
            try:
                events = await self.database.list_events(ledger.run_id)
                if not any(event["type"] == state.event_type for event in events):
                    await ledger.append(state.event_type, state.event_payload)
                if not receipt_path.exists():
                    snapshot = await ledger.seal(
                        state.envelope,
                        instruction_hash=INSTRUCTION_HASH,
                        tool_schema_hash=TOOL_SCHEMA_HASH,
                        verifier_version=VERIFIER_VERSION,
                        trace_id=state.trace_id,
                        error=state.error,
                        rehearsal=state.rehearsal_metadata,
                    )
                    receipt_path = snapshot.path
                else:
                    snapshot = load_sealed_receipt_snapshot(receipt_path)
                await self._commit_terminal(snapshot)
            except Exception as exc:  # noqa: BLE001 - persist a recoverable terminal state
                error = {
                    "type": "finalization_interrupted",
                    "message": str(exc)[:500],
                    "intended_status": state.envelope.status.value,
                }
                interrupted = self._envelope(
                    ledger.run_id,
                    state.envelope.question,
                    status=RunStatus.INTERRUPTED,
                    answer=None,
                    claims=[],
                )
                with suppress(Exception):
                    await ledger.append(
                        "run.interrupted",
                        {**error, "authoritative": False, "recoverable": True},
                    )
                receipt_sha256 = None
                persisted_receipt_path = None
                if snapshot is not None:
                    receipt_sha256 = snapshot.receipt_sha256
                    persisted_receipt_path = str(snapshot.path)
                elif receipt_path.exists():
                    with suppress(Exception):
                        recovered_snapshot = load_sealed_receipt_snapshot(receipt_path)
                        receipt_sha256 = recovered_snapshot.receipt_sha256
                        persisted_receipt_path = str(recovered_snapshot.path)
                await self.database.update_run(
                    ledger.run_id,
                    status=RunStatus.INTERRUPTED,
                    completed_at=utc_now_iso(),
                    answer=None,
                    result_json=interrupted.model_dump_json(),
                    receipt_path=persisted_receipt_path,
                    receipt_sha256=receipt_sha256,
                    error_json=json.dumps(error),
                )
        finally:
            if self.usage_limiter:
                with suppress(Exception):
                    await self.usage_limiter.release(ledger.run_id)

    async def _commit_terminal(
        self,
        snapshot: SealedReceiptSnapshot,
    ) -> None:
        receipt = snapshot.decode()
        envelope = _run_envelope_from_receipt(str(receipt["run_id"]), receipt)
        error = receipt.get("error")
        await self.database.update_run(
            envelope.run_id,
            status=envelope.status,
            completed_at=utc_now_iso(),
            answer=envelope.answer,
            result_json=envelope.model_dump_json(),
            receipt_path=str(snapshot.path),
            receipt_sha256=snapshot.receipt_sha256,
            error_json=json.dumps(error) if error else None,
        )

    def _envelope(
        self,
        run_id: str,
        question: str,
        *,
        status: RunStatus,
        answer: str | None,
        claims: list[VerifiedClaim],
    ) -> RunEnvelope:
        return RunEnvelope(
            run_id=run_id,
            status=status,
            question=question,
            answer=answer,
            corpus_id=self.corpus.corpus_id,
            corpus_version=self.corpus.corpus_version,
            corpus_manifest_sha256=self.corpus.manifest_sha256,
            claims=claims,
            receipt_url=f"/api/runs/{run_id}/receipt",
            receipt_json_url=f"/api/runs/{run_id}/receipt.json",
        )
