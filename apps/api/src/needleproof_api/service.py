from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from .agent import (
    INSTRUCTION_HASH,
    TOOL_SCHEMA_HASH,
    InvestigationContext,
    investigate,
)
from .config import Settings
from .db import AppDatabase
from .models import (
    ClaimStatus,
    RunCreateRequest,
    RunCreateResponse,
    RunEnvelope,
    RunStatus,
    VerifiedClaim,
)
from .receipt import RunLedger
from .retrieval import CorpusStore
from .util import new_run_id, utc_now_iso
from .verification import EvidenceVerifier

VERIFIER_VERSION = "deterministic-v1"


def compose_authoritative_answer(claims: list[VerifiedClaim], searches: int) -> str | None:
    accepted = [claim for claim in claims if claim.status != ClaimStatus.UNVERIFIED]
    if not accepted:
        return None

    sentences: list[str] = []
    for claim in accepted:
        statement = claim.statement.strip().rstrip(".")
        if claim.status == ClaimStatus.NOT_FOUND:
            sentences.append(f"{statement}. Not found after {searches} searches.")
        elif claim.status == ClaimStatus.CONFLICT:
            sentences.append(f"{statement}. The verified sources contain a conflict.")
        elif claim.status == ClaimStatus.DATE_VARIANT:
            sentences.append(f"{statement}. These are date variants, not a conflict.")
        else:
            sentences.append(f"{statement}.")
    return " ".join(sentences)


class InvestigationService:
    def __init__(self, settings: Settings, database: AppDatabase, corpus: CorpusStore):
        self.settings = settings
        self.database = database
        self.corpus = corpus
        self.verifier = EvidenceVerifier(corpus)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_runs)
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def create_run(self, request: RunCreateRequest) -> RunCreateResponse:
        run_id = new_run_id()
        await self.database.create_run(
            run_id=run_id,
            session_id=request.session_id,
            question=request.question.strip(),
            corpus_id=self.corpus.corpus_id,
            corpus_version=self.corpus.corpus_version,
            manifest_sha256=self.corpus.manifest_sha256,
        )
        task = asyncio.create_task(
            self._replay(run_id, request) if request.rehearsal else self._execute(run_id, request)
        )
        self._tasks[run_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(run_id, None))
        return RunCreateResponse(
            run_id=run_id,
            status=RunStatus.QUEUED,
            events_url=f"/api/runs/{run_id}/events",
            run_url=f"/api/runs/{run_id}",
        )

    async def cancel(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def _soft_timeout(self, ledger: RunLedger) -> None:
        await asyncio.sleep(self.settings.soft_timeout_seconds)
        await ledger.append(
            "run.soft_timeout",
            {
                "seconds": self.settings.soft_timeout_seconds,
                "message": "Sealing soon with available evidence.",
            },
        )

    async def _execute(self, run_id: str, request: RunCreateRequest) -> None:
        ledger = RunLedger(run_id, self.database, self.settings)
        await ledger.hydrate()
        envelope: RunEnvelope | None = None
        trace_id: str | None = None
        error: dict[str, Any] | None = None
        soft_timer: asyncio.Task[None] | None = None
        context = InvestigationContext(
            run_id=run_id,
            corpus=self.corpus,
            verifier=self.verifier,
            ledger=ledger,
            settings=self.settings,
        )
        try:
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
                    },
                )
                soft_timer = asyncio.create_task(self._soft_timeout(ledger))
                outcome = await asyncio.wait_for(
                    investigate(
                        request.question,
                        context=context,
                        session_id=f"{request.session_id or 'browser'}:{run_id}",
                    ),
                    timeout=self.settings.hard_timeout_seconds,
                )
                trace_id = outcome.trace_id
                await ledger.append("verification.authoritative_started", {})
                verification = self.verifier.verify_claims(
                    outcome.draft.claims, completed_searches=context.searches
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
                answer = compose_authoritative_answer(claims, context.searches)
                status = RunStatus.COMPLETED if answer else RunStatus.INCOMPLETE
                envelope = self._envelope(
                    run_id, request.question, status=status, answer=answer, claims=claims
                )
                terminal_event = (
                    "run.completed" if status == RunStatus.COMPLETED else "run.incomplete"
                )
                await ledger.append(
                    terminal_event,
                    {"status": status.value, "authoritative": bool(answer)},
                )
        except asyncio.CancelledError:
            error = {"type": "cancelled", "message": "The investigation was cancelled."}
            envelope = self._envelope(
                run_id,
                request.question,
                status=RunStatus.CANCELLED,
                answer=None,
                claims=[],
            )
            await ledger.append("run.cancelled", {"authoritative": False})
        except TimeoutError:
            error = {
                "type": "timeout",
                "message": f"Investigation reached the {self.settings.hard_timeout_seconds:g}-second limit.",
            }
            envelope = self._envelope(
                run_id, request.question, status=RunStatus.INCOMPLETE, answer=None, claims=[]
            )
            await ledger.append("run.timeout", {**error, "authoritative": False})
        except Exception as exc:  # noqa: BLE001 - every failure must seal a receipt
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            envelope = self._envelope(
                run_id, request.question, status=RunStatus.FAILED, answer=None, claims=[]
            )
            await ledger.append("run.failed", {**error, "authoritative": False})
        finally:
            if soft_timer:
                soft_timer.cancel()
                with suppress(asyncio.CancelledError):
                    await soft_timer
            if envelope is None:
                envelope = self._envelope(
                    run_id, request.question, status=RunStatus.FAILED, answer=None, claims=[]
                )
            receipt_path = await ledger.seal(
                envelope,
                instruction_hash=INSTRUCTION_HASH,
                tool_schema_hash=TOOL_SCHEMA_HASH,
                verifier_version=VERIFIER_VERSION,
                trace_id=trace_id,
                error=error,
            )
            await self.database.update_run(
                run_id,
                status=envelope.status,
                completed_at=utc_now_iso(),
                answer=envelope.answer,
                result_json=envelope.model_dump_json(),
                receipt_path=str(receipt_path),
                error_json=json.dumps(error) if error else None,
            )

    async def _replay(self, run_id: str, request: RunCreateRequest) -> None:
        ledger = RunLedger(run_id, self.database, self.settings)
        await ledger.hydrate()
        source = self.settings.rehearsal_path
        if not source.exists():
            await self._seal_replay_failure(run_id, request.question, ledger, source)
            return
        receipt = json.loads(source.read_text(encoding="utf-8"))
        await self.database.update_run(run_id, status=RunStatus.RUNNING, started_at=utc_now_iso())
        await ledger.append(
            "run.started",
            {"rehearsal": True, "source_receipt_sha256": receipt.get("receipt_sha256")},
        )
        for event in receipt.get("events", []):
            if event.get("type") in {"run.started", "run.completed"}:
                continue
            await asyncio.sleep(0.08)
            await ledger.append(event["type"], {**event.get("payload", {}), "replayed": True})
        envelope = self._envelope(
            run_id,
            receipt.get("question", request.question),
            status=RunStatus.COMPLETED,
            answer=receipt.get("answer"),
            claims=[VerifiedClaim.model_validate(claim) for claim in receipt.get("claims", [])],
        )
        await ledger.append("run.completed", {"status": "completed", "rehearsal": True})
        path = await ledger.seal(
            envelope,
            instruction_hash=INSTRUCTION_HASH,
            tool_schema_hash=TOOL_SCHEMA_HASH,
            verifier_version=VERIFIER_VERSION,
            trace_id=receipt.get("provenance", {}).get("trace_id"),
        )
        await self.database.update_run(
            run_id,
            status=RunStatus.COMPLETED,
            completed_at=utc_now_iso(),
            answer=envelope.answer,
            result_json=envelope.model_dump_json(),
            receipt_path=str(path),
        )

    async def _seal_replay_failure(
        self, run_id: str, question: str, ledger: RunLedger, source: Path
    ) -> None:
        error = {"type": "rehearsal_unavailable", "message": f"Missing {source}"}
        envelope = self._envelope(run_id, question, status=RunStatus.FAILED, answer=None, claims=[])
        await ledger.append("run.failed", {**error, "authoritative": False})
        path = await ledger.seal(
            envelope,
            instruction_hash=INSTRUCTION_HASH,
            tool_schema_hash=TOOL_SCHEMA_HASH,
            verifier_version=VERIFIER_VERSION,
            trace_id=None,
            error=error,
        )
        await self.database.update_run(
            run_id,
            status=RunStatus.FAILED,
            completed_at=utc_now_iso(),
            result_json=envelope.model_dump_json(),
            receipt_path=str(path),
            error_json=json.dumps(error),
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
