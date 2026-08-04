from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from needleproof_api.config import Settings
from needleproof_api.db import AppDatabase
from needleproof_api.models import ClaimStatus, RunStatus
from needleproof_api.receipt import validate_receipt
from needleproof_api.retrieval import CorpusStore
from needleproof_api.service import InvestigationService, ReceiptRecoveryOutcome
from needleproof_api.util import canonical_json, sha256_text


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_temporal_anchor", [None, "", "  \t"])
async def test_retained_schema_1_2_run_envelope_is_adapted_read_only(
    tmp_path,
    legacy_temporal_anchor,
):
    database = AppDatabase(tmp_path / "legacy.sqlite3")
    await database.initialize()
    run_id = "run_" + "1" * 32
    corpus_version = "v_" + "2" * 16
    manifest_sha256 = "3" * 64
    await database.create_run(
        run_id=run_id,
        session_id="legacy-owner",
        rehearsal=False,
        question="What was revenue?",
        corpus_id="legacy-corpus",
        corpus_version=corpus_version,
        manifest_sha256=manifest_sha256,
    )
    evidence_reference = {
        "chunk_id": "chk_8000000000000001",
        "metric_anchor": "Revenue",
        "exact_quote": "Revenue was $2 million.",
        "relation": "supports",
    }
    legacy_envelope = {
        "run_id": run_id,
        "status": "completed",
        "question": "What was revenue?",
        "answer": "Revenue was $2 million.",
        "corpus_id": "legacy-corpus",
        "corpus_version": corpus_version,
        "corpus_manifest_sha256": manifest_sha256,
        "claims": [
            {
                "statement": "Revenue was $2 million.",
                "metric": "Revenue",
                "status": "verified",
                "values": [
                    {
                        "value": "$2 million",
                        "temporal_anchor": legacy_temporal_anchor,
                        "evidence": [evidence_reference],
                    }
                ],
                "evidence": [
                    {
                        "chunk_id": "chk_8000000000000001",
                        "document_id": "doc_legacy",
                        "document_name": "Legacy report",
                        "physical_page_index": 1,
                        "printed_page_label": "1",
                        "metric_anchor": "Revenue",
                        "metric_anchor_found": True,
                        "temporal_anchors": (
                            [] if legacy_temporal_anchor is None else [legacy_temporal_anchor]
                        ),
                        "temporal_anchors_found": True,
                        "quote": "Revenue was $2 million.",
                        "normalized_quote": "Revenue was $2 million.",
                        "normalization_operations": [],
                        "relation": "supports",
                        "quote_found": True,
                        "value_found": True,
                        "chunk_sha256": "4" * 64,
                        "source_url": (
                            f"/api/corpora/{corpus_version}/documents/doc_legacy/pdf#page=1"
                        ),
                    }
                ],
                "verification_notes": [],
            }
        ],
        "receipt_url": f"/api/runs/{run_id}/receipt",
        "receipt_json_url": f"/api/runs/{run_id}/receipt.json",
    }
    stored_result = json.dumps(legacy_envelope)
    await database.update_run(
        run_id,
        status=RunStatus.COMPLETED,
        result_json=stored_result,
    )

    envelope = await database.get_envelope(run_id)

    assert envelope is not None
    assert envelope.status == RunStatus.COMPLETED
    assert envelope.claims[0].status == ClaimStatus.VERIFIED
    assert envelope.claims[0].observations[0].value_text == "$2 million"
    assert envelope.claims[0].observations[0].evidence[0].exact_assertion == (
        "Revenue was $2 million."
    )
    adapted_evidence = envelope.claims[0].evidence[0]
    assert adapted_evidence.temporal_anchor is None
    assert adapted_evidence.temporal_value_bound is False
    assert adapted_evidence.binding_profile is None
    assert "schema-1.2" in envelope.claims[0].verification_notes[-1]
    persisted = await database.get_run_row(run_id)
    assert persisted is not None
    assert persisted["result_json"] == stored_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy_exact_quotes",
    [
        ["Revenue was $2 million."],
        [""],
        ["  \t"],
        ["", "Revenue was $2 million."],
    ],
)
async def test_terminal_recovery_adapts_schema_1_2_claims_before_validation(
    tmp_path,
    legacy_exact_quotes,
):
    run_id = "run_" + "5" * 32
    corpus_version = "v_" + "6" * 16
    legacy_receipt = {
        "schema_version": "1.2",
        "run_id": run_id,
        "status": "completed",
        "question": "What was revenue?",
        "answer": "Revenue was $2 million.",
        "corpus_id": "legacy-corpus",
        "corpus_version": corpus_version,
        "corpus_manifest_sha256": "7" * 64,
        "claims": [
            {
                "statement": "Revenue was $2 million.",
                "metric": "Revenue",
                "status": "possible_conflict",
                "values": [
                    {
                        "value": "$2 million",
                        "temporal_anchor": " ",
                        "evidence": [
                            {
                                "chunk_id": "chk_8000000000000001",
                                "metric_anchor": "Revenue",
                                "exact_quote": legacy_exact_quote,
                                "relation": "supports",
                            }
                            for legacy_exact_quote in legacy_exact_quotes
                        ],
                    }
                ],
                "evidence": [],
                "verification_notes": [],
            }
        ],
    }

    previous_hash = "0" * 64
    event_body = {
        "run_id": run_id,
        "sequence": 1,
        "type": "run.completed",
        "occurred_at": "2026-08-01T12:00:00+00:00",
        "payload": {"status": "completed"},
        "previous_hash": previous_hash,
    }
    event_hash = sha256_text(previous_hash + canonical_json(event_body))
    legacy_receipt.update(
        {
            "events": [
                {
                    key: value
                    for key, value in {**event_body, "event_hash": event_hash}.items()
                    if key != "run_id"
                }
            ],
            "openai_calls": [],
            "configuration": {
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimensions": 768,
                "embedding_l2_normalized": True,
                "turbovec_version": "0.8.0",
                "turbovec_bit_width": 4,
                "retrieval_modes": ["dense", "lexical", "hybrid"],
                "parallel_tool_calls": False,
                "max_turns": 6,
                "trace_include_sensitive_data": False,
            },
            "provenance": {
                "application_version": "0.1.0",
                "git_commit_sha": None,
                "dependency_lock_digests": {},
                "agent_instruction_hash": "agent",
                "tool_schema_hash": "tools",
                "verifier_version": "deterministic-verifier-v4-bound-anchors",
                "trace_id": None,
                "sealed_at": "2026-08-01T12:00:00+00:00",
                "event_chain_head": event_hash,
            },
            "error": None,
            "rehearsal": None,
        }
    )
    legacy_receipt["receipt_sha256"] = sha256_text(canonical_json(legacy_receipt))
    assert validate_receipt(legacy_receipt) == []

    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    settings = Settings(data_dir=tmp_path)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    await database.create_run(
        run_id=run_id,
        session_id="legacy-owner",
        rehearsal=True,
        question=legacy_receipt["question"],
        corpus_id=legacy_receipt["corpus_id"],
        corpus_version=corpus_version,
        manifest_sha256=legacy_receipt["corpus_manifest_sha256"],
    )
    settings.receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = settings.receipts_dir / f"{run_id}.json"
    receipt_path.write_text(json.dumps(legacy_receipt), encoding="utf-8")
    await database.update_run(
        run_id,
        status=RunStatus.INTERRUPTED,
        receipt_path=str(receipt_path),
        receipt_sha256=legacy_receipt["receipt_sha256"],
    )
    service = InvestigationService(settings, database, CorpusStore(settings))

    outcome = await service.reconcile_pending_receipt(run_id)
    envelope = await database.get_envelope(run_id)

    assert outcome == ReceiptRecoveryOutcome.RECOVERED
    assert envelope is not None
    assert envelope.status == RunStatus.COMPLETED
    assert envelope.claims[0].status == ClaimStatus.POSSIBLE_CONFLICT
    valid_quotes = [quote for quote in legacy_exact_quotes if quote.strip()]
    if valid_quotes:
        assert envelope.claims[0].observations[0].value_text == "$2 million"
        assert envelope.claims[0].observations[0].temporal_anchor is None
        assert len(envelope.claims[0].observations[0].evidence) == len(valid_quotes)
    else:
        assert envelope.claims[0].observations == []
        assert any(
            "Omitted 1 legacy observation" in note for note in envelope.claims[0].verification_notes
        )
    assert any("schema-1.2" in note for note in envelope.claims[0].verification_notes)
    assert sum(
        "empty legacy evidence" in note for note in envelope.claims[0].verification_notes
    ) == (1 if len(valid_quotes) != len(legacy_exact_quotes) else 0)
