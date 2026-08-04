from __future__ import annotations

import json

import pytest
from needleproof_api.db import AppDatabase
from needleproof_api.models import ClaimStatus, RunStatus


@pytest.mark.asyncio
async def test_retained_schema_1_2_run_envelope_is_adapted_read_only(tmp_path):
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
                        "temporal_anchor": None,
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
                        "temporal_anchors": [],
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
    await database.update_run(
        run_id,
        status=RunStatus.COMPLETED,
        result_json=json.dumps(legacy_envelope),
    )

    envelope = await database.get_envelope(run_id)

    assert envelope is not None
    assert envelope.status == RunStatus.COMPLETED
    assert envelope.claims[0].status == ClaimStatus.VERIFIED
    assert envelope.claims[0].observations[0].value_text == "$2 million"
    assert envelope.claims[0].observations[0].evidence[0].exact_assertion == (
        "Revenue was $2 million."
    )
    assert "schema-1.2" in envelope.claims[0].verification_notes[-1]
