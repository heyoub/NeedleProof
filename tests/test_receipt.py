from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from needleproof_api.main import _validated_receipt
from needleproof_api.receipt import (
    ReceiptProvenance,
    _git_sha,
    _optional_env,
    _optional_sha256_env,
    receipt_contract_schema,
    validate_receipt,
)
from needleproof_api.util import canonical_json, sha256_text
from pydantic import TypeAdapter


def test_featured_rehearsal_receipt_is_sealed(settings):
    receipt = json.loads(settings.rehearsal_path.read_text(encoding="utf-8"))
    assert validate_receipt(receipt) == []
    assert receipt["status"] == "completed"
    statuses = {claim["status"] for claim in receipt["claims"]}
    assert "date_variant" in statuses
    assert "conflict" in statuses
    assert receipt["configuration"]["model"] == "gpt-5.6-terra"
    assert receipt["configuration"]["reasoning_effort"] == "medium"
    assert receipt["configuration"]["trace_include_sensitive_data"] is False
    assert receipt["provenance"]["receipt_derivation"] == "contract_migration"
    assert receipt["provenance"]["source_receipt_sha256"]
    assert receipt["provenance"]["source_verifier_version"]


def test_committed_receipt_schema_is_generated_from_pydantic_contract():
    committed = json.loads(
        Path("packages/contracts/receipt.schema.json").read_text(encoding="utf-8")
    )
    assert committed == receipt_contract_schema()


def test_receipt_schema_exposes_provenance_derivation_contract():
    schema = receipt_contract_schema()
    provenance = schema["properties"]["provenance"]
    assert provenance["discriminator"]["propertyName"] == "receipt_derivation"
    assert len(provenance["oneOf"]) == 2
    for definition_name in ("LiveReceiptProvenance", "MigratedReceiptProvenance"):
        definition = schema["$defs"][definition_name]
        assert "receipt_derivation" in definition["required"]
        assert "source_receipt_sha256" in definition["required"]
        assert "source_verifier_version" in definition["required"]


def test_receipt_contract_rejects_unknown_top_level_fields(settings):
    receipt = json.loads(settings.rehearsal_path.read_text(encoding="utf-8"))
    receipt["surprise"] = "not part of the contract"
    assert any("Extra inputs" in error for error in validate_receipt(receipt))


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        [],
        "receipt",
        7,
        {"events": None},
        {"events": [None]},
        {"events": [], "provenance": None},
    ],
)
def test_receipt_validation_is_total_for_arbitrary_json(receipt):
    errors = validate_receipt(receipt)

    assert errors
    assert all(isinstance(error, str) for error in errors)


json_scalar = st.none() | st.booleans() | st.integers() | st.text()
json_value = st.recursive(
    json_scalar,
    lambda children: (
        st.lists(children, max_size=8) | st.dictionaries(st.text(max_size=20), children, max_size=8)
    ),
    max_leaves=50,
)


@given(json_value)
def test_receipt_validation_never_raises_for_recursive_json(value):
    errors = validate_receipt(value)
    assert isinstance(errors, list)
    assert all(isinstance(error, str) for error in errors)


def test_git_sha_is_optional_when_git_executable_is_missing(monkeypatch):
    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("needleproof_api.receipt.subprocess.run", missing_git)
    assert _git_sha() is None


def test_empty_build_provenance_is_normalized_to_none(monkeypatch):
    monkeypatch.setenv("NEEDLEPROOF_IMAGE_REVISION", "")
    assert _optional_env("NEEDLEPROOF_IMAGE_REVISION") is None


def test_injected_lock_digest_must_be_canonical_sha256(monkeypatch):
    monkeypatch.setenv("NEEDLEPROOF_UV_LOCK_SHA256", "not-a-digest")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _optional_sha256_env("NEEDLEPROOF_UV_LOCK_SHA256")


def test_contract_migration_requires_source_provenance(settings):
    receipt = json.loads(settings.rehearsal_path.read_text(encoding="utf-8"))
    provenance = receipt["provenance"]
    provenance["source_receipt_sha256"] = None
    with pytest.raises(ValueError):
        TypeAdapter(ReceiptProvenance).validate_python(provenance)


def test_live_provenance_rejects_migration_source_identity(settings):
    receipt = json.loads(settings.rehearsal_path.read_text(encoding="utf-8"))
    provenance = receipt["provenance"]
    provenance["receipt_derivation"] = "live"
    with pytest.raises(ValueError):
        TypeAdapter(ReceiptProvenance).validate_python(provenance)


def test_final_ledger_event_must_be_terminal(settings):
    receipt = json.loads(settings.rehearsal_path.read_text(encoding="utf-8"))
    previous_hash = receipt["events"][-1]["event_hash"]
    sequence = len(receipt["events"]) + 1
    event_body = {
        "run_id": receipt["run_id"],
        "sequence": sequence,
        "type": "tool.search.completed",
        "occurred_at": receipt["events"][-1]["occurred_at"],
        "payload": {},
        "previous_hash": previous_hash,
    }
    event_hash = sha256_text(previous_hash + canonical_json(event_body))
    receipt["events"].append({**event_body, "event_hash": event_hash})
    receipt["provenance"]["event_chain_head"] = event_hash
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = sha256_text(canonical_json(unsigned))

    assert any("final terminal event" in error for error in validate_receipt(receipt))


@pytest.mark.parametrize("extended_configuration", [False, True])
@pytest.mark.parametrize("claim_status", ["verified", "possible_conflict"])
def test_retained_schema_1_2_receipt_remains_integrity_checkable(
    tmp_path, extended_configuration, claim_status
):
    run_id = "run_" + "1" * 32
    corpus_version = "v_" + "2" * 16
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
    evidence_reference = {
        "chunk_id": "chk_8000000000000001",
        "metric_anchor": "Revenue",
        "exact_quote": "Revenue was $2 million.",
        "relation": "supports",
    }
    receipt = {
        "schema_version": "1.2",
        "run_id": run_id,
        "status": "completed",
        "question": "What was revenue?",
        "answer": "Revenue was $2 million.",
        "corpus_id": "legacy-corpus",
        "corpus_version": corpus_version,
        "corpus_manifest_sha256": "3" * 64,
        "claims": [
            {
                "statement": "Revenue was $2 million.",
                "metric": "Revenue",
                "status": claim_status,
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
    if extended_configuration:
        receipt["configuration"].update(
            {
                "max_model_output_tokens_per_call": 8_000,
                "model_token_reservation_per_run": 100_000,
            }
        )
    receipt["receipt_sha256"] = sha256_text(canonical_json(receipt))

    assert validate_receipt(receipt) == []
    receipt_path = tmp_path / "legacy-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    _path, served = _validated_receipt(
        {
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
            "status": "completed",
        }
    )
    assert served["schema_version"] == "1.2"
