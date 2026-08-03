from __future__ import annotations

import json
from pathlib import Path

from needleproof_api.receipt import _git_sha, receipt_contract_schema, validate_receipt


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


def test_committed_receipt_schema_is_generated_from_pydantic_contract():
    committed = json.loads(
        Path("packages/contracts/receipt.schema.json").read_text(encoding="utf-8")
    )
    assert committed == receipt_contract_schema()


def test_receipt_contract_rejects_unknown_top_level_fields(settings):
    receipt = json.loads(settings.rehearsal_path.read_text(encoding="utf-8"))
    receipt["surprise"] = "not part of the contract"
    assert any("Extra inputs" in error for error in validate_receipt(receipt))


def test_git_sha_is_optional_when_git_executable_is_missing(monkeypatch):
    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("needleproof_api.receipt.subprocess.run", missing_git)
    assert _git_sha() is None
