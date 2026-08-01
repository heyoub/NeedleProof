from __future__ import annotations

import json

from needleproof_api.receipt import validate_receipt


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
