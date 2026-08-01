from __future__ import annotations

import pytest
from needleproof_api.config import Settings
from needleproof_api.readiness import ModelAvailability


@pytest.mark.asyncio
async def test_missing_model_credentials_degrade_readiness_without_process_failure(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    availability = ModelAvailability(Settings(), ttl_seconds=0)
    assert await availability.refresh() is False
    assert availability.error == "RuntimeError"
    with pytest.raises(RuntimeError, match="will not silently substitute"):
        await availability.require()
