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


@pytest.mark.asyncio
async def test_readiness_uses_configured_reasoning_and_closes_client(monkeypatch):
    observed: dict[str, object] = {}

    class FakeResponses:
        async def create(self, **kwargs):
            observed.update(kwargs)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.responses = FakeResponses()

        async def close(self):
            observed["closed"] = True

    monkeypatch.setenv("OPENAI_API_KEY", "configured-for-test")
    monkeypatch.setattr("needleproof_api.readiness.AsyncOpenAI", FakeClient)
    availability = ModelAvailability(Settings(reasoning_effort="high"), ttl_seconds=0)

    assert await availability.refresh() is True
    assert observed["reasoning"] == {"effort": "high"}
    assert observed["closed"] is True
