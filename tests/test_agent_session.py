from __future__ import annotations

from types import SimpleNamespace

import pytest
from needleproof_api import agent as agent_module
from needleproof_api.agent import (
    InvestigationContext,
    ReceiptHooks,
    investigate,
    record_tool_failures,
)
from needleproof_api.config import Settings
from needleproof_api.models import AgentDraft


@pytest.mark.asyncio
async def test_investigation_purges_and_closes_its_private_agent_session(tmp_path, monkeypatch):
    calls: list[str] = []

    class FakeSession:
        def __init__(self, _session_id, *, db_path):
            assert db_path == tmp_path / "needleproof.sqlite3"

        async def clear_session(self):
            calls.append("clear")

        async def close(self):
            calls.append("close")

    class FakeResult:
        final_output = AgentDraft(claims=[])

        async def stream_events(self):
            if False:
                yield None

    monkeypatch.setattr(agent_module, "AsyncSQLiteSession", FakeSession)
    monkeypatch.setattr(
        agent_module.Runner,
        "run_streamed",
        lambda *_args, **_kwargs: FakeResult(),
    )
    context = InvestigationContext(
        run_id="run_session_cleanup",
        corpus=SimpleNamespace(manifest_sha256="digest"),
        verifier=None,
        ledger=None,
        settings=Settings(data_dir=tmp_path),
    )

    await investigate("What is the metric?", context=context, session_id="private-session")

    assert calls == ["clear", "close"]


@pytest.mark.asyncio
async def test_tool_failures_are_recorded_before_the_error_escapes():
    events: list[tuple[str, dict[str, object]]] = []

    class Ledger:
        async def append(self, event_type, payload):
            events.append((event_type, payload))

    @record_tool_failures("sample")
    async def fail(context):
        del context
        raise RuntimeError("private database detail")

    context = SimpleNamespace(context=SimpleNamespace(ledger=Ledger()))
    with pytest.raises(RuntimeError, match="private database detail"):
        await fail(context)

    assert events[0][0] == "tool.sample.failed"
    assert events[0][1]["error"] == {
        "type": "RuntimeError",
        "message": "private database detail",
    }


@pytest.mark.asyncio
async def test_failed_model_attempt_is_written_to_openai_call_ledger():
    calls: list[dict[str, object]] = []

    class Ledger:
        async def reserve_model_call_capacity(self, _ceiling):
            return None

        async def append(self, _event_type, _payload):
            return None

        async def record_openai_call(self, record):
            calls.append(record)

    state = SimpleNamespace(
        settings=Settings(),
        ledger=Ledger(),
    )
    context = SimpleNamespace(context=state)
    hooks = ReceiptHooks()
    await hooks.on_llm_start(context, SimpleNamespace(model="gpt-5.6-terra"), None, [])
    await hooks.record_failed_model_call(state, "gpt-5.6-terra", RuntimeError("provider down"))

    assert len(calls) == 1
    assert calls[0]["operation"] == "model"
    assert calls[0]["error"] == {"type": "RuntimeError", "message": "provider down"}
