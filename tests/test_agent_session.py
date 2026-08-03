from __future__ import annotations

from types import SimpleNamespace

import pytest
from needleproof_api import agent as agent_module
from needleproof_api.agent import InvestigationContext, investigate
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
        final_output = AgentDraft(answer="", claims=[])

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
