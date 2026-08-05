from __future__ import annotations

import pytest
from needleproof_api import cli
from needleproof_api.config import Settings


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_golden_cli_closes_temporary_embedding_store(monkeypatch, failure):
    closed = 0

    class Store:
        async def close(self):
            nonlocal closed
            closed += 1

    store = Store()

    async def run_golden(_store):
        assert _store is store
        if failure:
            raise RuntimeError("injected golden retrieval failure")
        return {"passed": True}

    monkeypatch.setattr(cli, "CorpusStore", lambda _settings: store)
    monkeypatch.setattr(cli, "run_golden_retrieval", run_golden)

    if failure:
        with pytest.raises(RuntimeError, match="injected golden retrieval failure"):
            await cli._golden(Settings())
    else:
        assert await cli._golden(Settings()) == 0

    assert closed == 1
