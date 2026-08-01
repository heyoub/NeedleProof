from __future__ import annotations

import pytest
from needleproof_api.config import Settings
from needleproof_api.security import PublicUsageLimiter, UsageLimitError


class TokenDatabase:
    def __init__(self, tokens: int = 0):
        self.tokens = tokens

    async def sum_model_tokens_since(self, _started_at: str) -> int:
        return self.tokens


@pytest.mark.asyncio
async def test_session_and_ip_hourly_windows_are_bounded():
    settings = Settings(
        max_runs_per_session_per_hour=1,
        max_runs_per_ip_per_hour=2,
    )
    limiter = PublicUsageLimiter(settings, TokenDatabase())
    await limiter.admit(session_id="alice", client_ip="127.0.0.1", rehearsal=True)

    with pytest.raises(UsageLimitError, match="browser"):
        await limiter.admit(session_id="alice", client_ip="127.0.0.1", rehearsal=True)

    await limiter.admit(session_id="bob", client_ip="127.0.0.1", rehearsal=True)
    with pytest.raises(UsageLimitError, match="network"):
        await limiter.admit(session_id="carol", client_ip="127.0.0.1", rehearsal=True)


@pytest.mark.asyncio
async def test_model_budget_blocks_live_runs_but_not_rehearsal():
    settings = Settings(max_model_tokens_per_hour=100, max_model_tokens_per_day=100)
    limiter = PublicUsageLimiter(settings, TokenDatabase(tokens=100))

    with pytest.raises(UsageLimitError, match="model budget"):
        await limiter.admit(session_id="alice", client_ip="1.1.1.1", rehearsal=False)

    await limiter.admit(session_id="bob", client_ip="2.2.2.2", rehearsal=True)
