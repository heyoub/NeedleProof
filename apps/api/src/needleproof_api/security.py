from __future__ import annotations

import asyncio
import re
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .config import Settings
from .db import AppDatabase
from .util import sha256_text

_SESSION_TOKEN = re.compile(r"^[A-Za-z0-9_-]{40,128}$")


def new_browser_session() -> str:
    return secrets.token_urlsafe(48)


def session_digest(token: str) -> str:
    return sha256_text(token)


def valid_browser_session(token: str | None) -> bool:
    return bool(token and _SESSION_TOKEN.fullmatch(token))


class UsageLimitError(RuntimeError):
    def __init__(self, message: str, *, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class UsageDecision:
    session_attempts: int
    ip_attempts: int


class PublicUsageLimiter:
    """Single-process request windows plus persisted aggregate model-token budgets."""

    def __init__(self, settings: Settings, database: AppDatabase):
        self.settings = settings
        self.database = database
        self._session_attempts: defaultdict[str, deque[float]] = defaultdict(deque)
        self._ip_attempts: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    @staticmethod
    def _prune(window: deque[float], now: float) -> None:
        threshold = now - 3600
        while window and window[0] < threshold:
            window.popleft()

    async def admit(self, *, session_id: str, client_ip: str, rehearsal: bool) -> UsageDecision:
        now = time.monotonic()
        async with self._lock:
            session_window = self._session_attempts[session_id]
            ip_window = self._ip_attempts[client_ip]
            self._prune(session_window, now)
            self._prune(ip_window, now)
            if len(session_window) >= self.settings.max_runs_per_session_per_hour:
                raise UsageLimitError(
                    "This browser reached the hourly investigation limit.",
                    retry_after=3600,
                )
            if len(ip_window) >= self.settings.max_runs_per_ip_per_hour:
                raise UsageLimitError(
                    "This network reached the hourly investigation limit.",
                    retry_after=3600,
                )
            session_window.append(now)
            ip_window.append(now)

        if not rehearsal:
            current = datetime.now(UTC)
            hourly = await self.database.sum_model_tokens_since(
                (current - timedelta(hours=1)).isoformat()
            )
            daily = await self.database.sum_model_tokens_since(
                (current - timedelta(days=1)).isoformat()
            )
            if hourly >= self.settings.max_model_tokens_per_hour:
                raise UsageLimitError(
                    "The public hourly model budget is exhausted.", retry_after=3600
                )
            if daily >= self.settings.max_model_tokens_per_day:
                raise UsageLimitError(
                    "The public daily model budget is exhausted.", retry_after=86400
                )

        return UsageDecision(len(session_window), len(ip_window))
