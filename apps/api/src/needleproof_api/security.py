from __future__ import annotations

import asyncio
import re
import secrets
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address, ip_network
from typing import TypeGuard

from .config import Settings
from .db import AppDatabase
from .util import sha256_text

_SESSION_TOKEN = re.compile(r"^[A-Za-z0-9_-]{40,128}$")


def new_browser_session() -> str:
    return secrets.token_urlsafe(48)


def session_digest(token: str) -> str:
    return sha256_text(token)


def valid_browser_session(token: str | None) -> TypeGuard[str]:
    return bool(token and _SESSION_TOKEN.fullmatch(token))


def resolve_client_ip(
    peer_host: str | None,
    cf_connecting_ip: str | None,
    trusted_proxy_cidrs: list[str],
) -> str:
    """Honor Cloudflare's client header only when the transport peer is trusted."""

    if not peer_host:
        return "unknown"
    try:
        peer_address = ip_address(peer_host)
    except ValueError:
        # ASGI test clients may expose a hostname. It remains a stable direct-peer key,
        # and an unparseable peer can never become a trusted forwarding proxy.
        return peer_host
    peer_is_trusted = any(
        peer_address in ip_network(cidr, strict=False) for cidr in trusted_proxy_cidrs
    )
    if peer_is_trusted and cf_connecting_ip:
        try:
            return str(ip_address(cf_connecting_ip.strip()))
        except ValueError:
            pass
    return str(peer_address)


class UsageLimitError(RuntimeError):
    def __init__(self, message: str, *, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class UsageDecision:
    session_attempts: int
    ip_attempts: int
    reserved_tokens: int = 0


class PublicUsageLimiter:
    """Single-process request windows plus persisted aggregate model-token budgets."""

    def __init__(self, settings: Settings, database: AppDatabase):
        self.settings = settings
        self.database = database
        self._session_attempts: defaultdict[str, deque[float]] = defaultdict(deque)
        self._ip_attempts: defaultdict[str, deque[float]] = defaultdict(deque)
        self._attempt_by_run: dict[str, tuple[str, str, float]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _prune(window: deque[float], now: float) -> None:
        threshold = now - 3600
        while window and window[0] < threshold:
            window.popleft()

    def _prune_attempt_maps(self, now: float) -> None:
        for attempts in (self._session_attempts, self._ip_attempts):
            for key, window in list(attempts.items()):
                self._prune(window, now)
                if not window:
                    del attempts[key]
        threshold = now - 3600
        self._attempt_by_run = {
            run_id: attempt
            for run_id, attempt in self._attempt_by_run.items()
            if attempt[2] >= threshold
        }

    async def admit(
        self, *, run_id: str, session_id: str, client_ip: str, rehearsal: bool
    ) -> UsageDecision:
        now = time.monotonic()
        async with self._lock:
            self._prune_attempt_maps(now)
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
            self._attempt_by_run[run_id] = (session_id, client_ip, now)

        reserved_tokens = 0
        if not rehearsal:
            try:
                current = datetime.now(UTC)
                reserved_tokens = self.settings.model_token_reservation_per_run
                exceeded = await self.database.reserve_model_tokens(
                    run_id=run_id,
                    reserved_tokens=reserved_tokens,
                    created_at=current.isoformat(),
                    hourly_cutoff=(current - timedelta(hours=1)).isoformat(),
                    daily_cutoff=(current - timedelta(days=1)).isoformat(),
                    hourly_limit=self.settings.max_model_tokens_per_hour,
                    daily_limit=self.settings.max_model_tokens_per_day,
                )
                if exceeded == "hourly":
                    raise UsageLimitError(
                        "The public hourly model budget is exhausted.", retry_after=3600
                    )
                if exceeded == "daily":
                    raise UsageLimitError(
                        "The public daily model budget is exhausted.", retry_after=86400
                    )
            except BaseException:
                with suppress(Exception):
                    await self.database.release_model_token_reservation(run_id)
                await self._rollback_local_attempt(run_id)
                raise

        return UsageDecision(len(session_window), len(ip_window), reserved_tokens)

    async def release(self, run_id: str) -> None:
        await self.database.release_model_token_reservation(run_id)
        async with self._lock:
            self._attempt_by_run.pop(run_id, None)

    async def rollback(self, run_id: str) -> None:
        """Release model capacity and remove a pre-persistence local admission."""

        await self.database.release_model_token_reservation(run_id)
        await self._rollback_local_attempt(run_id)

    async def _rollback_local_attempt(self, run_id: str) -> None:
        async with self._lock:
            attempt = self._attempt_by_run.pop(run_id, None)
            if not attempt:
                return
            session_id, client_ip, admitted_at = attempt
            for window in (
                self._session_attempts.get(session_id),
                self._ip_attempts.get(client_ip),
            ):
                if window is None:
                    continue
                try:
                    window.remove(admitted_at)
                except ValueError:
                    pass
            if not self._session_attempts.get(session_id):
                self._session_attempts.pop(session_id, None)
            if not self._ip_attempts.get(client_ip):
                self._ip_attempts.pop(client_ip, None)
