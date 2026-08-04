from __future__ import annotations

import asyncio
import os
import time

from openai import AsyncOpenAI
from openai.types.shared_params import Reasoning

from .config import Settings


class ModelAvailability:
    def __init__(self, settings: Settings, *, ttl_seconds: float = 300.0):
        self.settings = settings
        self.ttl_seconds = ttl_seconds
        self.ready: bool | None = None
        self.error: str | None = None
        self.checked_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def stale(self) -> bool:
        return self.checked_at is None or time.monotonic() - self.checked_at >= self.ttl_seconds

    async def refresh(self) -> bool:
        async with self._lock:
            if not self.stale and self.ready is not None:
                return self.ready
            try:
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY is not configured")
                client = AsyncOpenAI(timeout=10.0)
                try:
                    await client.responses.create(
                        model=self.settings.model,
                        input="Reply with OK.",
                        reasoning=Reasoning(effort=self.settings.reasoning_effort),
                        max_output_tokens=16,
                        store=False,
                    )
                finally:
                    await client.close()
            except Exception as exc:  # noqa: BLE001 - readiness records sanitized failure
                self.ready = False
                self.error = type(exc).__name__
            else:
                self.ready = True
                self.error = None
            self.checked_at = time.monotonic()
            return bool(self.ready)

    async def require(self) -> None:
        if self.stale or self.ready is None:
            await self.refresh()
        if not self.ready:
            raise RuntimeError(
                f"Configured model {self.settings.model!r} is unavailable. "
                "NeedleProof will not silently substitute another model."
            )
