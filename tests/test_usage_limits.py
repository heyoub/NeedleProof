from __future__ import annotations

import asyncio

import pytest
from needleproof_api.config import Settings
from needleproof_api.db import AppDatabase
from needleproof_api.models import RunStatus
from needleproof_api.security import PublicUsageLimiter, UsageDecision, UsageLimitError


class RehearsalDatabase:
    """Rehearsal admissions never touch persisted model-token capacity."""


async def add_model_usage(
    database: AppDatabase,
    *,
    run_id: str,
    total_tokens: int,
    started_at: str | None = None,
) -> None:
    await database.create_run(
        run_id=run_id,
        session_id="seed",
        rehearsal=False,
        question="seed usage",
        corpus_id="corpus",
        corpus_version="version",
        manifest_sha256="digest",
    )
    record = {
        "operation": "model",
        "token_usage": {"total_tokens": total_tokens},
    }
    if started_at:
        record["started_at"] = started_at
    await database.append_openai_call(run_id, 1, record)


@pytest.mark.asyncio
async def test_model_budget_window_uses_call_time_not_run_creation_time(tmp_path):
    database = AppDatabase(tmp_path / "call-window.sqlite3")
    await database.initialize()
    await add_model_usage(
        database,
        run_id="run_old_call",
        total_tokens=75,
        started_at="2020-01-01T00:00:00+00:00",
    )

    assert await database.sum_model_tokens_since("2025-01-01T00:00:00+00:00") == 0


@pytest.mark.asyncio
async def test_session_and_ip_hourly_windows_are_bounded():
    settings = Settings(
        max_runs_per_session_per_hour=1,
        max_runs_per_ip_per_hour=2,
    )
    limiter = PublicUsageLimiter(settings, RehearsalDatabase())  # type: ignore[arg-type]
    await limiter.admit(
        run_id="run_alice_1",
        session_id="alice",
        client_ip="127.0.0.1",
        rehearsal=True,
    )

    with pytest.raises(UsageLimitError, match="browser"):
        await limiter.admit(
            run_id="run_alice_2",
            session_id="alice",
            client_ip="127.0.0.1",
            rehearsal=True,
        )

    await limiter.admit(
        run_id="run_bob",
        session_id="bob",
        client_ip="127.0.0.1",
        rehearsal=True,
    )
    with pytest.raises(UsageLimitError, match="network"):
        await limiter.admit(
            run_id="run_carol",
            session_id="carol",
            client_ip="127.0.0.1",
            rehearsal=True,
        )


@pytest.mark.asyncio
async def test_model_budget_blocks_live_runs_but_not_rehearsal(tmp_path):
    database = AppDatabase(tmp_path / "usage.sqlite3")
    await database.initialize()
    await add_model_usage(database, run_id="run_seed", total_tokens=100)
    settings = Settings(
        max_model_tokens_per_hour=100,
        max_model_tokens_per_day=100,
        model_token_reservation_per_run=10,
    )
    limiter = PublicUsageLimiter(settings, database)

    with pytest.raises(UsageLimitError, match="model budget"):
        await limiter.admit(
            run_id="run_live",
            session_id="alice",
            client_ip="1.1.1.1",
            rehearsal=False,
        )

    await limiter.admit(
        run_id="run_rehearsal",
        session_id="bob",
        client_ip="2.2.2.2",
        rehearsal=True,
    )

    daily_settings = Settings(
        max_model_tokens_per_hour=1_000,
        max_model_tokens_per_day=100,
        model_token_reservation_per_run=10,
    )
    daily_limiter = PublicUsageLimiter(daily_settings, database)
    with pytest.raises(UsageLimitError, match="daily model budget"):
        await daily_limiter.admit(
            run_id="run_daily",
            session_id="carol",
            client_ip="3.3.3.3",
            rehearsal=False,
        )


@pytest.mark.asyncio
async def test_concurrent_live_admission_atomically_reserves_remaining_budget(tmp_path):
    database = AppDatabase(tmp_path / "concurrent-usage.sqlite3")
    await database.initialize()
    await add_model_usage(database, run_id="run_seed", total_tokens=90)
    settings = Settings(
        max_model_tokens_per_hour=100,
        max_model_tokens_per_day=100,
        model_token_reservation_per_run=10,
    )
    # Separate limiter instances prove the persisted transaction, rather than an
    # in-process asyncio lock, protects the shared budget.
    limiters = [PublicUsageLimiter(settings, database), PublicUsageLimiter(settings, database)]
    run_ids = ["run_concurrent_a", "run_concurrent_b"]
    results = await asyncio.gather(
        *(
            limiter.admit(
                run_id=run_id,
                session_id=run_id,
                client_ip=f"192.0.2.{index + 1}",
                rehearsal=False,
            )
            for index, (limiter, run_id) in enumerate(zip(limiters, run_ids, strict=True))
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, UsageDecision) for result in results) == 1
    assert sum(isinstance(result, UsageLimitError) for result in results) == 1
    reservations = [await database.get_model_token_reservation(run_id) for run_id in run_ids]
    assert sorted(value for value in reservations if value is not None) == [10]

    winning_index = next(
        index for index, result in enumerate(results) if isinstance(result, UsageDecision)
    )
    winning_run_id = run_ids[winning_index]
    await add_model_usage(database, run_id=winning_run_id, total_tokens=10)
    await limiters[winning_index].release(winning_run_id)
    assert await database.get_model_token_reservation(winning_run_id) is None

    with pytest.raises(UsageLimitError, match="hourly model budget"):
        await PublicUsageLimiter(settings, database).admit(
            run_id="run_after_reconciliation",
            session_id="after",
            client_ip="192.0.2.3",
            rehearsal=False,
        )


@pytest.mark.asyncio
async def test_reservation_cleanup_preserves_active_runs_and_removes_stale_rows(tmp_path):
    database = AppDatabase(tmp_path / "reservation-cleanup.sqlite3")
    await database.initialize()
    settings = Settings(
        max_model_tokens_per_hour=100,
        max_model_tokens_per_day=100,
        model_token_reservation_per_run=10,
    )
    limiter = PublicUsageLimiter(settings, database)

    await limiter.admit(
        run_id="run_orphan",
        session_id="orphan",
        client_ip="192.0.2.10",
        rehearsal=False,
    )
    await limiter.admit(
        run_id="run_active",
        session_id="active",
        client_ip="192.0.2.11",
        rehearsal=False,
    )
    await database.create_run(
        run_id="run_active",
        session_id="active",
        rehearsal=False,
        question="active",
        corpus_id="corpus",
        corpus_version="version",
        manifest_sha256="digest",
    )

    assert await database.release_terminal_or_orphan_token_reservations() == 1
    assert await database.get_model_token_reservation("run_orphan") is None
    assert await database.get_model_token_reservation("run_active") == 10

    await database.update_run("run_active", status=RunStatus.FAILED)
    assert await database.release_terminal_or_orphan_token_reservations() == 1
    assert await database.get_model_token_reservation("run_active") is None
