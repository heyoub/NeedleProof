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


@pytest.mark.asyncio
async def test_recorded_model_usage_atomically_expands_active_reservation(tmp_path):
    database = AppDatabase(tmp_path / "actual-usage-reservation.sqlite3")
    await database.initialize()
    settings = Settings(
        max_model_tokens_per_hour=1_000_000,
        max_model_tokens_per_day=1_000_000,
        model_token_reservation_per_run=100_000,
    )
    limiter = PublicUsageLimiter(settings, database)
    run_id = "run_actual_usage"
    await limiter.admit(
        run_id=run_id,
        session_id="actual",
        client_ip="192.0.2.30",
        rehearsal=False,
    )
    await database.create_run(
        run_id=run_id,
        session_id="actual",
        rehearsal=False,
        question="record actual usage",
        corpus_id="corpus",
        corpus_version="version",
        manifest_sha256="digest",
    )

    for sequence in (1, 2):
        await database.append_openai_call(
            run_id,
            sequence,
            {"operation": "model", "token_usage": {"total_tokens": 60_000}},
        )

    assert await database.get_model_token_reservation(run_id) == 120_000


@pytest.mark.asyncio
async def test_next_model_call_expands_reservation_before_request(tmp_path):
    database = AppDatabase(tmp_path / "projected-call-reservation.sqlite3")
    await database.initialize()
    settings = Settings(
        max_model_tokens_per_hour=1_000_000,
        max_model_tokens_per_day=1_000_000,
        model_token_reservation_per_run=100_000,
    )
    limiter = PublicUsageLimiter(settings, database)
    run_id = "run_projected_call"
    await limiter.admit(
        run_id=run_id,
        session_id="projected",
        client_ip="192.0.2.31",
        rehearsal=False,
    )
    await database.create_run(
        run_id=run_id,
        session_id="projected",
        rehearsal=False,
        question="reserve next model call",
        corpus_id="corpus",
        corpus_version="version",
        manifest_sha256="digest",
    )
    await database.append_openai_call(
        run_id,
        1,
        {"operation": "model", "token_usage": {"total_tokens": 90_000}},
    )

    exceeded = await database.expand_model_token_reservation(
        run_id=run_id,
        call_token_ceiling=50_000,
        hourly_cutoff="2020-01-01T00:00:00+00:00",
        daily_cutoff="2020-01-01T00:00:00+00:00",
        hourly_limit=1_000_000,
        daily_limit=1_000_000,
    )

    assert exceeded is None
    assert await database.get_model_token_reservation(run_id) == 140_000


@pytest.mark.asyncio
async def test_same_run_usage_is_not_double_counted_during_expansion(tmp_path):
    database = AppDatabase(tmp_path / "same-run-expansion.sqlite3")
    await database.initialize()
    limiter = PublicUsageLimiter(
        Settings(
            max_model_tokens_per_hour=110,
            max_model_tokens_per_day=110,
            model_token_reservation_per_run=100,
        ),
        database,
    )
    run_id = "run_near_limit"
    await limiter.admit(
        run_id=run_id,
        session_id="near-limit",
        client_ip="192.0.2.32",
        rehearsal=False,
    )
    await database.create_run(
        run_id=run_id,
        session_id="near-limit",
        rehearsal=False,
        question="expand without double counting",
        corpus_id="corpus",
        corpus_version="version",
        manifest_sha256="digest",
    )
    await database.append_openai_call(
        run_id,
        1,
        {"operation": "model", "token_usage": {"total_tokens": 60}},
    )

    exceeded = await database.expand_model_token_reservation(
        run_id=run_id,
        call_token_ceiling=50,
        hourly_cutoff="2020-01-01T00:00:00+00:00",
        daily_cutoff="2020-01-01T00:00:00+00:00",
        hourly_limit=110,
        daily_limit=110,
    )

    assert exceeded is None
    assert await database.get_model_token_reservation(run_id) == 110


@pytest.mark.asyncio
async def test_active_run_usage_is_not_double_counted_for_new_admission(tmp_path):
    database = AppDatabase(tmp_path / "active-run-admission.sqlite3")
    await database.initialize()
    settings = Settings(
        max_model_tokens_per_hour=110,
        max_model_tokens_per_day=110,
        model_token_reservation_per_run=100,
    )
    first_limiter = PublicUsageLimiter(settings, database)
    await first_limiter.admit(
        run_id="run_active_usage",
        session_id="first",
        client_ip="192.0.2.33",
        rehearsal=False,
    )
    await database.create_run(
        run_id="run_active_usage",
        session_id="first",
        rehearsal=False,
        question="active usage",
        corpus_id="corpus",
        corpus_version="version",
        manifest_sha256="digest",
    )
    await database.append_openai_call(
        "run_active_usage",
        1,
        {"operation": "model", "token_usage": {"total_tokens": 60}},
    )

    second_limiter = PublicUsageLimiter(
        settings.model_copy(update={"model_token_reservation_per_run": 10}),
        database,
    )
    await second_limiter.admit(
        run_id="run_second",
        session_id="second",
        client_ip="192.0.2.34",
        rehearsal=False,
    )

    assert await database.get_model_token_reservation("run_second") == 10


@pytest.mark.asyncio
async def test_old_active_reservation_keeps_remaining_capacity_in_rolling_window(tmp_path):
    database = AppDatabase(tmp_path / "old-active-reservation.sqlite3")
    await database.initialize()
    old_started_at = "2020-01-01T00:00:00+00:00"
    cutoff = "2020-01-01T01:00:00+00:00"
    await database.create_run(
        run_id="run_old_active",
        session_id="old",
        rehearsal=False,
        question="old active run",
        corpus_id="corpus",
        corpus_version="version",
        manifest_sha256="digest",
    )
    assert (
        await database.reserve_model_tokens(
            run_id="run_old_active",
            reserved_tokens=100,
            created_at=old_started_at,
            hourly_cutoff=cutoff,
            daily_cutoff=cutoff,
            hourly_limit=100,
            daily_limit=100,
        )
        is None
    )
    await database.append_openai_call(
        "run_old_active",
        1,
        {
            "operation": "model",
            "started_at": old_started_at,
            "token_usage": {"total_tokens": 60},
        },
    )

    assert (
        await database.reserve_model_tokens(
            run_id="run_fits_exactly",
            reserved_tokens=60,
            created_at="2020-01-01T02:00:00+00:00",
            hourly_cutoff=cutoff,
            daily_cutoff=cutoff,
            hourly_limit=100,
            daily_limit=100,
        )
        is None
    )

    assert (
        await database.reserve_model_tokens(
            run_id="run_exceeds",
            reserved_tokens=1,
            created_at="2020-01-01T02:00:00+00:00",
            hourly_cutoff=cutoff,
            daily_cutoff=cutoff,
            hourly_limit=100,
            daily_limit=100,
        )
        == "hourly"
    )
