from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from needleproof_api.config import Settings
from needleproof_api.db import AppDatabase
from needleproof_api.main import app, document_media_type
from needleproof_api.models import RunCreateRequest
from needleproof_api.retrieval import CorpusStore
from needleproof_api.security import PublicUsageLimiter, resolve_client_ip
from needleproof_api.service import InvestigationService
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_browser_sessions_isolate_every_run_surface(tmp_path):
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    (tmp_path / "rehearsal").mkdir()
    shutil.copy2(Path("data/rehearsal/featured.json"), tmp_path / "rehearsal/featured.json")
    settings = Settings(data_dir=tmp_path, session_cookie_secure=False)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    corpus = CorpusStore(settings)
    limiter = PublicUsageLimiter(settings, database)
    service = InvestigationService(settings, database, corpus, limiter)
    app.state.settings = settings
    app.state.database = database
    app.state.corpus = corpus
    app.state.service = service
    app.state.usage_limiter = limiter

    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as alice,
        httpx.AsyncClient(transport=transport, base_url="http://test") as bob,
    ):
        created = await alice.post(
            "/api/runs",
            json={"question": "Replay isolated evidence", "rehearsal": True},
        )
        assert created.status_code == 202
        assert "HttpOnly" in created.headers["set-cookie"]
        assert "SameSite=lax" in created.headers["set-cookie"]
        run_id = created.json()["run_id"]
        await service._tasks[run_id]

        alice_run = await alice.get(f"/api/runs/{run_id}")
        assert alice_run.status_code == 200
        evidence = alice_run.json()["claims"][0]["evidence"][0]
        evidence_pdf = await alice.get(evidence["source_url"].split("#", 1)[0])
        assert evidence_pdf.status_code == 200

        assert (await bob.get(f"/api/runs/{run_id}")).status_code == 404
        assert (await bob.get(f"/api/runs/{run_id}/events")).status_code == 404
        assert (await bob.post(f"/api/runs/{run_id}/cancel")).status_code == 404
        assert (await bob.get(f"/api/runs/{run_id}/receipt")).status_code == 404
        assert (await bob.get(f"/api/runs/{run_id}/receipt.json")).status_code == 404
        assert (
            await bob.post(
                "/api/verify/quote",
                json={
                    "run_id": run_id,
                    "corpus_version": alice_run.json()["corpus_version"],
                    "chunk_id": evidence["chunk_id"],
                    "quote": f"{evidence['quote']} [altered]",
                },
            )
        ).status_code == 404

        assert (await alice.get(f"/api/runs/{run_id}/receipt")).status_code == 200
        assert (await alice.get(f"/api/runs/{run_id}/receipt.json")).status_code == 200
        challenge = await alice.post(
            "/api/verify/quote",
            json={
                "run_id": run_id,
                "corpus_version": alice_run.json()["corpus_version"],
                "chunk_id": evidence["chunk_id"],
                "quote": f"{evidence['quote']} [altered]",
            },
        )
        assert challenge.status_code == 200
        assert challenge.json()["status"] == "rejected"
        case_variant = await alice.post(
            "/api/verify/quote",
            json={
                "run_id": run_id,
                "corpus_version": alice_run.json()["corpus_version"],
                "chunk_id": evidence["chunk_id"],
                "quote": evidence["quote"].swapcase(),
            },
        )
        assert case_variant.status_code == 200
        assert case_variant.json()["status"] == "verified"
        empty_challenge = await alice.post(
            "/api/verify/quote",
            json={
                "run_id": run_id,
                "corpus_version": alice_run.json()["corpus_version"],
                "chunk_id": evidence["chunk_id"],
                "quote": " \t\n  ",
            },
        )
        assert empty_challenge.status_code == 422
        wrong_version = await alice.post(
            "/api/verify/quote",
            json={
                "run_id": run_id,
                "corpus_version": "v_0000000000000000",
                "chunk_id": evidence["chunk_id"],
                "quote": evidence["quote"],
            },
        )
        assert wrong_version.status_code == 404

        events = await database.list_events(run_id)
        replay = await alice.get(
            f"/api/runs/{run_id}/events",
            headers={"Last-Event-ID": str(events[-2]["sequence"])},
        )
        assert replay.status_code == 200
        assert f"id: {events[-1]['sequence']}" in replay.text
        assert f"id: {events[-2]['sequence']}\n" not in replay.text

        row = await database.get_run_row(run_id)
        assert row is not None
        assert row["session_id"] != alice.cookies[settings.session_cookie_name]
        receipt_path = Path(str(row["receipt_path"]))
        receipt_path.write_text(
            receipt_path.read_text(encoding="utf-8").replace('"answer":', '"answer_tampered":', 1),
            encoding="utf-8",
        )
        assert (await alice.get(f"/api/runs/{run_id}/receipt.json")).status_code == 409


def test_public_demo_configuration_requires_secure_cookies():
    with pytest.raises(ValueError, match="Secure"):
        Settings(public_demo=True, session_cookie_secure=False)


def test_document_media_type_matches_supported_source_format():
    assert document_media_type(Path("evidence.pdf")) == "application/pdf"
    assert document_media_type(Path("evidence.txt")) == "text/plain"


def test_public_demo_routes_follow_dotenv_configuration(tmp_path):
    (tmp_path / ".env").write_text(
        "NEEDLEPROOF_PUBLIC_DEMO=true\nNEEDLEPROOF_SESSION_COOKIE_SECURE=true\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("NEEDLEPROOF_PUBLIC_DEMO", None)
    environment.pop("NEEDLEPROOF_SESSION_COOKIE_SECURE", None)
    source_root = str(Path("apps/api/src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, environment.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from needleproof_api.main import app; "
                "assert app.docs_url is None; assert app.openapi_url is None"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_question_is_stripped_and_rejects_whitespace_only_input():
    assert RunCreateRequest(question="  What was AUM?  ").question == "What was AUM?"
    with pytest.raises(ValidationError, match="non-whitespace"):
        RunCreateRequest(question="   ")


def test_forwarded_client_ip_requires_a_trusted_transport_peer():
    assert resolve_client_ip("198.51.100.10", "203.0.113.1", []) == "198.51.100.10"
    assert resolve_client_ip("198.51.100.10", "203.0.113.1", ["198.51.100.0/24"]) == "203.0.113.1"
    assert (
        resolve_client_ip("198.51.100.10", "forged, invalid", ["198.51.100.0/24"])
        == "198.51.100.10"
    )


def test_trusted_proxy_configuration_rejects_invalid_cidrs():
    with pytest.raises(ValueError, match="Invalid trusted proxy CIDR"):
        Settings(trusted_proxy_cidrs=["not-a-network"])


@pytest.mark.asyncio
async def test_direct_client_cannot_rotate_ip_limit_with_cf_header(tmp_path):
    data_dir = tmp_path / "direct-client"
    shutil.copytree(Path("data/corpora"), data_dir / "corpora")
    (data_dir / "rehearsal").mkdir()
    shutil.copy2(Path("data/rehearsal/featured.json"), data_dir / "rehearsal/featured.json")
    settings = Settings(
        data_dir=data_dir,
        session_cookie_secure=False,
        max_runs_per_session_per_hour=10,
        max_runs_per_ip_per_hour=1,
        trusted_proxy_cidrs=[],
    )
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    corpus = CorpusStore(settings)
    limiter = PublicUsageLimiter(settings, database)
    service = InvestigationService(settings, database, corpus, limiter)
    app.state.settings = settings
    app.state.database = database
    app.state.corpus = corpus
    app.state.service = service
    app.state.usage_limiter = limiter

    transport = httpx.ASGITransport(app=app, client=("198.51.100.10", 43123))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/api/runs",
            headers={"CF-Connecting-IP": "203.0.113.1"},
            json={"question": "First direct request", "rehearsal": True},
        )
        assert first.status_code == 202
        await service._tasks[first.json()["run_id"]]

        forged = await client.post(
            "/api/runs",
            headers={"CF-Connecting-IP": "203.0.113.2"},
            json={"question": "Forged direct request", "rehearsal": True},
        )
        assert forged.status_code == 429
        assert "network" in forged.json()["detail"]
