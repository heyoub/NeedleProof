from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from needleproof_api.config import Settings
from needleproof_api.db import AppDatabase
from needleproof_api.models import RunCreateRequest, RunStatus
from needleproof_api.retrieval import CorpusStore
from needleproof_api.service import InvestigationService
from needleproof_api.util import new_run_id


@pytest.mark.asyncio
async def test_rehearsal_replays_and_seals_without_model_access(tmp_path):
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    (tmp_path / "rehearsal").mkdir()
    shutil.copy2(Path("data/rehearsal/featured.json"), tmp_path / "rehearsal/featured.json")

    settings = Settings(data_dir=tmp_path)
    database = AppDatabase(settings.app_db_path)
    await database.initialize()
    service = InvestigationService(settings, database, CorpusStore(settings))
    request = RunCreateRequest(question="Replay the featured investigation", rehearsal=True)
    run_id = new_run_id()
    await database.create_run(
        run_id=run_id,
        session_id=None,
        question=request.question,
        corpus_id=service.corpus.corpus_id,
        corpus_version=service.corpus.corpus_version,
        manifest_sha256=service.corpus.manifest_sha256,
    )
    await service._replay(run_id, request)

    envelope = await database.get_envelope(run_id)
    assert envelope is not None
    assert envelope.status == RunStatus.COMPLETED
    assert envelope.answer
    assert {claim.status.value for claim in envelope.claims} >= {"conflict", "date_variant"}
    assert settings.receipts_dir.joinpath(f"{run_id}.json").exists()
    events = await database.list_events(run_id)
    assert events[0]["type"] == "run.started"
    assert events[-1]["type"] == "run.completed"
