from __future__ import annotations

import pytest
from needleproof_api.config import Settings
from needleproof_api.retrieval import CorpusStore


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def corpus(settings: Settings) -> CorpusStore:
    return CorpusStore(settings)
