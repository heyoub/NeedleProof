from __future__ import annotations

import pytest
from needleproof_api.agent import InvestigationContext, build_agent
from needleproof_api.config import Settings
from needleproof_api.models import ClaimStatus, DraftClaim
from needleproof_api.verification import EvidenceVerifier


def context(settings: Settings, corpus) -> InvestigationContext:
    return InvestigationContext(
        run_id="run_budget",
        corpus=corpus,
        verifier=EvidenceVerifier(corpus),
        ledger=None,  # type: ignore[arg-type] - accounting test has no I/O
        settings=settings,
    )


def search_arguments(query: str = "total headcount") -> dict[str, object]:
    return {
        "query": query,
        "mode": "hybrid",
        "metric": "total headcount",
        "document_ids": None,
        "date_from": None,
        "date_to": None,
    }


def test_duplicate_searches_do_not_authorize_not_found(corpus):
    state = context(Settings(max_searches=4), corpus)
    for _ in range(4):
        state.begin_search()
        state.complete_search(search_arguments())

    assert state.attempted_searches == 4
    assert state.completed_searches == 4
    assert state.searches == 1
    claim = DraftClaim(metric="total headcount", status="not_found")
    verified = state.verifier.verify_claim(
        claim,
        completed_searches=state.searches,
        completed_search_records=state.completed_search_records,
    )
    assert verified.status == ClaimStatus.UNVERIFIED


def test_failed_search_does_not_count_as_completed(corpus):
    state = context(Settings(max_searches=4), corpus)
    state.begin_search()
    assert state.attempted_searches == 1
    assert state.completed_searches == 0
    assert state.searches == 0


def test_rejected_fifth_search_does_not_increment_attempt_count(corpus):
    state = context(Settings(max_searches=4), corpus)
    for index in range(4):
        state.begin_search()
        state.complete_search(search_arguments(f"headcount wording {index}"))

    with pytest.raises(ValueError, match="Search limit"):
        state.begin_search()
    assert state.attempted_searches == 4
    assert state.completed_searches == 4
    assert state.searches == 4


def test_searches_for_another_metric_do_not_authorize_not_found(corpus):
    state = context(Settings(max_searches=4), corpus)
    for index in range(4):
        arguments = search_arguments(f"assets under management wording {index}")
        arguments["metric"] = "total headcount"
        state.begin_search()
        state.complete_search(arguments)

    claim = DraftClaim(metric="total headcount", status="not_found")
    verified = state.verifier.verify_claim(
        claim,
        completed_searches=state.searches,
        completed_search_records=state.completed_search_records,
    )
    assert verified.status == ClaimStatus.UNVERIFIED


def test_agent_enforces_configured_output_token_cap():
    agent = build_agent(Settings(max_model_output_tokens_per_call=4_321))
    assert agent.model_settings.max_tokens == 4_321
