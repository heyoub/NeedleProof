from __future__ import annotations

import pytest
from needleproof_api.agent import InvestigationContext, build_agent
from needleproof_api.config import Settings
from needleproof_api.models import ClaimStatus, DraftClaim, SearchHit, SearchResult
from needleproof_api.verification import EvidenceVerifier
from pydantic import ValidationError


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


def complete_search(state: InvestigationContext, arguments: dict[str, object]) -> None:
    state.complete_search(
        arguments,
        SearchResult(
            query=str(arguments["query"]),
            mode=str(arguments["mode"]),  # type: ignore[arg-type]
            results=[],
            corpus_manifest_sha256=state.corpus.manifest_sha256,
        ),
    )


def test_duplicate_searches_do_not_authorize_not_found(corpus):
    state = context(Settings(max_searches=4), corpus)
    for _ in range(4):
        state.begin_search()
        complete_search(state, search_arguments())

    assert state.attempted_searches == 4
    assert state.completed_searches == 4
    assert state.searches == 1
    claim = DraftClaim(metric="total headcount", request_absence_probe=True)
    verified = state.verifier.verify_claim(
        claim,
        completed_searches=state.searches,
        completed_search_records=state.completed_search_records,
    )
    assert verified.status == ClaimStatus.UNVERIFIED


def test_punctuation_and_token_order_variants_are_one_search(corpus):
    state = context(Settings(max_searches=4), corpus)
    for query in (
        "total headcount",
        "Total headcount?",
        "headcount, total!",
        "total... headcount",
    ):
        state.begin_search()
        complete_search(state, search_arguments(query))

    assert state.completed_searches == 4
    assert state.searches == 1


def test_failed_search_does_not_count_as_completed(corpus):
    state = context(Settings(max_searches=4), corpus)
    state.begin_search()
    assert state.attempted_searches == 1
    assert state.completed_searches == 0
    assert state.searches == 0


def test_search_record_construction_failure_leaves_completed_accounting_unchanged(
    corpus,
    monkeypatch,
):
    state = context(Settings(max_searches=4), corpus)
    state.begin_search()

    def fail_chunk_lookup(_chunk_ids):
        raise RuntimeError("corpus lookup failed")

    monkeypatch.setattr(state.corpus, "get_chunks", fail_chunk_lookup)

    with pytest.raises(RuntimeError, match="corpus lookup failed"):
        complete_search(state, search_arguments())

    assert state.attempted_searches == 1
    assert state.completed_searches == 0
    assert state.completed_search_records == []
    assert state.unique_search_signatures == set()


def test_rejected_fifth_search_does_not_increment_attempt_count(corpus):
    state = context(Settings(max_searches=4), corpus)
    for index in range(4):
        state.begin_search()
        complete_search(state, search_arguments(f"headcount wording {index}"))

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
        complete_search(state, arguments)

    claim = DraftClaim(metric="total headcount", request_absence_probe=True)
    verified = state.verifier.verify_claim(
        claim,
        completed_searches=state.searches,
        completed_search_records=state.completed_search_records,
    )
    assert verified.status == ClaimStatus.UNVERIFIED


def test_scoped_searches_do_not_authorize_corpus_wide_not_found(corpus):
    state = context(Settings(max_searches=4), corpus)
    for index in range(4):
        arguments = search_arguments(f"total headcount wording {index}")
        arguments["document_ids"] = ["doc_outside_scope"]
        state.begin_search()
        complete_search(state, arguments)

    claim = DraftClaim(metric="total headcount", request_absence_probe=True)
    verified = state.verifier.verify_claim(
        claim,
        completed_searches=state.searches,
        completed_search_records=state.completed_search_records,
    )
    assert verified.status == ClaimStatus.UNVERIFIED


def test_search_diagnostics_normalize_line_broken_metric(corpus):
    state = context(Settings(max_searches=4), corpus)
    chunk = corpus.find_exact_metric_chunks("Fee-related earnings")[0]
    arguments = search_arguments("fee related earnings")
    arguments["metric"] = "Fee-related   earn-\nings"
    state.complete_search(
        arguments,
        SearchResult(
            query=str(arguments["query"]),
            mode="hybrid",
            results=[
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    score=1.0,
                    retrieval_mode="hybrid",
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    physical_page_index=chunk.physical_page_index,
                    printed_page_label=chunk.printed_page_label,
                    preview=chunk.normalized_text[:280],
                    sha256=chunk.sha256,
                )
            ],
            corpus_manifest_sha256=corpus.manifest_sha256,
        ),
    )

    assert state.completed_search_records[-1].exact_metric_hit_count == 1


def test_agent_enforces_configured_output_token_cap():
    agent = build_agent(Settings(max_model_output_tokens_per_call=4_321))
    assert agent.model_settings.max_tokens == 4_321


def test_top_k_configuration_cannot_exceed_public_tool_contract():
    with pytest.raises(ValidationError):
        Settings(max_top_k=21)
    with pytest.raises(ValidationError):
        Settings(top_k=9, max_top_k=8)
    with pytest.raises(ValidationError):
        Settings(top_k=4, max_top_k=4)
