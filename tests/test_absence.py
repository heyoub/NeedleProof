from __future__ import annotations

import pytest
from needleproof_api.absence import probe_metric_absence
from needleproof_api.binding import word_phrase_spans
from needleproof_api.chunk_ids import chunk_id_from_uint64
from needleproof_api.models import (
    AbsenceConclusion,
    ChunkRecord,
    ClaimStatus,
    DraftClaim,
    SearchHit,
    SearchResult,
)
from needleproof_api.service import _metric_key as service_metric_key
from needleproof_api.service import compose_authoritative_answer
from needleproof_api.util import canonical_metric_key
from needleproof_api.verification import EvidenceVerifier
from needleproof_api.verification import _canonical_metric as verifier_metric_key


@pytest.mark.asyncio
async def test_bounded_absence_probe_authorizes_seeded_missing_metric(corpus):
    probe = await probe_metric_absence(corpus, "total headcount")
    verified = EvidenceVerifier(corpus).verify_claim(
        DraftClaim(metric="total headcount", request_absence_probe=True),
        absence_probe=probe,
    )
    assert probe.conclusion == AbsenceConclusion.NOT_FOUND_IN_PROBE
    assert len(probe.searches) == 4
    assert all(search.top_k >= 8 for search in probe.searches)
    assert verified.status == ClaimStatus.NOT_FOUND
    assert verified.verification_notes[0].startswith("Not found after 4 searches")
    answer = compose_authoritative_answer([verified], 2, corpus.corpus_version)
    assert "Not found after 4 searches" in str(answer)
    assert "Not found after 2 searches" not in str(answer)


@pytest.mark.asyncio
async def test_bounded_absence_probe_rejects_metric_with_returned_value(corpus):
    probe = await probe_metric_absence(corpus, "Fee-related earnings")
    verified = EvidenceVerifier(corpus).verify_claim(
        DraftClaim(metric="Fee-related earnings", request_absence_probe=True),
        absence_probe=probe,
    )
    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert probe.supporting_value_candidates
    assert verified.status == ClaimStatus.UNVERIFIED


def test_metric_phrase_matching_does_not_match_inside_trauma():
    assert word_phrase_spans("AUM", "trauma") == []


def test_claim_and_absence_probe_share_one_metric_identity_normalizer():
    metric = "Fee-earn-\ning AUM_2026"
    expected = canonical_metric_key(metric)
    assert expected == "fee earning aum 2026"
    assert service_metric_key(metric) == expected
    assert verifier_metric_key(metric) == expected


def test_model_search_counts_cannot_authorize_not_found(corpus):
    verified = EvidenceVerifier(corpus).verify_claim(
        DraftClaim(metric="total headcount", request_absence_probe=True),
        completed_searches=4,
        completed_search_records=[{"query": "total headcount"}] * 4,
    )
    assert verified.status == ClaimStatus.UNVERIFIED


@pytest.mark.asyncio
async def test_unrecognized_metric_adjacent_number_requires_review():
    chunk_id = chunk_id_from_uint64(2**63 + 19)
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc_absence",
        document_name="Absence fixture",
        physical_page_index=1,
        chunk_position=0,
        text="Total headcount was approximately 500 employees.",
        normalized_text="Total headcount was approximately 500 employees.",
        sha256="a" * 64,
        token_estimate=8,
    )

    class Corpus:
        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="b" * 64,
                results=[
                    SearchHit(
                        chunk_id=chunk_id,
                        score=1.0,
                        lexical_score=1.0,
                        lexical_rank=1,
                        retrieval_mode=mode,
                        document_id="doc_absence",
                        document_name="Absence fixture",
                        physical_page_index=1,
                        preview=chunk.text,
                        sha256=chunk.sha256,
                    )
                ][:top_k],
            )

        def get_chunks(self, chunk_ids):
            return [chunk] if chunk_id in chunk_ids else []

    probe = await probe_metric_absence(Corpus(), "total headcount")  # type: ignore[arg-type]
    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert probe.supporting_value_candidates[0].binding_profile is None
    assert probe.supporting_value_candidates[0].binding_failure_reason
