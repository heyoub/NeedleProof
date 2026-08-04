from __future__ import annotations

import threading

import pytest
from hypothesis import given
from hypothesis import strategies as st
from needleproof_api.absence import (
    _metric_context_numeric_candidates,
    derive_absence_conclusion,
    probe_metric_absence,
)
from needleproof_api.binding import has_unresolved_metric_predicate, word_phrase_spans
from needleproof_api.chunk_ids import chunk_id_from_uint64
from needleproof_api.models import (
    AbsenceConclusion,
    ChunkRecord,
    ClaimStatus,
    DraftClaim,
    MetricOccurrence,
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
async def test_verifier_recomputes_absence_proof_instead_of_trusting_conclusion(corpus):
    complete = await probe_metric_absence(corpus, "total headcount")
    assert derive_absence_conclusion(complete) == AbsenceConclusion.NOT_FOUND_IN_PROBE

    mutations = [
        complete.model_copy(update={"searches": []}),
        complete.model_copy(update={"exact_metric_scan_completed": False}),
        complete.model_copy(update={"opened_chunk_ids": []}),
        complete.model_copy(update={"protocol_version": "fabricated-protocol"}),
    ]
    for mutated in mutations:
        forged = mutated.model_copy(update={"conclusion": AbsenceConclusion.NOT_FOUND_IN_PROBE})
        verified = EvidenceVerifier(corpus).verify_claim(
            DraftClaim(metric="total headcount", request_absence_probe=True),
            absence_probe=forged,
        )

        assert derive_absence_conclusion(forged) != AbsenceConclusion.NOT_FOUND_IN_PROBE
        assert verified.status == ClaimStatus.UNVERIFIED


@pytest.mark.asyncio
async def test_recomputed_absence_detects_omitted_reviewable_candidates(corpus):
    reviewable = await probe_metric_absence(corpus, "Fee-related earnings")
    forged = reviewable.model_copy(
        update={
            "supporting_value_candidates": [],
            "unresolved_predicate_occurrences": [],
            "conclusion": AbsenceConclusion.NOT_FOUND_IN_PROBE,
        }
    )

    verified = EvidenceVerifier(corpus).verify_claim(
        DraftClaim(metric="Fee-related earnings", request_absence_probe=True),
        absence_probe=forged,
    )

    assert derive_absence_conclusion(forged) == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert verified.status == ClaimStatus.UNVERIFIED


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


def test_exact_metric_scan_normalizes_pdf_linebreak_dehyphenation(corpus):
    chunks = corpus.find_exact_metric_chunks("Fee-earn-\ning AUM")

    assert chunks
    assert all(word_phrase_spans("fee earning aum", chunk.normalized_text) for chunk in chunks)


def test_metric_phrase_matching_does_not_match_inside_trauma():
    assert word_phrase_spans("AUM", "trauma") == []


@given(
    qualifiers=st.lists(
        st.sampled_from(
            ["for", "the", "current", "fiscal", "reporting", "annual", "period", "year"]
        ),
        min_size=1,
        max_size=8,
    )
)
def test_bounded_qualifier_shape_cannot_hide_positive_qualitative_predicate(qualifiers):
    qualifier = " ".join(qualifiers)
    assertion = f"Credit rating {qualifier} was stable."

    assert has_unresolved_metric_predicate("Credit rating", assertion)


@given(
    state=st.sampled_from(
        ["available", "disclosed", "flat", "reported", "stable", "stated", "unchanged"]
    ),
    lead=st.sampled_from(
        ["The company maintained a", "The filing described an", "Management reported a"]
    ),
    modifiers=st.lists(
        st.sampled_from(["current", "long", "term", "U.S."]),
        min_size=0,
        max_size=3,
    ),
)
def test_closed_pre_metric_qualitative_family_blocks_absence(state, lead, modifiers):
    qualifier = f" {' '.join(modifiers)}" if modifiers else ""
    assert has_unresolved_metric_predicate(
        "Credit rating",
        f"{lead} {state}{qualifier} Credit rating.",
    )


@pytest.mark.parametrize("boundary", [".", "?", "!", ";", ","])
def test_qualitative_predicate_detection_does_not_cross_punctuation(boundary):
    assertion = f"Credit rating appeared in the rubric{boundary} Revenue was stable."

    assert not has_unresolved_metric_predicate("Credit rating", assertion)


@pytest.mark.parametrize(
    "assertion",
    [
        "Credit rating appears in the rubric.",
        "Credit rating - source quote.",
        "Question: what was Credit rating?",
    ],
)
def test_bare_metric_mentions_do_not_become_qualitative_predicates(assertion):
    assert not has_unresolved_metric_predicate("Credit rating", assertion)


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
        text="Total headcount was approximately 500 employees. Total headcount was 500.",
        normalized_text=(
            "Total headcount was approximately 500 employees. Total headcount was 500."
        ),
        sha256="a" * 64,
        token_estimate=8,
    )

    class Corpus:
        corpus_version = "v_0000000000000001"

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

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk_id in chunk_ids else []

        def find_exact_metric_chunks(self, metric):
            del metric
            return [chunk]

    probe = await probe_metric_absence(Corpus(), "total headcount")
    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert len(probe.supporting_value_candidates) == 1
    assert probe.supporting_value_candidates[0].binding_profile is None
    assert probe.supporting_value_candidates[0].binding_failure_reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sentence",
    [
        "$2 million in revenue.",
        "The company reported $2 million revenue.",
        "The company reported $2 million — revenue.",
        "The company reported $2 million: revenue.",
        "The company reported $2 million, revenue.",
        "The company reported $2 million of revenue.",
        "At year end, $2 million in revenue was recorded.",
    ],
)
async def test_value_before_metric_prevents_authoritative_absence(sentence):
    chunk_id = chunk_id_from_uint64(2**63 + 21)
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc_value_first",
        document_name="Value-first absence fixture",
        physical_page_index=1,
        chunk_position=0,
        text=sentence,
        normalized_text=sentence,
        sha256="b" * 64,
        token_estimate=8,
    )

    class Corpus:
        corpus_version = "v_0000000000000008"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="c" * 64,
                results=[
                    SearchHit(
                        chunk_id=chunk_id,
                        score=1.0,
                        lexical_score=1.0,
                        lexical_rank=1,
                        retrieval_mode=mode,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        physical_page_index=1,
                        preview=chunk.text,
                        sha256=chunk.sha256,
                    )
                ][:top_k],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk_id in chunk_ids else []

        def find_exact_metric_chunks(self, metric):
            del metric
            return [chunk]

    corpus = Corpus()
    probe = await probe_metric_absence(corpus, "revenue")
    verified = EvidenceVerifier(corpus).verify_claim(
        DraftClaim(metric="revenue", request_absence_probe=True),
        absence_probe=probe,
    )

    assert probe.supporting_value_candidates
    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert verified.status == ClaimStatus.UNVERIFIED


@given(
    separator=st.sampled_from(
        [" ", "  ", " , ", " ; ", " : ", " - ", " – ", " — ", " (", " [", " {", ' "']
    ),
    leading_words=st.lists(
        st.sampled_from(["the", "company", "reported", "approximately"]),
        min_size=0,
        max_size=4,
    ),
)
def test_bounded_value_first_separator_family_is_always_reviewable(
    separator,
    leading_words,
):
    prefix = f"{' '.join(leading_words)} " if leading_words else ""
    sentence = f"{prefix}$2 million{separator}Revenue."
    metric_start = sentence.index("Revenue")
    occurrence = MetricOccurrence(
        chunk_id=chunk_id_from_uint64(2**63 + 22),
        sentence=sentence,
        span=(metric_start, metric_start + len("Revenue")),
    )

    assert _metric_context_numeric_candidates(occurrence)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sentence",
    [
        "Credit rating was stable.",
        "Credit rating: stable.",
        "Credit rating reached stable.",
        "Credit rating for the period was stable.",
        "Credit rating as of year end remained unchanged.",
        "Credit rating during the fiscal year was reported stable.",
        "The company maintained a stable Credit rating.",
        "An unchanged Credit rating was disclosed.",
        "Total revenue was reported by the U.S. Credit rating.",
    ],
)
async def test_exact_metric_with_qualitative_predicate_requires_review(sentence):
    chunk_id = chunk_id_from_uint64(2**63 + 23)
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc_absence_qualitative",
        document_name="Qualitative absence fixture",
        physical_page_index=1,
        chunk_position=0,
        text=sentence,
        normalized_text=sentence,
        sha256="c" * 64,
        token_estimate=5,
    )

    class Corpus:
        corpus_version = "v_0000000000000002"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="d" * 64,
                results=[
                    SearchHit(
                        chunk_id=chunk_id,
                        score=1.0,
                        lexical_score=1.0,
                        lexical_rank=1,
                        retrieval_mode=mode,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        physical_page_index=1,
                        preview=chunk.text,
                        sha256=chunk.sha256,
                    )
                ][:top_k],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk_id in chunk_ids else []

        def find_exact_metric_chunks(self, metric):
            del metric
            return [chunk]

    probe = await probe_metric_absence(Corpus(), "Credit rating")
    verified = EvidenceVerifier(Corpus()).verify_claim(
        DraftClaim(metric="Credit rating", request_absence_probe=True),
        absence_probe=probe,
    )

    assert probe.exact_metric_occurrences
    assert probe.unresolved_predicate_occurrences
    assert not probe.supporting_value_candidates
    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert verified.status == ClaimStatus.UNVERIFIED


@pytest.mark.asyncio
async def test_failed_search_makes_absence_probe_incomplete():
    class Corpus:
        corpus_version = "v_0000000000000003"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder, top_k
            if "fiscal year" in query:
                raise RuntimeError("injected search failure")
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="e" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del chunk_ids, neighbor_radius
            return []

        def find_exact_metric_chunks(self, metric):
            del metric
            return []

    corpus = Corpus()
    probe = await probe_metric_absence(corpus, "Total headcount")
    verified = EvidenceVerifier(corpus).verify_claim(
        DraftClaim(metric="Total headcount", request_absence_probe=True),
        absence_probe=probe,
    )

    assert probe.conclusion == AbsenceConclusion.INCOMPLETE_PROBE
    assert sum(search.completion_status == "completed" for search in probe.searches) == 3
    assert verified.status == ClaimStatus.UNVERIFIED


@pytest.mark.asyncio
async def test_exhaustive_metric_scan_defeats_top_k_or_token_displacement():
    answer_id = chunk_id_from_uint64(2**63 + 500)
    answer = ChunkRecord(
        chunk_id=answer_id,
        document_id="doc_answer",
        document_name="Exact answer",
        physical_page_index=1,
        chunk_position=0,
        text="Net revenue was $2 million.",
        normalized_text="Net revenue was $2 million.",
        sha256="f" * 64,
        token_estimate=6,
    )
    decoys = [
        ChunkRecord(
            chunk_id=chunk_id_from_uint64(2**63 + 600 + index),
            document_id=f"doc_decoy_{index}",
            document_name=f"Decoy {index}",
            physical_page_index=1,
            chunk_position=0,
            text=f"Net income commentary and gross revenue context {index}.",
            normalized_text=f"Net income commentary and gross revenue context {index}.",
            sha256=f"{index + 1:064x}",
            token_estimate=8,
        )
        for index in range(8)
    ]
    by_id = {chunk.chunk_id: chunk for chunk in [*decoys, answer]}
    scanned_metrics: list[str] = []

    class Corpus:
        corpus_version = "v_0000000000000004"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="1" * 64,
                results=[
                    SearchHit(
                        chunk_id=chunk.chunk_id,
                        score=1.0 - index / 100,
                        lexical_score=1.0 - index / 100,
                        lexical_rank=index + 1,
                        retrieval_mode=mode,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        physical_page_index=1,
                        preview=chunk.text,
                        sha256=chunk.sha256,
                    )
                    for index, chunk in enumerate(decoys[:top_k])
                ],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

        def find_exact_metric_chunks(self, metric):
            scanned_metrics.append(metric)
            return [answer, decoys[0]]

    probe = await probe_metric_absence(Corpus(), "net revenue")

    assert probe.exact_metric_scan_completed is True
    assert scanned_metrics == ["net revenue"]
    assert probe.exact_metric_scan_chunk_ids == [answer_id]
    assert answer_id in probe.opened_chunk_ids
    assert probe.supporting_value_candidates
    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW


@pytest.mark.asyncio
async def test_failed_exact_metric_scan_makes_absence_probe_incomplete():
    class Corpus:
        corpus_version = "v_0000000000000005"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder, top_k
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="2" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del chunk_ids, neighbor_radius
            return []

        def find_exact_metric_chunks(self, metric):
            del metric
            raise RuntimeError("injected exact scan failure")

    probe = await probe_metric_absence(Corpus(), "net revenue")

    assert probe.exact_metric_scan_completed is False
    assert probe.exact_metric_scan_error == "RuntimeError"
    assert probe.conclusion == AbsenceConclusion.INCOMPLETE_PROBE


@pytest.mark.asyncio
async def test_exact_metric_scan_runs_off_the_event_loop_thread():
    event_loop_thread = threading.get_ident()
    scan_threads: list[int] = []
    chunk_read_threads: list[int] = []

    class Corpus:
        corpus_version = "v_0000000000000008"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder, top_k
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="7" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del chunk_ids, neighbor_radius
            chunk_read_threads.append(threading.get_ident())
            return []

        def find_exact_metric_chunks(self, metric):
            del metric
            scan_threads.append(threading.get_ident())
            return []

    probe = await probe_metric_absence(Corpus(), "total headcount")

    assert probe.exact_metric_scan_completed is True
    assert scan_threads and scan_threads[0] != event_loop_thread
    assert chunk_read_threads and all(
        thread_id != event_loop_thread for thread_id in chunk_read_threads
    )


@pytest.mark.asyncio
async def test_probe_uses_normalized_metric_and_next_sentence_binding():
    chunk_id = chunk_id_from_uint64(2**63 + 700)
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc_anaphoric",
        document_name="Anaphoric answer",
        physical_page_index=1,
        chunk_position=0,
        text="Fee-earning AUM was stable. It remained at $82 billion.",
        normalized_text="Fee-earning AUM was stable. It remained at $82 billion.",
        sha256="3" * 64,
        token_estimate=10,
    )
    scanned_metrics: list[str] = []

    class Corpus:
        corpus_version = "v_0000000000000006"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="4" * 64,
                results=[
                    SearchHit(
                        chunk_id=chunk_id,
                        score=1.0,
                        lexical_score=1.0,
                        lexical_rank=1,
                        retrieval_mode=mode,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        physical_page_index=1,
                        preview=chunk.text,
                        sha256=chunk.sha256,
                    )
                ][:top_k],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk_id in chunk_ids else []

        def find_exact_metric_chunks(self, metric):
            scanned_metrics.append(metric)
            return [chunk]

    probe = await probe_metric_absence(Corpus(), "Fee-earn-\ning AUM")

    assert probe.exact_metric_occurrences
    assert scanned_metrics == ["Fee-earn-\ning AUM"]
    assert all(search.exact_metric_hit_count == 1 for search in probe.searches)
    assert probe.supporting_value_candidates
    assert all(
        candidate.binding_failure_reason == "numeric_candidate_in_metric_context_requires_review"
        for candidate in probe.supporting_value_candidates
    )
    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW


@pytest.mark.asyncio
async def test_abbreviated_metric_survives_absence_context_scanning():
    chunk_id = chunk_id_from_uint64(2**63 + 800)
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc_abbreviation",
        document_name="Abbreviated metric",
        physical_page_index=1,
        chunk_position=0,
        text="U.S. revenue was $2 million.",
        normalized_text="U.S. revenue was $2 million.",
        sha256="5" * 64,
        token_estimate=6,
    )
    scanned_metrics: list[str] = []

    class Corpus:
        corpus_version = "v_0000000000000007"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder, top_k
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="6" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk_id in chunk_ids else []

        def find_exact_metric_chunks(self, metric):
            scanned_metrics.append(metric)
            return [chunk]

    probe = await probe_metric_absence(Corpus(), "U.S. revenue")

    assert probe.exact_metric_occurrences
    assert scanned_metrics == ["U.S. revenue"]
    assert probe.supporting_value_candidates
    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
