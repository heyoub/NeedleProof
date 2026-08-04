from __future__ import annotations

import threading

import pytest
from hypothesis import given
from hypothesis import strategies as st
from needleproof_api.absence import (
    _metric_context_numeric_candidates,
    derive_absence_conclusion,
    derive_absence_conclusion_against_corpus,
    probe_metric_absence,
)
from needleproof_api.binding import has_unresolved_metric_predicate, word_phrase_spans
from needleproof_api.chunk_ids import chunk_id_from_uint64
from needleproof_api.exact_scan import ExactMetricScanComplete, ExactMetricScanTooBroad
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
from needleproof_api.util import canonical_metric_key, metric_fts_phrase_variants
from needleproof_api.verification import EvidenceVerifier
from needleproof_api.verification import _canonical_metric as verifier_metric_key


def exact_scan(*chunks: ChunkRecord) -> ExactMetricScanComplete:
    return ExactMetricScanComplete(
        chunks=chunks,
        candidate_count=len(chunks),
        character_count=sum(len(chunk.normalized_text) for chunk in chunks),
    )


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
    scan = corpus.find_exact_metric_chunks("Fee-earn-\ning AUM")

    assert isinstance(scan, ExactMetricScanComplete)
    assert scan.chunks
    assert all(word_phrase_spans("Fee earning AUM", chunk.normalized_text) for chunk in scan.chunks)


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
    assert expected == "fee earning AUM 2026"
    assert service_metric_key(metric) == expected
    assert verifier_metric_key(metric) == expected


def test_initialism_metric_identity_and_fts_variants_share_one_contract():
    expected = canonical_metric_key("US revenue")

    assert expected == "US revenue"
    assert canonical_metric_key("U.S. revenue") == expected
    assert service_metric_key("U.S. revenue") == expected
    assert verifier_metric_key("U.S. revenue") == expected
    assert metric_fts_phrase_variants("US revenue") == ("us revenue", "u s revenue")
    assert metric_fts_phrase_variants("U.S. revenue") == ("us revenue", "u s revenue")
    assert metric_fts_phrase_variants("us revenue") == ("us revenue",)


def test_initialism_and_ordinary_word_have_distinct_absence_identities_and_fts_shapes():
    assert service_metric_key("IT") != service_metric_key("It")
    assert verifier_metric_key("IT") != verifier_metric_key("It")
    assert metric_fts_phrase_variants("IT") == ("it", "i t")
    assert metric_fts_phrase_variants("It") == ("it",)


def test_many_initialisms_fail_over_to_bounded_exhaustive_scan():
    metric = " ".join(["US"] * 7)

    assert metric_fts_phrase_variants(metric) is None


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
            return exact_scan(chunk)

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
            return exact_scan(chunk)

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
            return exact_scan(chunk)

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
            return exact_scan()

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
@pytest.mark.parametrize(
    ("metric_text", "include_neighbor", "expected"),
    [
        ("Revenue", True, AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW),
        ("Revenue.", True, AbsenceConclusion.NOT_FOUND_IN_PROBE),
        ("Revenue", False, AbsenceConclusion.INCOMPLETE_PROBE),
    ],
)
async def test_absence_probe_handles_metric_values_split_across_chunk_edges(
    metric_text,
    include_neighbor,
    expected,
):
    metric_id = chunk_id_from_uint64(2**63 + 31)
    value_id = chunk_id_from_uint64(2**63 + 32)
    metric_chunk = ChunkRecord(
        chunk_id=metric_id,
        document_id="doc_split_metric",
        document_name="Split metric fixture",
        physical_page_index=1,
        chunk_position=0,
        text=metric_text,
        normalized_text=metric_text,
        next_chunk_id=value_id,
        sha256="1" * 64,
        token_estimate=1,
    )
    value_chunk = ChunkRecord(
        chunk_id=value_id,
        document_id="doc_split_metric",
        document_name="Split metric fixture",
        physical_page_index=1,
        chunk_position=1,
        text="was $2 million.",
        normalized_text="was $2 million.",
        previous_chunk_id=metric_id,
        sha256="2" * 64,
        token_estimate=4,
    )

    class Corpus:
        corpus_version = "v_0000000000000031"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder, top_k
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="3" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            if metric_id not in chunk_ids:
                return []
            if neighbor_radius and include_neighbor:
                return [metric_chunk, value_chunk]
            return [metric_chunk]

        def find_exact_metric_chunks(self, metric):
            del metric
            return exact_scan(metric_chunk)

    probe = await probe_metric_absence(Corpus(), "Revenue")

    assert probe.conclusion == expected
    if include_neighbor:
        assert value_id in probe.opened_chunk_ids
    else:
        assert probe.exact_metric_scan_error == "NeighborChunkMissing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("local_text", "linked_text", "expected_value", "expected_conclusion"),
    [
        (
            "Fee-earning AUM. It remained at",
            "$82 billion.",
            "$82 billion",
            AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW,
        ),
        (
            "Fee-earning AUM. It was",
            "500 dollars.",
            "500",
            AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW,
        ),
        (
            "Fee-earning AUM. It remained at",
            "the prior level. It was $82 billion.",
            None,
            AbsenceConclusion.INCOMPLETE_PROBE,
        ),
        (
            "Fee-earning AUM. It remained at",
            "the prior level. They were $82 billion.",
            None,
            AbsenceConclusion.INCOMPLETE_PROBE,
        ),
        (
            "Fee-earning AUM. It remained at",
            "the prior level. These figures were $82 billion.",
            None,
            AbsenceConclusion.INCOMPLETE_PROBE,
        ),
        (
            "Fee-earning AUM. It remained at",
            "the prior level. Those values were $82 billion.",
            None,
            AbsenceConclusion.INCOMPLETE_PROBE,
        ),
        (
            "Fee-earning AUM. It remained at",
            "the prior level. As of FY 2025, it was $82 billion.",
            None,
            AbsenceConclusion.INCOMPLETE_PROBE,
        ),
        (
            "Fee-earning AUM. It remained at",
            "the prior level. The metric was $82 billion.",
            None,
            AbsenceConclusion.INCOMPLETE_PROBE,
        ),
    ],
)
async def test_open_anaphoric_continuation_extends_into_linked_chunk(
    local_text,
    linked_text,
    expected_value,
    expected_conclusion,
):
    metric_id = chunk_id_from_uint64(2**63 + 41)
    value_id = chunk_id_from_uint64(2**63 + 42)
    metric_chunk = ChunkRecord(
        chunk_id=metric_id,
        document_id="doc_split_anaphor",
        document_name="Split anaphor fixture",
        physical_page_index=1,
        chunk_position=0,
        text=local_text,
        normalized_text=local_text,
        next_chunk_id=value_id,
        sha256="4" * 64,
        token_estimate=6,
    )
    value_chunk = ChunkRecord(
        chunk_id=value_id,
        document_id="doc_split_anaphor",
        document_name="Split anaphor fixture",
        physical_page_index=1,
        chunk_position=1,
        text=linked_text,
        normalized_text=linked_text,
        previous_chunk_id=metric_id,
        sha256="5" * 64,
        token_estimate=3,
    )

    class Corpus:
        corpus_version = "v_0000000000000041"

        async def search(self, query, *, mode, top_k, recorder=None):
            del query, recorder, top_k
            return SearchResult(
                query="Fee-earning AUM",
                mode=mode,
                corpus_manifest_sha256="6" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            chunks = []
            if metric_id in chunk_ids:
                chunks.append(metric_chunk)
            if value_id in chunk_ids or (neighbor_radius and metric_id in chunk_ids):
                chunks.append(value_chunk)
            return chunks

        def find_exact_metric_chunks(self, metric):
            assert metric == "Fee-earning AUM"
            return exact_scan(metric_chunk)

    corpus = Corpus()
    probe = await probe_metric_absence(corpus, "Fee-earning AUM")

    assert probe.conclusion == expected_conclusion
    assert (
        any(
            candidate.value_text == expected_value
            for candidate in probe.supporting_value_candidates
        )
        if expected_value
        else not probe.supporting_value_candidates
    )
    assert derive_absence_conclusion_against_corpus(probe, corpus) == expected_conclusion
    forged = probe.model_copy(
        update={
            "supporting_value_candidates": [],
            "unresolved_predicate_occurrences": [],
            "conclusion": AbsenceConclusion.NOT_FOUND_IN_PROBE,
        }
    )
    verified = EvidenceVerifier(corpus).verify_claim(
        DraftClaim(metric="Fee-earning AUM", request_absence_probe=True),
        absence_probe=forged,
    )
    assert verified.status == ClaimStatus.UNVERIFIED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "linked_text",
    [
        "It was $82 billion.",
        "They were $82 billion.",
        "These figures were $82 billion.",
        "Those values were $82 billion.",
        "As of FY 2025, it was $82 billion.",
        "The metric was $82 billion.",
    ],
)
async def test_metric_at_chunk_end_follows_linked_anaphoric_value(linked_text):
    metric_id = chunk_id_from_uint64(2**63 + 43)
    value_id = chunk_id_from_uint64(2**63 + 44)
    metric_chunk = ChunkRecord(
        chunk_id=metric_id,
        document_id="doc_linked_anaphor",
        document_name="Linked anaphor fixture",
        physical_page_index=1,
        chunk_position=0,
        text="Fee-earning AUM.",
        normalized_text="Fee-earning AUM.",
        next_chunk_id=value_id,
        sha256="6" * 64,
        token_estimate=3,
    )
    value_chunk = ChunkRecord(
        chunk_id=value_id,
        document_id="doc_linked_anaphor",
        document_name="Linked anaphor fixture",
        physical_page_index=1,
        chunk_position=1,
        text=linked_text,
        normalized_text=linked_text,
        previous_chunk_id=metric_id,
        sha256="7" * 64,
        token_estimate=5,
    )

    class Corpus:
        corpus_version = "v_0000000000000043"

        async def search(self, query, *, mode, top_k, recorder=None):
            del query, recorder, top_k
            return SearchResult(
                query="Fee-earning AUM",
                mode=mode,
                corpus_manifest_sha256="8" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            chunks = []
            if metric_id in chunk_ids:
                chunks.append(metric_chunk)
            if value_id in chunk_ids or (neighbor_radius and metric_id in chunk_ids):
                chunks.append(value_chunk)
            return chunks

        def find_exact_metric_chunks(self, metric):
            assert metric == "Fee-earning AUM"
            return exact_scan(metric_chunk)

    corpus = Corpus()
    probe = await probe_metric_absence(corpus, "Fee-earning AUM")

    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert any(
        candidate.value_text == "$82 billion" for candidate in probe.supporting_value_candidates
    )
    assert (
        derive_absence_conclusion_against_corpus(probe, corpus)
        == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_lead",
    [
        "It was",
        "They were",
        "These figures were",
        "Those values were",
        "As of FY 2025, it was",
        "The reported metric was",
    ],
)
async def test_second_same_chunk_anaphor_makes_absence_incomplete(second_lead):
    chunk_id = chunk_id_from_uint64(2**63 + 45)
    text = f"Fee-earning AUM. It remained at the prior level. {second_lead} $82 billion."
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc_anaphoric_chain",
        document_name="Anaphoric chain fixture",
        physical_page_index=1,
        chunk_position=0,
        text=text,
        normalized_text=text,
        sha256="9" * 64,
        token_estimate=14,
    )

    class Corpus:
        corpus_version = "v_0000000000000045"

        async def search(self, query, *, mode, top_k, recorder=None):
            del query, recorder, top_k
            return SearchResult(
                query="Fee-earning AUM",
                mode=mode,
                corpus_manifest_sha256="a" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk_id in chunk_ids else []

        def find_exact_metric_chunks(self, metric):
            assert metric == "Fee-earning AUM"
            return exact_scan(chunk)

    corpus = Corpus()
    probe = await probe_metric_absence(corpus, "Fee-earning AUM")

    assert probe.conclusion == AbsenceConclusion.INCOMPLETE_PROBE
    assert (
        derive_absence_conclusion_against_corpus(probe, corpus)
        == AbsenceConclusion.INCOMPLETE_PROBE
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Fee-earning AUM. It was $82 billion",
        "Fee-earning AUM. They were $82 billion",
        "Fee-earning AUM. These figures were $82 billion",
        "Fee-earning AUM. Those values were $82 billion",
        "Fee-earning AUM. As of FY 2025, it was $82 billion",
        "Fee-earning AUM. The metric was $82 billion",
    ],
)
async def test_terminal_unpunctuated_anaphor_is_analyzed_without_crashing(text):
    chunk_id = chunk_id_from_uint64(2**63 + 46)
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc_terminal_anaphor",
        document_name="Terminal anaphor fixture",
        physical_page_index=1,
        chunk_position=0,
        text=text,
        normalized_text=text,
        sha256="b" * 64,
        token_estimate=8,
    )

    class Corpus:
        corpus_version = "v_0000000000000046"

        async def search(self, query, *, mode, top_k, recorder=None):
            del query, recorder, top_k
            return SearchResult(
                query="Fee-earning AUM",
                mode=mode,
                corpus_manifest_sha256="c" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk_id in chunk_ids else []

        def find_exact_metric_chunks(self, metric):
            assert metric == "Fee-earning AUM"
            return exact_scan(chunk)

    corpus = Corpus()
    probe = await probe_metric_absence(corpus, "Fee-earning AUM")

    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert (
        derive_absence_conclusion_against_corpus(probe, corpus)
        == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "linked_text",
    [
        "It was $82 billion.",
        "They were $82 billion.",
        "These figures were $82 billion.",
        "Those values were $82 billion.",
        "During the period, they were $82 billion.",
        "The reported metric stood at $82 billion.",
    ],
)
async def test_completed_local_anaphor_checks_linked_coreferential_continuation(
    linked_text,
):
    metric_id = chunk_id_from_uint64(2**63 + 47)
    value_id = chunk_id_from_uint64(2**63 + 48)
    metric_chunk = ChunkRecord(
        chunk_id=metric_id,
        document_id="doc_completed_local_anaphor",
        document_name="Completed local anaphor fixture",
        physical_page_index=1,
        chunk_position=0,
        text="Fee-earning AUM. It remained at the prior level.",
        normalized_text="Fee-earning AUM. It remained at the prior level.",
        next_chunk_id=value_id,
        sha256="d" * 64,
        token_estimate=9,
    )
    value_chunk = ChunkRecord(
        chunk_id=value_id,
        document_id="doc_completed_local_anaphor",
        document_name="Completed local anaphor fixture",
        physical_page_index=1,
        chunk_position=1,
        text=linked_text,
        normalized_text=linked_text,
        previous_chunk_id=metric_id,
        sha256="e" * 64,
        token_estimate=5,
    )

    class Corpus:
        corpus_version = "v_0000000000000047"

        async def search(self, query, *, mode, top_k, recorder=None):
            del query, recorder, top_k
            return SearchResult(
                query="Fee-earning AUM",
                mode=mode,
                corpus_manifest_sha256="f" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            chunks = []
            if metric_id in chunk_ids:
                chunks.append(metric_chunk)
            if value_id in chunk_ids or (neighbor_radius and metric_id in chunk_ids):
                chunks.append(value_chunk)
            return chunks

        def find_exact_metric_chunks(self, metric):
            assert metric == "Fee-earning AUM"
            return exact_scan(metric_chunk)

    corpus = Corpus()
    probe = await probe_metric_absence(corpus, "Fee-earning AUM")

    assert probe.conclusion == AbsenceConclusion.INCOMPLETE_PROBE
    assert (
        derive_absence_conclusion_against_corpus(probe, corpus)
        == AbsenceConclusion.INCOMPLETE_PROBE
    )


@pytest.mark.asyncio
async def test_linked_anaphoric_chain_beyond_opened_neighbor_is_incomplete():
    metric_id = chunk_id_from_uint64(2**63 + 49)
    first_anaphor_id = chunk_id_from_uint64(2**63 + 50)
    hidden_value_id = chunk_id_from_uint64(2**63 + 51)
    metric_chunk = ChunkRecord(
        chunk_id=metric_id,
        document_id="doc_deep_anaphoric_chain",
        document_name="Deep anaphoric chain fixture",
        physical_page_index=1,
        chunk_position=0,
        text="Fee-earning AUM.",
        normalized_text="Fee-earning AUM.",
        next_chunk_id=first_anaphor_id,
        sha256="1" * 64,
        token_estimate=3,
    )
    first_anaphor = ChunkRecord(
        chunk_id=first_anaphor_id,
        document_id="doc_deep_anaphoric_chain",
        document_name="Deep anaphoric chain fixture",
        physical_page_index=1,
        chunk_position=1,
        text="It remained at the prior level.",
        normalized_text="It remained at the prior level.",
        previous_chunk_id=metric_id,
        next_chunk_id=hidden_value_id,
        sha256="2" * 64,
        token_estimate=7,
    )

    class Corpus:
        corpus_version = "v_0000000000000049"

        async def search(self, query, *, mode, top_k, recorder=None):
            del query, recorder, top_k
            return SearchResult(
                query="Fee-earning AUM",
                mode=mode,
                corpus_manifest_sha256="3" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            chunks = []
            if metric_id in chunk_ids:
                chunks.append(metric_chunk)
            if first_anaphor_id in chunk_ids or (neighbor_radius and metric_id in chunk_ids):
                chunks.append(first_anaphor)
            return chunks

        def find_exact_metric_chunks(self, metric):
            assert metric == "Fee-earning AUM"
            return exact_scan(metric_chunk)

    corpus = Corpus()
    probe = await probe_metric_absence(corpus, "Fee-earning AUM")

    assert probe.conclusion == AbsenceConclusion.INCOMPLETE_PROBE
    assert (
        derive_absence_conclusion_against_corpus(probe, corpus)
        == AbsenceConclusion.INCOMPLETE_PROBE
    )


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
            return exact_scan(answer, decoys[0])

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
            return exact_scan()

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
            return exact_scan(chunk)

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
@pytest.mark.parametrize("probe_metric", ["U.S. revenue", "US revenue"])
async def test_abbreviated_metric_survives_absence_context_scanning(probe_metric):
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
            return exact_scan(chunk)

    probe = await probe_metric_absence(Corpus(), probe_metric)

    assert probe.exact_metric_occurrences
    assert scanned_metrics == [probe_metric]
    assert probe.supporting_value_candidates
    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW


@pytest.mark.asyncio
@pytest.mark.parametrize("distance", [383, 384, 385, 610])
async def test_complete_same_chunk_context_blocks_absence_beyond_display_window(distance):
    chunk_id = chunk_id_from_uint64(2**63 + 900 + distance)
    metric = "Total headcount"
    gap = " " + ("x" * (distance - 6)) + " was "
    assert len(gap) == distance
    text = f"{metric}{gap}500 employees."
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc_long_context",
        document_name="Long context",
        physical_page_index=1,
        chunk_position=0,
        text=text,
        normalized_text=text,
        sha256="7" * 64,
        token_estimate=20,
    )

    class Corpus:
        corpus_version = "v_0000000000000008"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder, top_k
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="8" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk_id in chunk_ids else []

        def find_exact_metric_chunks(self, scanned_metric):
            assert scanned_metric == metric
            return exact_scan(chunk)

    probe = await probe_metric_absence(Corpus(), metric)

    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert probe.supporting_value_candidates
    forged = probe.model_copy(
        update={
            "supporting_value_candidates": [],
            "unresolved_predicate_occurrences": [],
            "conclusion": AbsenceConclusion.NOT_FOUND_IN_PROBE,
        }
    )
    verified = EvidenceVerifier(Corpus()).verify_claim(
        DraftClaim(metric=metric, request_absence_probe=True),
        absence_probe=forged,
    )
    assert verified.status == ClaimStatus.UNVERIFIED
    if distance > 384:
        assert "500" not in probe.exact_metric_occurrences[0].sentence


@pytest.mark.asyncio
async def test_too_broad_exact_scan_is_typed_incomplete_not_absence():
    class Corpus:
        corpus_version = "v_0000000000000008"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder, top_k
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="9" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del chunk_ids, neighbor_radius
            return []

        def find_exact_metric_chunks(self, metric):
            del metric
            return ExactMetricScanTooBroad(candidate_count=1001, character_count=50_000)

    probe = await probe_metric_absence(Corpus(), "rate")

    assert probe.exact_metric_scan_completed is False
    assert probe.exact_metric_scan_error == "ExactScanTooBroad"
    assert probe.conclusion == AbsenceConclusion.INCOMPLETE_PROBE


@pytest.mark.asyncio
async def test_kind_ambiguous_exact_scan_cannot_authorize_absence():
    class Corpus:
        corpus_version = "v_0000000000000008"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder, top_k
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="d" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del chunk_ids, neighbor_radius
            return []

        def find_exact_metric_chunks(self, metric):
            del metric
            return ExactMetricScanComplete(
                chunks=(),
                candidate_count=1,
                character_count=24,
                ambiguous_candidate_count=1,
            )

    probe = await probe_metric_absence(Corpus(), "Revenue")

    assert probe.exact_metric_scan_completed is False
    assert probe.exact_metric_scan_error == "MetricTokenKindAmbiguous"
    assert probe.conclusion == AbsenceConclusion.INCOMPLETE_PROBE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Total headcount at Acme Inc. was 500 employees.",
        "Total headcount was approx. 500 employees.",
    ],
)
async def test_ambiguous_abbreviation_punctuation_cannot_hide_same_chunk_value(text):
    chunk_id = chunk_id_from_uint64(2**63 + 1600)
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc_abbreviation_context",
        document_name="Abbreviation context",
        physical_page_index=1,
        chunk_position=0,
        text=text,
        normalized_text=text,
        sha256="a" * 64,
        token_estimate=10,
    )

    class Corpus:
        corpus_version = "v_0000000000000008"

        async def search(self, query, *, mode, top_k, recorder=None):
            del recorder, top_k
            return SearchResult(
                query=query,
                mode=mode,
                corpus_manifest_sha256="b" * 64,
                results=[],
            )

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk_id in chunk_ids else []

        def find_exact_metric_chunks(self, metric):
            del metric
            return exact_scan(chunk)

    probe = await probe_metric_absence(Corpus(), "Total headcount")

    assert probe.conclusion == AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    assert probe.supporting_value_candidates
