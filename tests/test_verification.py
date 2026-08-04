from __future__ import annotations

from decimal import Context, getcontext, setcontext

import pytest
from hypothesis import given
from hypothesis import strategies as st
from needleproof_api.agent import AGENT_INSTRUCTIONS
from needleproof_api.binding import (
    canonical_decimal_digits,
    canonical_numeric_signature,
    numeric_signature_sequence,
    word_phrase_spans,
)
from needleproof_api.chunk_ids import ChunkId, chunk_id_from_uint64
from needleproof_api.models import (
    ChunkRecord,
    ClaimStatus,
    DraftClaim,
    DraftObservation,
    EvidenceReference,
    EvidenceRelation,
    ObservationKind,
    VerifiedClaim,
)
from needleproof_api.service import compose_authoritative_answer
from needleproof_api.util import normalize_evidence_text
from needleproof_api.verification import (
    EvidenceVerifier,
    _compatible_measurements,
    _distinct_values,
    _measurements_bound_to_metric,
    _temporal_anchor_is_valid,
    _temporal_signature,
    reported_value_found,
    reported_value_linked_to_metric,
)

MEMO_CHUNK = chunk_id_from_uint64(2565635019366042796)
MEMO_CONTINUATION_CHUNK = chunk_id_from_uint64(6812131146285660789)


def reference(
    chunk_id: ChunkId,
    quote: str,
    metric_anchor: str,
    *,
    assertion: str | None = None,
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
) -> EvidenceReference:
    return EvidenceReference(
        chunk_id=chunk_id,
        metric_anchor=metric_anchor,
        exact_quote=quote,
        exact_assertion=assertion or quote,
        relation=relation,
    )


def observation(
    value: str,
    evidence: EvidenceReference,
    *,
    kind: ObservationKind = ObservationKind.REPORTED_LEVEL,
    temporal_anchor: str | None = None,
) -> DraftObservation:
    return DraftObservation(
        kind=kind,
        value_text=value,
        temporal_anchor=temporal_anchor,
        evidence=[evidence],
    )


def claim(metric: str, *observations: DraftObservation) -> DraftClaim:
    return DraftClaim(metric=metric, observations=list(observations))


def corpus_with_chunk(chunk: ChunkRecord):
    class Corpus:
        corpus_version = "v_0000000000000009"

        def get_chunks(self, chunk_ids, neighbor_radius=0):
            del neighbor_radius
            return [chunk] if chunk.chunk_id in chunk_ids else []

    return Corpus()


def test_normalization_records_pdf_linebreak_and_whitespace_operations():
    normalized, operations = normalize_evidence_text("fee-earn-\ning   assets\tunder management")
    assert normalized == "fee-earning assets under management"
    assert operations == ["pdf_linebreak_dehyphenation", "whitespace_folding"]


def test_numeric_value_preserves_full_signature_and_sign():
    quote = "The loss was ($4 billion), not $4 million."
    assert reported_value_found("($4 billion)", quote)
    assert not reported_value_found("$4 billion", quote)
    assert not reported_value_found("$4 million", "The loss was $4 billion.")


def test_direct_value_is_verified_with_diagnostic_binding(corpus):
    quote = (
        "Fee-related earnings were $345 million, up 25 percent, with the FRE margin "
        "improving to 50 percent from 48 percent."
    )
    evidence = reference(
        MEMO_CHUNK,
        quote,
        "Fee-related earnings",
        assertion="Fee-related earnings were $345 million",
    )
    verified = EvidenceVerifier(corpus).verify_claim(
        claim("fee-related earnings", observation("$345 million", evidence))
    )
    assert verified.status == ClaimStatus.VERIFIED
    assert verified.evidence[0].metric_value_bound
    assert verified.evidence[0].binding_profile == "direct_copula"
    assert verified.statement == "Fee-related earnings: $345 million"


@pytest.mark.parametrize(
    ("quote", "assertion"),
    [
        ("Forecast revenue was $2 million.", "revenue was $2 million."),
        ("Revenue was $2 million forecast.", "Revenue was $2 million"),
        ("Revenue was $2 million target.", "Revenue was $2 million"),
        ("Revenue was $2 million, a forecast for next year.", "Revenue was $2 million"),
        ("Revenue was $2 million. This was a forecast.", "Revenue was $2 million."),
        ("Revenue was $2 million. That amount was a target.", "Revenue was $2 million."),
        ("Revenue was $2 million. This was a forecast for 2026.", "Revenue was $2 million."),
        ("Revenue was $2 million. This was a forecast for 2026.", "Revenue was $2 million"),
    ],
)
def test_role_changing_quote_context_cannot_be_cropped_from_assertion(quote, assertion):
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 901),
        document_id="doc_cropped_context",
        document_name="Cropped context fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="9" * 64,
        token_estimate=7,
    )
    evidence = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion=assertion,
    )

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim("Revenue", observation("$2 million", evidence))
    )

    assert verified.status == ClaimStatus.UNVERIFIED
    assert not verified.evidence[0].assertion_found
    assert verified.evidence[0].binding_failure_reason == "assertion_not_bound_to_quote_context"


@given(
    subject=st.sampled_from(
        ["This", "That", "This value", "That amount", "It", "The figure", "The number"]
    ),
    copula=st.sampled_from(["is", "was", "remains"]),
    qualifier=st.from_regex(r"[A-Za-z]{2,18}(?: [A-Za-z]{2,18}){0,3}", fullmatch=True),
    assertion_terminal=st.sampled_from(["", "."]),
)
def test_nonnumeric_anaphoric_followup_cannot_be_cropped(
    subject,
    copula,
    qualifier,
    assertion_terminal,
):
    quote = f"Revenue was $2 million. {subject} {copula} {qualifier}."
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 909),
        document_id="doc_anaphoric_context_property",
        document_name="Anaphoric context property fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="4" * 64,
        token_estimate=12,
    )
    evidence = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion=f"Revenue was $2 million{assertion_terminal}",
    )

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim("Revenue", observation("$2 million", evidence))
    )

    assert verified.status == ClaimStatus.UNVERIFIED
    assert not verified.evidence[0].assertion_found
    assert verified.evidence[0].binding_failure_reason == "assertion_not_bound_to_quote_context"


@given(year=st.integers(min_value=1900, max_value=2100))
def test_temporal_digits_cannot_disguise_cropped_anaphoric_role_context(year):
    quote = f"Revenue was $2 million. This was a forecast for {year}."
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 911),
        document_id="doc_anaphoric_year_property",
        document_name="Anaphoric year property fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="6" * 64,
        token_estimate=11,
    )
    evidence = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million.",
    )

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim("Revenue", observation("$2 million", evidence))
    )

    assert verified.status == ClaimStatus.UNVERIFIED
    assert not verified.evidence[0].assertion_found
    assert verified.evidence[0].binding_failure_reason == "assertion_not_bound_to_quote_context"


def test_following_numeric_observation_does_not_reclassify_prior_assertion():
    quote = "Revenue was $2 million. It was $3 million in 2025."
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 908),
        document_id="doc_numeric_followup",
        document_name="Numeric followup fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="2" * 64,
        token_estimate=11,
    )
    evidence = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million.",
    )

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim("Revenue", observation("$2 million", evidence))
    )

    assert verified.status == ClaimStatus.VERIFIED


def test_modified_quote_is_rejected(corpus):
    quote = "Fee-related earnings were $346 million."
    evidence = reference(MEMO_CHUNK, quote, "Fee-related earnings")
    verified = EvidenceVerifier(corpus).verify_claim(
        claim("Fee-related earnings", observation("$346 million", evidence))
    )
    assert verified.status == ClaimStatus.UNVERIFIED
    assert not verified.evidence[0].quote_found
    assert any("exact quotations" in note for note in verified.verification_notes)


def test_any_evidence_reference_from_another_corpus_version_rejects_claim(corpus):
    quote = "Fee-related earnings were $345 million."
    valid = reference(MEMO_CHUNK, quote, "Fee-related earnings")
    missing = reference(chunk_id_from_uint64(999_999), quote, "Fee-related earnings")
    draft = claim("Fee-related earnings", observation("$345 million", valid))
    draft.context_evidence.append(missing)
    verified = EvidenceVerifier(corpus).verify_claim(draft)
    assert verified.status == ClaimStatus.UNVERIFIED
    assert any("do not resolve" in note for note in verified.verification_notes)


def test_right_number_attached_to_wrong_metric_is_rejected(corpus):
    quote = "Fee-related earnings were $345 million."
    evidence = reference(MEMO_CHUNK, quote, "Fee-related earnings")
    verified = EvidenceVerifier(corpus).verify_claim(
        claim("Revenue", observation("$345 million", evidence))
    )
    assert verified.status == ClaimStatus.UNVERIFIED
    assert not verified.evidence[0].metric_value_bound
    assert any("metric anchors" in note for note in verified.verification_notes)


def test_shared_reference_is_bound_independently_for_each_observation(corpus):
    quote = (
        "Fee-related earnings were $345 million, up 25 percent, with the FRE margin "
        "improving to 50 percent from 48 percent."
    )
    shared = reference(
        MEMO_CHUNK,
        quote,
        "Fee-related earnings",
        assertion="Fee-related earnings were $345 million",
    )
    verified = EvidenceVerifier(corpus).verify_claim(
        claim(
            "Fee-related earnings",
            observation("$999 million", shared),
            observation("$345 million", shared),
        )
    )

    assert verified.status == ClaimStatus.UNVERIFIED
    assert [item.value_text for item in verified.evidence] == [
        "$999 million",
        "$345 million",
    ]
    assert not verified.evidence[0].metric_value_bound
    assert verified.evidence[1].metric_value_bound


@pytest.mark.parametrize(
    "relation",
    [EvidenceRelation.CONTEXTUALIZES, EvidenceRelation.CONTRADICTS],
)
def test_non_supporting_evidence_cannot_authorize_observation(corpus, relation):
    quote = "Fee-related earnings were $345 million."
    evidence = reference(MEMO_CHUNK, quote, "Fee-related earnings", relation=relation)
    verified = EvidenceVerifier(corpus).verify_claim(
        claim("Fee-related earnings", observation("$345 million", evidence))
    )
    assert verified.status == ClaimStatus.UNVERIFIED


@pytest.mark.parametrize(
    "quote",
    [
        "Revenue was flat, increasing expenses by $2 million.",
        "Revenue was flat, unexpectedly increasing expenses by $2 million.",
        "Revenue was flat so operating expenses were $2 million.",
        "Revenue, including $2 million from services, was $10 million.",
        "Revenue, excluding $2 million of pass-through costs, was $10 million.",
        "Revenue was not $2 million.",
        "Revenue was expected to reach $2 million.",
        "Revenue could reach $2 million.",
        "Revenue targeted $2 million.",
        "Revenue was below $2 million.",
        "Revenue was more than $2 million.",
        "Revenue increased by $2 million.",
        "Revenue was flat after operating expenses reached $2 million.",
        "Revenue was flat or operating expenses reached $2 million.",
        "Revenue: $2 million forecast.",
        "Revenue: $2 million target.",
        "Revenue was $2 million expected.",
        "Revenue was $2 million projected.",
        "Revenue was $2 million maximum.",
        "Revenue was $2 million, forecast.",
        "Revenue was $2 million, not the reported actual.",
        "No revenue was $2 million.",
        "Forecast revenue was $2 million.",
        "Target revenue was $2 million.",
        "Projected revenue was $2 million.",
        "Adjusted revenue was $2 million.",
        "Non-GAAP revenue was $2 million.",
        "Company revenue was $2 million.",
    ],
)
def test_unknown_negated_modal_component_delta_and_competing_subject_forms_fail_closed(quote):
    assert reported_value_linked_to_metric("$2 million", "Revenue", quote) is None


@pytest.mark.parametrize(
    "quote",
    [
        "Revenue was $2 million.",
        "Revenue reached $2 million.",
        "Revenue stood at $2 million.",
        "Revenue ended the year at $2 million.",
        "The Revenue was $2 million.",
        "Revenue was stable. It remained at $2 million.",
    ],
)
def test_closed_positive_level_profiles_are_accepted(quote):
    assert reported_value_linked_to_metric("$2 million", "Revenue", quote) is not None


def test_trailing_temporal_anchor_is_part_of_the_same_atomic_binding():
    assert (
        reported_value_linked_to_metric(
            "$2 million",
            "Revenue",
            "Revenue was $2 million as of 31 December 2025.",
            temporal_anchor="31 December 2025",
        )
        is not None
    )


@pytest.mark.parametrize(
    ("quote", "temporal_anchor"),
    [
        ("At year end, Revenue was $2 million.", "year end"),
        ("February call has Revenue at $2 million.", "February call"),
    ],
)
def test_bound_leading_temporal_anchor_may_precede_complete_metric(quote, temporal_anchor):
    assert (
        reported_value_linked_to_metric(
            "$2 million",
            "Revenue",
            quote,
            temporal_anchor=temporal_anchor,
        )
        is not None
    )


@given(
    prefix=st.from_regex(r"[A-Za-z]{1,24}", fullmatch=True).filter(
        lambda value: value.casefold() != "the"
    )
)
def test_unknown_pre_metric_prefix_always_fails_closed(prefix):
    assert (
        reported_value_linked_to_metric(
            "$2 million",
            "Revenue",
            f"{prefix} Revenue was $2 million.",
        )
        is None
    )


@given(suffix=st.from_regex(r"[A-Za-z]{1,24}", fullmatch=True))
def test_unknown_post_value_suffix_always_fails_closed(suffix):
    assert (
        reported_value_linked_to_metric(
            "$2 million",
            "Revenue",
            f"Revenue was $2 million {suffix}.",
        )
        is None
    )


def test_compound_metric_anchor_is_not_split_at_and():
    metric = "Research and development expenses"
    quote = f"{metric} were $2 million."
    assert reported_value_linked_to_metric("$2 million", metric, quote) is not None
    assert reported_value_linked_to_metric("$2 million", "Research", quote) is None


@pytest.mark.parametrize("terminator", [".", "?", "!", ";", ":", ","])
def test_metric_phrase_cannot_cross_clause_or_sentence_terminator(terminator):
    assertion = f"Revenue{terminator} Growth was 10%."

    assert word_phrase_spans("Revenue Growth", assertion) == []
    assert reported_value_linked_to_metric("10%", "Revenue Growth", assertion) is None


@pytest.mark.parametrize("separator", [" ", "\n", "-", "–", "/", " & ", " ( "])
def test_metric_phrase_accepts_only_named_intra_phrase_formatting(separator):
    assertion = f"Fee{separator}related earnings were $2 million."

    assert word_phrase_spans("Fee related earnings", assertion)


def test_metric_phrase_preserves_multi_initial_abbreviation():
    assertion = "U.S. revenue was $2 million."

    assert word_phrase_spans("U S revenue", assertion)
    assert reported_value_linked_to_metric("$2 million", "U.S. revenue", assertion) is not None


@pytest.mark.parametrize(
    "quote,value",
    [
        ("Revenue was not reported and it was $5.", "$5"),
        ("Revenue was $2 million forecast. It was $5 million.", "$5 million"),
        (
            "Revenue was stable. In the next paragraph, operating income was discussed. It was $2 million.",
            "$2 million",
        ),
    ],
)
def test_anaphoric_binding_requires_immediate_positive_antecedent(quote, value):
    assert reported_value_linked_to_metric(value, "Revenue", quote) is None


@pytest.mark.parametrize("boundary", ["?", "!"])
def test_interrogative_or_exclamatory_antecedent_cannot_authorize_anaphora(boundary):
    quote = f"Revenue was $100{boundary} It ended the year at $5."

    assert reported_value_linked_to_metric("$5", "Revenue", quote) is None


@pytest.mark.parametrize(
    "quote,value",
    [
        ("Revenue was reported and it was $5.", "$5"),
        ("Revenue was stable. It remained at $2 million.", "$2 million"),
    ],
)
def test_anaphoric_binding_accepts_closed_positive_profiles(quote, value):
    assert reported_value_linked_to_metric(value, "Revenue", quote) is not None


@pytest.mark.parametrize(
    ("quote", "value", "expected"),
    [
        ("Revenue was $2 million. It was $2 million.", "$2 million", True),
        ("Revenue was $2.0 million. It was $2 million.", "$2 million", True),
        ("Revenue was $2 million. It was $3 million.", "$3 million", False),
        ("Revenue was $2 million. It was 2 million euros.", "2 million euros", False),
    ],
)
def test_numeric_anaphora_requires_the_same_canonical_value(quote, value, expected):
    match = reported_value_linked_to_metric(value, "Revenue", quote)
    assert bool(match) is expected


def test_new_bound_period_allows_a_distinct_anaphoric_value():
    match = reported_value_linked_to_metric(
        "$3 million",
        "Revenue",
        "Revenue was $2 million in 2024. In 2025 it was $3 million.",
        temporal_anchor="2025",
    )
    assert match is not None
    assert match.temporal_span is not None


def test_next_sentence_temporal_lead_cannot_hide_competing_subject():
    quote = (
        "Revenue was $2 million as of 2024. "
        "In 2025 operating income was discussed, and it was $5 million."
    )

    assert (
        reported_value_linked_to_metric(
            "$5 million",
            "Revenue",
            quote,
            temporal_anchor="2025",
        )
        is None
    )


@pytest.mark.parametrize(
    "quote,temporal_anchor",
    [
        ("Forecast Revenue was $2 million.", "Forecast"),
        ("Revenue was $2 million forecast.", "forecast"),
        ("Revenue was $2 million target.", "target"),
        ("Forecast 2025 Revenue was $2 million.", "Forecast 2025"),
        ("Revenue was $2 million in a 2025 forecast.", "2025 forecast"),
        ("February target Revenue was $2 million.", "February target"),
    ],
)
def test_role_qualifiers_cannot_masquerade_as_temporal_anchors(quote, temporal_anchor):
    assert (
        reported_value_linked_to_metric(
            "$2 million",
            "Revenue",
            quote,
            temporal_anchor=temporal_anchor,
        )
        is None
    )
    assert not _temporal_anchor_is_valid(temporal_anchor)


@pytest.mark.parametrize(
    "temporal_anchor",
    [
        "2025",
        "FY 2025",
        "Q1 2026",
        "as of 31 December 2025",
        "for the fiscal year ended March 31, 2026",
        "ended the year",
        "February call",
    ],
)
def test_closed_temporal_anchor_profiles_remain_authorized(temporal_anchor):
    assert _temporal_anchor_is_valid(temporal_anchor)


@given(
    intervening_words=st.lists(
        st.from_regex(r"[A-Za-z]{2,12}", fullmatch=True),
        min_size=1,
        max_size=8,
    )
)
def test_any_intervening_sentence_breaks_next_sentence_anaphora(intervening_words):
    intervening = " ".join(intervening_words)
    quote = f"Revenue was stable. {intervening}. It was $2 million."

    assert reported_value_linked_to_metric("$2 million", "Revenue", quote) is None


@pytest.mark.parametrize("separator", [",", ":"])
def test_qualitative_predicate_does_not_leak_across_metrics(separator):
    quote = f"Revenue was flat{separator} operating expenses were stable."
    assert reported_value_linked_to_metric("stable", "Revenue", quote) is None
    assert reported_value_linked_to_metric("stable", "operating expenses", quote) is None
    assert (
        reported_value_linked_to_metric(
            "stable",
            "operating expenses",
            "operating expenses were stable.",
        )
        is not None
    )


def test_compound_numeric_value_is_rejected_as_ambiguous():
    assert (
        reported_value_linked_to_metric(
            "$2 million and $3 million",
            "Revenue",
            "Revenue was $2 million and $3 million.",
        )
        is None
    )


def test_repeated_identical_values_cannot_exchange_temporal_anchors():
    quote = "Revenue was $2 million in 2024. Operating expenses were $2 million in 2025."
    assert (
        reported_value_linked_to_metric(
            "$2 million",
            "Revenue",
            quote,
            temporal_anchor="2025",
        )
        is None
    )


def test_seeded_aum_values_are_date_variants(corpus):
    quote = (
        "Assets under management were $146.1 billion as of 31 December 2025. "
        "By the fiscal year end on 31 March 2026 the figure was $142 billion, which is "
        "up about $4 billion or 3 percent against the prior year even though it is down "
        "against December."
    )
    first = reference(
        MEMO_CHUNK,
        quote,
        "Assets under management",
        assertion="Assets under management were $146.1 billion as of 31 December 2025.",
    )
    second = reference(
        MEMO_CHUNK,
        quote,
        "Assets under management",
        assertion=(
            "Assets under management were $146.1 billion as of 31 December 2025. "
            "By the fiscal year end on 31 March 2026 the figure was $142 billion"
        ),
    )
    repeated = observation("$146.1 billion", first, temporal_anchor="31 December 2025")
    verified = EvidenceVerifier(corpus).verify_claim(
        claim(
            "Assets under management",
            repeated,
            repeated.model_copy(deep=True),
            observation("$142 billion", second, temporal_anchor="31 March 2026"),
        )
    )
    assert verified.status == ClaimStatus.DATE_VARIANT
    assert verified.statement.count("$146.1 billion") == 1


@given(repetitions=st.integers(min_value=1, max_value=8))
def test_duplicate_observations_cannot_change_temporal_classification(repetitions):
    quote = (
        "Assets under management were $146.1 billion as of 31 December 2025. "
        "By the fiscal year end on 31 March 2026 the figure was $142 billion."
    )
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 910),
        document_id="doc_duplicate_observation_property",
        document_name="Duplicate observation property fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="5" * 64,
        token_estimate=18,
    )
    first_assertion = "Assets under management were $146.1 billion as of 31 December 2025."
    second_assertion = (
        "Assets under management were $146.1 billion as of 31 December 2025. "
        "By the fiscal year end on 31 March 2026 the figure was $142 billion."
    )
    first = reference(
        chunk.chunk_id,
        quote,
        "Assets under management",
        assertion=first_assertion,
    )
    second = reference(
        chunk.chunk_id,
        quote,
        "Assets under management",
        assertion=second_assertion,
    )
    repeated = observation("$146.1 billion", first, temporal_anchor="31 December 2025")
    observations = [repeated.model_copy(deep=True) for _ in range(repetitions)]
    observations.append(observation("$142 billion", second, temporal_anchor="31 March 2026"))

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim("Assets under management", *observations)
    )

    assert verified.status == ClaimStatus.DATE_VARIANT
    assert verified.statement.count("$146.1 billion") == 1


def test_seeded_fee_aum_source_characterization_is_conflict(corpus):
    first_quote = (
        "Fee-earning AUM is the number that actually matters for revenue and it ended "
        "the year at $82 billion, up $9 billion or 13 percent."
    )
    second_quote = (
        "First, my note from the February call has fee-earning AUM at $8.2 billion, "
        "which cannot be right alongside the $82 billion figure above, and I have not "
        "been able to work out which of my two sources introduced the error."
    )
    first = reference(
        MEMO_CHUNK,
        first_quote,
        "Fee-earning AUM",
        assertion=(
            "Fee-earning AUM is the number that actually matters for revenue and it ended "
            "the year at $82 billion"
        ),
    )
    second = reference(
        MEMO_CONTINUATION_CHUNK,
        second_quote,
        "fee-earning AUM",
        assertion="February call has fee-earning AUM at $8.2 billion",
    )
    verified = EvidenceVerifier(corpus).verify_claim(
        claim(
            "Fee-earning AUM",
            observation("$82 billion", first, temporal_anchor="ended the year"),
            observation("$8.2 billion", second, temporal_anchor="February call"),
        )
    )
    assert verified.status == ClaimStatus.CONFLICT


@pytest.mark.parametrize("relation", list(EvidenceRelation))
def test_explicit_conflict_evidence_blocks_single_value_authorization(corpus, relation):
    value_quote = (
        "Fee-earning AUM is the number that actually matters for revenue and it ended "
        "the year at $82 billion, up $9 billion or 13 percent."
    )
    conflict_quote = (
        "First, my note from the February call has fee-earning AUM at $8.2 billion, "
        "which cannot be right alongside the $82 billion figure above, and I have not "
        "been able to work out which of my two sources introduced the error."
    )
    value_evidence = reference(
        MEMO_CHUNK,
        value_quote,
        "Fee-earning AUM",
        assertion=(
            "Fee-earning AUM is the number that actually matters for revenue and it ended "
            "the year at $82 billion"
        ),
    )
    conflict_evidence = reference(
        MEMO_CONTINUATION_CHUNK,
        conflict_quote,
        "fee-earning AUM",
        assertion="February call has fee-earning AUM at $8.2 billion",
        relation=relation,
    )
    draft = claim(
        "Fee-earning AUM",
        observation("$82 billion", value_evidence, temporal_anchor="ended the year"),
    )
    draft.context_evidence.append(conflict_evidence)

    verified = EvidenceVerifier(corpus).verify_claim(draft)

    assert verified.status == ClaimStatus.POSSIBLE_CONFLICT
    assert compose_authoritative_answer([verified], 2, corpus.corpus_version) is None
    assert any("conflict evidence" in note for note in verified.verification_notes)


@pytest.mark.parametrize(
    "quote",
    [
        (
            "Revenue was $2 million in 2024. Revenue was $3 million in 2025. "
            "Our systems are incompatible."
        ),
        (
            "Revenue was $2 million in 2024. Revenue was $3 million in 2025, "
            "while our systems are incompatible."
        ),
        (
            "Revenue was $2 million in 2024. Revenue was $3 million in 2025, "
            "while teams have an unresolved conflict."
        ),
        (
            "Revenue was $2 million in 2024. Revenue was $3 million in 2025, "
            "because dependencies introduced the error."
        ),
        (
            "Revenue was $2 million in 2024. Revenue was $3 million in 2025, "
            "although interfaces are conflicting."
        ),
    ],
)
def test_unrelated_incompatible_systems_do_not_turn_date_variants_into_conflict(quote):
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 901),
        document_id="doc_conflict_locality",
        document_name="Conflict locality fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="9" * 64,
        token_estimate=16,
    )
    first = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million in 2024.",
    )
    second = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $3 million in 2025",
    )

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim(
            "Revenue",
            observation("$2 million", first, temporal_anchor="2024"),
            observation("$3 million", second, temporal_anchor="2025"),
        )
    )

    assert verified.status == ClaimStatus.DATE_VARIANT


@pytest.mark.parametrize("separator", [", and ", "; "])
def test_typed_conflict_characterization_binds_local_distinct_values(separator):
    quote = (
        f"Revenue was $2 million{separator}Revenue was $3 million; the figures are incompatible."
    )
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 903),
        document_id="doc_typed_conflict",
        document_name="Typed conflict fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="7" * 64,
        token_estimate=13,
    )
    first = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million",
    )
    second = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $3 million",
    )

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim(
            "Revenue",
            observation("$2 million", first),
            observation("$3 million", second),
        )
    )

    assert verified.status == ClaimStatus.CONFLICT


def test_year_is_not_mistaken_for_second_unitless_conflict_value():
    quote = "Headcount was 100 in 2024; the figures are incompatible."
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 904),
        document_id="doc_unitless_conflict",
        document_name="Unitless conflict fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="6" * 64,
        token_estimate=9,
    )
    evidence = reference(
        chunk.chunk_id,
        quote,
        "Headcount",
        assertion="Headcount was 100 in 2024",
    )

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim("Headcount", observation("100", evidence, temporal_anchor="2024"))
    )

    assert verified.status == ClaimStatus.VERIFIED


@pytest.mark.parametrize(
    "quote",
    [
        (
            "Revenue was $2 million in 2024. Revenue was $3 million in 2025; "
            "operating expense figures are incompatible."
        ),
        (
            "Revenue was $2 million in 2024. Revenue was $3 million in 2025; "
            "the reports are incompatible with our software."
        ),
    ],
)
def test_collective_conflict_profile_rejects_competing_subjects_and_tails(quote):
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 905),
        document_id="doc_collective_conflict_scope",
        document_name="Collective conflict scope fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="5" * 64,
        token_estimate=16,
    )
    first = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million in 2024.",
    )
    second = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $3 million in 2025",
    )

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim(
            "Revenue",
            observation("$2 million", first, temporal_anchor="2024"),
            observation("$3 million", second, temporal_anchor="2025"),
        )
    )

    assert verified.status == ClaimStatus.DATE_VARIANT


def test_conflict_bridge_cannot_borrow_competing_metric_value():
    quote = (
        "Revenue was $2 million, which cannot be right alongside operating expenses of $3 million."
    )
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 906),
        document_id="doc_bridge_conflict_scope",
        document_name="Bridge conflict scope fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="4" * 64,
        token_estimate=14,
    )
    value_evidence = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million",
    )
    context = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million",
        relation=EvidenceRelation.CONTEXTUALIZES,
    )
    draft = claim("Revenue", observation("$2 million", value_evidence))
    draft.context_evidence.append(context)

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(draft)

    assert verified.status == ClaimStatus.VERIFIED


def test_conflict_bridge_requires_both_disputed_authorized_values():
    quote = (
        "Revenue was $1 million. Revenue was $2 million. "
        "Revenue was $1 million, which cannot be right alongside $3 million."
    )
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 907),
        document_id="doc_bridge_decoy_value",
        document_name="Conflict bridge decoy fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="3" * 64,
        token_estimate=18,
    )
    first = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $1 million.",
    )
    second = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million.",
    )
    context = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        relation=EvidenceRelation.CONTEXTUALIZES,
    )
    draft = claim(
        "Revenue",
        observation("$1 million", first),
        observation("$2 million", second),
    )
    draft.context_evidence.append(context)

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(draft)

    assert verified.status == ClaimStatus.POSSIBLE_CONFLICT


@pytest.mark.parametrize(
    "context_quote",
    [
        "Revenue was $1 million, operating expenses were $2 million; the figures are conflicting.",
        "Revenue was $1 million, headcount costs were $2 million; the values are incompatible.",
    ],
)
def test_collective_conflict_requires_each_measurement_to_belong_to_metric(context_quote):
    quote = f"Revenue was $1 million. Revenue was $2 million. {context_quote}"
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 912),
        document_id="doc_collective_metric_ownership",
        document_name="Collective metric ownership fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="7" * 64,
        token_estimate=24,
    )
    first = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $1 million.",
    )
    second = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million.",
    )
    context = reference(
        chunk.chunk_id,
        context_quote,
        "Revenue",
        relation=EvidenceRelation.CONTEXTUALIZES,
    )
    draft = claim(
        "Revenue",
        observation("$1 million", first),
        observation("$2 million", second),
    )
    draft.context_evidence.append(context)

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(draft)

    assert verified.status == ClaimStatus.POSSIBLE_CONFLICT


def test_collective_conflict_accepts_two_locally_bound_same_metric_measurements():
    support_quote = "Revenue was $1 million. Revenue was $2 million."
    context_quote = "Revenue was $1 million, Revenue was $2 million; the figures are conflicting."
    quote = f"{support_quote} {context_quote}"
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 913),
        document_id="doc_collective_same_metric",
        document_name="Collective same-metric fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="8" * 64,
        token_estimate=21,
    )
    first = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $1 million.",
    )
    second = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million.",
    )
    context = reference(
        chunk.chunk_id,
        context_quote,
        "Revenue",
        relation=EvidenceRelation.CONTEXTUALIZES,
    )
    draft = claim(
        "Revenue",
        observation("$1 million", first),
        observation("$2 million", second),
    )
    draft.context_evidence.append(context)

    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(draft)

    assert verified.status == ClaimStatus.CONFLICT


@given(
    competing_metric=st.from_regex(
        r"[A-Za-z]{3,14}(?: [A-Za-z]{3,14}){0,2}",
        fullmatch=True,
    ).filter(lambda value: "revenue" not in value.casefold())
)
def test_competing_metric_mutations_never_own_collective_revenue_value(competing_metric):
    sentence = (
        f"Revenue was $1 million, {competing_metric} was $2 million; the figures are conflicting."
    )
    one = canonical_numeric_signature(numeric_signature_sequence("$1 million")[0])
    two = canonical_numeric_signature(numeric_signature_sequence("$2 million")[0])
    measurements = _compatible_measurements(sentence, {one, two})

    owned = _measurements_bound_to_metric(
        sentence,
        "Revenue",
        measurements,
        {
            one: {ObservationKind.REPORTED_LEVEL},
            two: {ObservationKind.REPORTED_LEVEL},
        },
    )

    assert {signature for signature, _span in owned} == {one}


def test_equivalent_quarter_anchors_classify_distinct_values_as_conflict():
    quote = "Revenue was $2 million in Q1 2025. Revenue was $3 million in first quarter 2025."
    chunk = ChunkRecord(
        chunk_id=chunk_id_from_uint64(2**63 + 902),
        document_id="doc_temporal_equivalence",
        document_name="Temporal equivalence fixture",
        physical_page_index=1,
        chunk_position=0,
        text=quote,
        normalized_text=quote,
        sha256="8" * 64,
        token_estimate=14,
    )
    first = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $2 million in Q1 2025.",
    )
    second = reference(
        chunk.chunk_id,
        quote,
        "Revenue",
        assertion="Revenue was $3 million in first quarter 2025.",
    )

    repeated = observation("$2 million", first, temporal_anchor="Q1 2025")
    verified = EvidenceVerifier(corpus_with_chunk(chunk)).verify_claim(
        claim(
            "Revenue",
            repeated,
            repeated.model_copy(deep=True),
            observation("$3 million", second, temporal_anchor="first quarter 2025"),
        )
    )

    assert verified.status == ClaimStatus.CONFLICT
    assert verified.statement.count("$2 million") == 1


def test_unresolved_distinct_values_do_not_enter_authoritative_answer(corpus):
    unresolved = VerifiedClaim(
        statement="Fee-earning AUM has unresolved reported values: $82 billion and $8.2 billion",
        metric="Fee-earning AUM",
        status=ClaimStatus.POSSIBLE_CONFLICT,
    )
    assert compose_authoritative_answer([unresolved], 2, corpus.corpus_version) is None


def test_impossible_dates_are_not_temporal_signatures():
    assert _temporal_signature("31 February 2025") is None
    assert _temporal_signature("2025-13-01") is None
    assert not _temporal_anchor_is_valid("31 February 2025")
    assert not _temporal_anchor_is_valid("2025-13-01")


def test_equivalent_date_formats_have_one_signature():
    assert _temporal_signature("31 December 2025") == _temporal_signature("2025-12-31")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Q1 2025", "first quarter 2025"),
        ("Q2 2025", "second quarter 2025"),
        ("Q3 2025", "third quarter 2025"),
        ("Q4 2025", "fourth quarter 2025"),
        ("Q1", "first quarter"),
        ("2025", "calendar year 2025"),
        ("2025", "year 2025"),
        ("FY2025", "fiscal year 2025"),
        ("FY", "fiscal year"),
        ("year", "calendar year"),
        ("February 2025", "February 2025 month"),
        ("February 2025", "in February 2025"),
        ("at year end", "year end"),
        ("during the fiscal quarter", "the fiscal quarter"),
    ],
)
def test_equivalent_named_periods_have_one_signature(left, right):
    assert _temporal_signature(left) == _temporal_signature(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Q1 2025", "Q2 2025"),
        ("FY2025", "calendar year 2025"),
        ("February 2025", "March 2025"),
        ("February call", "February 2025"),
    ],
)
def test_semantically_distinct_period_shapes_remain_distinct(left, right):
    assert _temporal_signature(left) != _temporal_signature(right)


def test_decimal_digit_canonicalization_is_context_independent():
    original = getcontext().copy()
    try:
        setcontext(Context(prec=2))
        assert canonical_decimal_digits("00012345678901234567890.000") == "12345678901234567890"
    finally:
        setcontext(original)


def test_two_hundred_digit_adjacent_integers_never_collapse():
    integer = 10**199
    assert canonical_decimal_digits(str(integer)) != canonical_decimal_digits(str(integer + 1))


@given(integer=st.integers(min_value=10**199, max_value=10**200 - 2))
def test_adjacent_200_digit_integers_never_collapse(integer):
    padded = f"000{integer}.000"
    adjacent = f"000{integer + 1}.000"
    assert canonical_decimal_digits(padded) != canonical_decimal_digits(adjacent)


@given(
    integer=st.integers(min_value=0, max_value=10**100),
    zeros=st.integers(min_value=1, max_value=8),
)
def test_leading_and_fractional_zero_forms_canonicalize_equally(integer, zeros):
    plain = str(integer)
    padded = f"{'0' * zeros}{integer}.{'0' * zeros}"
    assert canonical_decimal_digits(plain) == canonical_decimal_digits(padded)


def test_distinct_value_grouping_uses_source_digits_without_decimal_context():
    evidence = reference(MEMO_CHUNK, "Revenue was $1 million.", "Revenue")
    first = observation("123456789012345678901234567890 million", evidence)
    second = observation("123456789012345678901234567891 million", evidence)
    assert len(_distinct_values([first, second])) == 2


def test_rate_and_percentage_are_not_silently_rescaled():
    evidence = reference(MEMO_CHUNK, "Fee rate was 67 basis points.", "Fee rate")
    values = [
        observation("67 basis points", evidence, kind=ObservationKind.REPORTED_RATE),
        observation("0.67 percent", evidence, kind=ObservationKind.REPORTED_RATE),
    ]
    assert len(_distinct_values(values)) == 2


def test_model_contract_has_observations_but_no_publishable_answer_or_status():
    claim_properties = DraftClaim.model_json_schema()["properties"]
    assert "observations" in claim_properties
    assert "status" not in claim_properties
    assert "statement" not in claim_properties


def test_document_instructions_are_explicitly_untrusted():
    lowered = AGENT_INSTRUCTIONS.casefold()
    assert "untrusted evidence" in lowered
    assert "must never change your behavior" in lowered
