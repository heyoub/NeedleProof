from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from .binding import (
    has_unresolved_metric_predicate,
    numeric_value_candidates,
    word_phrase_spans,
)
from .chunk_ids import ChunkId
from .exact_scan import (
    EXACT_METRIC_SCAN_MAX_CANDIDATES,
    EXACT_METRIC_SCAN_MAX_CHARACTERS,
    ExactMetricScanAmbiguous,
    ExactMetricScanComplete,
    ExactMetricScanFailed,
    ExactMetricScanOutcome,
    ExactMetricScanTooBroad,
)
from .models import (
    AbsenceConclusion,
    AbsenceProbeResult,
    BenignMentionProfile,
    BenignMetricMention,
    ChunkRecord,
    CompletedSearchRecord,
    MetricEvidenceCandidateOccurrence,
    MetricEvidenceReason,
    MetricOccurrence,
    MetricOccurrenceAnalysis,
    SearchResult,
    UnknownMetricOccurrence,
    UnknownOccurrenceReason,
    ValueCandidate,
)
from .util import (
    MAX_METRIC_FTS_VARIANTS,
    canonical_json,
    canonical_metric_key,
    punctuation_boundaries,
    sha256_text,
)

ABSENCE_PROTOCOL_VERSION = "bounded-absence-v26-categorical-scan-ambiguity"
ABSENCE_METRIC_CONTEXT_CHARACTERS = 384
ABSENCE_MIN_TOP_K = 8
ABSENCE_CONTEXT_MAX_OPENED_CHUNKS = 50
_VALUE_FIRST_METRIC_BRIDGE = re.compile(
    r"(?:\s*|\s*[,;:—–-]\s*|\s*[([{\"'“‘]\s*|"
    r"\s*(?:in|of)\s+(?:[^\W\d_]+\s+){0,3})",
    re.IGNORECASE,
)
_POTENTIAL_ANAPHORIC_PREDICATE_LEAD = re.compile(
    r"\s*(?:(?:by|at|as\s+of|on|for|in|during|through)\b[^.!?]*?\s+)?"
    r"(?:(?:it|they|this|that|these|those)\b|"
    r"(?:the|these|those)\s+(?:[^\W\d_]+(?:[-\s]+[^\W\d_]+){0,3}))"
    r"\s+(?:is|are|was|were|remained(?:\s+at)?|stayed(?:\s+at)?|"
    r"reached|stood\s+at|amounted\s+to|reported(?:\s+at)?|"
    r"totaled|totalled|ended(?:\s+the\s+(?:year|quarter|month|period))?\s+at)\b",
    re.IGNORECASE,
)
_QUALITATIVE_WORD = re.compile(r"[^\W\d_]+")
_BENIGN_RUBRIC_REFERENCE = re.compile(
    r"\b(?:rubric|instruction|guide|prompt|request(?:ed)?|extract|"
    r"mention(?:ed|s)?|list(?:ed|s)?)\b",
    re.IGNORECASE,
)
_BENIGN_SOURCE_LABEL = re.compile(
    r"^\s*[-—–:]\s*[\"'“‘]?(?:source\s+)?(?:quote|passage|text)\b",
    re.IGNORECASE,
)
_BENIGN_STANDALONE_REMAINDER = re.compile(r"^[\s\d.)\]}>:—–-]*$")
_POTENTIAL_UNKNOWN_STRUCTURAL_REFERENCE_LEAD = re.compile(
    r"\s*(?:(?:by|at|as\s+of|on|for|in|during|through)\b[^.!?]*?\s+)?"
    r"(?:(?:this|that|these|those|its|their|his|her|such)\b|"
    r"(?:the\s+)?(?:[^\W\d_]+(?:[-\s]+[^\W\d_]+){0,3})(?:'s|’s)\b)",
    re.IGNORECASE,
)
_POTENTIAL_FOLLOWING_ASSERTION = re.compile(
    r"\s*(?:(?:by|at|as\s+of|on|for|in|during|through)\b[^.!?]*?\s+)?"
    r"(?:[^\W\d_]+(?:[-\s]+[^\W\d_]+){0,5})\s+"
    r"(?:is|are|was|were|became|changed|declined|decreased|equaled|equalled|equals|"
    r"fell|grew|increased|reached|remained|reported|rose|stood|stayed|totaled|totalled)\b",
    re.IGNORECASE,
)
AbsenceMode = Literal["lexical"]
ABSENCE_PROBE_TEMPLATES: tuple[tuple[str, AbsenceMode], ...] = (
    ("{metric}", "lexical"),
    ('"{metric}" reported value', "lexical"),
    ("{metric} fiscal year period", "lexical"),
    ("{metric} total company metric", "lexical"),
)
ABSENCE_PROTOCOL_SPEC = {
    "version": ABSENCE_PROTOCOL_VERSION,
    "minimum_searches": 4,
    "minimum_top_k": ABSENCE_MIN_TOP_K,
    "requires_exact_lexical_probe": True,
    "requires_exhaustive_exact_metric_scan": True,
    "exact_metric_scan": {
        "engine": "sqlite_fts5_phrase",
        "top_k": None,
        "post_filter": "shared_complete_word_phrase_with_canonical_initialisms",
        "initialism_variant_cap": MAX_METRIC_FTS_VARIANTS,
        "variant_overflow": "full_sqlite_chunk_scan",
        "maximum_candidates": EXACT_METRIC_SCAN_MAX_CANDIDATES,
        "maximum_characters": EXACT_METRIC_SCAN_MAX_CHARACTERS,
        "breadth_limit": "typed_incomplete_probe",
        "token_kind_mismatch": "typed_incomplete_probe",
        "failure": "incomplete_probe",
    },
    "requires_all_unique_candidates_opened": True,
    "metric_matching": "complete_word_phrase",
    "probe_templates": ABSENCE_PROBE_TEMPLATES,
    "numeric_candidate_policy": (
        "same_assertion_or_structural_anaphoric_value_and_closed_value_first_shape"
    ),
    "qualitative_predicate_policy": (
        "known_positive_connector_after_bounded_punctuation_free_qualifiers_requires_review"
    ),
    "occurrence_authority_policy": (
        "every_exact_occurrence_must_be_typed_benign;"
        "reviewable_evidence_blocks_absence;unknown_structure_requires_review"
    ),
    "benign_rubric_reference_pattern": _BENIGN_RUBRIC_REFERENCE.pattern,
    "benign_source_label_pattern": _BENIGN_SOURCE_LABEL.pattern,
    "benign_standalone_remainder_pattern": _BENIGN_STANDALONE_REMAINDER.pattern,
    "potential_unknown_reference_pattern": _POTENTIAL_UNKNOWN_STRUCTURAL_REFERENCE_LEAD.pattern,
    "potential_following_assertion_pattern": _POTENTIAL_FOLLOWING_ASSERTION.pattern,
    "metric_context_characters": ABSENCE_METRIC_CONTEXT_CHARACTERS,
    "metric_context_policy": ("full_opened_source_chain_with_bounded_display_excerpt"),
    "authorization_revalidation": (
        "rerun_exhaustive_scan_and_rebuild_all_metric_contexts_from_bound_corpus_snapshot"
    ),
    "neighbor_radius": 1,
    "maximum_context_chunks": ABSENCE_CONTEXT_MAX_OPENED_CHUNKS,
    "cross_chunk_context": (
        "expand_source_links_to_document_edge_within_context_budget;"
        "missing_or_over_budget_link_is_incomplete"
    ),
    "conclusion": "bounded_not_global",
}
ABSENCE_PROTOCOL_SHA256 = sha256_text(canonical_json(ABSENCE_PROTOCOL_SPEC))

CallRecorder = Callable[[dict[str, object]], Awaitable[None]]


class AbsenceCorpus(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int,
        mode: AbsenceMode,
        recorder: CallRecorder | None = None,
    ) -> SearchResult: ...

    def get_chunks(
        self,
        chunk_ids: list[ChunkId],
        neighbor_radius: int = 0,
    ) -> list[ChunkRecord]: ...

    def find_exact_metric_chunks(self, metric: str) -> ExactMetricScanOutcome: ...


class AbsenceContextCorpus(Protocol):
    def get_chunks(
        self,
        chunk_ids: list[ChunkId],
        neighbor_radius: int = 0,
    ) -> list[ChunkRecord]: ...

    def find_exact_metric_chunks(self, metric: str) -> ExactMetricScanOutcome: ...


def _normalized_query(value: str) -> str:
    return canonical_metric_key(value)


def _query_signature_text(value: str) -> str:
    return " ".join(sorted(canonical_metric_key(value).split()))


def search_signature(
    *,
    query: str,
    metric: str,
    mode: str,
    top_k: int,
    document_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "query": _query_signature_text(query),
                "metric": _normalized_query(metric),
                "mode": mode,
                "top_k": top_k,
                "document_ids": sorted(document_ids or []),
                "date_from": date_from,
                "date_to": date_to,
            }
        )
    )


def derive_absence_conclusion(probe: AbsenceProbeResult) -> AbsenceConclusion:
    """Recompute the bounded conclusion from the complete typed proof record."""

    expected_probes = tuple(
        (template.format(metric=probe.metric), mode) for template, mode in ABSENCE_PROBE_TEMPLATES
    )
    if probe.protocol_version != ABSENCE_PROTOCOL_VERSION or len(probe.searches) != len(
        expected_probes
    ):
        return AbsenceConclusion.INCOMPLETE_PROBE
    for search, (expected_query, expected_mode) in zip(
        probe.searches,
        expected_probes,
        strict=True,
    ):
        if (
            search.completion_status != "completed"
            or search.mode != expected_mode
            or search.normalized_query != _normalized_query(expected_query)
            or canonical_metric_key(search.metric or "") != canonical_metric_key(probe.metric)
            or search.top_k < ABSENCE_MIN_TOP_K
            or search.document_ids
            or search.date_from is not None
            or search.date_to is not None
            or search.result_count != len(search.result_chunk_ids)
            or len(set(search.result_chunk_ids)) != len(search.result_chunk_ids)
            or search.signature
            != search_signature(
                query=expected_query,
                metric=probe.metric,
                mode=expected_mode,
                top_k=search.top_k,
            )
        ):
            return AbsenceConclusion.INCOMPLETE_PROBE

    candidate_ids = probe.unique_candidate_chunk_ids
    opened_ids = probe.opened_chunk_ids
    candidate_set = set(candidate_ids)
    opened_set = set(opened_ids)
    occurrence_ids = {occurrence.chunk_id for occurrence in probe.exact_metric_occurrences}
    result_ids = {chunk_id for search in probe.searches for chunk_id in search.result_chunk_ids}
    exact_scan_ids = set(probe.exact_metric_scan_chunk_ids)
    reviewable_occurrence = any(
        has_unresolved_metric_predicate(probe.metric, occurrence.sentence)
        or _has_structural_anaphoric_qualitative_predicate(occurrence)
        or bool(_metric_context_numeric_candidates(occurrence))
        for occurrence in probe.exact_metric_occurrences
    )
    if (
        probe.unresolved_predicate_occurrences
        or probe.supporting_value_candidates
        or reviewable_occurrence
    ):
        # Known reviewable evidence dominates an incomplete breadth/context
        # proof. Either outcome blocks absence authority, while preserving the
        # evidence-specific result makes the receipt useful.
        return AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    if (
        not probe.exact_metric_scan_completed
        or probe.exact_metric_scan_error is not None
        or len(candidate_set) != len(candidate_ids)
        or len(opened_set) != len(opened_ids)
        or candidate_set != opened_set
        or not result_ids.issubset(candidate_set)
        or not exact_scan_ids.issubset(candidate_set)
        or not exact_scan_ids.issubset(occurrence_ids)
        or not occurrence_ids.issubset(opened_set)
    ):
        return AbsenceConclusion.INCOMPLETE_PROBE
    return AbsenceConclusion.NOT_FOUND_IN_PROBE


def _benign_metric_mention_profile(
    occurrence: MetricOccurrence,
    structural_context: tuple[int, tuple[tuple[int, int], ...]] | None = None,
) -> BenignMentionProfile | None:
    """Recognize the closed set of non-evidentiary metric mention shapes."""

    metric_start, metric_end = occurrence.span
    metric_sentence_end, _anaphoric_spans = structural_context or _structural_anaphoric_context(
        occurrence
    )
    metric_sentence = occurrence.sentence[:metric_sentence_end]
    prefix = metric_sentence[:metric_start]
    suffix = metric_sentence[metric_end:]
    following = occurrence.sentence[metric_sentence_end:].strip()
    if following:
        following_boundaries = punctuation_boundaries(following)
        following_end = following_boundaries[0][1] if following_boundaries else len(following)
        following_sentence = following[:following_end]
        if (
            _POTENTIAL_UNKNOWN_STRUCTURAL_REFERENCE_LEAD.match(following_sentence)
            or _POTENTIAL_FOLLOWING_ASSERTION.match(following_sentence)
            or numeric_value_candidates(following_sentence)
        ):
            return None
    if "?" in metric_sentence:
        return BenignMentionProfile.QUESTION
    if _BENIGN_SOURCE_LABEL.match(suffix):
        return BenignMentionProfile.SOURCE_LABEL
    if _BENIGN_RUBRIC_REFERENCE.search(prefix) or _BENIGN_RUBRIC_REFERENCE.search(suffix):
        return BenignMentionProfile.RUBRIC_REFERENCE
    if not _BENIGN_STANDALONE_REMAINDER.fullmatch(prefix + suffix):
        return None
    if following:
        return None
    return BenignMentionProfile.STANDALONE_LABEL


def _analyze_metric_occurrence(
    metric: str,
    display_occurrence: MetricOccurrence,
    proof_occurrence: MetricOccurrence,
) -> MetricOccurrenceAnalysis:
    """Classify one exact occurrence; unknown syntax never counts as benign."""

    structural_context = _structural_anaphoric_context(proof_occurrence)
    reasons: list[MetricEvidenceReason] = []
    if _metric_context_numeric_candidates(proof_occurrence, structural_context):
        reasons.append(MetricEvidenceReason.NUMERIC_CANDIDATE)
    if _has_structural_anaphoric_qualitative_predicate(proof_occurrence, structural_context):
        reasons.append(MetricEvidenceReason.QUALITATIVE_PREDICATE)
    if has_unresolved_metric_predicate(metric, proof_occurrence.sentence):
        reasons.append(MetricEvidenceReason.UNRESOLVED_PREDICATE)
    if reasons:
        return MetricEvidenceCandidateOccurrence(
            occurrence=display_occurrence,
            reasons=list(dict.fromkeys(reasons)),
        )
    profile = _benign_metric_mention_profile(proof_occurrence, structural_context)
    if profile is not None:
        return BenignMetricMention(occurrence=display_occurrence, profile=profile)

    metric_sentence_end, _structural_chain = structural_context
    reason = (
        UnknownOccurrenceReason.UNKNOWN_STRUCTURAL_CONTINUATION
        if proof_occurrence.sentence[metric_sentence_end:].strip()
        else UnknownOccurrenceReason.UNCLASSIFIED_METRIC_CONTEXT
    )
    return UnknownMetricOccurrence(occurrence=display_occurrence, reason=reason)


def _structural_anaphoric_context(
    occurrence: MetricOccurrence,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Return the metric sentence end and its contiguous structural reference chain."""

    _metric_start, metric_end = occurrence.span
    boundaries = punctuation_boundaries(occurrence.sentence)
    metric_sentence_end = next(
        (end for start, end in boundaries if start >= metric_end),
        len(occurrence.sentence),
    )
    anaphoric_sentence_spans: list[tuple[int, int]] = []
    sentence_start = metric_sentence_end
    following_sentence_ends = [end for _start, end in boundaries if end > metric_sentence_end]
    if not following_sentence_ends or following_sentence_ends[-1] < len(occurrence.sentence):
        following_sentence_ends.append(len(occurrence.sentence))
    for sentence_end in following_sentence_ends:
        sentence = occurrence.sentence[sentence_start:sentence_end]
        if not _POTENTIAL_ANAPHORIC_PREDICATE_LEAD.match(sentence):
            break
        anaphoric_sentence_spans.append((sentence_start, sentence_end))
        sentence_start = sentence_end
    return metric_sentence_end, tuple(anaphoric_sentence_spans)


def _has_structural_anaphoric_qualitative_predicate(
    occurrence: MetricOccurrence,
    structural_context: tuple[int, tuple[tuple[int, int], ...]] | None = None,
) -> bool:
    """Detect a qualitative value in the same structural chain used for numbers."""

    _metric_sentence_end, anaphoric_sentence_spans = (
        structural_context or _structural_anaphoric_context(occurrence)
    )
    for start, end in anaphoric_sentence_spans:
        sentence = occurrence.sentence[start:end]
        relationship = _POTENTIAL_ANAPHORIC_PREDICATE_LEAD.match(sentence)
        if relationship is not None and _QUALITATIVE_WORD.search(sentence[relationship.end() :]):
            return True
    return False


def _metric_context_numeric_candidates(
    occurrence: MetricOccurrence,
    structural_context: tuple[int, tuple[tuple[int, int], ...]] | None = None,
) -> list[tuple[str, tuple[int, int]]]:
    """Return same-assertion, structural-anaphoric, and closed value-first values."""

    metric_start, metric_end = occurrence.span
    metric_sentence_end, anaphoric_sentence_spans = (
        structural_context or _structural_anaphoric_context(occurrence)
    )

    candidates = []
    for value_text, value_span in numeric_value_candidates(occurrence.sentence):
        if metric_end <= value_span[0] < metric_sentence_end:
            candidates.append((value_text, value_span))
            continue
        if any(
            start <= value_span[0] and value_span[1] <= end
            for start, end in anaphoric_sentence_spans
        ):
            candidates.append((value_text, value_span))
            continue
        if value_span[1] <= metric_start and _VALUE_FIRST_METRIC_BRIDGE.fullmatch(
            occurrence.sentence[value_span[1] : metric_start]
        ):
            candidates.append((value_text, value_span))
    return candidates


def derive_absence_conclusion_against_corpus(
    probe: AbsenceProbeResult,
    corpus: AbsenceContextCorpus,
) -> AbsenceConclusion:
    """Rebuild proof contexts from the bound corpus before authorizing absence."""

    structural = derive_absence_conclusion(probe)
    if structural is not AbsenceConclusion.NOT_FOUND_IN_PROBE:
        return structural
    # The sealed probe is untrusted input at authorization time. Re-run the
    # exhaustive primitive against the bound corpus snapshot and require the
    # independently returned complete-match set to agree exactly with the
    # probe. A partial, failed, or over-budget revalidation can never prove
    # absence.
    revalidated_scan = corpus.find_exact_metric_chunks(probe.metric)
    if not isinstance(revalidated_scan, ExactMetricScanComplete):
        return AbsenceConclusion.INCOMPLETE_PROBE
    revalidated_scan_ids = [chunk.chunk_id for chunk in revalidated_scan.chunks]
    if (
        len(revalidated_scan_ids) != len(set(revalidated_scan_ids))
        or set(revalidated_scan_ids) != set(probe.exact_metric_scan_chunk_ids)
        or revalidated_scan.candidate_count < len(revalidated_scan.chunks)
    ):
        return AbsenceConclusion.INCOMPLETE_PROBE
    # ``opened_chunk_ids`` already includes the neighbors selected by the probe.
    # Expanding radius again would silently change the sealed proof set.
    chunks = corpus.get_chunks(probe.opened_chunk_ids, 0)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if set(chunks_by_id) != set(probe.opened_chunk_ids):
        return AbsenceConclusion.INCOMPLETE_PROBE

    expected_occurrences: set[tuple[ChunkId, str, tuple[int, int]]] = set()
    review_required = False
    unknown_occurrence = False
    for chunk in chunks:
        for metric_start, metric_end in word_phrase_spans(probe.metric, chunk.normalized_text):
            analyzed = _metric_occurrence_with_open_neighbors(
                chunk,
                chunks_by_id,
                metric_start,
                metric_end,
            )
            if not analyzed.complete:
                return AbsenceConclusion.INCOMPLETE_PROBE
            occurrence = analyzed.display_occurrence
            expected_occurrences.add((occurrence.chunk_id, occurrence.sentence, occurrence.span))
            occurrence_analysis = _analyze_metric_occurrence(
                probe.metric,
                analyzed.display_occurrence,
                analyzed.proof_occurrence,
            )
            review_required = review_required or isinstance(
                occurrence_analysis,
                MetricEvidenceCandidateOccurrence,
            )
            unknown_occurrence = unknown_occurrence or isinstance(
                occurrence_analysis,
                UnknownMetricOccurrence,
            )

    recorded_occurrences = {
        (occurrence.chunk_id, occurrence.sentence, occurrence.span)
        for occurrence in probe.exact_metric_occurrences
    }
    if recorded_occurrences != expected_occurrences:
        return AbsenceConclusion.INCOMPLETE_PROBE
    if review_required:
        return AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    if unknown_occurrence:
        return AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    return AbsenceConclusion.NOT_FOUND_IN_PROBE


def _join_source_fragments(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    separator = "" if left[-1].isspace() or right[0].isspace() else " "
    return f"{left}{separator}{right}"


@dataclass(frozen=True, slots=True)
class _AnalyzedMetricContext:
    display_occurrence: MetricOccurrence
    proof_occurrence: MetricOccurrence
    complete: bool


def _metric_occurrence_with_open_neighbors(
    chunk: ChunkRecord,
    chunks_by_id: Mapping[ChunkId, ChunkRecord],
    metric_start: int,
    metric_end: int,
) -> _AnalyzedMetricContext:
    """Analyze full opened source context while bounding only its display excerpt."""

    before = chunk.normalized_text[:metric_start]
    after = chunk.normalized_text[metric_end:]
    complete = True

    prior_boundaries = punctuation_boundaries(before)
    if prior_boundaries:
        before = before[prior_boundaries[-1][1] :]
    else:
        previous_id = chunk.previous_chunk_id
        visited_previous = {chunk.chunk_id}
        while previous_id is not None:
            if previous_id in visited_previous:
                complete = False
                break
            visited_previous.add(previous_id)
            previous = chunks_by_id.get(previous_id)
            if previous is None:
                complete = False
                break
            previous_boundaries = punctuation_boundaries(previous.normalized_text)
            previous_fragment = (
                previous.normalized_text[previous_boundaries[-1][1] :]
                if previous_boundaries
                else previous.normalized_text
            )
            before = _join_source_fragments(previous_fragment, before)
            if previous_boundaries:
                break
            previous_id = previous.previous_chunk_id

    following_id = chunk.next_chunk_id
    visited_following = {chunk.chunk_id}
    while following_id is not None:
        if following_id in visited_following:
            complete = False
            break
        visited_following.add(following_id)
        following = chunks_by_id.get(following_id)
        if following is None:
            complete = False
            break
        after = _join_source_fragments(after, following.normalized_text)
        following_id = following.next_chunk_id

    metric_text = chunk.normalized_text[metric_start:metric_end]
    proof_context = _join_source_fragments(before, metric_text)
    proof_span_end = len(proof_context)
    proof_span_start = proof_span_end - len(metric_text)
    proof_context = _join_source_fragments(proof_context, after)
    proof_occurrence = MetricOccurrence(
        chunk_id=chunk.chunk_id,
        sentence=proof_context,
        span=(proof_span_start, proof_span_end),
    )

    display_start = max(0, proof_span_start - ABSENCE_METRIC_CONTEXT_CHARACTERS)
    display_end = min(
        len(proof_context),
        proof_span_end + ABSENCE_METRIC_CONTEXT_CHARACTERS,
    )
    display_text = proof_context[display_start:display_end]
    display_occurrence = MetricOccurrence(
        chunk_id=chunk.chunk_id,
        sentence=display_text,
        span=(proof_span_start - display_start, proof_span_end - display_start),
    )
    return _AnalyzedMetricContext(display_occurrence, proof_occurrence, complete)


async def probe_metric_absence(
    corpus: AbsenceCorpus,
    metric: str,
    *,
    top_k: int = 8,
    recorder: CallRecorder | None = None,
) -> AbsenceProbeResult:
    """Run a server-owned bounded probe; never infer absence from model omissions."""

    top_k = max(ABSENCE_MIN_TOP_K, min(top_k, 20))
    metric_phrase = canonical_metric_key(metric)
    probes = tuple(
        (template.format(metric=metric), mode) for template, mode in ABSENCE_PROBE_TEMPLATES
    )
    searches: list[CompletedSearchRecord] = []
    candidate_ids = []
    for query, mode in probes:
        signature = search_signature(
            query=query,
            metric=metric,
            mode=mode,
            top_k=top_k,
        )
        try:
            result = await corpus.search(
                query,
                top_k=top_k,
                mode=mode,
                recorder=recorder,
            )
        except Exception:  # noqa: BLE001 - a failed probe is a typed incomplete outcome
            searches.append(
                CompletedSearchRecord(
                    query=query,
                    normalized_query=_normalized_query(query),
                    metric=metric,
                    mode=mode,
                    top_k=top_k,
                    signature=signature,
                    result_count=0,
                    exact_metric_hit_count=0,
                    completion_status="failed",
                )
            )
            continue
        chunks = await asyncio.to_thread(
            corpus.get_chunks,
            [hit.chunk_id for hit in result.results],
        )
        exact_hits = sum(bool(word_phrase_spans(metric, chunk.normalized_text)) for chunk in chunks)
        result_ids = [hit.chunk_id for hit in result.results]
        candidate_ids.extend(result_ids)
        searches.append(
            CompletedSearchRecord(
                query=query,
                normalized_query=_normalized_query(query),
                metric=metric,
                mode=mode,
                top_k=top_k,
                signature=signature,
                result_chunk_ids=result_ids,
                result_count=len(result_ids),
                exact_metric_hit_count=exact_hits,
                completion_status="completed",
            )
        )

    exact_metric_scan_completed = False
    exact_metric_scan_chunk_ids: list[ChunkId] = []
    exact_metric_scan_error: str | None = None
    if metric_phrase:
        try:
            exact_scan = await asyncio.to_thread(corpus.find_exact_metric_chunks, metric)
        except Exception as error:  # noqa: BLE001 - absence fails closed on scan failure
            exact_scan = ExactMetricScanFailed(type(error).__name__)
        if isinstance(exact_scan, ExactMetricScanComplete):
            exact_metric_scan_completed = True
            exact_metric_scan_chunk_ids = [
                chunk.chunk_id
                for chunk in exact_scan.chunks
                if word_phrase_spans(metric, chunk.normalized_text)
            ]
            candidate_ids.extend(exact_metric_scan_chunk_ids)
        elif isinstance(exact_scan, ExactMetricScanAmbiguous):
            exact_metric_scan_error = "MetricTokenKindAmbiguous"
        elif isinstance(exact_scan, ExactMetricScanTooBroad):
            exact_metric_scan_error = "ExactScanTooBroad"
        else:
            exact_metric_scan_error = exact_scan.error_type

    unique_ids = list(dict.fromkeys(candidate_ids))
    context_expansion_limited = len(unique_ids) > ABSENCE_CONTEXT_MAX_OPENED_CHUNKS
    initial_ids = unique_ids[:ABSENCE_CONTEXT_MAX_OPENED_CHUNKS]
    chunks = await asyncio.to_thread(corpus.get_chunks, initial_ids, 0)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    while not context_expansion_limited:
        linked_ids = {
            linked_id
            for opened in chunks_by_id.values()
            for linked_id in (opened.previous_chunk_id, opened.next_chunk_id)
            if linked_id is not None and linked_id not in chunks_by_id
        }
        if not linked_ids:
            break
        if len(chunks_by_id) + len(linked_ids) > ABSENCE_CONTEXT_MAX_OPENED_CHUNKS:
            context_expansion_limited = True
            break
        expanded = await asyncio.to_thread(
            corpus.get_chunks,
            list(linked_ids),
            0,
        )
        added = False
        for expanded_chunk in expanded:
            if expanded_chunk.chunk_id in chunks_by_id:
                continue
            if len(chunks_by_id) >= ABSENCE_CONTEXT_MAX_OPENED_CHUNKS:
                context_expansion_limited = True
                break
            chunks_by_id[expanded_chunk.chunk_id] = expanded_chunk
            added = True
        if context_expansion_limited or not added:
            break
    chunks = list(chunks_by_id.values())
    unique_ids = list(dict.fromkeys([*unique_ids, *chunks_by_id]))
    occurrences: list[MetricOccurrence] = []
    unresolved_predicates: list[MetricOccurrence] = []
    value_candidates: list[ValueCandidate] = []
    value_candidate_keys: set[tuple[ChunkId, str]] = set()
    missing_open_edge_neighbor = context_expansion_limited
    for chunk in chunks:
        for metric_start, metric_end in word_phrase_spans(metric, chunk.normalized_text):
            analyzed = _metric_occurrence_with_open_neighbors(
                chunk,
                chunks_by_id,
                metric_start,
                metric_end,
            )
            missing_open_edge_neighbor = missing_open_edge_neighbor or not analyzed.complete
            occurrences.append(analyzed.display_occurrence)
            occurrence_analysis = _analyze_metric_occurrence(
                metric,
                analyzed.display_occurrence,
                analyzed.proof_occurrence,
            )
            if (
                isinstance(occurrence_analysis, UnknownMetricOccurrence)
                or has_unresolved_metric_predicate(
                    metric,
                    analyzed.proof_occurrence.sentence,
                )
                or _has_structural_anaphoric_qualitative_predicate(analyzed.proof_occurrence)
            ):
                unresolved_predicates.append(analyzed.display_occurrence)
            for value_text, _span in _metric_context_numeric_candidates(analyzed.proof_occurrence):
                candidate_key = (chunk.chunk_id, value_text)
                if candidate_key in value_candidate_keys:
                    continue
                value_candidate_keys.add(candidate_key)
                value_candidates.append(
                    ValueCandidate(
                        chunk_id=chunk.chunk_id,
                        metric_anchor=metric,
                        value_text=value_text,
                        binding_failure_reason=(
                            "numeric_candidate_in_metric_context_requires_review"
                        ),
                    )
                )

    if missing_open_edge_neighbor:
        exact_metric_scan_completed = False
        exact_metric_scan_error = (
            "ContextExpansionLimitExceeded" if context_expansion_limited else "NeighborChunkMissing"
        )
    probe = AbsenceProbeResult(
        protocol_version=ABSENCE_PROTOCOL_VERSION,
        metric=metric,
        searches=searches,
        exact_metric_scan_completed=exact_metric_scan_completed,
        exact_metric_scan_chunk_ids=exact_metric_scan_chunk_ids,
        exact_metric_scan_error=exact_metric_scan_error,
        unique_candidate_chunk_ids=unique_ids,
        opened_chunk_ids=[chunk.chunk_id for chunk in chunks],
        exact_metric_occurrences=occurrences,
        unresolved_predicate_occurrences=unresolved_predicates,
        supporting_value_candidates=value_candidates,
        conclusion=AbsenceConclusion.INCOMPLETE_PROBE,
    )
    return probe.model_copy(update={"conclusion": derive_absence_conclusion(probe)})
