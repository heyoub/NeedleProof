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
    ExactMetricScanComplete,
    ExactMetricScanFailed,
    ExactMetricScanOutcome,
    ExactMetricScanTooBroad,
)
from .models import (
    AbsenceConclusion,
    AbsenceProbeResult,
    ChunkRecord,
    CompletedSearchRecord,
    MetricOccurrence,
    SearchResult,
    ValueCandidate,
)
from .util import (
    MAX_METRIC_FTS_VARIANTS,
    canonical_json,
    canonical_metric_key,
    punctuation_boundaries,
    sha256_text,
)

ABSENCE_PROTOCOL_VERSION = "bounded-absence-v18-total-anaphoric-boundary-states"
ABSENCE_METRIC_CONTEXT_CHARACTERS = 384
ABSENCE_MIN_TOP_K = 8
_VALUE_FIRST_METRIC_BRIDGE = re.compile(
    r"(?:\s*|\s*[,;:—–-]\s*|\s*[([{\"'“‘]\s*|"
    r"\s*(?:in|of)\s+(?:[^\W\d_]+\s+){0,3})",
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
        "bidirectional_metric_context_with_bounded_separator_or_enclosure_requires_review"
    ),
    "qualitative_predicate_policy": (
        "known_positive_connector_after_bounded_punctuation_free_qualifiers_requires_review"
    ),
    "metric_context_characters": ABSENCE_METRIC_CONTEXT_CHARACTERS,
    "metric_context_policy": (
        "complete_positive_boundary_context_with_ambiguous_punctuation_expanding_fail_closed"
    ),
    "authorization_revalidation": "rebuild_all_metric_contexts_from_bound_corpus_snapshot",
    "neighbor_radius": 1,
    "cross_chunk_context": (
        "one_source_linked_sentence_when_local_clause_or_anaphoric_assertion_edge_is_open;"
        "immediate_linked_anaphoric_chain_is_incomplete"
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

    reviewable_occurrence = any(
        has_unresolved_metric_predicate(probe.metric, occurrence.sentence)
        or bool(_metric_context_numeric_candidates(occurrence))
        for occurrence in probe.exact_metric_occurrences
    )
    if (
        probe.unresolved_predicate_occurrences
        or probe.supporting_value_candidates
        or reviewable_occurrence
    ):
        return AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    return AbsenceConclusion.NOT_FOUND_IN_PROBE


def _metric_context_numeric_candidates(
    occurrence: MetricOccurrence,
) -> list[tuple[str, tuple[int, int]]]:
    """Return conservative forward values and closed value-first metric shapes."""

    metric_start, metric_end = occurrence.span
    candidates = []
    for value_text, value_span in numeric_value_candidates(occurrence.sentence):
        if value_span[0] >= metric_end:
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
    # ``opened_chunk_ids`` already includes the neighbors selected by the probe.
    # Expanding radius again would silently change the sealed proof set.
    chunks = corpus.get_chunks(probe.opened_chunk_ids, 0)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if set(chunks_by_id) != set(probe.opened_chunk_ids):
        return AbsenceConclusion.INCOMPLETE_PROBE

    expected_occurrences: set[tuple[ChunkId, str, tuple[int, int]]] = set()
    review_required = False
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
            review_required = (
                review_required
                or has_unresolved_metric_predicate(
                    probe.metric,
                    analyzed.proof_occurrence.sentence,
                )
                or bool(_metric_context_numeric_candidates(analyzed.proof_occurrence))
            )

    recorded_occurrences = {
        (occurrence.chunk_id, occurrence.sentence, occurrence.span)
        for occurrence in probe.exact_metric_occurrences
    }
    if recorded_occurrences != expected_occurrences:
        return AbsenceConclusion.INCOMPLETE_PROBE
    if review_required:
        return AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    return AbsenceConclusion.NOT_FOUND_IN_PROBE


def _join_source_fragments(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    separator = "" if left[-1].isspace() or right[0].isspace() else " "
    return f"{left}{separator}{right}"


_ANAPHORIC_SENTENCE_LEAD = re.compile(
    r"\s*(?:(?:it|they|this|that|these|those)\b|"
    r"(?:the|these|those)\s+(?:figures?|values?|amounts?|numbers?)\b)",
    re.IGNORECASE,
)


def _linked_following_fragment(
    chunk: ChunkRecord,
    chunks_by_id: Mapping[ChunkId, ChunkRecord],
) -> tuple[str, bool]:
    """Return one linked source sentence fragment and whether it is complete."""

    if chunk.next_chunk_id is None:
        return "", True
    following = chunks_by_id.get(chunk.next_chunk_id)
    if following is None:
        return "", False
    boundaries = punctuation_boundaries(following.normalized_text)
    if boundaries:
        boundary_end = boundaries[0][1]
        remainder = following.normalized_text[boundary_end:]
        # One linked sentence is the maximum supported proof profile. An
        # immediately following anaphoric sentence may still belong to the
        # metric assertion, so omitting it cannot prove absence. Keep that
        # shape deliberately unsupported and fail closed.
        complete = _ANAPHORIC_SENTENCE_LEAD.match(remainder) is None
        return following.normalized_text[:boundary_end], complete
    return following.normalized_text, following.next_chunk_id is None


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
    """Analyze one complete, conservatively bounded source proposition."""

    before = chunk.normalized_text[:metric_start]
    after = chunk.normalized_text[metric_end:]
    complete = True

    prior_boundaries = punctuation_boundaries(before)
    if prior_boundaries:
        before = before[prior_boundaries[-1][1] :]
    elif chunk.previous_chunk_id is not None:
        previous = chunks_by_id.get(chunk.previous_chunk_id)
        if previous is None:
            complete = False
        else:
            previous_boundaries = punctuation_boundaries(previous.normalized_text)
            previous_fragment = (
                previous.normalized_text[previous_boundaries[-1][1] :]
                if previous_boundaries
                else previous.normalized_text
            )
            before = _join_source_fragments(previous_fragment, before)
            if not previous_boundaries and previous.previous_chunk_id is not None:
                complete = False

    following_boundaries = punctuation_boundaries(after)
    following_boundary = following_boundaries[0] if following_boundaries else None
    if following_boundary:
        after = after[: following_boundary[1]]
        remainder = chunk.normalized_text[metric_end + following_boundary[1] :]
        if _ANAPHORIC_SENTENCE_LEAD.match(remainder):
            anaphoric_boundaries = punctuation_boundaries(remainder)
            after = _join_source_fragments(
                after,
                remainder[: anaphoric_boundaries[0][1]] if anaphoric_boundaries else remainder,
            )
            if anaphoric_boundaries:
                if _ANAPHORIC_SENTENCE_LEAD.match(remainder[anaphoric_boundaries[0][1] :]):
                    # One anaphoric continuation is the maximum supported proof
                    # profile. A second coreferential sentence may still own a
                    # value for the metric, so excluding it cannot prove absence.
                    complete = False
            elif chunk.next_chunk_id is not None:
                following_fragment, following_complete = _linked_following_fragment(
                    chunk,
                    chunks_by_id,
                )
                after = _join_source_fragments(after, following_fragment)
                complete = complete and following_complete
        elif not remainder.strip() and chunk.next_chunk_id is not None:
            following = chunks_by_id.get(chunk.next_chunk_id)
            if following is None:
                complete = False
            elif _ANAPHORIC_SENTENCE_LEAD.match(following.normalized_text):
                following_fragment, following_complete = _linked_following_fragment(
                    chunk,
                    chunks_by_id,
                )
                after = _join_source_fragments(after, following_fragment)
                complete = complete and following_complete
    elif chunk.next_chunk_id is not None:
        following_fragment, following_complete = _linked_following_fragment(
            chunk,
            chunks_by_id,
        )
        after = _join_source_fragments(after, following_fragment)
        complete = complete and following_complete

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
            if exact_scan.ambiguous_candidate_count:
                exact_metric_scan_error = "MetricTokenKindAmbiguous"
            else:
                exact_metric_scan_completed = True
                exact_metric_scan_chunk_ids = [
                    chunk.chunk_id
                    for chunk in exact_scan.chunks
                    if word_phrase_spans(metric, chunk.normalized_text)
                ]
                candidate_ids.extend(exact_metric_scan_chunk_ids)
        elif isinstance(exact_scan, ExactMetricScanTooBroad):
            exact_metric_scan_error = "ExactScanTooBroad"
        else:
            exact_metric_scan_error = exact_scan.error_type

    unique_ids = list(dict.fromkeys(candidate_ids))
    chunks = await asyncio.to_thread(corpus.get_chunks, unique_ids, 1)
    unique_ids = list(dict.fromkeys([*unique_ids, *(chunk.chunk_id for chunk in chunks)]))
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    occurrences: list[MetricOccurrence] = []
    unresolved_predicates: list[MetricOccurrence] = []
    value_candidates: list[ValueCandidate] = []
    value_candidate_keys: set[tuple[ChunkId, str]] = set()
    missing_open_edge_neighbor = False
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
            if has_unresolved_metric_predicate(metric, analyzed.proof_occurrence.sentence):
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
        exact_metric_scan_error = "NeighborChunkMissing"

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
