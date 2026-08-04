from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

from .binding import (
    has_unresolved_metric_predicate,
    numeric_value_candidates,
    word_phrase_spans,
)
from .chunk_ids import ChunkId
from .models import (
    AbsenceConclusion,
    AbsenceProbeResult,
    ChunkRecord,
    CompletedSearchRecord,
    MetricOccurrence,
    SearchResult,
    ValueCandidate,
)
from .util import canonical_json, canonical_metric_key, normalize_evidence_text, sha256_text

ABSENCE_PROTOCOL_VERSION = "bounded-absence-v6-qualified-predicate-windows"
ABSENCE_METRIC_CONTEXT_CHARACTERS = 384
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
    "minimum_top_k": 8,
    "requires_exact_lexical_probe": True,
    "requires_exhaustive_exact_metric_scan": True,
    "exact_metric_scan": {
        "engine": "sqlite_fts5_phrase",
        "top_k": None,
        "post_filter": "complete_word_phrase",
        "failure": "incomplete_probe",
    },
    "requires_all_unique_candidates_opened": True,
    "metric_matching": "complete_word_phrase",
    "probe_templates": ABSENCE_PROBE_TEMPLATES,
    "numeric_candidate_policy": "requires_review_even_without_positive_binding",
    "qualitative_predicate_policy": (
        "known_positive_connector_after_bounded_punctuation_free_qualifiers_requires_review"
    ),
    "metric_context_characters": ABSENCE_METRIC_CONTEXT_CHARACTERS,
    "metric_context_policy": "forward_window_from_complete_metric_without_sentence_parsing",
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

    def find_exact_metric_chunks(self, metric: str) -> list[ChunkRecord]: ...


def _normalized_query(value: str) -> str:
    normalized, _ = normalize_evidence_text(value)
    return " ".join(normalized.casefold().split())


def _query_signature_text(value: str) -> str:
    normalized, _ = normalize_evidence_text(value)
    return " ".join(sorted(word.casefold() for word in re.findall(r"[^\W_]+", normalized)))


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


async def probe_metric_absence(
    corpus: AbsenceCorpus,
    metric: str,
    *,
    top_k: int = 8,
    recorder: CallRecorder | None = None,
) -> AbsenceProbeResult:
    """Run a server-owned bounded probe; never infer absence from model omissions."""

    top_k = max(8, min(top_k, 20))
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
        exact_hits = sum(
            bool(word_phrase_spans(metric_phrase, chunk.normalized_text)) for chunk in chunks
        )
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
            exact_metric_chunks = await asyncio.to_thread(corpus.find_exact_metric_chunks, metric)
        except Exception as error:  # noqa: BLE001 - absence fails closed on scan failure
            exact_metric_chunks = []
            exact_metric_scan_error = type(error).__name__
        else:
            exact_metric_scan_completed = True
            exact_metric_scan_chunk_ids = [
                chunk.chunk_id
                for chunk in exact_metric_chunks
                if word_phrase_spans(metric_phrase, chunk.normalized_text)
            ]
            candidate_ids.extend(exact_metric_scan_chunk_ids)

    unique_ids = list(dict.fromkeys(candidate_ids))
    chunks = await asyncio.to_thread(corpus.get_chunks, unique_ids)
    occurrences: list[MetricOccurrence] = []
    unresolved_predicates: list[MetricOccurrence] = []
    value_candidates: list[ValueCandidate] = []
    value_candidate_keys: set[tuple[ChunkId, str]] = set()
    for chunk in chunks:
        for metric_start, metric_end in word_phrase_spans(metric_phrase, chunk.normalized_text):
            context_end = min(
                len(chunk.normalized_text),
                metric_end + ABSENCE_METRIC_CONTEXT_CHARACTERS,
            )
            context = chunk.normalized_text[metric_start:context_end]
            occurrence = MetricOccurrence(
                chunk_id=chunk.chunk_id,
                sentence=context,
                span=(metric_start, metric_end),
            )
            occurrences.append(occurrence)
            if has_unresolved_metric_predicate(metric_phrase, context):
                unresolved_predicates.append(occurrence)
            for value_text, _span in numeric_value_candidates(context):
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

    completed = [search for search in searches if search.completion_status == "completed"]
    has_exact_lexical = any(
        search.mode == "lexical" and search.normalized_query == _normalized_query(metric)
        for search in completed
    )
    if (
        len(completed) < 4
        or not has_exact_lexical
        or not exact_metric_scan_completed
        or len(chunks) != len(unique_ids)
    ):
        conclusion = AbsenceConclusion.INCOMPLETE_PROBE
    elif unresolved_predicates or value_candidates:
        conclusion = AbsenceConclusion.EVIDENCE_REQUIRES_REVIEW
    else:
        conclusion = AbsenceConclusion.NOT_FOUND_IN_PROBE
    return AbsenceProbeResult(
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
        conclusion=conclusion,
    )
