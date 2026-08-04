from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from .binding import (
    BINDING_CONTRACT_SHA256,
    BindingMatch,
    bind_observation,
    canonical_numeric_signature,
    numeric_signature_sequence,
    numeric_signatures,
)
from .chunk_ids import ChunkId
from .models import (
    AbsenceConclusion,
    AbsenceProbeResult,
    ClaimStatus,
    DraftClaim,
    DraftObservation,
    EvidenceReference,
    EvidenceRelation,
    EvidenceVerificationResult,
    ObservationKind,
    VerifiedClaim,
    VerifiedEvidence,
)
from .retrieval import CorpusStore
from .util import evidence_text_contains, normalize_evidence_text

_WORD = re.compile(r"[^\W_]+")
_MONTHS = {month.casefold(): index for index, month in enumerate(calendar.month_name) if month}
_EXPLICIT_CONFLICT_PHRASES = (
    "cannot be right",
    "incompatible",
    "introduced the error",
    "unresolved conflict",
)


@dataclass(frozen=True, slots=True)
class _VerifiedObservation:
    observation: DraftObservation
    evidence: tuple[VerifiedEvidence, ...]
    authorized: bool


def _canonical_metric(value: str) -> str:
    normalized, _ = normalize_evidence_text(value)
    return " ".join(_WORD.findall(normalized.casefold()))


def _word_phrase_found(needle: str, haystack: str) -> bool:
    expected = _WORD.findall(needle.casefold())
    observed = _WORD.findall(haystack.casefold())
    width = len(expected)
    return bool(
        expected
        and any(
            observed[index : index + width] == expected
            for index in range(len(observed) - width + 1)
        )
    )


def reported_value_found(value: str, quote: str) -> bool:
    """Return whether the exact qualitative phrase or numeric signature occurs."""

    normalized_value, _ = normalize_evidence_text(value)
    normalized_quote, _ = normalize_evidence_text(quote)
    expected = numeric_signatures(normalized_value)
    if expected:
        return expected.issubset(numeric_signatures(normalized_quote))
    return _word_phrase_found(normalized_value, normalized_quote)


def reported_value_linked_to_metric(
    value: str,
    metric_anchor: str,
    quote: str,
    *,
    kind: ObservationKind = ObservationKind.REPORTED_LEVEL,
    temporal_anchor: str | None = None,
) -> BindingMatch | None:
    """Return the proved positive binding, never an absence-of-known-errors guess."""

    return bind_observation(
        metric_anchor=metric_anchor,
        value_text=value,
        kind=kind,
        temporal_anchor=temporal_anchor,
        assertion=quote,
    ).match


def temporal_anchor_linked_to_value(
    value: str,
    temporal_anchor: str,
    metric_anchor: str,
    quote: str,
    *,
    kind: ObservationKind = ObservationKind.REPORTED_LEVEL,
) -> bool:
    """Compatibility diagnostic backed by one exact binding match."""

    match = reported_value_linked_to_metric(
        value,
        metric_anchor,
        quote,
        kind=kind,
        temporal_anchor=temporal_anchor,
    )
    return bool(match and match.temporal_span)


def _canonical_value(value: str) -> str:
    signatures = numeric_signature_sequence(value)
    if signatures:
        return repr(tuple(canonical_numeric_signature(signature) for signature in signatures))
    return normalize_evidence_text(value)[0].casefold()


def _distinct_values(observations: Iterable[DraftObservation]) -> set[str]:
    return {_canonical_value(observation.value_text) for observation in observations}


def _valid_date(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _temporal_signature(value: str) -> str | None:
    normalized = normalize_evidence_text(value)[0].casefold()
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", normalized)
    if iso:
        year, month, day = (int(part) for part in iso.groups())
        return f"date:{year:04d}-{month:02d}-{day:02d}" if _valid_date(year, month, day) else None
    day_first = re.search(
        r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b",
        normalized,
    )
    if day_first:
        day_text, month_text, year_text = day_first.groups()
        year, month, day = int(year_text), _MONTHS[month_text], int(day_text)
        return f"date:{year:04d}-{month:02d}-{day:02d}" if _valid_date(year, month, day) else None
    month_first = re.search(
        r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b",
        normalized,
    )
    if month_first:
        month_text, day_text, year_text = month_first.groups()
        year, month, day = int(year_text), _MONTHS[month_text], int(day_text)
        return f"date:{year:04d}-{month:02d}-{day:02d}" if _valid_date(year, month, day) else None
    fiscal_period = re.search(
        r"\b(?:fy|fiscal\s+year)\s*(\d{4})\b|\bq([1-4])\s*(\d{4})\b",
        normalized,
    )
    if fiscal_period:
        fiscal_year, quarter, quarter_year = fiscal_period.groups()
        return f"fiscal-year:{fiscal_year}" if fiscal_year else f"quarter:{quarter_year}-q{quarter}"
    return None


def _temporal_anchor_is_valid(value: str | None) -> bool:
    if value is None:
        return True
    normalized = normalize_evidence_text(value)[0].casefold()
    looks_like_calendar_date = bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", normalized)
        or re.search(r"\b\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")\s+\d{4}\b", normalized)
        or re.search(
            r"\b(?:" + "|".join(_MONTHS) + r")\s+\d{1,2},?\s+\d{4}\b",
            normalized,
        )
    )
    return not looks_like_calendar_date or _temporal_signature(value) is not None


def _distinct_temporal_anchors(observations: Iterable[DraftObservation]) -> set[str]:
    return {
        signature
        for observation in observations
        if observation.temporal_anchor
        and (signature := _temporal_signature(observation.temporal_anchor)) is not None
    }


def _reference_key(reference: EvidenceReference) -> tuple[ChunkId, str, str, str, EvidenceRelation]:
    return (
        reference.chunk_id,
        reference.metric_anchor,
        reference.exact_assertion,
        reference.exact_quote,
        reference.relation,
    )


def _unique_references(claim: DraftClaim) -> list[EvidenceReference]:
    references = list(claim.context_evidence)
    for observation in claim.observations:
        references.extend(observation.evidence)
    output: list[EvidenceReference] = []
    seen: set[tuple[ChunkId, str, str, str, EvidenceRelation]] = set()
    for reference in references:
        key = _reference_key(reference)
        if key not in seen:
            seen.add(key)
            output.append(reference)
    return output


def _has_explicit_conflict(evidence: Iterable[VerifiedEvidence], metric: str) -> bool:
    return any(
        item.relation == EvidenceRelation.SUPPORTS
        and item.quote_found
        and item.assertion_found
        and item.metric_anchor_found
        and _canonical_metric(item.metric_anchor) == _canonical_metric(metric)
        and any(phrase in item.normalized_quote.casefold() for phrase in _EXPLICIT_CONFLICT_PHRASES)
        for item in evidence
    )


def _render_observation(observation: DraftObservation) -> str:
    if observation.temporal_anchor:
        return f"{observation.value_text} ({observation.temporal_anchor})"
    return observation.value_text


def _authoritative_statement(
    metric: str,
    status: ClaimStatus,
    observations: list[DraftObservation],
) -> str:
    metric_anchor = metric.strip()
    for observation in observations:
        if observation.evidence:
            metric_anchor = observation.evidence[0].metric_anchor.strip()
            break
    rendered = [_render_observation(observation) for observation in observations]
    values_text = (
        " and ".join(rendered)
        if len(rendered) <= 2
        else ", ".join(rendered[:-1]) + f", and {rendered[-1]}"
    )
    if status == ClaimStatus.NOT_FOUND:
        return metric_anchor
    if status == ClaimStatus.CONFLICT:
        return f"{metric_anchor} is reported with conflicting values: {values_text}"
    if status == ClaimStatus.DATE_VARIANT:
        return f"{metric_anchor} is reported at different dates: {values_text}"
    if status == ClaimStatus.POSSIBLE_CONFLICT:
        return f"{metric_anchor} has unresolved reported values: {values_text}"
    if status == ClaimStatus.VERIFIED:
        return f"{metric_anchor}: {values_text}"
    return f"Unverified claim about {metric_anchor}"


class EvidenceVerifier:
    version = "deterministic-verifier-v5-positive-bindings"
    binding_contract_sha256 = BINDING_CONTRACT_SHA256

    def __init__(self, corpus: CorpusStore):
        self.corpus = corpus

    def verify_claims(
        self,
        claims: list[DraftClaim],
        *,
        absence_probes: dict[str, AbsenceProbeResult] | None = None,
        completed_searches: int | None = None,
        completed_search_records: list[dict[str, object]] | None = None,
    ) -> EvidenceVerificationResult:
        del completed_searches, completed_search_records
        verified = [
            self.verify_claim(
                claim,
                absence_probe=(absence_probes or {}).get(_canonical_metric(claim.metric)),
            )
            for claim in claims
        ]
        authoritative = all(
            claim.status
            in {
                ClaimStatus.VERIFIED,
                ClaimStatus.CONFLICT,
                ClaimStatus.DATE_VARIANT,
                ClaimStatus.NOT_FOUND,
            }
            for claim in verified
        )
        return EvidenceVerificationResult(claims=verified, all_claims_authoritative=authoritative)

    def verify_claim(
        self,
        claim: DraftClaim,
        *,
        absence_probe: AbsenceProbeResult | None = None,
        completed_searches: int | None = None,
        completed_search_records: list[dict[str, object]] | None = None,
    ) -> VerifiedClaim:
        del completed_searches, completed_search_records
        notes: list[str] = []
        references = _unique_references(claim)
        chunks = self.corpus.get_chunks([reference.chunk_id for reference in references])
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        verified_by_reference: dict[
            tuple[ChunkId, str, str, str, EvidenceRelation], VerifiedEvidence
        ] = {}

        observation_by_reference: dict[
            tuple[ChunkId, str, str, str, EvidenceRelation], DraftObservation
        ] = {}
        for observation in claim.observations:
            for reference in observation.evidence:
                observation_by_reference[_reference_key(reference)] = observation

        for reference in references:
            key = _reference_key(reference)
            chunk = chunks_by_id.get(reference.chunk_id)
            normalized_quote, operations = normalize_evidence_text(reference.exact_quote)
            operations = [*operations, "unicode_casefold_comparison"]
            quote_found = bool(
                chunk and evidence_text_contains(reference.exact_quote, chunk.normalized_text)
            )
            assertion_found = evidence_text_contains(
                reference.exact_assertion,
                reference.exact_quote,
            )
            metric_anchor_found = _word_phrase_found(
                reference.metric_anchor,
                reference.exact_assertion,
            )
            observation = observation_by_reference.get(key)
            binding = None
            value_text_found = False
            failure_reason: str | None = None
            if observation is not None:
                result = bind_observation(
                    metric_anchor=reference.metric_anchor,
                    value_text=observation.value_text,
                    kind=observation.kind,
                    temporal_anchor=observation.temporal_anchor,
                    assertion=reference.exact_assertion,
                )
                binding = result.match
                value_text_found = result.value_text_found
                failure_reason = result.failure_reason
            if chunk is None:
                notes.append(f"Chunk {reference.chunk_id} is not part of this corpus version.")
                continue
            metric_matches_claim = _canonical_metric(reference.metric_anchor) == _canonical_metric(
                claim.metric
            )
            verified_by_reference[key] = VerifiedEvidence(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                physical_page_index=chunk.physical_page_index,
                printed_page_label=chunk.printed_page_label,
                metric_anchor=reference.metric_anchor,
                metric_anchor_found=metric_anchor_found,
                assertion=reference.exact_assertion,
                assertion_found=assertion_found,
                temporal_anchor=observation.temporal_anchor if observation else None,
                temporal_value_bound=bool(
                    binding
                    and _temporal_anchor_is_valid(
                        observation.temporal_anchor if observation else None
                    )
                    and (
                        observation is None
                        or observation.temporal_anchor is None
                        or binding.temporal_span is not None
                    )
                ),
                quote=reference.exact_quote,
                normalized_quote=normalized_quote,
                normalization_operations=operations,
                relation=reference.relation,
                quote_found=quote_found,
                value_text_found=value_text_found,
                metric_value_bound=bool(binding and metric_matches_claim),
                value_role_authorized=bool(binding),
                binding_profile=binding.profile if binding else None,
                binding_failure_reason=(
                    failure_reason
                    if metric_matches_claim
                    else "metric_anchor_does_not_match_canonical_metric"
                ),
                chunk_sha256=chunk.sha256,
                source_url=(
                    f"/api/corpora/{self.corpus.corpus_version}/documents/"
                    f"{chunk.document_id}/pdf#page={chunk.physical_page_index}"
                ),
            )

        evidence = list(verified_by_reference.values())
        verified_observations: list[_VerifiedObservation] = []
        for observation in claim.observations:
            observation_evidence = tuple(
                verified_by_reference[key]
                for reference in observation.evidence
                if (key := _reference_key(reference)) in verified_by_reference
            )
            supporting = any(
                item.relation == EvidenceRelation.SUPPORTS
                and item.quote_found
                and item.assertion_found
                and item.metric_anchor_found
                and item.value_text_found
                and item.metric_value_bound
                and item.temporal_value_bound
                and item.value_role_authorized
                for item in observation_evidence
            )
            verified_observations.append(
                _VerifiedObservation(observation, observation_evidence, supporting)
            )

        authorized = [item.observation for item in verified_observations if item.authorized]
        all_observations_authorized = bool(claim.observations) and len(authorized) == len(
            claim.observations
        )
        distinct_values = _distinct_values(authorized)
        distinct_temporal = _distinct_temporal_anchors(authorized)
        temporal_count = sum(bool(observation.temporal_anchor) for observation in authorized)
        explicit_conflict = _has_explicit_conflict(evidence, claim.metric)
        all_references_valid = len(evidence) == len(references) and all(
            item.quote_found
            and item.assertion_found
            and item.metric_anchor_found
            and _canonical_metric(item.metric_anchor) == _canonical_metric(claim.metric)
            for item in evidence
        )

        if claim.request_absence_probe:
            if claim.observations or references:
                status = ClaimStatus.UNVERIFIED
                notes.append("An absence request cannot also propose observations or evidence.")
            elif absence_probe and absence_probe.conclusion == AbsenceConclusion.NOT_FOUND_IN_PROBE:
                status = ClaimStatus.NOT_FOUND
                notes.append(
                    f"Not found after {len(absence_probe.searches)} searches and "
                    f"{len(absence_probe.opened_chunk_ids)} inspected candidates across "
                    f"corpus version {self.corpus.corpus_version} under "
                    f"{absence_probe.protocol_version}."
                )
            else:
                status = ClaimStatus.UNVERIFIED
                notes.append("The server-owned bounded absence probe did not establish not found.")
        elif not claim.observations:
            status = ClaimStatus.UNVERIFIED
            notes.append("An authoritative factual claim requires at least one observation.")
        elif not all_references_valid or not all_observations_authorized:
            status = ClaimStatus.UNVERIFIED
            if not all_references_valid:
                notes.append(
                    "Every cited evidence reference must resolve inside this corpus version."
                )
            notes.append(
                "Every observation requires a supporting quotation with one authorized positive binding profile."
            )
        elif len(distinct_values) >= 2:
            if explicit_conflict:
                status = ClaimStatus.CONFLICT
            elif temporal_count == len(authorized) and len(distinct_temporal) == len(authorized):
                status = ClaimStatus.DATE_VARIANT
            elif temporal_count == len(authorized) and len(distinct_temporal) == 1:
                status = ClaimStatus.CONFLICT
            else:
                status = ClaimStatus.POSSIBLE_CONFLICT
                notes.append(
                    "Distinct values lack a fully verified common or distinct temporal relationship."
                )
        elif len(distinct_values) == 1:
            status = ClaimStatus.VERIFIED
        else:
            status = ClaimStatus.UNVERIFIED

        return VerifiedClaim(
            statement=_authoritative_statement(claim.metric, status, authorized),
            metric=claim.metric,
            status=status,
            observations=claim.observations,
            evidence=evidence,
            absence_probe=absence_probe,
            verification_notes=notes,
        )
