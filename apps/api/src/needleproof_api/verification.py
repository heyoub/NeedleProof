from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from .absence import derive_absence_conclusion
from .binding import (
    BINDING_CONTRACT_SHA256,
    BindingMatch,
    bind_observation,
    canonical_numeric_signature,
    is_authorized_temporal_anchor,
    numeric_signature_sequence,
    numeric_signatures,
    numeric_value_candidates,
)
from .chunk_ids import ChunkId
from .models import (
    AbsenceConclusion,
    AbsenceProbeResult,
    ChunkRecord,
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
from .util import canonical_metric_key, evidence_text_contains, normalize_evidence_text

_WORD = re.compile(r"[^\W_]+")
_MONTHS = {month.casefold(): index for index, month in enumerate(calendar.month_name) if month}
_BOUND_CONFLICT_RELATION = re.compile(
    r"\b(?:"
    r"(?P<bridge>cannot\s+be\s+right\s+(?:when\s+)?alongside)|"
    r"(?P<typed>(?:values|figures|amounts|numbers|reports|sources)\s+"
    r"(?:are|remain|seem|appear)?\s*(?:incompatible|conflicting)|"
    r"(?:incompatible|conflicting)\s+"
    r"(?:values|figures|amounts|numbers|reports|sources))|"
    r"(?P<unresolved>(?:values|figures|amounts|numbers|reports|sources)\s+"
    r"(?:form|create|represent|show)?\s*(?:an?\s+)?unresolved\s+conflict|"
    r"unresolved\s+conflict\s+(?:between|among))"
    r")\b",
    re.IGNORECASE,
)
_TEMPORAL_LEAD = re.compile(r"^(?:as\s+of|at|by|during|for|in|on|through)\s+", re.IGNORECASE)
_QUARTER_NUMBERS = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
_CONFLICT_BRIDGE_LEFT_GAP = re.compile(r"\s*,?\s*(?:which\s+)?", re.IGNORECASE)
_CONFLICT_BRIDGE_RIGHT_GAP = re.compile(r"\s*(?:the\s*)?", re.IGNORECASE)
_CONFLICT_COLLECTIVE_LEAD_GAP = re.compile(
    r"\s*[,;:]?\s*(?:(?:and|but)\s+)?(?:the\s+)?",
    re.IGNORECASE,
)
_CONFLICT_COLLECTIVE_TAIL = re.compile(
    r"\s*(?:with\s+(?:one\s+another|each\s+other))?\s*[.!?]?\s*",
    re.IGNORECASE,
)
_DIRECT_ASSERTION_QUOTE_BOUNDARY = re.compile(
    r"(?:[.!?;]|[,;:]\s*(?:and|but|while|whereas))\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _VerifiedObservation:
    observation: DraftObservation
    evidence: tuple[VerifiedEvidence, ...]
    authorized: bool


class VerificationCorpus(Protocol):
    @property
    def corpus_version(self) -> str: ...

    def get_chunks(
        self,
        chunk_ids: list[ChunkId],
        neighbor_radius: int = 0,
    ) -> list[ChunkRecord]: ...


def _canonical_metric(value: str) -> str:
    return canonical_metric_key(value)


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


def _binding_respects_quote_boundaries(
    assertion: str,
    quote: str,
    binding: BindingMatch,
) -> bool:
    """Reject a clean relationship cropped out of role-changing quote context."""

    normalized_assertion = normalize_evidence_text(assertion)[0].casefold()
    normalized_quote = normalize_evidence_text(quote)[0].casefold()
    direct_metric_subject = bool(
        re.fullmatch(r"\s*(?:the\s+)?", normalized_assertion[: binding.metric_span[0]])
    )
    for occurrence in re.finditer(re.escape(normalized_assertion), normalized_quote):
        prefix = normalized_quote[: occurrence.start()].rstrip()
        suffix = normalized_quote[occurrence.end() :]
        if (
            direct_metric_subject
            and prefix
            and _DIRECT_ASSERTION_QUOTE_BOUNDARY.search(prefix) is None
        ):
            continue
        if (
            suffix
            and normalized_assertion.rstrip()[-1:] not in ".!?"
            and re.match(r"\s*[,.;:!?]", suffix) is None
        ):
            continue
        return True
    return False


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
    core = _TEMPORAL_LEAD.sub("", normalized, count=1)
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
    fiscal_year = re.fullmatch(r"(?:fy|fiscal\s+year)(?:\s*(\d{4}))?", core)
    if fiscal_year:
        return f"fiscal-year:{fiscal_year.group(1) or 'unspecified'}"
    quarter = re.fullmatch(
        r"(?:q([1-4])|(first|second|third|fourth)\s+quarter)(?:\s+(\d{4}))?",
        core,
    )
    if quarter:
        number, named, year = quarter.groups()
        return f"quarter:{year or 'unspecified'}-q{number or _QUARTER_NUMBERS[named]}"
    year = re.fullmatch(r"(?:(?:calendar\s+)?year\s+)?((?:19|20)\d{2})", core)
    if year:
        return f"year:{year.group(1)}"
    if re.fullmatch(r"(?:calendar\s+)?year", core):
        return "year:unspecified"
    month_period = re.fullmatch(
        r"(" + "|".join(_MONTHS) + r")(?:\s+((?:19|20)\d{2}))?(?:\s+(?:month|period))?",
        core,
    )
    if month_period:
        month_name, year_text = month_period.groups()
        return f"month:{year_text or 'unspecified'}-{_MONTHS[month_name]:02d}"
    month_event = re.fullmatch(
        r"(" + "|".join(_MONTHS) + r")(?:\s+((?:19|20)\d{2}))?\s+(call)",
        core,
    )
    if month_event:
        month_name, year_text, event = month_event.groups()
        return f"event:{event}:{year_text or 'unspecified'}-{_MONTHS[month_name]:02d}"
    if is_authorized_temporal_anchor(value):
        return f"period:{core}"
    return None


def _temporal_anchor_is_valid(value: str | None) -> bool:
    if not is_authorized_temporal_anchor(value):
        return False
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


def _compatible_measurements(
    sentence: str,
    authorized_values: set[tuple[str, str, str, str]],
) -> list[tuple[tuple[str, str, str, str], tuple[int, int]]]:
    measurements = []
    for value_text, span in numeric_value_candidates(sentence):
        signatures = numeric_signature_sequence(value_text)
        if len(signatures) != 1:
            continue
        signature = canonical_numeric_signature(signatures[0])
        if any(
            signature[1] == authorized[1] and signature[3] == authorized[3]
            for authorized in authorized_values
        ):
            measurements.append((signature, span))
    return measurements


def _relation_binds_distinct_measurements(
    sentence: str,
    relation: re.Match[str],
    measurements: list[tuple[tuple[str, str, str, str], tuple[int, int]]],
    authorized_values: set[tuple[str, str, str, str]],
) -> bool:
    if relation.group("bridge"):
        before = [item for item in measurements if item[1][1] <= relation.start()]
        after = [item for item in measurements if item[1][0] >= relation.end()]
        if not before or not after:
            return False
        left, right = before[-1], after[0]
        if not _CONFLICT_BRIDGE_LEFT_GAP.fullmatch(sentence[left[1][1] : relation.start()]):
            return False
        if not _CONFLICT_BRIDGE_RIGHT_GAP.fullmatch(sentence[relation.end() : right[1][0]]):
            return False
        candidates = (left[0], right[0])
        return len(set(candidates)) >= 2 and any(value in candidates for value in authorized_values)
    observed_authorized = [item for item in measurements if item[0] in authorized_values]
    if len({signature for signature, _span in observed_authorized}) < 2:
        return False
    last_value_end = observed_authorized[-1][1][1]
    return bool(
        last_value_end <= relation.start()
        and _CONFLICT_COLLECTIVE_LEAD_GAP.fullmatch(sentence[last_value_end : relation.start()])
        and _CONFLICT_COLLECTIVE_TAIL.fullmatch(sentence[relation.end() :])
    )


def _has_explicit_conflict(
    evidence: Iterable[VerifiedEvidence],
    metric: str,
    observations: Iterable[DraftObservation],
) -> bool:
    """Recognize only source conflict language bound locally to disputed values."""

    authorized_values = {
        canonical_numeric_signature(signature)
        for observation in observations
        for signature in numeric_signature_sequence(observation.value_text)
    }
    if not authorized_values:
        return False
    for item in evidence:
        if not (
            item.quote_found
            and item.assertion_found
            and item.metric_anchor_found
            and _canonical_metric(item.metric_anchor) == _canonical_metric(metric)
        ):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", item.normalized_quote.casefold()):
            relation = _BOUND_CONFLICT_RELATION.search(sentence)
            if not relation or not _word_phrase_found(metric, sentence):
                continue
            measurements = _compatible_measurements(sentence, authorized_values)
            if _relation_binds_distinct_measurements(
                sentence,
                relation,
                measurements,
                authorized_values,
            ):
                return True
    return False


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
    version = "deterministic-verifier-v16-value-consistent-anaphora"
    binding_contract_sha256 = BINDING_CONTRACT_SHA256

    def __init__(self, corpus: VerificationCorpus):
        self.corpus = corpus

    def verify_claims(
        self,
        claims: list[DraftClaim],
        *,
        absence_probes: dict[str, AbsenceProbeResult] | None = None,
        completed_searches: int | None = None,
        completed_search_records: list[dict[str, object]] | None = None,
    ) -> EvidenceVerificationResult:
        """Verify claims; deprecated model-search counters never authorize absence."""

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
        """Verify one claim; deprecated model-search counters are intentionally ignored."""

        del completed_searches, completed_search_records
        notes: list[str] = []
        reference_pairs: list[tuple[EvidenceReference, DraftObservation | None]] = [
            (reference, None) for reference in claim.context_evidence
        ]
        reference_pairs.extend(
            (reference, observation)
            for observation in claim.observations
            for reference in observation.evidence
        )
        references = [reference for reference, _observation in reference_pairs]
        chunks = self.corpus.get_chunks([reference.chunk_id for reference in references])
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        missing_chunk_ids: set[str] = set()

        def verify_reference(
            reference: EvidenceReference,
            observation: DraftObservation | None,
        ) -> VerifiedEvidence | None:
            chunk = chunks_by_id.get(reference.chunk_id)
            if chunk is None:
                missing_chunk_ids.add(str(reference.chunk_id))
                return None
            normalized_quote, operations = normalize_evidence_text(reference.exact_quote)
            operations = [*operations, "unicode_casefold_comparison"]
            quote_found = evidence_text_contains(
                reference.exact_quote,
                chunk.normalized_text,
            )
            assertion_found = evidence_text_contains(
                reference.exact_assertion,
                reference.exact_quote,
            )
            metric_anchor_found = _word_phrase_found(
                reference.metric_anchor,
                reference.exact_assertion,
            )
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
                if binding and not _binding_respects_quote_boundaries(
                    reference.exact_assertion,
                    reference.exact_quote,
                    binding,
                ):
                    binding = None
                    assertion_found = False
                    failure_reason = "assertion_not_bound_to_quote_context"
            metric_matches_claim = _canonical_metric(reference.metric_anchor) == _canonical_metric(
                claim.metric
            )
            return VerifiedEvidence(
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
                    and observation
                    and _temporal_anchor_is_valid(observation.temporal_anchor)
                    and (observation.temporal_anchor is None or binding.temporal_span is not None)
                ),
                observation_kind=observation.kind if observation else None,
                value_text=observation.value_text if observation else None,
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

        context_evidence = [
            verified
            for reference in claim.context_evidence
            if (verified := verify_reference(reference, None)) is not None
        ]
        verified_observations: list[_VerifiedObservation] = []
        for observation in claim.observations:
            observation_evidence = tuple(
                verified
                for reference in observation.evidence
                if (verified := verify_reference(reference, observation)) is not None
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

        evidence = [
            *context_evidence,
            *(item for observation in verified_observations for item in observation.evidence),
        ]

        authorized = [item.observation for item in verified_observations if item.authorized]
        all_observations_authorized = bool(claim.observations) and len(authorized) == len(
            claim.observations
        )
        distinct_values = _distinct_values(authorized)
        distinct_temporal = _distinct_temporal_anchors(authorized)
        temporal_count = sum(bool(observation.temporal_anchor) for observation in authorized)
        explicit_conflict = _has_explicit_conflict(evidence, claim.metric, authorized)
        unresolved_references = len(evidence) != len(references)
        missing_quotes = any(not item.quote_found for item in evidence)
        unmatched_assertions = any(not item.assertion_found for item in evidence)
        invalid_metric_anchors = any(
            not item.metric_anchor_found
            or _canonical_metric(item.metric_anchor) != _canonical_metric(claim.metric)
            for item in evidence
        )
        all_references_valid = not any(
            (
                unresolved_references,
                missing_quotes,
                unmatched_assertions,
                invalid_metric_anchors,
            )
        )

        if claim.request_absence_probe:
            if claim.observations or references:
                status = ClaimStatus.UNVERIFIED
                notes.append("An absence request cannot also propose observations or evidence.")
            elif (
                absence_probe
                and _canonical_metric(absence_probe.metric) == _canonical_metric(claim.metric)
                and absence_probe.conclusion == AbsenceConclusion.NOT_FOUND_IN_PROBE
                and derive_absence_conclusion(absence_probe) == AbsenceConclusion.NOT_FOUND_IN_PROBE
            ):
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
            if unresolved_references:
                notes.append(
                    "Evidence references do not resolve inside this corpus version: "
                    + ", ".join(sorted(missing_chunk_ids))
                )
            if missing_quotes:
                notes.append("One or more exact quotations were not found in their cited chunks.")
            if unmatched_assertions:
                notes.append("One or more exact assertions were not found inside their quotations.")
            if invalid_metric_anchors:
                notes.append(
                    "One or more metric anchors were absent from the assertion or did not match the canonical claim metric."
                )
            if not all_observations_authorized:
                notes.append(
                    "Every observation requires a supporting quotation with one authorized positive binding profile."
                )
        elif explicit_conflict and len(distinct_values) < 2:
            status = ClaimStatus.POSSIBLE_CONFLICT
            notes.append(
                "Recognized same-metric conflict evidence prevents single-value authorization."
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
            context_evidence=claim.context_evidence,
            evidence=evidence,
            absence_probe=absence_probe,
            verification_notes=notes,
        )
