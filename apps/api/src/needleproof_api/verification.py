from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation

from .chunk_ids import ChunkId
from .models import (
    ClaimStatus,
    DraftClaim,
    EvidenceReference,
    EvidenceRelation,
    EvidenceVerificationResult,
    ReportedValue,
    VerifiedClaim,
    VerifiedEvidence,
)
from .retrieval import CorpusStore
from .util import evidence_text_contains, normalize_evidence_text

_NUMERIC = re.compile(
    r"(?P<open>\()?\s*(?P<sign_before>[+-])?\s*(?P<currency>[$£€])?\s*"
    r"(?P<sign_after>[+-])?\s*(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>billion|million|thousand|basis\s+points?|bps|percent|%|per\s+share)?"
    r"\s*(?P<close>\))?",
    re.IGNORECASE,
)


def numeric_signatures(text: str) -> set[tuple[str, str, str, str]]:
    signatures: set[tuple[str, str, str, str]] = set()
    for match in _NUMERIC.finditer(text):
        parenthesized = bool(match.group("open") and match.group("close"))
        explicit_sign = match.group("sign_before") or match.group("sign_after")
        sign = "-" if parenthesized or explicit_sign == "-" else "+"
        currency = match.group("currency") or ""
        number = match.group("number").replace(",", "")
        unit = re.sub(r"\s+", " ", (match.group("unit") or "").lower())
        signatures.add((sign, currency, number, unit))
    return signatures


def reported_value_found(value: str, quote: str) -> bool:
    normalized_value, _ = normalize_evidence_text(value)
    normalized_quote, _ = normalize_evidence_text(quote)
    expected = numeric_signatures(normalized_value)
    if expected:
        observed = numeric_signatures(normalized_quote)
        return expected.issubset(observed)
    return normalized_value.casefold() in normalized_quote.casefold()


def _all_evidence(claim: DraftClaim) -> list[EvidenceReference]:
    evidence: list[EvidenceReference] = list(claim.evidence)
    for value in claim.values:
        evidence.extend(value.evidence)
    output: list[EvidenceReference] = []
    seen: set[tuple[ChunkId, str, str, EvidenceRelation]] = set()
    for reference in evidence:
        key = (
            reference.chunk_id,
            reference.metric_anchor,
            reference.exact_quote,
            reference.relation,
        )
        if key not in seen:
            seen.add(key)
            output.append(reference)
    return output


def _distinct_values(values: Iterable[ReportedValue]) -> set[str]:
    output: set[str] = set()
    for value in values:
        signatures = numeric_signatures(value.value)
        if signatures:
            canonical: list[tuple[str, str, str, str]] = []
            for sign, currency, number, unit in signatures:
                try:
                    number = format(Decimal(number).normalize(), "f")
                except InvalidOperation:
                    pass
                unit = {
                    "bps": "basis points",
                    "basis point": "basis points",
                    "%": "percent",
                }.get(unit, unit)
                canonical.append((sign, currency, number, unit))
            output.add(repr(sorted(canonical)))
        else:
            output.add(normalize_evidence_text(value.value)[0].casefold())
    return output


_MONTHS = {
    month.casefold(): index
    for index, month in enumerate(
        (
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
    )
    if month
}


def _temporal_signature(value: str) -> str | None:
    normalized = normalize_evidence_text(value)[0].casefold()
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", normalized)
    if iso:
        return f"date:{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"
    day_first = re.search(
        r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b",
        normalized,
    )
    if day_first:
        day, month, year = day_first.groups()
        return f"date:{year}-{_MONTHS[month]:02d}-{int(day):02d}"
    month_first = re.search(
        r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b",
        normalized,
    )
    if month_first:
        month, day, year = month_first.groups()
        return f"date:{year}-{_MONTHS[month]:02d}-{int(day):02d}"
    fiscal_period = re.search(
        r"\b(?:fy|fiscal\s+year)\s*(\d{4})\b|\bq([1-4])\s*(\d{4})\b",
        normalized,
    )
    if fiscal_period:
        fiscal_year, quarter, quarter_year = fiscal_period.groups()
        if fiscal_year:
            return f"fiscal-year:{fiscal_year}"
        return f"quarter:{quarter_year}-q{quarter}"
    return None


def _distinct_temporal_anchors(values: Iterable[ReportedValue]) -> set[str]:
    signatures = (
        _temporal_signature(value.temporal_anchor) for value in values if value.temporal_anchor
    )
    return {signature for signature in signatures if signature is not None}


def _canonical_metric(value: str) -> str:
    normalized, _ = normalize_evidence_text(value)
    return re.sub(r"[^a-z0-9]+", " ", normalized.casefold()).strip()


def _text_found(needle: str, haystack: str) -> bool:
    return evidence_text_contains(needle, haystack)


def _reference_key(
    reference: EvidenceReference,
) -> tuple[ChunkId, str, str, EvidenceRelation]:
    return (
        reference.chunk_id,
        reference.metric_anchor,
        reference.exact_quote,
        reference.relation,
    )


def _authoritative_statement(claim: DraftClaim, status: ClaimStatus) -> str:
    if status == ClaimStatus.NOT_FOUND:
        return claim.metric.strip()
    if status == ClaimStatus.UNVERIFIED:
        return f"Unverified claim about {claim.metric.strip()}"

    metric_anchor = claim.metric.strip()
    for value in claim.values:
        if value.evidence:
            metric_anchor = value.evidence[0].metric_anchor.strip()
            break
    rendered_values = [
        f"{value.value} ({value.temporal_anchor})" if value.temporal_anchor else value.value
        for value in claim.values
    ]
    values_text = ", ".join(rendered_values[:-1])
    if len(rendered_values) > 1:
        values_text = f"{values_text} and {rendered_values[-1]}"
    elif rendered_values:
        values_text = rendered_values[0]

    if status == ClaimStatus.CONFLICT:
        return f"{metric_anchor} is reported with conflicting values: {values_text}"
    if status == ClaimStatus.DATE_VARIANT:
        return f"{metric_anchor} is reported at different dates: {values_text}"
    if status == ClaimStatus.POSSIBLE_CONFLICT:
        return f"{metric_anchor} has unresolved reported values: {values_text}"
    return f"{metric_anchor}: {values_text}"


def _has_explicit_conflict(evidence: list[VerifiedEvidence]) -> bool:
    conflict_phrases = (
        "cannot be right",
        "incompatible",
        "introduced the error",
        "unresolved conflict",
    )
    return any(
        item.relation == EvidenceRelation.SUPPORTS
        and any(phrase in item.normalized_quote.casefold() for phrase in conflict_phrases)
        for item in evidence
    )


class EvidenceVerifier:
    version = "deterministic-verifier-v3-strict-signatures"

    def __init__(self, corpus: CorpusStore):
        self.corpus = corpus

    def verify_claims(
        self,
        claims: list[DraftClaim],
        *,
        completed_searches: int,
        completed_search_records: list[dict[str, object]] | None = None,
    ) -> EvidenceVerificationResult:
        verified = [
            self.verify_claim(
                claim,
                completed_searches=completed_searches,
                completed_search_records=completed_search_records,
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
        completed_searches: int,
        completed_search_records: list[dict[str, object]] | None = None,
    ) -> VerifiedClaim:
        notes: list[str] = []
        references = _all_evidence(claim)
        chunks = self.corpus.get_chunks([reference.chunk_id for reference in references])
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        evidence: list[VerifiedEvidence] = []
        missing_reference = False
        valid_reference_keys: set[tuple[ChunkId, str, str, EvidenceRelation]] = set()
        metric_linked_keys: set[tuple[ChunkId, str, str, EvidenceRelation]] = set()

        for reference in references:
            chunk = chunks_by_id.get(reference.chunk_id)
            normalized_quote, operations = normalize_evidence_text(reference.exact_quote)
            quote_found = bool(
                chunk and evidence_text_contains(reference.exact_quote, chunk.normalized_text)
            )
            relevant_values = [
                value
                for value in claim.values
                if any(
                    nested.chunk_id == reference.chunk_id
                    and nested.exact_quote == reference.exact_quote
                    for nested in value.evidence
                )
            ]
            value_found = all(
                reported_value_found(value.value, reference.exact_quote)
                for value in relevant_values
            )
            if not relevant_values:
                value_found = True
            metric_anchor_found = bool(
                chunk and _text_found(reference.metric_anchor, reference.exact_quote)
            )
            metric_linked = _canonical_metric(reference.metric_anchor) == _canonical_metric(
                claim.metric
            )
            temporal_anchors = [
                value.temporal_anchor for value in relevant_values if value.temporal_anchor
            ]
            temporal_anchors_found = all(
                _text_found(anchor, reference.exact_quote) for anchor in temporal_anchors
            )
            if not chunk:
                missing_reference = True
                notes.append(f"Chunk {reference.chunk_id} is not part of this corpus version.")
                continue
            evidence.append(
                VerifiedEvidence(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    physical_page_index=chunk.physical_page_index,
                    printed_page_label=chunk.printed_page_label,
                    metric_anchor=reference.metric_anchor,
                    metric_anchor_found=metric_anchor_found,
                    temporal_anchors=temporal_anchors,
                    temporal_anchors_found=temporal_anchors_found,
                    quote=reference.exact_quote,
                    normalized_quote=normalized_quote,
                    normalization_operations=operations,
                    relation=reference.relation,
                    quote_found=quote_found,
                    value_found=value_found,
                    chunk_sha256=chunk.sha256,
                    source_url=(
                        f"/api/corpora/{self.corpus.corpus_version}/documents/"
                        f"{chunk.document_id}/pdf#page={chunk.physical_page_index}"
                    ),
                )
            )
            if quote_found and value_found and metric_anchor_found and temporal_anchors_found:
                valid_reference_keys.add(_reference_key(reference))
                if metric_linked:
                    metric_linked_keys.add(_reference_key(reference))

        all_quotes_found = (
            not missing_reference and bool(evidence) and all(item.quote_found for item in evidence)
        )
        all_values_found = not missing_reference and all(item.value_found for item in evidence)
        all_metric_anchors_found = not missing_reference and all(
            item.metric_anchor_found for item in evidence
        )
        all_temporal_anchors_found = not missing_reference and all(
            item.temporal_anchors_found for item in evidence
        )
        has_supporting_evidence = any(
            reference.relation == EvidenceRelation.SUPPORTS
            and _reference_key(reference) in valid_reference_keys
            and _canonical_metric(reference.metric_anchor) == _canonical_metric(claim.metric)
            for reference in references
        )
        all_values_metric_linked = all(
            any(_reference_key(reference) in metric_linked_keys for reference in value.evidence)
            for value in claim.values
        )
        all_values_supported = all(
            any(
                reference.relation == EvidenceRelation.SUPPORTS
                and _reference_key(reference) in metric_linked_keys
                for reference in value.evidence
            )
            for value in claim.values
        )
        distinct_values = _distinct_values(claim.values)
        distinct_temporal_anchors = _distinct_temporal_anchors(claim.values)
        explicit_conflict = _has_explicit_conflict(evidence)

        if claim.status == "not_found":
            canonical_claim_metric = _canonical_metric(claim.metric)
            relevant_searches = {
                str(record.get("signature"))
                for record in completed_search_records or []
                if _canonical_metric(str(record.get("metric") or "")) == canonical_claim_metric
                and canonical_claim_metric in _canonical_metric(str(record.get("query") or ""))
            }
            if len(relevant_searches) >= 4 and not references and not claim.values:
                status = ClaimStatus.NOT_FOUND
                notes.append(
                    f"Not found after {len(relevant_searches)} metric-targeted searches "
                    f"across corpus version {self.corpus.corpus_version}."
                )
            else:
                status = ClaimStatus.UNVERIFIED
                notes.append(
                    "A not-found conclusion requires four distinct completed searches tagged "
                    "with the claimed metric and no evidence."
                )
        elif not claim.values:
            status = ClaimStatus.UNVERIFIED
            notes.append("An authoritative factual claim requires at least one reported value.")
        elif (
            not all_quotes_found
            or not all_values_found
            or not all_metric_anchors_found
            or not all_temporal_anchors_found
            or not all_values_metric_linked
            or not has_supporting_evidence
        ):
            status = ClaimStatus.UNVERIFIED
            if not all_quotes_found:
                notes.append(
                    "At least one quotation was not found verbatim after allowed normalization."
                )
            if not all_values_found:
                notes.append("At least one reported value was not present in its cited quotation.")
            if not all_metric_anchors_found:
                notes.append("At least one metric anchor was not present in its cited quotation.")
            if not all_temporal_anchors_found:
                notes.append("At least one temporal anchor was not present in its cited quotation.")
            if not all_values_metric_linked:
                notes.append(
                    "At least one evidence metric anchor did not match the canonical metric."
                )
            if not has_supporting_evidence:
                notes.append(
                    "Contextual or contradicting evidence cannot independently authorize a claim."
                )
        elif claim.status == "date_variant":
            if explicit_conflict and len(distinct_values) >= 2:
                status = ClaimStatus.CONFLICT
                notes.append(
                    "A verified quotation explicitly characterizes the values as erroneous or incompatible."
                )
            elif (
                len(distinct_values) >= 2
                and len(distinct_temporal_anchors) >= 2
                and len(distinct_temporal_anchors) == len(claim.values)
            ):
                status = ClaimStatus.DATE_VARIANT
            else:
                status = ClaimStatus.POSSIBLE_CONFLICT
                notes.append("Distinct values were not tied to at least two explicit dates.")
        elif claim.status in {"conflict", "possible_conflict"}:
            if len(distinct_values) < 2:
                status = ClaimStatus.UNVERIFIED
                notes.append("A conflict requires at least two distinct reported values.")
            elif explicit_conflict:
                status = ClaimStatus.CONFLICT
                notes.append(
                    "A verified quotation explicitly characterizes the values as erroneous or incompatible."
                )
            elif len(distinct_temporal_anchors) >= 2 and len(distinct_temporal_anchors) == len(
                claim.values
            ):
                status = ClaimStatus.DATE_VARIANT
                notes.append(
                    "All differing values have distinct dates; classified as date variants."
                )
            elif len(distinct_temporal_anchors) == 1 and all(
                value.temporal_anchor for value in claim.values
            ):
                status = ClaimStatus.CONFLICT
            else:
                status = ClaimStatus.POSSIBLE_CONFLICT
        elif not all_values_supported:
            status = ClaimStatus.UNVERIFIED
            notes.append(
                "Each value in a supported claim requires its own supporting evidence item."
            )
        elif all_quotes_found and all_values_found:
            status = ClaimStatus.VERIFIED
        else:
            status = ClaimStatus.UNVERIFIED

        return VerifiedClaim(
            statement=_authoritative_statement(claim, status),
            metric=claim.metric,
            status=status,
            values=claim.values,
            evidence=evidence,
            verification_notes=notes,
        )
