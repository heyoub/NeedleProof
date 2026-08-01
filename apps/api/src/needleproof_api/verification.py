from __future__ import annotations

import re
from collections.abc import Iterable

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
from .util import normalize_evidence_text

_NUMERIC = re.compile(
    r"(?P<currency>[$£€])?\s*(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>billion|million|thousand|basis\s+points?|bps|percent|%|per\s+share)?",
    re.IGNORECASE,
)


def numeric_signatures(text: str) -> set[tuple[str, str, str]]:
    signatures: set[tuple[str, str, str]] = set()
    for match in _NUMERIC.finditer(text):
        currency = match.group("currency") or ""
        number = match.group("number").replace(",", "")
        unit = re.sub(r"\s+", " ", (match.group("unit") or "").lower())
        signatures.add((currency, number, unit))
    return signatures


def reported_value_found(value: str, quote: str) -> bool:
    normalized_value, _ = normalize_evidence_text(value)
    normalized_quote, _ = normalize_evidence_text(quote)
    if normalized_value.casefold() in normalized_quote.casefold():
        return True
    expected = numeric_signatures(normalized_value)
    observed = numeric_signatures(normalized_quote)
    return bool(expected) and expected.issubset(observed)


def _all_evidence(claim: DraftClaim) -> list[EvidenceReference]:
    evidence: list[EvidenceReference] = list(claim.evidence)
    for value in claim.values:
        evidence.extend(value.evidence)
    output: list[EvidenceReference] = []
    seen: set[tuple[int, str, EvidenceRelation]] = set()
    for reference in evidence:
        key = (reference.chunk_id, reference.exact_quote, reference.relation)
        if key not in seen:
            seen.add(key)
            output.append(reference)
    return output


def _distinct_values(values: Iterable[ReportedValue]) -> set[str]:
    return {normalize_evidence_text(value.value)[0].casefold() for value in values}


def _distinct_dates(values: Iterable[ReportedValue]) -> set[str]:
    return {
        normalize_evidence_text(value.as_of_date)[0].casefold()
        for value in values
        if value.as_of_date
    }


def _has_explicit_conflict(evidence: list[VerifiedEvidence]) -> bool:
    conflict_phrases = (
        "cannot be right",
        "incompatible",
        "introduced the error",
        "unresolved conflict",
    )
    return any(
        item.relation == EvidenceRelation.CONTRADICTS
        and any(phrase in item.normalized_quote.casefold() for phrase in conflict_phrases)
        for item in evidence
    )


class EvidenceVerifier:
    version = "deterministic-verifier-v1"

    def __init__(self, corpus: CorpusStore):
        self.corpus = corpus

    def verify_claims(
        self, claims: list[DraftClaim], *, completed_searches: int
    ) -> EvidenceVerificationResult:
        verified = [
            self.verify_claim(claim, completed_searches=completed_searches) for claim in claims
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

    def verify_claim(self, claim: DraftClaim, *, completed_searches: int) -> VerifiedClaim:
        notes: list[str] = []
        references = _all_evidence(claim)
        chunks = self.corpus.get_chunks([reference.chunk_id for reference in references])
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        evidence: list[VerifiedEvidence] = []
        missing_reference = False

        for reference in references:
            chunk = chunks_by_id.get(reference.chunk_id)
            normalized_quote, operations = normalize_evidence_text(reference.exact_quote)
            quote_found = bool(
                chunk and normalized_quote.casefold() in chunk.normalized_text.casefold()
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
                    quote=reference.exact_quote,
                    normalized_quote=normalized_quote,
                    normalization_operations=operations,
                    relation=reference.relation,
                    quote_found=quote_found,
                    value_found=value_found,
                    chunk_sha256=chunk.sha256,
                    source_url=f"/api/documents/{chunk.document_id}/pdf#page={chunk.physical_page_index}",
                )
            )

        all_quotes_found = (
            not missing_reference
            and bool(evidence)
            and all(item.quote_found for item in evidence)
        )
        all_values_found = not missing_reference and all(
            item.value_found for item in evidence
        )
        distinct_values = _distinct_values(claim.values)
        distinct_dates = _distinct_dates(claim.values)
        explicit_conflict = _has_explicit_conflict(evidence)

        if claim.status == "not_found":
            if completed_searches >= 4 and not references and not claim.values:
                status = ClaimStatus.NOT_FOUND
                notes.append(
                    f"Not found after {completed_searches} searches across corpus version {self.corpus.corpus_version}."
                )
            else:
                status = ClaimStatus.UNVERIFIED
                notes.append(
                    "A not-found conclusion requires four completed searches and no evidence."
                )
        elif not all_quotes_found or not all_values_found:
            status = ClaimStatus.UNVERIFIED
            if not all_quotes_found:
                notes.append(
                    "At least one quotation was not found verbatim after allowed normalization."
                )
            if not all_values_found:
                notes.append("At least one reported value was not present in its cited quotation.")
        elif claim.status == "date_variant":
            if explicit_conflict and len(distinct_values) >= 2:
                status = ClaimStatus.CONFLICT
                notes.append(
                    "A verified quotation explicitly characterizes the values as erroneous or incompatible."
                )
            elif len(distinct_values) >= 2 and len(distinct_dates) >= 2:
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
            elif len(distinct_dates) >= 2 and len(distinct_dates) == len(claim.values):
                status = ClaimStatus.DATE_VARIANT
                notes.append(
                    "All differing values have distinct dates; classified as date variants."
                )
            elif claim.status == "conflict":
                status = ClaimStatus.CONFLICT
            else:
                status = ClaimStatus.POSSIBLE_CONFLICT
        elif all_quotes_found and all_values_found:
            status = ClaimStatus.VERIFIED
        else:
            status = ClaimStatus.UNVERIFIED

        return VerifiedClaim(
            statement=claim.statement,
            metric=claim.metric,
            status=status,
            values=claim.values,
            evidence=evidence,
            verification_notes=notes,
        )
