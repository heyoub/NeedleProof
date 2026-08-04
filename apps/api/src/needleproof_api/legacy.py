from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .chunk_ids import ChunkId
from .models import (
    ClaimStatus,
    DraftObservation,
    EvidenceReference,
    EvidenceRelation,
    ObservationKind,
    RunEnvelope,
    RunStatus,
    VerifiedClaim,
    VerifiedEvidence,
)

Sha256Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class LegacyEvidenceReferenceV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: ChunkId
    metric_anchor: str = Field(min_length=1)
    exact_quote: str
    relation: EvidenceRelation


class LegacyReportedValueV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    temporal_anchor: str | None
    evidence: list[LegacyEvidenceReferenceV12] = Field(min_length=1)


class LegacyVerifiedEvidenceV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: ChunkId
    document_id: str
    document_name: str
    physical_page_index: int
    printed_page_label: str | None
    metric_anchor: str
    metric_anchor_found: bool
    temporal_anchors: list[str]
    temporal_anchors_found: bool
    quote: str
    normalized_quote: str
    normalization_operations: list[str]
    relation: EvidenceRelation
    quote_found: bool
    value_found: bool
    chunk_sha256: Sha256Digest
    source_url: str


class LegacyVerifiedClaimV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    metric: str
    status: ClaimStatus
    values: list[LegacyReportedValueV12]
    evidence: list[LegacyVerifiedEvidenceV12]
    verification_notes: list[str]


class LegacyReceiptEventV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    type: str
    occurred_at: str
    payload: dict[str, Any]
    previous_hash: Sha256Digest
    event_hash: Sha256Digest


class LegacyReceiptOpenAICallV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    operation: Literal["embedding", "model"]
    model: str
    response_id: str | None
    request_id: str | None
    started_at: str
    ended_at: str
    duration_ms: float = Field(ge=0)
    token_usage: dict[str, Any]
    retry_count: int = Field(ge=0)
    error: dict[str, Any] | None


class LegacyReceiptRehearsalV12(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_receipt_sha256: Sha256Digest
    source_run_id: str
    validated: Literal[True]
    reverified: Literal[True]
    source_verifier_version: str
    current_verifier_version: str
    version_drift: bool


class LegacyRunEnvelopeV12(BaseModel):
    """Exact persisted run-envelope contract used before receipt schema 1.4."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    question: str
    answer: str | None = None
    corpus_id: str
    corpus_version: str
    corpus_manifest_sha256: str
    claims: list[LegacyVerifiedClaimV12] = Field(default_factory=list)
    receipt_url: str
    receipt_json_url: str


def _legacy_observation_kind(value: str) -> ObservationKind:
    # Keep this frozen classifier independent from the live binding contract. It adapts
    # retained schema-1.2 results without retroactively regrading historical receipts.
    normalized = " ".join(value.casefold().split())
    if "per share" in normalized:
        return ObservationKind.REPORTED_PER_SHARE
    if (
        "%" in normalized
        or "percent" in normalized
        or "basis point" in normalized
        or "bps" in normalized
    ):
        return ObservationKind.REPORTED_RATE
    return ObservationKind.REPORTED_LEVEL


def _legacy_optional_temporal_anchor(value: str | None) -> str | None:
    """Map legacy empty optional text to the current explicit-null contract."""

    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _adapt_reference(reference: LegacyEvidenceReferenceV12) -> EvidenceReference:
    return EvidenceReference(
        chunk_id=reference.chunk_id,
        metric_anchor=reference.metric_anchor,
        exact_quote=reference.exact_quote,
        exact_assertion=reference.exact_quote,
        relation=reference.relation,
    )


def _adapt_verified_evidence(evidence: LegacyVerifiedEvidenceV12) -> VerifiedEvidence:
    temporal_anchor = (
        _legacy_optional_temporal_anchor(evidence.temporal_anchors[0])
        if len(evidence.temporal_anchors) == 1
        else None
    )
    legacy_binding_supported = (
        evidence.quote_found and evidence.metric_anchor_found and evidence.value_found
    )
    return VerifiedEvidence(
        chunk_id=evidence.chunk_id,
        document_id=evidence.document_id,
        document_name=evidence.document_name,
        physical_page_index=evidence.physical_page_index,
        printed_page_label=evidence.printed_page_label,
        metric_anchor=evidence.metric_anchor,
        metric_anchor_found=evidence.metric_anchor_found,
        assertion=evidence.quote,
        assertion_found=evidence.quote_found,
        temporal_anchor=temporal_anchor,
        temporal_value_bound=bool(temporal_anchor) and evidence.temporal_anchors_found,
        observation_kind=None,
        value_text=None,
        quote=evidence.quote,
        normalized_quote=evidence.normalized_quote,
        normalization_operations=evidence.normalization_operations,
        relation=evidence.relation,
        quote_found=evidence.quote_found,
        value_text_found=evidence.value_found,
        metric_value_bound=legacy_binding_supported,
        value_role_authorized=legacy_binding_supported,
        binding_profile=None,
        binding_failure_reason=(
            None if legacy_binding_supported else "legacy_schema_1_2_evidence_not_authorized"
        ),
        chunk_sha256=evidence.chunk_sha256,
        source_url=evidence.source_url,
    )


def adapt_legacy_run_envelope(value: Any) -> RunEnvelope:
    """Validate and adapt one retained envelope without regrading its historical result."""

    legacy = LegacyRunEnvelopeV12.model_validate(value)
    claims = []
    for claim in legacy.claims:
        observations = [
            DraftObservation(
                kind=_legacy_observation_kind(reported.value),
                value_text=reported.value,
                temporal_anchor=_legacy_optional_temporal_anchor(reported.temporal_anchor),
                evidence=[_adapt_reference(reference) for reference in reported.evidence],
            )
            for reported in claim.values
        ]
        claims.append(
            VerifiedClaim(
                statement=claim.statement,
                metric=claim.metric,
                status=claim.status,
                observations=observations,
                context_evidence=[],
                evidence=[_adapt_verified_evidence(item) for item in claim.evidence],
                absence_probe=None,
                verification_notes=[
                    *claim.verification_notes,
                    "Read-only result adapted from the schema-1.2 evidence-location contract.",
                ],
            )
        )
    return RunEnvelope(
        run_id=legacy.run_id,
        status=legacy.status,
        question=legacy.question,
        answer=legacy.answer,
        corpus_id=legacy.corpus_id,
        corpus_version=legacy.corpus_version,
        corpus_manifest_sha256=legacy.corpus_manifest_sha256,
        claims=claims,
        receipt_url=legacy.receipt_url,
        receipt_json_url=legacy.receipt_json_url,
    )
