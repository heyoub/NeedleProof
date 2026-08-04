from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from .chunk_ids import ChunkId


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ClaimStatus(StrEnum):
    VERIFIED = "verified"
    CONFLICT = "conflict"
    DATE_VARIANT = "date_variant"
    NOT_FOUND = "not_found"
    UNVERIFIED = "unverified"
    POSSIBLE_CONFLICT = "possible_conflict"


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"


class ObservationKind(StrEnum):
    REPORTED_LEVEL = "reported_level"
    REPORTED_RATE = "reported_rate"
    REPORTED_PER_SHARE = "reported_per_share"
    REPORTED_DELTA = "reported_delta"
    REPORTED_BOUND = "reported_bound"
    TARGET = "target"
    FORECAST = "forecast"
    COMPONENT = "component"
    RANGE = "range"


class BindingProfile(StrEnum):
    DIRECT_COPULA = "direct_copula"
    DIRECT_REPORTED = "direct_reported"
    COLON = "colon"
    DATED_DIRECT = "dated_direct"
    SAME_SENTENCE_ANAPHORIC = "same_sentence_anaphoric"
    NEXT_SENTENCE_ANAPHORIC = "next_sentence_anaphoric"


class AbsenceConclusion(StrEnum):
    NOT_FOUND_IN_PROBE = "not_found_in_probe"
    EVIDENCE_REQUIRES_REVIEW = "evidence_requires_review"
    INCOMPLETE_PROBE = "incomplete_probe"


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DocumentRecord(BaseModel):
    document_id: str
    display_name: str
    source_path: str
    sha256: str
    publication_date: str | None = None
    reporting_period_date: str | None = None
    file_creation_date: str | None = None
    included_pages: list[int] = Field(default_factory=list)
    page_count: int
    chunk_count: int


class ChunkRecord(BaseModel):
    internal_id: int | None = None
    chunk_id: ChunkId
    document_id: str
    document_name: str
    physical_page_index: int
    printed_page_label: str | None = None
    chunk_position: int
    text: str
    normalized_text: str
    previous_chunk_id: ChunkId | None = None
    next_chunk_id: ChunkId | None = None
    sha256: str
    token_estimate: int


class SearchHit(BaseModel):
    chunk_id: ChunkId
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    retrieval_mode: Literal["dense", "lexical", "hybrid"]
    document_id: str
    document_name: str
    physical_page_index: int
    printed_page_label: str | None = None
    preview: str
    sha256: str


class SearchResult(BaseModel):
    query: str
    mode: Literal["dense", "lexical", "hybrid"]
    results: list[SearchHit]
    corpus_manifest_sha256: str


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: ChunkId
    metric_anchor: NonEmptyText
    exact_quote: NonEmptyText
    exact_assertion: NonEmptyText
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS


class DraftObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ObservationKind
    value_text: NonEmptyText
    temporal_anchor: NonEmptyText | None = None
    evidence: list[EvidenceReference] = Field(min_length=1)


class DraftClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: NonEmptyText
    observations: list[DraftObservation] = Field(default_factory=list)
    context_evidence: list[EvidenceReference] = Field(default_factory=list)
    request_absence_probe: bool = False


class AgentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[DraftClaim]
    unresolved_questions: list[str] = Field(default_factory=list)


class CompletedSearchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: NonEmptyText
    normalized_query: NonEmptyText
    metric: NonEmptyText | None = None
    mode: Literal["dense", "lexical", "hybrid"]
    top_k: int = Field(ge=1, le=20)
    document_ids: list[str] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    signature: NonEmptyText
    result_chunk_ids: list[ChunkId] = Field(default_factory=list)
    result_count: int = Field(ge=0)
    exact_metric_hit_count: int = Field(ge=0)
    completion_status: Literal["completed", "failed"]


class MetricOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: ChunkId
    sentence: str
    span: tuple[int, int]


class ValueCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: ChunkId
    metric_anchor: str
    value_text: str
    binding_profile: BindingProfile | None = None
    binding_failure_reason: str | None = None


class AbsenceProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str
    metric: str
    searches: list[CompletedSearchRecord]
    unique_candidate_chunk_ids: list[ChunkId]
    opened_chunk_ids: list[ChunkId]
    exact_metric_occurrences: list[MetricOccurrence]
    unresolved_predicate_occurrences: list[MetricOccurrence]
    supporting_value_candidates: list[ValueCandidate]
    conclusion: AbsenceConclusion


class VerifiedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: ChunkId
    document_id: str
    document_name: str
    physical_page_index: int
    printed_page_label: str | None = None
    metric_anchor: str
    metric_anchor_found: bool
    assertion: str
    assertion_found: bool
    temporal_anchor: str | None = None
    temporal_value_bound: bool
    observation_kind: ObservationKind | None = None
    value_text: str | None = None
    quote: str
    normalized_quote: str
    normalization_operations: list[str] = Field(default_factory=list)
    relation: EvidenceRelation
    quote_found: bool
    value_text_found: bool
    metric_value_bound: bool
    value_role_authorized: bool
    binding_profile: BindingProfile | None = None
    binding_failure_reason: str | None = None
    chunk_sha256: str
    source_url: str


class VerifiedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    metric: str
    status: ClaimStatus
    observations: list[DraftObservation] = Field(default_factory=list)
    context_evidence: list[EvidenceReference] = Field(default_factory=list)
    evidence: list[VerifiedEvidence] = Field(default_factory=list)
    absence_probe: AbsenceProbeResult | None = None
    verification_notes: list[str] = Field(default_factory=list)


class EvidenceVerificationResult(BaseModel):
    claims: list[VerifiedClaim]
    all_claims_authoritative: bool


class OpenAICallRecord(BaseModel):
    sequence: int
    operation: Literal["embedding", "model"]
    model: str
    response_id: str | None = None
    request_id: str | None = None
    started_at: str
    ended_at: str
    duration_ms: float
    token_usage: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    error: dict[str, Any] | None = None


class LedgerEvent(BaseModel):
    sequence: int
    type: str
    occurred_at: str
    payload: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str
    event_hash: str


class RunEnvelope(BaseModel):
    run_id: str
    status: RunStatus
    question: str
    answer: str | None = None
    corpus_id: str
    corpus_version: str
    corpus_manifest_sha256: str
    claims: list[VerifiedClaim] = Field(default_factory=list)
    receipt_url: str
    receipt_json_url: str


class RunCreateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    rehearsal: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3:
            raise ValueError("question must contain at least three non-whitespace characters")
        return normalized


class RunCreateResponse(BaseModel):
    run_id: str
    status: RunStatus
    events_url: str
    run_url: str


class CorpusSummary(BaseModel):
    corpus_id: str
    display_name: str
    corpus_version: str
    manifest_sha256: str
    document_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimensions: int
    turbovec_version: str
