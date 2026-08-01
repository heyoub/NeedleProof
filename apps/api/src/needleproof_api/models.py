from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    metric_anchor: str = Field(min_length=1)
    exact_quote: str
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS


class ReportedValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    temporal_anchor: str | None = None
    evidence: list[EvidenceReference] = Field(min_length=1)


DraftStatus = Literal[
    "supported",
    "possible_conflict",
    "conflict",
    "date_variant",
    "not_found",
    "insufficient_evidence",
]


class DraftClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    status: DraftStatus
    values: list[ReportedValue] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class AgentDraft(BaseModel):
    answer: str
    claims: list[DraftClaim]
    unresolved_questions: list[str] = Field(default_factory=list)


class VerifiedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: ChunkId
    document_id: str
    document_name: str
    physical_page_index: int
    printed_page_label: str | None = None
    metric_anchor: str
    metric_anchor_found: bool
    temporal_anchors: list[str] = Field(default_factory=list)
    temporal_anchors_found: bool
    quote: str
    normalized_quote: str
    normalization_operations: list[str] = Field(default_factory=list)
    relation: EvidenceRelation
    quote_found: bool
    value_found: bool
    chunk_sha256: str
    source_url: str


class VerifiedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    metric: str
    status: ClaimStatus
    values: list[ReportedValue] = Field(default_factory=list)
    evidence: list[VerifiedEvidence] = Field(default_factory=list)
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
