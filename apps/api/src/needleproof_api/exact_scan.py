from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .models import ChunkRecord

EXACT_METRIC_SCAN_MAX_CANDIDATES = 1_000
EXACT_METRIC_SCAN_MAX_CHARACTERS = 5_000_000


@dataclass(frozen=True, slots=True)
class ExactMetricScanComplete:
    chunks: tuple[ChunkRecord, ...]
    candidate_count: int
    character_count: int


@dataclass(frozen=True, slots=True)
class ExactMetricScanAmbiguous:
    candidate_count: int
    character_count: int
    ambiguous_candidate_count: int


@dataclass(frozen=True, slots=True)
class ExactMetricScanTooBroad:
    candidate_count: int
    character_count: int


@dataclass(frozen=True, slots=True)
class ExactMetricScanFailed:
    error_type: str


ExactMetricScanOutcome: TypeAlias = (
    ExactMetricScanComplete
    | ExactMetricScanAmbiguous
    | ExactMetricScanTooBroad
    | ExactMetricScanFailed
)
