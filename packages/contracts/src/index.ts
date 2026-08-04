export type RunStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'incomplete'
  | 'cancelled'
  | 'failed'
  | 'interrupted';

export type ClaimStatus =
  | 'verified'
  | 'conflict'
  | 'date_variant'
  | 'not_found'
  | 'unverified'
  | 'possible_conflict';

export type ObservationKind =
  | 'reported_level'
  | 'reported_rate'
  | 'reported_per_share'
  | 'reported_delta'
  | 'reported_bound'
  | 'target'
  | 'forecast'
  | 'component'
  | 'range';

export type BindingProfile =
  | 'direct_copula'
  | 'direct_reported'
  | 'colon'
  | 'dated_direct'
  | 'same_sentence_anaphoric'
  | 'next_sentence_anaphoric';

export interface Evidence {
  chunk_id: string;
  document_id: string;
  document_name: string;
  physical_page_index: number;
  printed_page_label: string | null;
  metric_anchor: string;
  metric_anchor_found: boolean;
  assertion: string;
  assertion_found: boolean;
  temporal_anchor: string | null;
  temporal_value_bound: boolean;
  observation_kind: ObservationKind | null;
  value_text: string | null;
  quote: string;
  normalized_quote: string;
  normalization_operations: string[];
  relation: 'supports' | 'contradicts' | 'contextualizes';
  quote_found: boolean;
  value_text_found: boolean;
  metric_value_bound: boolean;
  value_role_authorized: boolean;
  binding_profile: BindingProfile | null;
  binding_failure_reason: string | null;
  chunk_sha256: string;
  source_url: string;
}

export interface EvidenceReference {
  chunk_id: string;
  metric_anchor: string;
  exact_quote: string;
  exact_assertion: string;
  relation: 'supports' | 'contradicts' | 'contextualizes';
}

export interface DraftObservation {
  kind: ObservationKind;
  value_text: string;
  temporal_anchor: string | null;
  evidence: EvidenceReference[];
}

export interface CompletedSearchRecord {
  query: string;
  normalized_query: string;
  metric: string | null;
  mode: 'dense' | 'lexical' | 'hybrid';
  top_k: number;
  document_ids: string[];
  date_from: string | null;
  date_to: string | null;
  signature: string;
  result_chunk_ids: string[];
  result_count: number;
  exact_metric_hit_count: number;
  completion_status: 'completed' | 'failed';
}

export interface MetricOccurrence {
  chunk_id: string;
  sentence: string;
  span: [number, number];
}

export interface ValueCandidate {
  chunk_id: string;
  metric_anchor: string;
  value_text: string;
  binding_profile: BindingProfile | null;
  binding_failure_reason: string | null;
}

export interface AbsenceProbeResult {
  protocol_version: string;
  metric: string;
  searches: CompletedSearchRecord[];
  unique_candidate_chunk_ids: string[];
  opened_chunk_ids: string[];
  exact_metric_occurrences: MetricOccurrence[];
  unresolved_predicate_occurrences: MetricOccurrence[];
  supporting_value_candidates: ValueCandidate[];
  conclusion: 'not_found_in_probe' | 'evidence_requires_review' | 'incomplete_probe';
}

export interface VerifiedClaim {
  statement: string;
  metric: string;
  status: ClaimStatus;
  observations: DraftObservation[];
  context_evidence: EvidenceReference[];
  evidence: Evidence[];
  absence_probe: AbsenceProbeResult | null;
  verification_notes: string[];
}

export interface RunEnvelope {
  run_id: string;
  status: RunStatus;
  question: string;
  answer: string | null;
  corpus_id: string;
  corpus_version: string;
  corpus_manifest_sha256: string;
  claims: VerifiedClaim[];
  receipt_url: string;
  receipt_json_url: string;
}

export interface LedgerEvent {
  sequence: number;
  type: string;
  occurred_at: string;
  payload: Record<string, unknown>;
  previous_hash: string;
  event_hash: string;
}
