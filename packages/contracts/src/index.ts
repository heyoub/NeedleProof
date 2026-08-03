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

export interface Evidence {
  chunk_id: string;
  document_id: string;
  document_name: string;
  physical_page_index: number;
  printed_page_label: string | null;
  metric_anchor: string;
  metric_anchor_found: boolean;
  temporal_anchors: string[];
  temporal_anchors_found: boolean;
  quote: string;
  normalized_quote: string;
  normalization_operations: string[];
  relation: 'supports' | 'contradicts' | 'contextualizes';
  quote_found: boolean;
  value_found: boolean;
  chunk_sha256: string;
  source_url: string;
}

export interface EvidenceReference {
  chunk_id: string;
  metric_anchor: string;
  exact_quote: string;
  relation: 'supports' | 'contradicts' | 'contextualizes';
}

export interface VerifiedClaim {
  statement: string;
  metric: string;
  status: ClaimStatus;
  values: Array<{
    value: string;
    temporal_anchor: string | null;
    evidence: EvidenceReference[];
  }>;
  evidence: Evidence[];
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
