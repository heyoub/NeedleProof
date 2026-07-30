export type RunStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'incomplete'
  | 'cancelled'
  | 'failed';

export type ClaimStatus =
  | 'verified'
  | 'conflict'
  | 'date_variant'
  | 'not_found'
  | 'unverified'
  | 'possible_conflict';

export interface Evidence {
  chunk_id: number;
  document_id: string;
  document_name: string;
  physical_page_index: number;
  printed_page_label: string | null;
  quote: string;
  normalized_quote: string;
  relation: 'supports' | 'contradicts' | 'contextualizes';
  quote_found: boolean;
  value_found: boolean;
  source_url: string;
}

export interface VerifiedClaim {
  statement: string;
  metric: string;
  status: ClaimStatus;
  values: Array<{ value: string; as_of_date: string | null; reporting_period: string | null }>;
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
  event_hash: string;
}
