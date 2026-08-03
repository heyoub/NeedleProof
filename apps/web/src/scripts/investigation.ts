import { SSE, type SSEMessage } from '@czap/web';
import { Millis } from '@czap/core';
import { Effect, Fiber } from 'effect';
import type {
  Evidence,
  LedgerEvent,
  RunEnvelope,
  RunStatus,
} from '@needleproof/contracts';

const terminalEvents = new Set([
  'run.completed',
  'run.incomplete',
  'run.cancelled',
  'run.failed',
  'run.timeout',
  'run.interrupted',
]);
const terminalStatuses = new Set<RunStatus>([
  'completed',
  'incomplete',
  'cancelled',
  'failed',
  'interrupted',
]);
const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const questionForm = el<HTMLFormElement>('question-form');
const question = el<HTMLTextAreaElement>('question');
const askButton = el<HTMLButtonElement>('ask-button');
const investigation = el<HTMLElement>('investigation');
const activityLedger = el<HTMLOListElement>('activity-ledger');
const answerWaiting = el<HTMLElement>('answer-waiting');
const answerContent = el<HTMLElement>('answer-content');
const answerStatus = el<HTMLElement>('answer-status');
const answerCopy = el<HTMLElement>('answer-copy');
const claims = el<HTMLElement>('claims');
const receiptLink = el<HTMLAnchorElement>('receipt-link');
const receiptJsonLink = el<HTMLAnchorElement>('receipt-json-link');
const cancelButton = el<HTMLButtonElement>('cancel-button');
const connectionState = el<HTMLElement>('connection-state');
const announcer = el<HTMLElement>('announcer');
const rehearsalButton = el<HTMLButtonElement>('rehearsal-button');
const evidenceDialog = el<HTMLDialogElement>('evidence-dialog');
const quoteChallenge = el<HTMLButtonElement>('quote-challenge');
const quoteChallengeResult = el<HTMLElement>('quote-challenge-result');

let activeRunId: string | null = null;
let activeCorpusVersion: string | null = null;
let streamFiber: ReturnType<typeof Effect.runFork> | null = null;
let previousFocus: HTMLElement | null = null;
let challengeEvidence: Evidence | null = null;
let terminalFetchInFlight = false;
let fallbackPolling = false;

function describeEvent(event: LedgerEvent): string {
  const payload = event.payload;
  const resultCount = Array.isArray(payload.results) ? payload.results.length : 0;
  const labels: Record<string, string> = {
    'run.started': 'Investigation opened against an immutable corpus version',
    'agent.model.started': 'Investigator is choosing the next evidence action',
    'agent.model.completed': 'Agent turn completed',
    'tool.search.started': `Searching “${String(payload.query ?? '')}”`,
    'tool.search.completed': `Search returned ${resultCount} ranked passages`,
    'tool.read.started': 'Opening exact passages and neighboring context',
    'tool.read.completed': `Read ${String(payload.unique_opened_chunks ?? 0)} unique source passages`,
    'tool.inspect.started': 'Widening inspection across the source document',
    'tool.inspect.completed': 'Document inspection completed',
    'verification.started': 'Checking exact quotations and reported values',
    'verification.completed': 'Agent verification pass completed',
    'verification.authoritative_started': 'Running authoritative server-side verification',
    'claim.verified': 'Claim accepted with verified evidence',
    'claim.conflict': 'Conflicting values preserved and verified',
    'claim.date_variant': 'Differing values tied to distinct dates',
    'claim.not_found': 'Bounded missing-evidence conclusion accepted',
    'claim.unverified': 'Claim rejected by deterministic verification',
    'run.completed': 'Receipt sealed — authoritative answer ready',
    'run.incomplete': 'Investigation sealed with incomplete evidence',
    'run.timeout': 'Time limit reached — partial receipt sealed',
    'run.cancelled': 'Investigation cancelled — partial receipt sealed',
    'run.failed': 'Investigation failed — diagnostic receipt sealed',
    'run.interrupted': 'Investigation interrupted — recovery receipt sealed',
  };
  return labels[event.type] ?? event.type.replaceAll('.', ' ');
}

function addActivity(event: LedgerEvent): void {
  if (activityLedger.querySelector(`[data-sequence="${event.sequence}"]`)) return;
  const item = document.createElement('li');
  item.dataset.sequence = String(event.sequence);
  item.dataset.state = event.type.includes('failed') || event.type.includes('unverified') ? 'error' : 'done';
  item.textContent = describeEvent(event);
  const meta = document.createElement('code');
  meta.textContent = `#${String(event.sequence).padStart(2, '0')} · ${event.event_hash.slice(0, 12)}…`;
  item.append(meta);
  activityLedger.append(item);
}

function statusText(status: string): string {
  return status.replaceAll('_', ' ');
}

function evidenceButton(evidence: Evidence, index: number): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'evidence-button';
  button.textContent = `Evidence ${index + 1} · p. ${evidence.printed_page_label ?? evidence.physical_page_index}`;
  button.addEventListener('click', () => openEvidence(evidence, button));
  return button;
}

function renderRun(run: RunEnvelope): void {
  activeCorpusVersion = run.corpus_version;
  answerWaiting.hidden = true;
  answerContent.hidden = false;
  answerStatus.dataset.status = run.status;
  answerStatus.textContent =
    run.status === 'completed'
      ? 'Authoritative · verified'
      : run.status === 'incomplete' && run.answer
        ? 'Verified subset · investigation incomplete'
        : `${statusText(run.status)} · non-authoritative`;
  answerCopy.textContent = run.answer ?? 'No authoritative answer was released. Inspect the partial receipt for the evidence gathered and terminal status.';
  claims.replaceChildren();
  challengeEvidence = run.claims.flatMap((claim) => claim.evidence)[0] ?? null;
  quoteChallenge.hidden = challengeEvidence === null;
  quoteChallengeResult.textContent = '';

  run.claims.forEach((claim) => {
    const card = document.createElement('section');
    card.className = 'claim';
    const head = document.createElement('div');
    head.className = 'claim-head';
    const title = document.createElement('h3');
    title.textContent = claim.statement;
    const status = document.createElement('span');
    status.className = 'status-label';
    status.dataset.status = claim.status;
    status.textContent = statusText(claim.status);
    head.append(title, status);
    card.append(head);
    if (claim.evidence.length) {
      const list = document.createElement('div');
      list.className = 'evidence-list';
      claim.evidence.forEach((evidence, index) => list.append(evidenceButton(evidence, index)));
      card.append(list);
    }
    claims.append(card);
  });

  receiptLink.href = run.receipt_url;
  receiptJsonLink.href = run.receipt_json_url;
  cancelButton.hidden = true;
  announcer.textContent = `Investigation ${run.status}. ${run.answer ?? 'No authoritative answer released.'}`;
}

async function loadRun(runId: string): Promise<RunEnvelope> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) throw new Error(`Run fetch failed (${response.status})`);
  return (await response.json()) as RunEnvelope;
}

async function fetchRun(runId: string): Promise<RunEnvelope> {
  const run = await loadRun(runId);
  renderRun(run);
  return run;
}

async function fetchUntilTerminal(runId: string): Promise<void> {
  if (terminalFetchInFlight) return;
  terminalFetchInFlight = true;
  try {
    for (let attempt = 0; attempt < 8; attempt += 1) {
      const run = await loadRun(runId);
      if (terminalStatuses.has(run.status)) {
        renderRun(run);
        stopStream();
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, Math.min(100 * 2 ** attempt, 1_500)));
    }
    throw new Error('The terminal event arrived before the completed run could be loaded. Reconnect or refresh to resume.');
  } finally {
    terminalFetchInFlight = false;
  }
}

async function pollUntilTerminal(runId: string): Promise<void> {
  if (fallbackPolling) return;
  fallbackPolling = true;
  connectionState.textContent = 'Polling for terminal state';
  try {
    for (let attempt = 0; attempt < 45; attempt += 1) {
      const run = await loadRun(runId);
      if (terminalStatuses.has(run.status)) {
        renderRun(run);
        stopStream();
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1_500));
    }
    throw new Error('Unable to recover the terminal investigation state. Refresh to retry.');
  } finally {
    fallbackPolling = false;
  }
}

function stopStream(): void {
  if (streamFiber) {
    Effect.runFork(Fiber.interrupt(streamFiber));
    streamFiber = null;
  }
}

function connectStream(runId: string): void {
  stopStream();
  const program = Effect.scoped(
    Effect.gen(function* () {
      yield* SSE.create({
        url: `/api/runs/${encodeURIComponent(runId)}/events`,
        heartbeatInterval: Millis(30_000),
        reconnect: {
          maxAttempts: 8,
          initialDelay: Millis(500),
          maxDelay: Millis(5_000),
          factor: 1.7,
        },
        onStateChange: (state) => {
          connectionState.textContent = state;
          if (state === 'error') {
            void pollUntilTerminal(runId).catch(showFailure);
          }
        },
        onMessage: (message: SSEMessage) => {
          const event = message as unknown as LedgerEvent;
          addActivity(event);
          if (terminalEvents.has(event.type)) {
            void fetchUntilTerminal(runId).catch(showFailure);
          }
        },
      });
      yield* Effect.never;
    }),
  );
  streamFiber = Effect.runFork(program);
}

function resetRunUi(): void {
  investigation.hidden = false;
  answerWaiting.hidden = false;
  answerContent.hidden = true;
  activityLedger.replaceChildren();
  claims.replaceChildren();
  challengeEvidence = null;
  activeCorpusVersion = null;
  quoteChallenge.hidden = true;
  quoteChallengeResult.textContent = '';
  terminalFetchInFlight = false;
  fallbackPolling = false;
  cancelButton.hidden = false;
  askButton.disabled = true;
  connectionState.textContent = 'Connecting';
  investigation.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function beginRun(rehearsal = false): Promise<void> {
  const prompt = rehearsal
    ? 'Summarize the company’s scale and fee economics and identify anything questionable.'
    : question.value.trim();
  if (prompt.length < 3) {
    question.focus();
    return;
  }
  resetRunUi();
  try {
    const response = await fetch('/api/runs', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: prompt, rehearsal }),
    });
    if (!response.ok) throw new Error(`Unable to start investigation (${response.status})`);
    const created = (await response.json()) as { run_id: string };
    activeRunId = created.run_id;
    const url = new URL(window.location.href);
    url.searchParams.set('run', created.run_id);
    window.history.replaceState(null, '', url);
    connectStream(created.run_id);
  } catch (error) {
    showFailure(error);
  } finally {
    askButton.disabled = false;
  }
}

function showFailure(error: unknown): void {
  answerWaiting.hidden = true;
  answerContent.hidden = false;
  answerStatus.dataset.status = 'failed';
  answerStatus.textContent = 'Connection error · non-authoritative';
  answerCopy.textContent = error instanceof Error ? error.message : 'The investigation could not be loaded.';
  cancelButton.hidden = true;
  announcer.textContent = answerCopy.textContent;
}

function openEvidence(evidence: Evidence, trigger: HTMLElement): void {
  previousFocus = trigger;
  el<HTMLElement>('evidence-relation').textContent = evidence.relation;
  el<HTMLElement>('evidence-relation').dataset.status = evidence.relation;
  el<HTMLElement>('evidence-location').textContent = `${evidence.document_name} · page ${evidence.printed_page_label ?? evidence.physical_page_index}`;
  el<HTMLElement>('evidence-chunk').textContent = `chunk ${evidence.chunk_id}`;
  el<HTMLElement>('evidence-quote').textContent = evidence.quote;
  el<HTMLElement>('evidence-quote-match').textContent = evidence.quote_found ? 'Verified exact match' : 'Rejected';
  el<HTMLElement>('evidence-value-match').textContent = evidence.value_found ? 'Reported value present' : 'Rejected';
  el<HTMLElement>('evidence-sha').textContent = evidence.chunk_sha256;
  el<HTMLIFrameElement>('evidence-pdf').src = evidence.source_url;
  evidenceDialog.showModal();
  el<HTMLButtonElement>('close-evidence').focus();
}

function closeEvidence(): void {
  evidenceDialog.close();
  el<HTMLIFrameElement>('evidence-pdf').src = 'about:blank';
  previousFocus?.focus();
}

questionForm.addEventListener('submit', (event) => {
  event.preventDefault();
  void beginRun(false);
});

document.querySelectorAll<HTMLButtonElement>('[data-prompt]').forEach((button) => {
  button.addEventListener('click', () => {
    question.value = button.dataset.prompt ?? '';
    question.focus();
  });
});

cancelButton.addEventListener('click', async () => {
  if (!activeRunId) return;
  cancelButton.disabled = true;
  try {
    await fetch(`/api/runs/${encodeURIComponent(activeRunId)}/cancel`, { method: 'POST' });
  } finally {
    cancelButton.disabled = false;
  }
});

rehearsalButton.addEventListener('click', () => void beginRun(true));
quoteChallenge.addEventListener('click', async () => {
  if (!challengeEvidence) return;
  quoteChallenge.disabled = true;
  quoteChallengeResult.textContent = 'Testing a deliberately altered quotation…';
  try {
    const response = await fetch('/api/verify/quote', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        run_id: activeRunId,
        corpus_version: activeCorpusVersion,
        chunk_id: challengeEvidence.chunk_id,
        quote: `${challengeEvidence.quote} [altered]`,
      }),
    });
    if (!response.ok) {
      throw new Error(`Quote verification failed with HTTP ${response.status}`);
    }
    const result = (await response.json()) as { status: string; quote_found: boolean };
    quoteChallengeResult.textContent = result.quote_found
      ? 'Unexpected match — inspect the verifier response.'
      : 'Rejected as expected: the altered quote does not exist in the cited chunk.';
  } catch (error) {
    quoteChallengeResult.textContent =
      error instanceof Error
        ? error.message
        : 'The quote challenge could not reach the verifier.';
  } finally {
    quoteChallenge.disabled = false;
  }
});
el<HTMLButtonElement>('close-evidence').addEventListener('click', closeEvidence);
evidenceDialog.addEventListener('click', (event) => {
  if (event.target === evidenceDialog) closeEvidence();
});
evidenceDialog.addEventListener('cancel', (event) => {
  event.preventDefault();
  closeEvidence();
});

const restoredRunId = new URL(window.location.href).searchParams.get('run');
if (/^run_[0-9a-f]{32}$/.test(restoredRunId ?? '')) {
  activeRunId = restoredRunId;
  resetRunUi();
  void fetchRun(restoredRunId!).then((run) => {
    if (!terminalStatuses.has(run.status)) connectStream(restoredRunId!);
  }).catch(showFailure).finally(() => {
    askButton.disabled = false;
  });
}

void fetch('/api/corpus')
  .then((response) => response.json())
  .then((corpus: { display_name: string; manifest_sha256: string }) => {
    el<HTMLElement>('corpus-label').textContent = `${corpus.display_name} · ${corpus.manifest_sha256.slice(0, 10)}…`;
  })
  .catch(() => {
    el<HTMLElement>('corpus-label').textContent = 'Corpus status unavailable';
  });
