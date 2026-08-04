# Verification boundary

NeedleProof verifies source integrity and a deliberately conservative structured evidence
contract. It does not claim to prove arbitrary English entailment.

An authoritative value must satisfy all of these checks:

- the cited chunk belongs to the run's immutable corpus version;
- the exact quotation exists after only the documented normalization operations;
- the canonical metric matches the copied metric anchor;
- the reported value occurs with that metric anchor under the lexical binding policy;
- any temporal anchor occurs in the same source sentence as its value;
- supporting evidence, rather than contextual or contradicting evidence alone, authorizes the
  value;
- conflict and date-variant classifications follow from the verified value and temporal sets.

The binding policy fails closed when coordination is ambiguous. For example, a number-neutral
predicate in `Revenue from products and services reached ...` cannot prove whether `services` is a
modifier or a second subject when the claimed metric is only `Revenue`. The agent must copy the
complete metric phrase, or the claim remains unverified. Complex prose may therefore produce a
false negative; it must not produce an authoritative guess.

The policy treats conjunctions and punctuation as a closed grammatical class and exercises them
as a cross-product in tests. It does not maintain an ever-growing list of financial predicate
verbs. If the product later needs broader semantic coverage, the evidence contract should gain an
explicit source-span binding representation rather than turning this verifier into a general
English parser.

## Numeric integrity

Financial values are preserved as source strings. Equality and conflict grouping parse exact
sign, currency, digits, and units, then normalize digits with `Decimal`; money, percentages, and
basis points are never converted to binary floating point or silently rescaled.

Floating point is appropriate only outside financial claim truth:

- embedding vectors and L2 normalization;
- dense and reciprocal-rank-fusion scores;
- latency, timeout, and retry-backoff clocks;
- retrieval recall ratios;
- approximate token-count heuristics.

Arithmetic involving ranking or backoff uses explicit parentheses. No authoritative financial
claim is calculated from those values.

## Static analysis

Ruff formats and lints the Python code; it is not a type checker. CI separately runs `ty` over the
production API package on Python 3.11 and 3.13. Tests use dynamic doubles heavily and remain under
runtime, Ruff, and focused contract coverage rather than weakening the production type gate with
broad ignores.
