# Verification boundary

NeedleProof verifies source integrity and a deliberately small structured evidence language. It
does not claim to prove arbitrary English entailment.

The model emits typed observations, never authoritative prose or its own verification status. Each
observation contains a canonical metric, value text, value role, optional temporal anchor, and one
or more evidence references. A reference contains the enclosing exact quotation and the smallest
contiguous exact assertion that expresses the proposed relationship.

An authoritative observation must prove all of these facts:

- the cited chunk belongs to the run's immutable corpus version;
- the quote and assertion exist under the documented normalization contract;
- the complete metric anchor occurs in that assertion and matches the canonical metric;
- the exact value occurs in the assertion;
- one closed positive binding profile connects that metric span to that value span;
- the exact assertion ends at the value or its bound trailing temporal anchor, apart from terminal
  punctuation, so a later target, forecast, bound, component, delta, or negation cannot change the
  value role after the matched prefix;
- any temporal anchor is valid and belongs to that same binding match;
- the value role is currently authorized; and
- at least one independently sufficient evidence item has the `supports` relation.

The authorized profiles are deliberately narrow: direct copula levels, direct reported levels,
colon-delimited values, dated direct values, and bounded same/next-sentence anaphora. Unknown
syntax fails closed. Negation, modality, forecasts, targets, components, bounds, ranges, and deltas
cannot become ordinary reported levels merely because the same number appears after the metric.

This inversion is the core safety property:

```text
old: no recognized separator bug -> authorize
new: one complete positive profile proved -> authorize
```

Complex but true prose can therefore produce a false negative. Extending coverage means adding a
named profile, a source-span contract, adversarial counterexamples, and a new contract hash—not an
open-ended conjunction, noun, or financial-verb blacklist.

## Server-owned classification

The server derives `verified`, `conflict`, `date_variant`, `possible_conflict`, and `unverified`
from the verified observation set. The model does not propose those statuses. Distinct supported
values with distinct valid temporal signatures become date variants; values tied to one common
period or explicitly characterized by the source as incompatible become conflicts. Unresolved
relationships never enter authoritative prose.

## Bounded absence

`not_found` is not inferred from model-controlled search counts. A post-run server protocol issues
four corpus-wide probes, including an exact lexical metric probe, records result IDs and counts,
opens every unique candidate, and finds complete-word metric occurrences and nearby numeric
candidates. Failed probes, scoped searches, duplicate wording, uninspected candidates, or plausible
metric-adjacent values cannot authorize absence. Any exact metric occurrence also requires review,
even when the predicate is qualitative and yields no numeric candidate. The conclusion remains bounded to the recorded
probe and exact corpus version; it never claims that the corpus proves nonexistence.

## Numeric integrity

Financial values remain source strings. Equality and conflict grouping parse sign, currency,
fixed-point digits, and units. Digit canonicalization is string-based: it removes only insignificant
leading integer zeros and trailing fractional zeros. It does not use `float`, `Decimal.normalize()`,
ambient decimal context, unit conversion, or arithmetic. Adjacent integers remain distinct even at
200 digits; basis points and percentages are never silently rescaled.

Floating point is used only outside authoritative financial claim truth:

- embeddings and L2 normalization;
- dense scores and reciprocal-rank fusion;
- latency and timeout clocks;
- retrieval quality ratios; and
- approximate token-count heuristics.

## Normalization and provenance

Quote containment uses Unicode NFKC, PDF line-break dehyphenation, whitespace folding, and Unicode
case-folded comparison. The receipt records the operations actually applied. Receipt schema 1.4
records separate quote, assertion, metric, value, temporal, role, and binding-profile outcomes,
plus hashes for the binding and bounded-absence contracts. Checked-in rehearsal receipts that are
re-verified after a contract change record `receipt_derivation=contract_migration` together with
the original sealed receipt digest and verifier version; they are not represented as fresh model runs.

## Static analysis

Ruff formats and lints Python; it is not a type checker. CI runs `ty` explicitly against Python
3.11 and 3.13 language semantics, executes the suite on both interpreters, checks Astro and the
browser contracts, and runs an offline Python 3.12 container smoke. Financial truth still comes
from explicit runtime contracts and mutation/property tests rather than type hints alone.
