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
- the complete metric anchor occurs in that assertion and matches the canonical metric; its words
  use only named intra-phrase formatting and cannot cross comma, clause, or sentence punctuation,
  while complete multi-initial abbreviations such as `U.S.` remain valid;
- the assertion begins at that complete metric subject, an optional article, or a bound leading
  temporal anchor. “Complete metric subject” includes every preceding forecast, target,
  adjustment, scope qualifier, or temporal qualifier required to preserve the source's meaning,
  so those terms cannot disappear from the authoritative metric;
- the exact value occurs in the assertion;
- one closed positive binding profile connects that metric span to that value span;
- an explicitly negated qualitative predicate such as `not available` or `never stable` cannot
  authorize an ordinary positive qualitative observation;
- nonnumeric value identity uses the same normalized, case-folded word-token sequence as the
  binder, so permitted punctuation and formatting cannot manufacture distinct values;
- the exact assertion ends at the value or its bound trailing temporal anchor, apart from terminal
  punctuation, so a later target, forecast, bound, component, delta, or negation cannot change the
  value role after the matched prefix;
- any temporal anchor matches a closed date, named period, or named temporal-event profile and
  belongs to that same binding match; role words such as `forecast` and `target` are never accepted
  as dates merely because the model placed them in the temporal field;
- the value role is currently authorized; and
- at least one independently sufficient evidence item has the `supports` relation.

The authorized profiles are deliberately narrow: direct copula levels, direct reported levels,
colon-delimited values, dated direct values, and bounded same/next-sentence anaphora. Anaphora must
follow one closed positive antecedent shape; next-sentence anaphora must occur in the immediately
following declarative sentence separated by a period, and any leading temporal phrase must contain
the observation's bound temporal anchor. A numeric antecedent must have the same canonical currency,
digits, magnitude, and unit as the selected anaphoric value unless the following sentence has its own
exact bound temporal lead. Without that new period, a different value is competing evidence, not a
reference. Questions, exclamations, and unknown syntax fail closed.
Negation, modality, forecasts, targets, components, bounds, ranges, deltas, competing subjects,
and compound values cannot become ordinary reported levels merely because the same number appears
after a partial or ambiguous metric phrase.

The selected assertion must preserve its structural boundary inside the exact quotation, and the
quotation must preserve the same metric boundary inside the immutable chunk. A direct metric
subject, atomic observation, or context assertion beginning with a leading temporal anchor cannot
be cropped out of a preceding word-level prefix at either layer. A closed attribution prefix must
match the entire omitted prefix rather than a benign suffix inside role-changing text. A cropped tail
must begin at a terminal boundary or one named comma-commentary/coordination shape. Therefore a draft cannot
turn `Forecast revenue was $2 million` or `Revenue was $2 million, a forecast for next year` into a
clean reported level by cropping the role qualifier. Authorized comparative commentary remains
outside the atomic assertion but inside the exact quote and receipt. An immediately following
nonnumeric anaphoric sentence is also role context and cannot be cropped; a closed numeric followup
remains a separate observation candidate only when a recognized anaphoric reporting connector is
followed immediately by a measurement and an optional temporal tail. Unrelated digits such as a
forecast year do not make cropped role context safe. The same rule applies when the draft omits the
terminal punctuation from its assertion while that punctuation remains in the exact quote.

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
values become date variants only when their closed temporal signatures are pairwise provably
disjoint; an underspecified period such as `Q1` may overlap `Q1 2025` and therefore remains
unresolved. Equivalent quarter, year, and month spellings are canonicalized before comparison.
Repeated canonical kind/value/period
observations are deduplicated for classification and authoritative prose while remaining visible in
the receipt. Values tied to one common period or explicitly
characterized by the source as incompatible become conflicts. Explicit conflict is itself a closed
positive profile: the canonical metric, conflict relationship, and two distinct compatible numeric
or qualitative values must occur in the same local sentence. Qualitative identity uses the same
canonical word-token representation as ordinary classification. A direct bridge must sit immediately between
the two values, its left operand must have its own local positive binding to the claimed metric,
and two modeled operands must belong to the authorized disputed observation set; a
collective characterization must immediately follow two authorized values and
end the sentence. Each collective operand must also have its own local positive binding to the
claimed metric; a competing metric cannot lend its value to the conflict. Merely finding a word
such as `incompatible` elsewhere in the enclosing
quotation proves nothing. Recognized same-metric conflict evidence
prevents a lone reported value from becoming authoritative, regardless of the model-supplied
evidence relation. Unresolved relationships never enter authoritative prose.

For a single qualitative observation, an immediate bounded qualitative operand on the right side
of a locally owned direct conflict bridge is recorded only as an unmodeled conflict signal. A
right-hand phrase containing its own reporting predicate is a competing clause, not an operand. The signal cannot
become a verified or published observation, but a distinct operand prevents the known value from
being published alone. Repeating the same canonical qualitative operand does not manufacture a
conflict, and a bridge owned by a competing metric proves nothing about the claim metric.

## Bounded absence

`not_found` is not inferred from model-controlled search counts. Only a pure absence request with no
observations or context evidence can enter the pending server-probe state; mixed drafts remain
rejected during model self-correction. A post-run server protocol issues
four corpus-wide ranked probes, then separately performs an exhaustive FTS phrase scan for the
complete normalized metric. Compact and dotted initialisms such as `US` and `U.S.` share one
span-preserving token identity; the FTS narrowing query emits both SQLite token representations,
then the exhaustive scan is filtered again by the same complete-word matcher used by verification.
Variant generation is capped; a pathologically acronym-heavy metric falls back to a full
chunk-table scan instead of allocating an exponential query. Exact scans first count candidate
rows and characters; a configured contract limit produces a typed incomplete result before source
rows are materialized. The protocol records both result sets, opens every unique candidate, and
expands source links deterministically to the document edge within a 50-chunk context budget.
Proof analysis uses that full opened source chain; only the separate fixed 384-character window on
both sides of every complete metric occurrence is bounded for the receipt and UI. A missing link,
cycle, or context-budget overflow makes the probe incomplete. This deliberately accepts some
false-negative review outcomes rather than treating an omitted source tail as proof of absence.
Every exact occurrence is then classified into one closed typed outcome: a named benign mention,
a reviewable evidence candidate, or unknown structure. Only questions, source labels, rubric or
instruction references, and terminal standalone labels are currently benign. Reviewable evidence
and unknown structure both require review. The classifier therefore proves a
mention harmless rather than treating an unrecognized reference such as `This value`, `Its value`,
or `The metric's value` as evidence of absence.
Values before the metric require a closed
`VALUE in/of METRIC` bridge. Forward values require either the metric's own assertion or a
contiguous chain of structurally recognized anaphoric reporting assertions; reference phrases are
matched by shape rather than a financial-noun whitelist. Immediate `VALUE METRIC` adjacency also
makes the occurrence reviewable, including bounded comma/colon/dash separators and opening
parentheses, brackets, braces, or quotes. Closed
qualitative states are inspected on both sides of the complete metric phrase, with bounded plain or
initialism qualifier tokens. This does
not attempt sentence parsing, so punctuation inside abbreviations such as
“U.S.” cannot hide a metric/value candidate. Compact initialisms retain their token kind, so `IT`
matches `I.T.` but never the pronoun `It`. If all-caps typography creates a word/initialism
ambiguity, the exact scan fails closed instead of post-filtering the candidate into `not_found`.
For positive evidence, a bare compact metric inside an entirely all-caps assertion is also
ambiguous and cannot authorize a value; an explicit dotted source form, ordinary mixed-case
predicate, or complete multi-token metric phrase disambiguates it.
Failed probes, a failed or over-broad exact scan, scoped searches,
duplicate wording, uninspected candidates, or plausible metric-adjacent values cannot authorize
absence. An exact metric occurrence with a numeric candidate or known positive qualitative
predicate requires review. Qualitative predicate detection permits at most eight punctuation-free
qualifier words between the exact metric and a closed positive connector and follows the same
contiguous structural-reference chain as numeric candidates, so both `Credit rating for the period
was stable` and `Credit rating. It remained stable.` cannot authorize absence. A bare rubric or instruction mention does not by
itself claim a value and therefore does not block the bounded conclusion. The conclusion remains
bounded to the recorded probe and exact corpus version. Complete metric context is inspected in
both directions so value-first and metric-first source shapes both fail closed; the result never
claims that the corpus proves
nonexistence. The verifier independently re-derives that conclusion from the typed search records,
signatures, exact-scan state, candidate/opened ID coverage, and metric occurrences, then rebuilds
every complete analysis context from the run-bound corpus snapshot; it never trusts the producer's
stored conclusion field by itself.

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
plus hashes for the binding and bounded-absence contracts. Provenance uses a schema-visible
discriminated union: live receipts must have null source identity, while contract migrations must
carry the source receipt digest and verifier version. Retained schema-1.2 receipts remain readable
through an exact read-only legacy validator; they cannot be replayed as current receipts. Checked-in rehearsal receipts that are
re-verified after a contract change record `receipt_derivation=contract_migration` together with
the original sealed receipt digest and verifier version; they are not represented as fresh model runs.
The application accepts that lineage only when a migration record is registered in code, the
retained hash-addressed source artifact reproduces the source digest, the source identity matches,
and the complete migrated receipt reproduces the record's canonical target digest. That target
digest covers answer, claims, evidence, events, configuration, and provenance, avoiding a
default-open field projection. This is an application trust boundary, not a third-party signature.
Receipt downloads are produced from the same in-memory serialized snapshot whose schema, canonical
digest, database digest, and terminal status were checked. The HTTP route never reopens the mutable
path after validation, so a replacement cannot swap different bytes into the response.

## Static analysis

Ruff formats and lints Python; it is not a type checker. CI runs `ty` explicitly against Python
3.11 and 3.13 language semantics, executes the suite on both interpreters, checks Astro and the
browser contracts, and runs an offline Python 3.12 container smoke. Financial truth still comes
from explicit runtime contracts and mutation/property tests rather than type hints alone.
