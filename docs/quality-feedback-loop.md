# Quality feedback loop

NeedleProof treats valid review feedback as input to the test and contract system, not as a queue of
sentences to patch. Each accepted finding must leave the repository better able to detect adjacent
failures without another reviewer naming them.

## Required loop

1. Reproduce the reported behavior against the exact reviewed commit.
2. State the violated invariant and assign it to a shape family.
3. Fix the lowest shared contract that owns the invariant.
4. Add the reported example plus adjacent positive, negative, and boundary shapes.
5. Add mutation, metamorphic, or property coverage when inputs form an open family.
6. Tighten types, schemas, version hashes, or provenance when the defect crossed a boundary.
7. Run the smallest focused suite, then every affected repository gate.
8. Push one exact head and inspect semantic-review payloads, not only check badges.

A review suggestion is not accepted merely because a bot produced it. It must reproduce or expose a
credible invariant break. Conversely, a green build does not overrule a reproducible correctness
failure; the missing shape becomes a new enforced test.

## Shape families

Current regression families include:

- positive observation bindings and fail-closed role, scope, modality, and predicate mutations;
- assertion/quotation boundary preservation and coordinated-clause boundaries;
- numeric injectivity, sign/unit identity, and decimal-context independence;
- temporal equivalence, distinction, and exact observation-span binding;
- conflict locality, value adjacency, and cross-metric decoys;
- bounded absence completeness, bidirectional metric context, and forged-proof mutations;
- receipt schema totality, version compatibility, event-chain consistency, and provenance;
- run lifecycle cancellation, persistence faults, reconciliation, reconnect, and idempotence;
- opaque identifier serialization, browser ownership isolation, and terminal-state recovery;
- retrieval ranking, filter behavior, artifact integrity, and real SQLite batching boundaries.

## Choosing the enforcement layer

- Use static types for impossible application states and API/tool boundaries.
- Use strict Pydantic and JSON Schema validation for decoded or persisted data.
- Use deterministic runtime checks for evidence, hashes, numeric strings, and lifecycle state.
- Use property tests for large input families and browser tests for JavaScript/network boundaries.
- Use fixed-point source-digit normalization for reported financial values; do not introduce binary
  floating-point arithmetic where the product requires exact source identity.
- Do not encode arbitrary English entailment in types or an expanding blacklist. Unknown semantic
  shapes fail closed until a named positive contract and its counterexamples are added.

The goal is cumulative: every real failure should permanently narrow the space in which the same
class of bug can return.
