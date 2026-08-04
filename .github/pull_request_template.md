## Change

Describe the user-visible outcome and the lowest shared contract changed.

## Validation

- [ ] Focused regression suite
- [ ] Full backend suite
- [ ] Ruff format and lint
- [ ] Python floor/ceiling type checks
- [ ] Receipt validation when receipt behavior changes
- [ ] Web build and browser contracts when an API/browser boundary changes

## Review-feedback closure

For every accepted review finding, record the cumulative guard it added. A sentence-level patch is
not complete until the adjacent failure family is constrained.

| Finding | Violated invariant / shape family | Root contract fixed | Adjacent or property coverage | CI gate |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

- [ ] Reproduced against the exact reviewed commit or established a credible invariant break
- [ ] Added the reported example and neighboring positive, negative, and boundary shapes
- [ ] Added mutation, metamorphic, or property coverage for an open-ended input family
- [ ] Tightened types, schemas, hashes, or provenance where the defect crossed a boundary
- [ ] Inspected the exact-head semantic review payload rather than relying on a green badge

See `docs/quality-feedback-loop.md` for the governing policy.
