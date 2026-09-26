# ADR 0001: Explainable heuristics over opaque ML for routing

- Status: accepted
- Date: 2026-09-26

## Context

The router decides, per prompt, whether to spend frontier-model money or
cheap-model money. The tempting implementation is a small fine-tuned
classifier or an embedding-based scorer. That buys accuracy in principle —
but at a price this toolkit is explicitly built to refuse:

- A learned model needs training data most teams don't have, infrastructure
  they don't want, and a dependency graph that breaks the "installs
  anywhere" promise.
- Every routing decision must survive a code review and a finops question.
  "The classifier said so" is not an answer; "the prompt contains three
  reasoning signals and demanded structured output, scoring 0.75 against a
  0.50 threshold" is.
- Token governance is a *trust* feature. The first time the router ships a
  hard prompt to a cheap model and the answer is visibly worse, engineers
  will route everything to the frontier model "to be safe" — and the
  governance loop is dead.

## Decision

Routing is a weighted, hand-tuned heuristic over six documented signals
(reasoning demand, structured-output demand, domain specificity, multi-part
task, long context, multiple questions). Every decision returns a
`ModelChoice` carrying the fired signals, the score, and the threshold, in a
single written sentence.

The heuristic is a swappable seam, not a commitment: `Router(classifier_fn=...)`
accepts any `prompt -> score` callable, so a team can graduate to a real
classifier without touching tracking, budgets, caching, or reporting.

## Consequences

- Routing quality is bounded by regex craftsmanship — acceptable for v1,
  where the frontier tier being slightly over-inclusive costs far less than
  silent misclassification costs trust.
- All routing decisions are debuggable from logs alone.
- A future learned classifier plugs into a one-argument constructor.
