# ADR 0002: Contextvar-scoped tracking

- Status: accepted
- Date: 2026-09-26

## Context

`track()` must attribute tokens to the right request when many requests share
a process — asyncio tasks in a web service, threads in a worker pool, nested
instrumented regions inside one request. The alternatives:

- **Pass the tracker explicitly everywhere.** Precise, but it contaminates
  every function signature in the call path, and the first utility that
  drops the parameter silently misattributes spend.
- **Module-level mutable globals.** Hopeless under concurrency.
- **Thread-local storage.** Works for threads, but asyncio tasks share one
  thread; attributions would bleed across concurrent requests.

## Decision

The active tracker lives in a `contextvars.ContextVar`. Entering a
`track()` context sets it; exiting resets it with the returned token. Each
asyncio task copies the context at creation, so concurrent tasks hold
independent trackers; threads do the same via `threading` semantics. A
`current_tracker()` accessor lets library code deep in the stack report
tokens without threading a parameter through every layer.

Aggregation into the session ledger is deliberately *not* context-scoped:
one process-wide `SessionLedger` guarded by a single `threading.Lock`
serializes `add()`, so cross-thread and cross-task records interleave safely.
Reads iterate a snapshot (`records` returns a copy), keeping reporting
correct while requests are in flight.

Records are written on context exit — not per `count_*` call — so a request
that raises still records nothing, matching the rule that cost follows
completed, billable work.

## Consequences

- Nested `track()` regions each produce their own ledger record; inner
  regions never double-count into the outer one.
- Library integrations can report through `current_tracker()` with zero
  signature changes.
- The ledger lock is a single contention point; at the kilohertz rates this
  toolkit targets, lock hold time (list append + budget fan-out) is
  negligible. If a future deployment needs lock-free ingestion, the ledger
  interface — `add(record)` — is the only seam to replace.
