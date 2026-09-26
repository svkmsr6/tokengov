# tokengov — Low-Level Design

## 1. Pricing (`pricing.py`)

```python
@dataclass(frozen=True, slots=True)
class ModelPricing:
    model: str
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None
```

- `PRICING` is a module-level dict keyed by model id. `register_pricing()`
  mutates it at runtime; `get_pricing()` raises `KeyError` on unknown models.
- `calc_cost(model, input, output, cached_input=0)` prices fresh input at the
  standard rate and the cached portion at the reduced rate:
  `fresh*in + cached*cached_in + output*out`, all per million.
- `count_tokens(text)` centralizes the encoding (`cl100k_base`) so tracker,
  linter, and tests agree on what a token is.

## 2. Tracking (`tracker.py`)

**Ledger schema.** `SessionLedger` holds an append-only list of frozen
`TokenRecord`s:

| field                | type  | source                          |
|----------------------|-------|---------------------------------|
| `model`              | str   | router / caller                 |
| `workflow`           | str   | caller label                    |
| `input_tokens`       | int   | reported via tracker            |
| `output_tokens`      | int   | reported via tracker            |
| `cached_input_tokens`| int   | reported via tracker            |
| `cost`               | float | `calc_cost` on exit             |
| `timestamp`          | float | `time.time()` on exit           |

**Aggregation.** A single `threading.Lock` serializes `add()`, `reset()`,
and budget attachment. Grouped views (`by_model`, `by_workflow`) fold records
into `{requests, input_tokens, output_tokens, cached_input_tokens, cost}` via
a private `_Aggregate` accumulator; reads iterate over a snapshot copy.

**Scoping.** `track()` returns a `TokenTracker` used as a context manager.
Entering pushes the tracker onto a module-level `ContextVar`; `count_*`
methods accumulate; exiting prices the totals and appends one record. The
ContextVar gives each asyncio task and thread its own active tracker. Nested
regions each record their own totals — the inner region's tokens are not
double-counted in the outer record because counting happens against the
tracker the code explicitly holds.

**Budget wiring.** A tracker created with `budget=` attaches it to the ledger
on enter and detaches on exit; the ledger charges every `add()` against all
attached budgets, so a breach surfaces inside the tracked region.

## 3. Budgets (`budget.py`)

`Budget` is a plain dataclass: `limit`, `warn_at` (default 0.8), `spent`,
`_warned`. `consume(cost)` adds spend, fires a one-shot
`warnings.warn` on crossing the soft threshold, and raises
`BudgetExceeded` past the hard limit. `budget_guard` attaches/detaches a
budget to a ledger; `budgeted` composes guard + `track` as a function
decorator that injects the tracker as the first argument.

## 4. Cache (`cache.py`)

**Tiers.**
- *Tier 1 (exact):* `OrderedDict[sha256(model \x00 prompt), CacheEntry]`.
- *Tier 2 (semantic):* `OrderedDict[seq, CacheEntry]` scanned linearly per
  lookup, restricted to the same model, with cosine similarity against the
  query's hashing-vector embedding.

**Embedding.** `hashing_embedding(text, dim=256)` lowercases, splits on
`[a-z0-9]+`, and accumulates token counts into buckets chosen by
`blake2b(token) % dim`, then L2-normalizes. Deterministic, stateless,
offline — lexical overlap is sufficient for near-duplicate prompts (retries,
reworded requests, template drift). `embed_fn` replaces it wholesale.

**Lookup policy.** Tier 1 hit → move-to-end, count. Else scan tier 2:
stale entries (age ≥ `ttl_seconds`) are evicted lazily; if the best
similarity ≥ `similarity_threshold`, promote the entry to tier 1 under the
new exact key (future identical requests become free), count a semantic hit.
Otherwise count a miss.

**Eviction.** On every `set()` and promotion, each tier is trimmed from the
front (LRU) to `max_entries`; evictions are counted in `CacheStats`.

## 5. Routing (`router.py`)

`Router.score(prompt)` returns `(score, fired_signals)`:

| signal              | detector                                   | weight |
|---------------------|--------------------------------------------|--------|
| `reasoning-demand`  | regex: prove/derive/trade-offs/debug/...   | 0.30   |
| `structured-output` | regex: json/schema/table/typescript/...    | 0.20   |
| `domain-specific`   | regex: raft/paxos/fourier/p99/...          | 0.25   |
| `multi-part`        | regex: for each/and also/finally           | 0.15   |
| `long-context`      | word count > 120 (+0.15) or > 400 (+0.25)  | 0.15–0.25 |
| `multi-question`    | ≥ 2 question marks                         | 0.10   |

Score clamps to [0, 1]; `threshold` (default 0.5) selects the tier.
`route()` returns a frozen `ModelChoice(model, rationale, score, signals)`
where `rationale` is a human sentence listing the fired signals. A
`classifier_fn(prompt) -> float` constructor argument bypasses the heuristic
entirely (rationale tagged `external-classifier`).

## 6. Reports (`report.py`)

`build_report(ledger)` snapshots aggregates into a frozen `Report`:
`rows_by_model`, `rows_by_workflow`, totals, elapsed session seconds, and
`projected_monthly_cost = cost * (30d / elapsed)` — a deliberately naive,
clearly labeled extrapolation. `elapsed` is floored at 60 seconds so
sub-minute sessions don't explode the projection; the floor is a display
convenience, not a forecasting claim. Rendering is a fixed-width ASCII table built
from max column widths; `to_json()` emits a schema validated by the pydantic
`ReportExport` model (the CLI parses `--input` files through it, so corrupt
exports fail with field-level errors).

## 7. Linting (`lint.py`)

Three rules over prompt text:
1. **Filler phrases** — per-occurrence regexes (politeness padding,
   assistant disclaimers, hedging); tokens wasted = tiktoken count of matches.
2. **Duplicated instructions** — non-trivial lines (>20 chars) seen twice;
   wasted tokens = the duplicate line's count.
3. **Oversized system prompt** — total tokens over a configurable ceiling;
   wasted = overflow, priced per call at a cheap model's input rate so the
   finding reads in dollars.

Findings sort by `tokens_wasted` descending; `total_savings()` feeds the CLI
summary. Exit code: 0 clean, 1 if any `error`-severity finding (CI-friendly).

## 8. CLI (`cli.py`)

argparse with three subcommands wired to `pyproject.toml`'s `tokengov`
script entry: `prices` (sorted rate card), `report [--input F] [--json|--csv]`
(empty in-process ledger → stderr guidance, exit 1), `lint <path>
[--max-tokens N]`.

## 9. Testing strategy

Unit tests cover: pricing arithmetic incl. cached rates and unknown-model
errors; tracker aggregation, contextvar isolation, nested regions, and
workflow grouping; budget soft-warn/hard-breach/reset; cache exact hits,
semantic near-duplicate hits, TTL expiry, LRU eviction, and stats; router
easy/hard classification, rationale content, threshold behavior, and the
external-classifier hook; report JSON round-trip through the pydantic schema;
and lint rule firing on a bloated fixture. All offline.
