"""Context-local token tracking with a thread-safe session ledger.

``track()`` opens a context that attributes every token your code reports to
a specific model and workflow. On exit, the record is priced and written to a
process-wide ledger that aggregates per model and per workflow — the raw
material for cost reports and budget enforcement.

Tracking is scoped with ``contextvars``, so concurrent tasks (asyncio) and
threads each see their own active tracker, while aggregation into the ledger
is serialized under a lock.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextvars import ContextVar
from contextvars import Token as CtxToken
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .pricing import calc_cost

if TYPE_CHECKING:
    from .budget import Budget

__all__ = ["TokenRecord", "SessionLedger", "TokenTracker", "get_ledger", "track"]


@dataclass(frozen=True, slots=True)
class TokenRecord:
    """One priced request, as written to the ledger."""

    model: str
    workflow: str
    input_tokens: int
    output_tokens: int
    cost: float
    cached_input_tokens: int = 0
    timestamp: float = 0.0

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "workflow": self.workflow,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cost": self.cost,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class _Aggregate:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost: float = 0.0


class SessionLedger:
    """Thread-safe aggregation of token records for a session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[TokenRecord] = []
        self._started_at = time.monotonic()
        self._budgets: list[Budget] = []

    # -- recording ---------------------------------------------------------

    def add(self, record: TokenRecord) -> None:
        """Append a record and notify any attached budgets.

        Raises:
            BudgetExceeded: If a :class:`~tokengov.budget.Budget` attached to
                this ledger breaches its limit on this record.
        """
        with self._lock:
            self._records.append(record)
            budgets = list(self._budgets)
        for budget in budgets:
            budget.consume(record.cost)

    def attach_budget(self, budget: Budget) -> None:
        """Attach a budget so every subsequent record is charged against it.

        Idempotent: attaching the same budget twice charges it once.
        """
        with self._lock:
            if budget not in self._budgets:
                self._budgets.append(budget)

    def detach_budget(self, budget: Budget) -> None:
        """Remove a previously attached budget (no-op if not attached)."""
        with self._lock:
            if budget in self._budgets:
                self._budgets.remove(budget)

    # -- queries -----------------------------------------------------------

    @property
    def records(self) -> list[TokenRecord]:
        with self._lock:
            return list(self._records)

    def totals(self) -> dict:
        """Aggregate totals across all records."""
        agg = _Aggregate()
        for r in self._records:
            agg.requests += 1
            agg.input_tokens += r.input_tokens
            agg.output_tokens += r.output_tokens
            agg.cached_input_tokens += r.cached_input_tokens
            agg.cost += r.cost
        return {
            "requests": agg.requests,
            "input_tokens": agg.input_tokens,
            "output_tokens": agg.output_tokens,
            "cached_input_tokens": agg.cached_input_tokens,
            "cost": round(agg.cost, 10),
            "elapsed_seconds": round(time.monotonic() - self._started_at, 3),
        }

    def by_model(self) -> dict[str, dict]:
        """Aggregates keyed by model."""
        return self._group_by(lambda r: r.model)

    def by_workflow(self) -> dict[str, dict]:
        """Aggregates keyed by workflow."""
        return self._group_by(lambda r: r.workflow)

    def _group_by(self, key) -> dict[str, dict]:
        groups: dict[str, _Aggregate] = defaultdict(_Aggregate)
        for r in self._records:
            agg = groups[key(r)]
            agg.requests += 1
            agg.input_tokens += r.input_tokens
            agg.output_tokens += r.output_tokens
            agg.cached_input_tokens += r.cached_input_tokens
            agg.cost += r.cost
        return {
            k: {
                "requests": a.requests,
                "input_tokens": a.input_tokens,
                "output_tokens": a.output_tokens,
                "cached_input_tokens": a.cached_input_tokens,
                "cost": round(a.cost, 10),
            }
            for k, a in sorted(groups.items())
        }

    def reset(self) -> None:
        """Clear all records and restart the session clock."""
        with self._lock:
            self._records.clear()
            self._started_at = time.monotonic()


_default_ledger = SessionLedger()


def get_ledger() -> SessionLedger:
    """Return the process-wide default ledger."""
    return _default_ledger


_current_tracker: ContextVar[TokenTracker | None] = ContextVar(
    "tokengov_current_tracker", default=None
)


@dataclass
class TokenTracker:
    """Accumulates tokens for one tracked region, then prices and records them.

    Use via :func:`track`, not directly.
    """

    model: str
    workflow: str = "default"
    ledger: SessionLedger = field(default_factory=get_ledger)
    budget: Budget | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    _ctx_token: CtxToken | None = field(default=None, repr=False)

    def count_input(self, text_or_tokens) -> int:
        """Report input tokens (int) or text (str, encoded via tiktoken)."""
        n = _coerce_tokens(text_or_tokens)
        self.input_tokens += n
        return n

    def count_output(self, text_or_tokens) -> int:
        """Report output tokens (int) or text (str, encoded via tiktoken)."""
        n = _coerce_tokens(text_or_tokens)
        self.output_tokens += n
        return n

    def count_cached_input(self, text_or_tokens) -> int:
        """Report input tokens that were served from cache."""
        n = _coerce_tokens(text_or_tokens)
        self.cached_input_tokens += n
        return n

    def __enter__(self) -> TokenTracker:
        self._ctx_token = _current_tracker.set(self)
        if self.budget is not None:
            self.ledger.attach_budget(self.budget)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._ctx_token is not None:
            _current_tracker.reset(self._ctx_token)
        try:
            if exc_type is None and (self.input_tokens or self.output_tokens):
                self.ledger.add(
                    TokenRecord(
                        model=self.model,
                        workflow=self.workflow,
                        input_tokens=self.input_tokens,
                        output_tokens=self.output_tokens,
                        cached_input_tokens=self.cached_input_tokens,
                        cost=calc_cost(
                            self.model,
                            self.input_tokens,
                            self.output_tokens,
                            self.cached_input_tokens,
                        ),
                        timestamp=time.time(),
                    )
                )
        finally:
            if self.budget is not None:
                self.ledger.detach_budget(self.budget)
        return False


def track(
    model: str,
    workflow: str = "default",
    *,
    ledger: SessionLedger | None = None,
    budget: Budget | None = None,
) -> TokenTracker:
    """Open a tracked region; every reported token is priced for ``model``.

    Args:
        model: Registered model identifier (see :mod:`tokengov.pricing`).
        workflow: Free-form workflow label (e.g. ``"support-bot"``), used for
            per-workflow budgets and report breakdowns.
        ledger: Ledger to record into (defaults to the process-wide ledger).
        budget: Optional budget charged on record write; on breach,
            :class:`~tokengov.budget.BudgetExceeded` propagates out of the
            ``with`` block.

    Example:
        >>> with track("gpt-4o-mini", workflow="summarizer") as t:
        ...     t.count_input("hello world")
        ...     t.count_output("hi there")
    """
    return TokenTracker(
        model=model,
        workflow=workflow,
        ledger=ledger or get_ledger(),
        budget=budget,
    )


def current_tracker() -> TokenTracker | None:
    """Return the tracker active in this context, if any."""
    return _current_tracker.get()


def _coerce_tokens(text_or_tokens) -> int:
    if isinstance(text_or_tokens, str):
        from .pricing import count_tokens

        return count_tokens(text_or_tokens)
    if isinstance(text_or_tokens, int) and text_or_tokens >= 0:
        return text_or_tokens
    raise TypeError(f"Expected token count (int) or text (str), got {type(text_or_tokens)!r}")
