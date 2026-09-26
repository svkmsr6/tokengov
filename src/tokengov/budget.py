"""Per-workflow and per-session budgets with soft warnings.

Budgets are charged in dollars as records land in the ledger. Crossing the
soft threshold (default 80%) emits a warning once; crossing the hard limit
raises :class:`BudgetExceeded`, which fails the tracked region — the same
contract a CI check or an HTTP 402 gives you, but local to your process.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field

from .tracker import SessionLedger, get_ledger, track

__all__ = ["Budget", "BudgetExceeded", "budgeted", "budget_guard"]


class BudgetExceeded(RuntimeError):
    """Raised when a budget's hard limit is breached."""


@dataclass
class Budget:
    """A dollar-denominated spend ceiling.

    Args:
        limit: Hard limit in USD. Breaching it raises :class:`BudgetExceeded`.
        warn_at: Fraction of the limit (0–1) at which a soft warning fires.
        label: Human-readable name for messages and reports.

    Example:
        >>> budget = Budget(limit=0.50, label="nightly-etl")
        >>> with budget_guard(budget):
        ...     ...  # tracked regions inside are charged against the budget
    """

    limit: float
    warn_at: float = 0.8
    label: str = "budget"
    spent: float = field(default=0.0, init=False)
    _warned: bool = field(default=False, init=False, repr=False)

    @property
    def remaining(self) -> float:
        """Dollars left before the hard limit."""
        return max(0.0, self.limit - self.spent)

    @property
    def utilization(self) -> float:
        """Fraction of the limit consumed (can exceed 1.0 after breach)."""
        return self.spent / self.limit if self.limit else float("inf")

    def consume(self, cost: float) -> None:
        """Charge ``cost`` dollars against this budget.

        Emits a soft warning exactly once when crossing ``warn_at``, then
        raises :class:`BudgetExceeded` when the hard limit is crossed.
        """
        self.spent += cost
        if not self._warned and self.spent >= self.limit * self.warn_at:
            self._warned = True
            warnings.warn(
                f"Budget {self.label!r} at {self.utilization:.0%} "
                f"(${self.spent:.6f} of ${self.limit:g})",
                stacklevel=2,
            )
        if self.spent > self.limit:
            raise BudgetExceeded(
                f"Budget {self.label!r} exceeded: ${self.spent:.6f} spent "
                f"against ${self.limit:g} limit"
            )

    def reset(self) -> None:
        """Zero out spend and re-arm the soft warning."""
        self.spent = 0.0
        self._warned = False


def budget_guard(budget: Budget, ledger: SessionLedger | None = None):
    """Context manager: charge every subsequent ledger record against ``budget``.

    Example:
        >>> with budget_guard(Budget(limit=1.00, label="chat")):
        ...     with track("gpt-4o-mini", workflow="chat"):
        ...         ...
    """
    (ledger or get_ledger()).attach_budget(budget)
    return _BudgetScope(budget, ledger or get_ledger())


class _BudgetScope:
    def __init__(self, budget: Budget, ledger: SessionLedger) -> None:
        self._budget = budget
        self._ledger = ledger

    def __enter__(self) -> Budget:
        return self._budget

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._ledger.detach_budget(self._budget)
        return False


def budgeted(
    model: str,
    limit: float,
    *,
    workflow: str = "default",
    warn_at: float = 0.8,
    ledger: SessionLedger | None = None,
):
    """Decorator: run ``func`` under tracking with a dollar budget.

    The wrapped call is tracked (tokens counted by the decorated function
    via the tracker passed as the first positional arg) and charged against a
    fresh :class:`Budget` of ``limit`` dollars.

    Example:
        >>> @budgeted("gpt-4o-mini", limit=0.01, workflow="summarize")
        ... def summarize(tracker, text: str) -> str:
        ...     tracker.count_input(text)
        ...     ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            budget = Budget(limit=limit, warn_at=warn_at, label=func.__name__)
            with budget_guard(budget, ledger), track(model, workflow, ledger=ledger) as tracker:
                return func(tracker, *args, **kwargs)

        return wrapper

    return decorator
