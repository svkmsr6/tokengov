"""tokengov — treat LLM spend like cloud spend.

Lightweight governance primitives for LLM workflows: token tracking with
per-model pricing, per-workflow budgets, two-tier response caching,
complexity-based model routing, cost reports, and a prompt-bloat linter.
"""

from .budget import Budget, BudgetExceeded, budget_guard, budgeted
from .cache import CacheStats, TwoTierCache, cosine, hashing_embedding
from .lint import LintFinding, lint_file, lint_text
from .pricing import PRICING, ModelPricing, calc_cost, count_tokens, register_pricing
from .report import Report, build_report
from .router import ModelChoice, Router
from .tracker import (
    SessionLedger,
    TokenRecord,
    TokenTracker,
    current_tracker,
    get_ledger,
    track,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # pricing
    "PRICING",
    "ModelPricing",
    "calc_cost",
    "count_tokens",
    "register_pricing",
    # tracker
    "SessionLedger",
    "TokenRecord",
    "TokenTracker",
    "current_tracker",
    "get_ledger",
    "track",
    # budget
    "Budget",
    "BudgetExceeded",
    "budget_guard",
    "budgeted",
    # cache
    "CacheStats",
    "TwoTierCache",
    "cosine",
    "hashing_embedding",
    # router
    "ModelChoice",
    "Router",
    # report
    "Report",
    "build_report",
    # lint
    "LintFinding",
    "lint_file",
    "lint_text",
]
