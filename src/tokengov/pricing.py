"""Model pricing table and cost calculation.

Prices are per million tokens (USD), sourced from public provider pricing
pages. The table is intentionally a plain dict of dataclasses so anyone can
register their own internal models or negotiated rates at runtime — your
finops team owns this table, not the library.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-million-token pricing for a single model.

    Attributes:
        model: Canonical model identifier (e.g. ``"gpt-4o"``).
        input_per_million: Cost of one million input (prompt) tokens, in USD.
        output_per_million: Cost of one million output (completion) tokens, in USD.
        cached_input_per_million: Cost of one million cache-hit input tokens.
            Falls back to ``input_per_million`` when ``None``.
    """

    model: str
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None

    def effective_cached_input(self) -> float:
        """Return the per-million rate charged for cache-hit input tokens."""
        return self.cached_input_per_million or self.input_per_million


# Pricing snapshot (USD per 1M tokens), based on publicly listed rates.
# Override or extend at runtime with ``register_pricing`` — e.g. for internal
# models, batch rates, or negotiated enterprise discounts.
PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        input_per_million=2.50,
        output_per_million=10.00,
        cached_input_per_million=1.25,
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        input_per_million=0.15,
        output_per_million=0.60,
    ),
    "gpt-4.1": ModelPricing(
        model="gpt-4.1",
        input_per_million=2.00,
        output_per_million=8.00,
        cached_input_per_million=0.50,
    ),
    "gpt-4.1-mini": ModelPricing(
        model="gpt-4.1-mini",
        input_per_million=0.40,
        output_per_million=1.60,
    ),
    "claude-sonnet-4": ModelPricing(
        model="claude-sonnet-4",
        input_per_million=3.00,
        output_per_million=15.00,
        cached_input_per_million=0.30,
    ),
    "claude-haiku-4-5": ModelPricing(
        model="claude-haiku-4-5",
        input_per_million=1.00,
        output_per_million=5.00,
        cached_input_per_million=0.10,
    ),
}


def register_pricing(pricing: ModelPricing) -> None:
    """Register or override pricing for a model at runtime.

    Args:
        pricing: The :class:`ModelPricing` entry to add or replace.
    """
    PRICING[pricing.model] = pricing


def get_pricing(model: str) -> ModelPricing:
    """Look up pricing for ``model``.

    Raises:
        KeyError: If the model is not registered. This is deliberate: silently
            pricing an unknown model at $0 makes cost governance useless.
    """
    try:
        return PRICING[model]
    except KeyError as exc:
        raise KeyError(
            f"No pricing registered for model {model!r}. "
            "Call register_pricing(ModelPricing(...)) first."
        ) from exc


def calc_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Calculate the USD cost of a single request.

    Args:
        model: Registered model identifier.
        input_tokens: Total prompt tokens (including any cached portion).
        output_tokens: Generated completion tokens.
        cached_input_tokens: Prompt tokens served from cache, billed at the
            reduced cached rate when the model defines one.
    """
    p = get_pricing(model)
    cached = min(cached_input_tokens, input_tokens)
    fresh = input_tokens - cached
    cost = fresh * p.input_per_million / 1_000_000
    cost += cached * p.effective_cached_input() / 1_000_000
    cost += output_tokens * p.output_per_million / 1_000_000
    return round(cost, 10)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in ``text`` with a tiktoken encoding.

    Used across the toolkit (linting, tracker convenience, tests) so every
    module agrees on what a "token" is.
    """
    import tiktoken

    return len(tiktoken.get_encoding(encoding).encode(text))
