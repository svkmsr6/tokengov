"""Complexity-based model routing: cheap model for easy, frontier for hard.

The classifier is a deliberately explainable heuristic. Each signal maps to a
documented weight; the score is a weighted sum in [0, 1]; the threshold picks
the model. Every decision carries a written rationale listing the signals that
fired — you can answer "why did this prompt go to the expensive model?" in a
code review without opening a notebook.

``classifier_fn`` is the escape hatch: pass any ``prompt -> score`` callable
to swap the heuristic for a real classifier later, keeping routing, tracking,
and budgeting untouched.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["ModelChoice", "Router"]


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """The router's decision, with its justification attached."""

    model: str
    rationale: str
    complexity_score: float
    signals: tuple[str, ...] = field(default=())


# Signals: (name, compiled pattern, weight). Weights sum with length/question
# bonuses to a 0–1 score; a prompt needs ~0.5 to reach the frontier tier.
_REASONING = re.compile(
    r"\b(prove|proof|derive|reason step by step|step[- ]by[- ]step|"
    r"why does|why is|trade[- ]?offs?|root cause|formally|optimize|"
    r"edge cases|analyze|analyse|critique|refactor|debug)\b",
    re.IGNORECASE,
)
_STRUCTURED = re.compile(
    r"\b(json|xml|yaml|schema|validate|strict format|table|"
    r"typescript|pydantic|rfc \d+)\b",
    re.IGNORECASE,
)
_DOMAIN = re.compile(
    r"\b(distributed systems|kubernetes|postgres|consensus|paxos|raft|"
    r"matrix|eigenvalue|laplace|fourier|black[- ]?scholes|kalman|"
    r"derivative|integral|latency percentile|p99|backpressure|"
    r"two-phase commit|cap theorem)\b",
    re.IGNORECASE,
)
_MULTITASK = re.compile(r"\b(for each|and also|additionally|finally,)\b", re.IGNORECASE)

_SIGNALS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("reasoning-demand", _REASONING, 0.30),
    ("structured-output", _STRUCTURED, 0.20),
    ("domain-specific", _DOMAIN, 0.25),
    ("multi-part", _MULTITASK, 0.15),
)


class Router:
    """Routes prompts to a cheap or a frontier model based on complexity.

    Args:
        cheap_model: Default model for routine prompts.
        frontier_model: Model for prompts that cross ``threshold``.
        threshold: Complexity score in [0, 1] that selects the frontier tier.
        classifier_fn: Optional replacement heuristic. Receives the prompt,
            returns a float score in [0, 1]. When provided, signal detection
            is skipped and the rationale notes the external classifier.

    Example:
        >>> router = Router(cheap_model="gpt-4o-mini", frontier_model="gpt-4o")
        >>> choice = router.route("Summarize this paragraph in one sentence.")
        >>> choice.model
        'gpt-4o-mini'
    """

    def __init__(
        self,
        cheap_model: str,
        frontier_model: str,
        *,
        threshold: float = 0.5,
        classifier_fn: Callable[[str], float] | None = None,
    ) -> None:
        self.cheap_model = cheap_model
        self.frontier_model = frontier_model
        self.threshold = threshold
        self._classifier_fn = classifier_fn

    # -- heuristic ----------------------------------------------------------

    def score(self, prompt: str) -> tuple[float, tuple[str, ...]]:
        """Return (complexity score, fired signal names) for ``prompt``."""
        if self._classifier_fn is not None:
            return self._clamp(self._classifier_fn(prompt)), ("external-classifier",)

        score = 0.0
        fired: list[str] = []
        lowered = prompt.lower()

        for name, pattern, weight in _SIGNALS:
            if pattern.search(prompt):
                score += weight
                fired.append(name)

        # Length signal: prompts carrying lots of context tend to demand more.
        words = len(lowered.split())
        if words > 400:
            score += 0.25
            fired.append("long-context")
        elif words > 120:
            score += 0.15
            fired.append("long-context")

        # Multiple distinct questions usually mean a harder task.
        if prompt.count("?") >= 2:
            score += 0.10
            fired.append("multi-question")

        return self._clamp(score), tuple(fired)

    # -- decision -----------------------------------------------------------

    def route(self, prompt: str) -> ModelChoice:
        """Choose a model for ``prompt`` with a written rationale."""
        score, fired = self.score(prompt)
        model = self.frontier_model if score >= self.threshold else self.cheap_model
        tier = "frontier" if model == self.frontier_model else "cheap"

        if fired:
            why = ", ".join(fired)
            rationale = (
                f"score {score:.2f} >= {self.threshold:.2f}: {why}"
                if (model == self.frontier_model)
                else f"score {score:.2f} < {self.threshold:.2f}: {why}"
            )
        else:
            rationale = f"no complexity signals fired (score {score:.2f})"

        return ModelChoice(
            model=model,
            rationale=f"[{tier}] {rationale}",
            complexity_score=score,
            signals=fired,
        )

    @staticmethod
    def _clamp(score: float) -> float:
        return max(0.0, min(1.0, score))
