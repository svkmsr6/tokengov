"""End-to-end tokengov demo with a mock LLM client — zero API keys required.

Runs a small governed chat loop over a fixed set of prompts. Each turn goes
through the full governance stack:

    route (cheap vs frontier) -> cache lookup -> (mock) generation -> track
    tokens -> enforce budget -> report

Run from the repo root:

    python examples/governed_chat.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Allow running straight from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokengov import (  # noqa: E402
    Budget,
    BudgetExceeded,
    Router,
    TwoTierCache,
    budget_guard,
    build_report,
    track,
)
from tokengov.pricing import count_tokens  # noqa: E402

PROMPTS = [
    "Summarize this paragraph in one sentence.",
    "Summarize this paragraph in one sentence.",  # exact cache replay
    "Summarize this paragraph in a single sentence please.",  # semantic hit
    "What is the capital of France?",
    "Derive the trade-offs between two-phase commit and consensus protocols "
    "like Raft in distributed systems, and analyze the failure modes under "
    "network partitions. Provide your answer as a JSON table.",
    "Give me a one-word answer: what color is the sky?",
    "Refactor this Python function to reduce its cyclomatic complexity and "
    "explain why does the original version cause bugs. Output strict JSON "
    "with the fields 'code' and 'rationale'.",
    "Rewrite this sentence to be more concise.",
]


class MockLLMClient:
    """Deterministic stand-in for a chat-completions client.

    ``complete`` returns a canned response sized by a seed, and reports the
    prompt/response token counts through the active tracker — the exact seam
    where a real client (OpenAI, Anthropic, etc.) would report usage.
    """

    OPENERS = (
        "Based on the request",
        "Here is a careful analysis",
        "Certainly",
        "Taking everything into account",
    )

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)

    def complete(self, tracker, prompt: str) -> str:
        tracker.count_input(prompt)
        # Frontier answers are longer — that's part of why they cost more.
        target_words = 60 if "gpt-4o" in tracker.model else 15
        response = " ".join(
            [self._rng.choice(self.OPENERS)]
            + [self._rng.choice(WORDS) for _ in range(target_words)]
        )
        tracker.count_output(response)
        return response


WORDS = (
    "the model considers token budgets governance caching routing "
    "workflows prompts latency throughput cost analysis design"
).split()


def main() -> int:
    router = Router(cheap_model="gpt-4o-mini", frontier_model="gpt-4o")
    cache = TwoTierCache(ttl_seconds=300, max_entries=128, similarity_threshold=0.7)
    budget = Budget(limit=0.01, label="demo-session")  # one cent of frontier spend
    client = MockLLMClient()

    print(f"Session budget: ${budget.limit:.2f} (warns at {budget.warn_at:.0%})\n")
    for i, prompt in enumerate(PROMPTS, start=1):
        choice = router.route(prompt)
        print(f"[{i}/{len(PROMPTS)}] {prompt[:64]}{'...' if len(prompt) > 64 else ''}")
        print(f"    route -> {choice.model} | {choice.rationale}")

        cached = cache.get(prompt, choice.model)
        if cached is not None:
            print("    cache hit — $0.000000\n")
            continue

        try:
            with budget_guard(budget), track(choice.model, workflow="governed-chat") as tracker:
                response = client.complete(tracker, prompt)
        except BudgetExceeded as exc:
            print(f"    BUDGET STOP: {exc}\n")
            break
        cache.set(prompt, choice.model, response)
        print(
            f"    generated {count_tokens(response)} tokens "
            f"(spent ${budget.spent:.6f} of ${budget.limit:.2f})\n"
        )

    print("=" * 72)
    print(build_report().render())
    print(
        f"\ncache stats: {cache.stats.hits} hits / {cache.stats.misses} misses "
        f"(exact {cache.stats.exact_hits}, semantic {cache.stats.semantic_hits})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
