"""Router classification tests."""

from tokengov import Router

EASY = [
    "Summarize this paragraph in one sentence.",
    "What is the capital of France?",
    "Give me a one-word answer: what color is the sky?",
    "Rewrite this sentence to be more concise.",
]

HARD = [
    "Derive the trade-offs between two-phase commit and consensus protocols "
    "like Raft in distributed systems, and analyze the failure modes under "
    "network partitions. Provide your answer as a JSON table.",
    "Prove that a distributed system cannot simultaneously guarantee "
    "consistency, availability, and partition tolerance, and explain why "
    "does the CAP theorem hold under message loss.",
    "Refactor this payment-processing function to handle every edge case, "
    "debug the existing race condition, and output strict JSON matching the "
    "provided pydantic schema.",
]


def make_router(**kw):
    return Router(cheap_model="cheap", frontier_model="frontier", **kw)


def test_easy_prompts_route_cheap():
    router = make_router()
    for prompt in EASY:
        choice = router.route(prompt)
        assert choice.model == "cheap", prompt
        assert choice.complexity_score < router.threshold


def test_hard_prompts_route_frontier():
    router = make_router()
    for prompt in HARD:
        choice = router.route(prompt)
        assert choice.model == "frontier", prompt
        assert choice.complexity_score >= router.threshold


def test_rationale_lists_fired_signals():
    router = make_router()
    choice = router.route(HARD[0])
    assert "reasoning-demand" in choice.signals
    assert "structured-output" in choice.signals
    assert "json" in choice.rationale.lower() or "structured" in choice.rationale


def test_no_signal_rationale_for_easy_prompts():
    router = make_router()
    choice = router.route("Hi.")
    assert choice.signals == ()
    assert "no complexity signals fired" in choice.rationale


def test_score_clamped_to_unit_interval():
    router = make_router(threshold=0.4)
    choice = router.route("prove derive analyze trade-offs json " * 20)
    assert 0.0 <= choice.complexity_score <= 1.0
    assert choice.model == "frontier"


def test_custom_threshold():
    router = make_router(threshold=1.0)  # nothing routes frontier
    assert router.route(HARD[0]).model == "cheap"


def test_external_classifier_override():
    router = make_router(classifier_fn=lambda prompt: 0.9 if "banana" in prompt else 0.0)
    assert router.route("banana?").model == "frontier"
    assert router.route("quantum chromodynamics proof").model == "cheap"
    assert router.route("banana?").signals == ("external-classifier",)
