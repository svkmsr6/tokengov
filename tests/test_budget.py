"""Budget tests."""

import warnings

import pytest

from tokengov import (
    Budget,
    BudgetExceeded,
    SessionLedger,
    budget_guard,
    budgeted,
    track,
)


@pytest.fixture()
def ledger():
    return SessionLedger()


def _spend(ledger, model, input_tokens, output_tokens, workflow="wf"):
    with track(model, workflow=workflow, ledger=ledger) as t:
        t.count_input(input_tokens)
        t.count_output(output_tokens)


def test_budget_warns_once_at_soft_threshold(ledger):
    budget = Budget(limit=1.00, warn_at=0.8, label="t")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with budget_guard(budget, ledger):
            _spend(ledger, "gpt-4o", 350_000, 0)  # $0.875 >= 80% of $1.00
        # Another spend while still over the soft threshold: no re-warn.
        with budget_guard(budget, ledger):
            _spend(ledger, "gpt-4o", 10_000, 0)
    soft = [w for w in caught if "Budget 't' at" in str(w.message)]
    assert len(soft) == 1


def test_budget_hard_breach_raises(ledger):
    budget = Budget(limit=0.20, label="tight")
    with pytest.raises(BudgetExceeded, match="exceeded"):
        with budget_guard(budget, ledger):
            _spend(ledger, "gpt-4o", 100_000, 10_000)  # $0.35 > $0.20


def test_budget_tracks_spend_and_remaining(ledger):
    budget = Budget(limit=1.00, label="t")
    assert budget.remaining == 1.00
    budget.consume(0.30)
    assert budget.spent == pytest.approx(0.30)
    assert budget.remaining == pytest.approx(0.70)
    assert budget.utilization == pytest.approx(0.30)


def test_budget_reset_rearms(ledger):
    budget = Budget(limit=1.00, warn_at=0.5, label="t")
    budget.consume(0.6)  # triggers warning internally
    budget.reset()
    assert budget.spent == 0.0
    assert budget._warned is False


def test_budgeted_decorator(ledger):
    calls = []

    @budgeted("gpt-4o-mini", limit=0.001, workflow="sum", ledger=ledger)
    def summarize(tracker, text):
        calls.append(text)
        tracker.count_input(text)
        return "ok"

    assert summarize("hello world") == "ok"
    assert calls == ["hello world"]
    assert len(ledger.records) == 1
    assert ledger.records[0].workflow == "sum"


def test_budgeted_decorator_breach_propagates(ledger):
    @budgeted("gpt-4o", limit=0.0001, ledger=ledger)
    def expensive(tracker):
        tracker.count_input(100_000)

    with pytest.raises(BudgetExceeded):
        expensive()


def test_track_budget_param_attaches_and_detaches(ledger):
    budget = Budget(limit=100.0, label="via-track")
    with track("gpt-4o-mini", ledger=ledger, budget=budget) as t:
        t.count_input(10)
    assert ledger._budgets == []  # detached on exit
    assert budget.spent > 0
