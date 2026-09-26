"""Tracker and ledger tests."""

import asyncio

import pytest

from tokengov import SessionLedger, current_tracker, track
from tokengov.pricing import calc_cost


@pytest.fixture()
def ledger():
    return SessionLedger()


def test_track_records_priced_record(ledger):
    with track("gpt-4o-mini", workflow="wf", ledger=ledger) as t:
        assert current_tracker() is t
        t.count_input(1000)
        t.count_output(500)
    assert current_tracker() is None
    records = ledger.records
    assert len(records) == 1
    r = records[0]
    assert r.model == "gpt-4o-mini"
    assert r.workflow == "wf"
    assert r.input_tokens == 1000
    assert r.output_tokens == 500
    assert r.cost == pytest.approx(calc_cost("gpt-4o-mini", 1000, 500))


def test_count_accepts_text(ledger):
    with track("gpt-4o-mini", ledger=ledger) as t:
        t.count_input("hello world")
    assert ledger.records[0].input_tokens == 2


def test_nested_regions_do_not_double_count(ledger):
    with track("gpt-4o-mini", workflow="outer", ledger=ledger) as outer:
        outer.count_input(100)
        with track("gpt-4o", workflow="inner", ledger=ledger) as inner:
            inner.count_input(50)
            inner.count_output(10)
        outer.count_output(5)
    assert len(ledger.records) == 2
    by_wf = {r.workflow: r for r in ledger.records}
    assert by_wf["outer"].input_tokens == 100
    assert by_wf["outer"].output_tokens == 5
    assert by_wf["inner"].input_tokens == 50


def test_exception_skips_record(ledger):
    with pytest.raises(ValueError), track("gpt-4o-mini", ledger=ledger):
        raise ValueError("boom")
    assert ledger.records == []


def test_totals_and_grouping(ledger):
    with track("gpt-4o-mini", workflow="a", ledger=ledger) as t:
        t.count_input(100)
        t.count_output(100)
    with track("gpt-4o", workflow="a", ledger=ledger) as t:
        t.count_input(200)
        t.count_output(200)
    with track("gpt-4o-mini", workflow="b", ledger=ledger) as t:
        t.count_input(300)
    totals = ledger.totals()
    assert totals["requests"] == 3
    assert totals["input_tokens"] == 600
    assert totals["cost"] == pytest.approx(
        calc_cost("gpt-4o-mini", 100, 100)
        + calc_cost("gpt-4o", 200, 200)
        + calc_cost("gpt-4o-mini", 300, 0)
    )
    by_model = ledger.by_model()
    assert by_model["gpt-4o-mini"]["requests"] == 2
    assert by_model["gpt-4o"]["input_tokens"] == 200
    by_wf = ledger.by_workflow()
    assert by_wf["a"]["requests"] == 2
    assert by_wf["b"]["output_tokens"] == 0


def test_concurrent_tasks_isolated(ledger):
    async def worker(model: str, n: int) -> None:
        with track(model, workflow=model, ledger=ledger) as t:
            await asyncio.sleep(0)  # force interleaving
            t.count_input(n)
            t.count_output(n)

    async def run():
        await asyncio.gather(
            worker("gpt-4o-mini", 10),
            worker("gpt-4o", 20),
            worker("gpt-4o-mini", 30),
        )

    asyncio.run(run())
    assert len(ledger.records) == 3
    assert sum(r.input_tokens for r in ledger.records) == 60
    assert {r.model for r in ledger.records} == {"gpt-4o-mini", "gpt-4o"}


def test_reset_clears_ledger(ledger):
    with track("gpt-4o-mini", ledger=ledger) as t:
        t.count_input(10)
    assert len(ledger.records) == 1
    ledger.reset()
    assert ledger.records == []
    assert ledger.totals()["requests"] == 0
