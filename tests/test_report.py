"""Report building and serialization tests."""

import json

import pytest

from tokengov import SessionLedger, build_report, track
from tokengov.report import ReportExport


@pytest.fixture()
def ledger():
    led = SessionLedger()
    with track("gpt-4o-mini", workflow="a", ledger=led) as t:
        t.count_input(1000)
        t.count_output(500)
    with track("gpt-4o", workflow="b", ledger=led) as t:
        t.count_input(2000)
        t.count_output(1000)
    return led


def test_report_aggregates_by_model_and_workflow(ledger):
    report = build_report(ledger)
    models = {r.label: r for r in report.rows_by_model}
    assert set(models) == {"gpt-4o-mini", "gpt-4o"}
    assert models["gpt-4o-mini"].input_tokens == 1000
    workflows = {r.label: r for r in report.rows_by_workflow}
    assert workflows["a"].requests == 1
    assert workflows["b"].requests == 1
    assert report.total_requests == 2
    assert report.total_cost > 0


def test_json_round_trip_validates_against_schema(ledger):
    report = build_report(ledger)
    data = json.loads(report.to_json())
    export = ReportExport.model_validate(data)
    assert export.totals.requests == 2
    assert len(export.by_model) == 2


def test_malformed_export_rejected():
    with pytest.raises(Exception):  # pydantic ValidationError
        ReportExport.model_validate({"totals": {"requests": "nope"}})


def test_csv_output(ledger):
    report = build_report(ledger)
    csv_out = report.to_csv()
    lines = csv_out.strip().splitlines()
    assert lines[0].startswith("dimension,label")
    assert len(lines) == 1 + 4  # header + 2 models + 2 workflows
    assert "gpt-4o" in csv_out


def test_render_contains_totals(ledger):
    report = build_report(ledger)
    out = report.render()
    assert "By model" in out
    assert "By workflow" in out
    assert "gpt-4o" in out
    assert "Projected 30-day spend" in out


def test_projection_is_positive(ledger):
    report = build_report(ledger)
    assert report.projected_monthly_cost >= report.total_cost
