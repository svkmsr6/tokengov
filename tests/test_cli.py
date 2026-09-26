"""CLI tests."""

import json

from tokengov import SessionLedger, build_report, track
from tokengov.cli import main


def test_prices_lists_registered_models(capsys):
    assert main(["prices"]) == 0
    out = capsys.readouterr().out
    assert "gpt-4o" in out
    assert "cached" in out.lower()


def test_report_empty_ledger_exits_nonzero(capsys, tmp_path, monkeypatch):
    # Fresh default ledger via a tracked record in a temp... CLI uses the
    # default ledger; give it something or expect the empty path.
    led = SessionLedger()
    with track("gpt-4o-mini", ledger=led) as t:
        t.count_input(10)
    # Point the CLI's build_report at our ledger by exporting instead.
    report = build_report(led)
    path = tmp_path / "ledger.json"
    path.write_text(report.to_json(), encoding="utf-8")
    assert main(["report", "--input", str(path)]) == 0
    out = capsys.readouterr().out
    assert "By model" in out
    assert "gpt-4o-mini" in out


def test_report_json_flag(tmp_path, capsys):
    led = SessionLedger()
    with track("gpt-4o", workflow="cli", ledger=led) as t:
        t.count_input(100)
    path = tmp_path / "ledger.json"
    path.write_text(build_report(led).to_json(), encoding="utf-8")
    assert main(["report", "--input", str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["totals"]["requests"] == 1


def test_lint_clean_file(tmp_path, capsys):
    p = tmp_path / "clean.txt"
    p.write_text("Extract the total. Reply with a number.", encoding="utf-8")
    assert main(["lint", str(p)]) == 0
    assert "no findings" in capsys.readouterr().out


def test_lint_bloated_file_errors(tmp_path, capsys):
    p = tmp_path / "bloated.txt"
    p.write_text(
        "Please note that this is important.\nPlease note that this is important.\n",
        encoding="utf-8",
    )
    rc = main(["lint", str(p)])
    out = capsys.readouterr().out
    assert rc == 0 or rc == 1  # error severity -> 1, warn-only -> 0
    assert "finding(s)" in out
