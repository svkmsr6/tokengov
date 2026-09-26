"""Prompt-bloat linter tests."""

from tokengov import lint_text
from tokengov.lint import total_savings

BLOATED = """\
You are a helpful assistant. As an AI, please follow these rules carefully.
Always answer in a professional tone, and please be concise.
Always answer in a professional tone, and please be concise.
It is important to note that accuracy matters more than speed in all cases.
In conclusion, remember that users depend on you every single day for the
answers that they need most urgently in their work and their lives today.
"""


def test_detects_filler_phrases():
    findings = lint_text(BLOATED)
    filler = [f for f in findings if f.rule.startswith("filler:")]
    assert filler, "expected filler findings"
    assert any("assistant-disclaimer" == f.rule.split(":", 1)[1] for f in filler)
    assert all(f.tokens_wasted > 0 for f in filler)


def test_detects_duplicated_instructions():
    findings = lint_text(BLOATED)
    dupes = [f for f in findings if f.rule == "duplicated-instruction"]
    assert len(dupes) == 1
    assert dupes[0].line == 3  # the second occurrence is flagged
    assert "line 2" in dupes[0].message


def test_detects_oversized_prompt():
    findings = lint_text(BLOATED, system_prompt_ceiling=20)
    oversized = [f for f in findings if f.rule == "oversized-system-prompt"]
    assert len(oversized) == 1
    assert oversized[0].severity == "error"
    assert "$" in oversized[0].message  # priced saving, not just tokens


def test_clean_prompt_has_no_findings():
    lean = "Extract the invoice total. Reply with a number only."
    assert lint_text(lean, system_prompt_ceiling=512) == []


def test_findings_sorted_by_tokens_wasted():
    findings = lint_text(BLOATED, system_prompt_ceiling=20)
    wasted = [f.tokens_wasted for f in findings]
    assert wasted == sorted(wasted, reverse=True)


def test_total_savings_sums_waste():
    findings = lint_text(BLOATED, system_prompt_ceiling=20)
    assert total_savings(findings) == sum(f.tokens_wasted for f in findings)


def test_threshold_respected_when_under_ceiling():
    text = "Summarize the input. Use at most three sentences."
    findings = lint_text(text, system_prompt_ceiling=100)
    assert not any(f.rule == "oversized-system-prompt" for f in findings)


def test_lint_file(tmp_path):
    from tokengov import lint_file

    p = tmp_path / "prompt.txt"
    p.write_text(BLOATED, encoding="utf-8")
    findings = lint_file(p, system_prompt_ceiling=20)
    assert len(findings) >= 3
