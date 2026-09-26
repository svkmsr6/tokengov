"""Prompt-bloat linter.

Scans prompt files and strings for the three failure modes I see most often
in production system prompts: filler phrases, duplicated instructions, and
system prompts that quietly ballooned past a sane token ceiling. Every
finding carries an estimated token saving so you can trade politeness against
a concrete dollar figure in review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .pricing import count_tokens, get_pricing

__all__ = ["LintFinding", "lint_file", "lint_text"]

# Phrases that cost tokens and add nothing the model can act on.
FILLER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("politeness-padding", re.compile(r"\b(please|kindly)\b[,.]?", re.IGNORECASE)),
    ("assistant-disclaimer", re.compile(r"\bas an ai\b", re.IGNORECASE)),
    ("meta-instruction", re.compile(r"\bit is important to\b", re.IGNORECASE)),
    ("hedging", re.compile(r"\bit should be noted that\b", re.IGNORECASE)),
    ("fluff-transition", re.compile(r"\bin conclusion\b", re.IGNORECASE)),
)

DEFAULT_SYSTEM_PROMPT_TOKEN_CEILING = 512


@dataclass(frozen=True, slots=True)
class LintFinding:
    """One rule violation with its estimated token saving."""

    rule: str
    severity: str  # "info" | "warn" | "error"
    message: str
    line: int
    tokens_wasted: int

    def __str__(self) -> str:
        return (
            f"{self.severity.upper():5} {self.rule} (line {self.line}): "
            f"{self.message} [~{self.tokens_wasted} tokens]"
        )


def lint_text(
    text: str,
    *,
    system_prompt_ceiling: int = DEFAULT_SYSTEM_PROMPT_TOKEN_CEILING,
    cost_model: str = "gpt-4o-mini",
) -> list[LintFinding]:
    """Lint a prompt string and return findings, sorted by tokens_wasted desc.

    Rules:
        * ``filler-phrase``: known zero-signal phrases (per occurrence).
        * ``duplicated-instruction``: a line appearing more than once.
        * ``oversized-system-prompt``: total tokens above the ceiling.

    Args:
        text: The prompt text to lint.
        system_prompt_ceiling: Token budget for the whole prompt.
        cost_model: Registered model used to price the potential saving.
    """
    findings: list[LintFinding] = []
    lines = text.splitlines()

    for rule, pattern in FILLER_PATTERNS:
        for i, line in enumerate(lines, start=1):
            hits = pattern.findall(line)
            if hits:
                findings.append(
                    LintFinding(
                        rule=f"filler:{rule}",
                        severity="warn",
                        message=f"{len(hits)}x occurrence(s) of a known "
                        f"filler phrase on this line",
                        line=i,
                        tokens_wasted=sum(count_tokens(m) for m in hits),
                    )
                )

    seen: dict[str, int] = {}
    for i, line in enumerate(lines, start=1):
        stripped = line.strip().lower()
        if len(stripped) < 20:
            continue  # too short to be a meaningful "instruction"
        if stripped in seen:
            findings.append(
                LintFinding(
                    rule="duplicated-instruction",
                    severity="error",
                    message=f"line duplicates line {seen[stripped]}: {stripped[:60]!r}",
                    line=i,
                    tokens_wasted=count_tokens(line),
                )
            )
        else:
            seen[stripped] = i

    total_tokens = count_tokens(text)
    if total_tokens > system_prompt_ceiling:
        overflow = total_tokens - system_prompt_ceiling
        price = get_pricing(cost_model)
        per_call = overflow * price.input_per_million / 1_000_000
        findings.append(
            LintFinding(
                rule="oversized-system-prompt",
                severity="error",
                message=f"prompt is {total_tokens} tokens, {overflow} over the "
                f"{system_prompt_ceiling}-token ceiling (~${per_call:.6f}/call "
                f"at {cost_model} input rates)",
                line=1,
                tokens_wasted=overflow,
            )
        )

    return sorted(findings, key=lambda f: f.tokens_wasted, reverse=True)


def lint_file(path: str | Path, **kwargs) -> list[LintFinding]:
    """Lint the file at ``path`` (UTF-8)."""
    return lint_text(Path(path).read_text(encoding="utf-8"), **kwargs)


def total_savings(findings: list[LintFinding]) -> int:
    """Sum of estimated tokens saved if every finding were fixed."""
    return sum(f.tokens_wasted for f in findings)
