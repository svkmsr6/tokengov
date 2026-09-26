"""Turn the session ledger into a cost report.

Aggregates per model and per workflow, renders an ASCII table for the CLI,
and exports JSON/CSV for dashboards. Projection is deliberately naive and
clearly labeled as such: session spend scaled to a 30-day month — good enough
to catch a runaway workflow before finance does. Sessions shorter than one
minute are floored to one minute before extrapolation, so a lightning CI run
doesn't produce a comical projection.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .tracker import SessionLedger, get_ledger

__all__ = ["Report", "ReportExport", "build_report"]

_SECONDS_PER_MONTH = 30 * 24 * 3600


class ReportRowModel(BaseModel):
    """One aggregated row in an exported report (JSON round-trip schema)."""

    label: str
    requests: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost: float


class TotalsModel(BaseModel):
    requests: int
    input_tokens: int
    output_tokens: int
    cost: float


class ReportExport(BaseModel):
    """Validated schema for ``Report.to_json()`` output.

    The CLI and any downstream tooling parse exported reports through this
    model so a malformed ledger file fails loudly with field-level errors
    instead of producing a silently wrong cost table.
    """

    totals: TotalsModel
    projected_monthly_cost: float
    by_model: list[ReportRowModel] = Field(default_factory=list)
    by_workflow: list[ReportRowModel] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ReportRow:
    label: str
    requests: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost: float


@dataclass(slots=True)
class Report:
    """An immutable snapshot of ledger spend."""

    rows_by_model: tuple[ReportRow, ...]
    rows_by_workflow: tuple[ReportRow, ...]
    total_cost: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    projected_monthly_cost: float
    elapsed_seconds: float

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "totals": {
                "requests": self.total_requests,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cost": round(self.total_cost, 10),
            },
            "projected_monthly_cost": round(self.projected_monthly_cost, 10),
            "elapsed_seconds": self.elapsed_seconds,
            "by_model": [
                {
                    "label": r.label,
                    "requests": r.requests,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cached_input_tokens": r.cached_input_tokens,
                    "cost": r.cost,
                }
                for r in self.rows_by_model
            ],
            "by_workflow": [
                {
                    "label": r.label,
                    "requests": r.requests,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cached_input_tokens": r.cached_input_tokens,
                    "cost": r.cost,
                }
                for r in self.rows_by_workflow
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "dimension",
                "label",
                "requests",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "cost_usd",
            ]
        )
        for dim, rows in (("model", self.rows_by_model), ("workflow", self.rows_by_workflow)):
            for r in rows:
                writer.writerow(
                    [
                        dim,
                        r.label,
                        r.requests,
                        r.input_tokens,
                        r.output_tokens,
                        r.cached_input_tokens,
                        f"{r.cost:.10f}",
                    ]
                )
        return buf.getvalue()

    # -- rendering ----------------------------------------------------------

    def render(self) -> str:
        """Render the report as aligned ASCII tables for terminal output."""
        lines = [
            _render_table("By model", self.rows_by_model),
            "",
            _render_table("By workflow", self.rows_by_workflow),
            "",
            f"Total: {self.total_requests} requests | "
            f"{self.total_input_tokens:,} in / {self.total_output_tokens:,} out | "
            f"${self.total_cost:.6f}",
            f"Session elapsed: {self.elapsed_seconds:.1f}s | "
            f"Projected 30-day spend: ${self.projected_monthly_cost:.2f} "
            "(naive extrapolation)",
        ]
        return "\n".join(lines)


def _render_table(title: str, rows: tuple[ReportRow, ...]) -> str:
    header = ("label", "requests", "input", "output", "cached", "cost")
    body = [
        (
            r.label,
            str(r.requests),
            f"{r.input_tokens:,}",
            f"{r.output_tokens:,}",
            f"{r.cached_input_tokens:,}",
            f"${r.cost:.6f}",
        )
        for r in rows
    ]
    if not body:
        body = [("-", "0", "0", "0", "0", "$0.000000")]
    widths = [max(len(header[i]), *(len(row[i]) for row in body)) for i in range(6)]
    sep = "-+-".join("-" * w for w in widths)
    fmt = " | ".join(f"{{{i}:<{widths[i]}}}" for i in range(6))
    out = [f"== {title} ".ljust(len(sep) + 4, "="), fmt.format(*header), sep]
    out += [fmt.format(*row) for row in body]
    return "\n".join(out)


def build_report(ledger: SessionLedger | None = None) -> Report:
    """Snapshot ``ledger`` (default: process-wide) into a :class:`Report`."""
    ledger = ledger or get_ledger()
    totals = ledger.totals()

    def rows(groups: dict[str, dict]) -> tuple[ReportRow, ...]:
        return tuple(
            ReportRow(
                label=label,
                requests=g["requests"],
                input_tokens=g["input_tokens"],
                output_tokens=g["output_tokens"],
                cached_input_tokens=g["cached_input_tokens"],
                cost=g["cost"],
            )
            for label, g in groups.items()
        )

    elapsed = max(totals["elapsed_seconds"], 60.0)  # floor: see lld.md §6
    return Report(
        rows_by_model=rows(ledger.by_model()),
        rows_by_workflow=rows(ledger.by_workflow()),
        total_cost=totals["cost"],
        total_requests=totals["requests"],
        total_input_tokens=totals["input_tokens"],
        total_output_tokens=totals["output_tokens"],
        projected_monthly_cost=totals["cost"] * (_SECONDS_PER_MONTH / elapsed),
        elapsed_seconds=totals["elapsed_seconds"],
    )
