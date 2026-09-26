"""``tokengov`` command-line interface.

Subcommands:
    prices              Show the registered model pricing table.
    report [--input F]  Render a cost report from an exported ledger JSON
                        (as produced by ``Report.to_json``), or from the
                        bundled sample run.
    lint <path>         Lint a prompt file for bloat, with token savings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .lint import lint_file, total_savings
from .pricing import PRICING
from .report import Report, ReportExport, ReportRow, build_report


def _cmd_prices(_args: argparse.Namespace) -> int:
    rows = sorted(PRICING.values(), key=lambda p: p.input_per_million)
    header = ("model", "$/1M in", "$/1M out", "$/1M cached")
    body = [
        (
            p.model,
            f"{p.input_per_million:.2f}",
            f"{p.output_per_million:.2f}",
            f"{p.effective_cached_input():.2f}",
        )
        for p in rows
    ]
    widths = [max(len(header[i]), *(len(r[i]) for r in body)) for i in range(4)]
    fmt = "  ".join(f"{{{i}:<{widths[i]}}}" for i in range(4))
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in body:
        print(fmt.format(*r))
    print("\nExtend with: register_pricing(ModelPricing(...))")
    return 0


def _report_from_dict(data: dict) -> Report:
    export = ReportExport.model_validate(data)  # fail loudly on bad ledgers
    mk = lambda rows: tuple(  # noqa: E731
        ReportRow(
            label=r.label,
            requests=r.requests,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            cached_input_tokens=r.cached_input_tokens,
            cost=r.cost,
        )
        for r in rows
    )
    t = export.totals
    return Report(
        rows_by_model=mk(export.by_model),
        rows_by_workflow=mk(export.by_workflow),
        total_cost=t.cost,
        total_requests=t.requests,
        total_input_tokens=t.input_tokens,
        total_output_tokens=t.output_tokens,
        projected_monthly_cost=export.projected_monthly_cost,
        elapsed_seconds=export.elapsed_seconds,
    )


def _cmd_report(args: argparse.Namespace) -> int:
    if args.input:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        report = _report_from_dict(data)
    else:
        # No export given: render whatever this process has tracked so far.
        report = build_report()
        if report.total_requests == 0:
            print(
                "Ledger is empty in this process. Run with --input <ledger.json> "
                "to render an exported report, or track calls before invoking "
                "'tokengov report'.",
                file=sys.stderr,
            )
            return 1
    if args.json:
        print(report.to_json())
    elif args.csv:
        print(report.to_csv())
    else:
        print(report.render())
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    findings = lint_file(
        args.path,
        system_prompt_ceiling=args.max_tokens,
    )
    if not findings:
        print(f"{args.path}: no findings — prompt is lean.")
        return 0
    for f in findings:
        print(f"{args.path}:{f}")
    print(f"\n{len(findings)} finding(s), ~{total_savings(findings)} tokens recoverable.")
    return 1 if any(f.severity == "error" for f in findings) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokengov",
        description="Treat LLM spend like cloud spend.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prices = sub.add_parser("prices", help="Show model pricing table.")
    p_prices.set_defaults(func=_cmd_prices)

    p_report = sub.add_parser("report", help="Render a cost report.")
    p_report.add_argument(
        "--input",
        metavar="LEDGER.json",
        help="Exported ledger JSON (Report.to_json output).",
    )
    p_report.add_argument("--json", action="store_true", help="Emit JSON.")
    p_report.add_argument("--csv", action="store_true", help="Emit CSV.")
    p_report.set_defaults(func=_cmd_report)

    p_lint = sub.add_parser("lint", help="Lint a prompt file for bloat.")
    p_lint.add_argument("path", help="Path to the prompt file.")
    p_lint.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Token ceiling for the prompt (default: 512).",
    )
    p_lint.set_defaults(func=_cmd_lint)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
