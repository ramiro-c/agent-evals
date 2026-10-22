"""Rendering: a human-readable table for the terminal and a JSON report for CI."""

from __future__ import annotations

import json
from pathlib import Path

from .runner import SuiteReport

_WIDTH = 96


def render_report(report: SuiteReport) -> str:
    """Human-readable summary: one line per case + one line per failing check."""
    lines = [
        f"SUITE  pass {report.passed}/{report.total}  rate {report.pass_rate:.2%}  "
        f"threshold {report.threshold:.2%}  exit {report.exit_code}",
        "-" * _WIDTH,
    ]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"[{status}] {result.case_id}")
        if not result.passed:
            for check in result.checks:
                if not check.passed:
                    lines.append(f"       x {check.check}: {check.detail}")
    lines.append("-" * _WIDTH)
    if report.ok:
        lines.append("OK     pass rate meets the threshold")
    else:
        lines.append("FAIL   pass rate is below the threshold")
    return "\n".join(lines)


def write_json_report(report: SuiteReport, path: str | Path) -> Path:
    """Machine-readable mirror of the report, for logs, dashboards and regates."""
    payload = {
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "pass_rate": report.pass_rate,
        "threshold": report.threshold,
        "exit_code": report.exit_code,
        "cases": [
            {
                "id": result.case_id,
                "passed": result.passed,
                "checks": [
                    {"check": check.check, "passed": check.passed, "detail": check.detail}
                    for check in result.checks
                ],
            }
            for result in report.results
        ],
    }
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
