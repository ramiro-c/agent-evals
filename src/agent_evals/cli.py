"""CLI: run a golden-case suite and exit non-zero when the pass rate drops."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, TextIO

from agent_evals.case import Case, load_cases
from agent_evals.report import render_report, write_json_report
from agent_evals.runner import AgentFunction, run_suite


def _parse_agent_spec(spec: str) -> (str, str):
    if ":" not in spec:
        raise ValueError("--agent expects the format <path>:<factory_name>")
    path, factory_name = spec.rsplit(":", 1)
    if not path or not factory_name:
        raise ValueError("--agent expects the format <path>:<factory_name>")
    return path, factory_name


def _load_agent(spec: str) -> AgentFunction:
    path, factory_name = _parse_agent_spec(spec)
    module_name = "agent_evals_external_agent"
    agent_path = Path(path)
    if not agent_path.exists():
        raise FileNotFoundError(f"agent module not found: {path}")
    module_spec = importlib.util.spec_from_file_location(module_name, agent_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"could not load agent module: {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    factory = getattr(module, factory_name)
    return factory()  # type: ignore[misc, no-any-return]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-evals",
        description="Run a golden-case suite against an agent and gate on the pass rate.",
    )
    parser.add_argument("cases", help="path to the JSON file containing the golden cases")
    parser.add_argument("--threshold", type=float, default=1.0, help="minimum pass rate to exit 0 (default: 1.0)")
    parser.add_argument(
        "--agent",
        help="agent factory as <path>:<factory_name>; omitted means replay the turns recorded in each case",
    )
    parser.add_argument("--report", help="write a JSON report to this path")
    return parser


def main(argv: TextIO | list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = load_cases(args.cases)
        agent = _load_agent(args.agent) if args.agent else None
        report = run_suite(cases, agent=agent, threshold=args.threshold)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render_report(report))
    if args.report:
        write_json_report(report, args.report)
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
