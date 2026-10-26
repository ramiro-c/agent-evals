"""Suite runner: gather checks per case, compute the pass rate, decide the exit code."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .case import Case, Turn
from .graders import ALL_GRADERS, CheckResult


AgentFunction = Callable[[Case], list[Turn]]


class CaseResult:
    """All grader outcomes for one case, plus the derived verdict."""

    __slots__ = ("case_id", "checks", "passed", "turns")

    def __init__(self, case_id: str, checks: Sequence[CheckResult], turns: Sequence[Turn] = ()) -> None:
        self.case_id = case_id
        self.checks: list[CheckResult] = list(checks)
        self.passed = all(check.passed for check in self.checks)
        self.turns: list[Turn] = list(turns)


class SuiteReport:
    """Aggregate report over a full suite run."""

    __slots__ = ("results", "threshold")

    def __init__(self, results: Iterable[CaseResult], threshold: float) -> None:
        self.results: list[CaseResult] = list(results)
        self.threshold = threshold

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if not result.passed)

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def ok(self) -> bool:
        # A suite that ran zero cases checks nothing: it must fail, even at threshold 0.
        return self.total > 0 and self.pass_rate >= self.threshold

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1


def run_suite(
    cases: Iterable[Case],
    agent: AgentFunction | None = None,
    threshold: float = 1.0,
) -> SuiteReport:
    """Run every case through every grader.

    ``agent`` is a callable that receives a :class:`~agent_evals.case.Case` and
    returns the conversation turns it produced. When it is omitted, the runner
    replays the turns already recorded on each case.
    """
    collected: list[CaseResult] = []
    for case in cases:
        if agent is None:
            if not case.turns:
                raise ValueError(
                    f"case {case.id!r} has no recorded turns and no agent was provided; "
                    "record turns in the case file or pass an agent"
                )
            produced = list(case.turns)
        else:
            produced = agent(case)
        checks = [grader(case, produced) for grader in ALL_GRADERS]
        collected.append(CaseResult(case.id, checks, produced))
    return SuiteReport(collected, threshold)
