"""Runner contract: agent mode, replay mode, pass-rate threshold and exit code."""

import unittest
from dataclasses import replace

from agent_evals.case import Case, ToolCall, Turn
from agent_evals.graders import ALL_GRADERS
from agent_evals.runner import run_suite


def _case(case_id="ok", **overrides):
    base = dict(
        id=case_id,
        prompt="¿cuál es el estado de mi reembolso?",
        turns=[],
        expected_tools=("kb_search",),
        expected_numbers=("1234",),
        expected_language="es",
    )
    base.update(overrides)
    return Case(**base)


def _good_turns(case):
    call = ToolCall(name="kb_search", args={"query": "reembolso"}, result="reembolso 1234: aprobado")
    return [Turn(prompt=case.prompt, response="Tu reembolso 1234 fue aprobado el dia de hoy.", tool_calls=[call])]


def _agent_turns(case):
    if case.id == "bad":
        call = ToolCall(name="kb_search", args={}, result="reembolso 1234: aprobado")
        return [Turn(prompt=case.prompt, response="Tu reembolso 9999 fue aprobado.", tool_calls=[call])]
    return _good_turns(case)


class RunnerModeTests(unittest.TestCase):
    def test_agent_mode_runs_every_grader_per_case(self):
        report = run_suite([_case()], _agent_turns)
        self.assertEqual(len(report.results), 1)
        self.assertEqual(len(report.results[0].checks), len(ALL_GRADERS))
        self.assertTrue(report.results[0].passed)

    def test_case_result_keeps_the_graded_turns(self):
        case = _case()
        report = run_suite([case], _agent_turns)
        self.assertEqual(report.results[0].turns, _agent_turns(case))

    def test_replay_mode_without_agent(self):
        case = _case()
        case = replace(case, turns=_good_turns(case))
        report = run_suite([case], agent=None)
        self.assertTrue(report.results[0].passed)

    def test_agent_and_replay_agree(self):
        case = _case("bad")
        replay_case = replace(case, turns=_agent_turns(case))
        self.assertEqual(
            run_suite([case], _agent_turns).results[0].passed,
            run_suite([replay_case], agent=None).results[0].passed,
        )

    def test_replay_without_turns_and_agent_raises(self):
        with self.assertRaises(ValueError):
            run_suite([_case()], agent=None)


class ThresholdTests(unittest.TestCase):
    def test_exit_code_zero_when_rate_meets_threshold(self):
        report = run_suite([_case("a"), _case("b", expected_numbers=("9999",))], _agent_turns, threshold=0.5)
        self.assertEqual(report.pass_rate, 0.5)
        self.assertTrue(report.ok)
        self.assertEqual(report.exit_code, 0)

    def test_exit_code_one_when_rate_below_threshold(self):
        report = run_suite([_case("a")], _agent_turns, threshold=1.0)
        self.assertTrue(report.results[0].passed)
        report = run_suite([_case("a"), _case("b", expected_numbers=("9999",))], _agent_turns, threshold=1.0)
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.pass_rate, 0.5)
        self.assertFalse(report.ok)
        self.assertEqual(report.exit_code, 1)

    def test_empty_suite_fails_safely(self):
        report = run_suite([], _agent_turns, threshold=0.0)
        self.assertEqual(report.total, 0)
        self.assertEqual(report.pass_rate, 0.0)
        self.assertEqual(report.exit_code, 1)


if __name__ == "__main__":
    unittest.main()
