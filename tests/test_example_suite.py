"""Golden example suite: 1 clean case, 7 single-failure cases, agent determinism."""

import importlib.util
import unittest
from pathlib import Path

from agent_evals.case import load_cases
from agent_evals.runner import run_suite

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = REPO_ROOT / "examples" / "support_agent" / "cases.json"
AGENT_PATH = REPO_ROOT / "examples" / "support_agent" / "agent.py"

FILLING_REASON = {
    "invented-number": "numeric_grounding",
    "unsupported-claim": "fact_grounding",
    "wrong-tool": "tool_usage",
    "forbidden-phrase": "no_forbidden_content",
    "wrong-language": "language",
    "forbidden-tool": "tool_usage",
    "wrong-args": "tool_args",
}


class ExampleSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(CASES_PATH)
        cls.report = run_suite(cls.cases, agent=None)

        spec = importlib.util.spec_from_file_location("support_agent", AGENT_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_suite_shape(self):
        self.assertEqual(len(self.cases), 8)
        self.assertEqual(self.report.total, 8)
        self.assertEqual(self.report.passed, 1)
        self.assertEqual(self.report.failed, 7)
        self.assertEqual(self.report.exit_code, 1)
        self.assertEqual(run_suite(self.cases, agent=None, threshold=0.0).exit_code, 0)

    def test_each_bad_case_fails_only_for_its_expected_reason(self):
        by_id = {r.case_id: r for r in self.report.results}
        for case_id, reason in FILLING_REASON.items():
            with self.subTest(case=case_id):
                result = by_id[case_id]
                self.assertFalse(result.passed, case_id)
                failed = [c for c in result.checks if not c.passed]
                self.assertTrue(
                    all(c.check == reason for c in failed),
                    f"{case_id}: failures {[c.check for c in failed]} expected all '{reason}'",
                )

    def test_clean_case_passes_every_grader(self):
        by_id = {r.case_id: r for r in self.report.results}
        result = by_id["refund-status-number"]
        self.assertTrue(result.passed)
        self.assertTrue(all(c.passed for c in result.checks))

    def test_agent_reproduces_recorded_turns_exactly(self):
        agent = self.module.build_agent()
        for case in self.cases:
            with self.subTest(case=case.id):
                self.assertEqual(tuple(agent(case)), tuple(case.turns))


import os


if __name__ == "__main__":
    unittest.main()
