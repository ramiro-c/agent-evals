"""CLI contract: exit code propagation, JSON report, agent loading from a file."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "examples" / "support_agent" / "cases.json"
PASS_ONLY = REPO_ROOT / "examples" / "support_agent" / "cases_positive.json"
AGENT = REPO_ROOT / "examples" / "support_agent" / "agent.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "agent_evals.cli", *args],
        cwd=REPO_ROOT,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not AGENT.exists():
            raise unittest.SkipTest("example support_agent not built yet")

    def test_threshold_one_on_mixed_suite_exits_nonzero(self):
        proc = _run(str(CASES), "--threshold", "1.0")
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_threshold_zero_exits_zero(self):
        proc = _run(str(CASES), "--threshold", "0.0")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_agent_mode_positive_suite_exits_zero(self):
        proc = _run(str(PASS_ONLY), "--agent", f"{AGENT}:build_agent", "--threshold", "1.0")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_json_report_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            proc = _run(str(CASES), "--threshold", "1.0", "--report", str(out))
            self.assertTrue(out.exists(), proc.stdout + proc.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["total"], 8)
            self.assertEqual(data["passed"], 1)
            self.assertEqual(data["threshold"], 1.0)
            self.assertEqual(data["exit_code"], 1)
            case_ids = {c["id"] for c in data["cases"]}
            self.assertIn("refund-status-number", case_ids)

    def test_missing_agent_for_never_recorded_case_errors(self):
        bad = {"id": "no-turns", "prompt": "p", "expected_language": "es"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps([bad]), encoding="utf-8")
            proc = _run(str(path))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("turns", (proc.stdout + proc.stderr).lower())


if __name__ == "__main__":
    unittest.main()
