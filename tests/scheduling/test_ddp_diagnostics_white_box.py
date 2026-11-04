"""Diagnostic capture preserves native output and cannot turn a failure into a pass."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.scheduling._ddp_diagnostics import create_log_directory, print_rank_failures


class DDPDiagnosticsWhiteBoxTests(unittest.TestCase):
    def test_log_directories_are_unique_and_persist_under_configured_artifact_root(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as rendezvous:
            with patch.dict(os.environ, {"DEEPOPT_DDP_LOG_DIR": root}):
                first = create_log_directory(rendezvous, 4)
                second = create_log_directory(rendezvous, 4)
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, Path(root))
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_without_artifact_root_logs_live_with_temporary_rendezvous(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"DEEPOPT_DDP_LOG_DIR": ""}):
                logs = create_log_directory(directory, 2)
            self.assertEqual(logs.parent, Path(directory))

    def test_worker_capture_includes_native_streams_and_restores_descriptors_on_exception(self):
        # Run the descriptor redirection in a child so the test runner's own
        # capture streams and fault handler are never replaced.
        with tempfile.TemporaryDirectory() as directory:
            script = f"""
import faulthandler
import os
import sys
from tests.scheduling._ddp_diagnostics import rank_diagnostics

before = faulthandler.is_enabled()
print("before capture")
try:
    with rank_diagnostics({directory!r}, 0, 2) as record:
        assert faulthandler.is_enabled()
        print("python stdout")
        print("python stderr", file=sys.stderr)
        os.write(1, b"native stdout\\n")
        os.write(2, b"native stderr\\n")
        record("train_epoch.start", accumulation=4, enabled=True)
        raise ValueError("original failure")
except ValueError as exc:
    assert str(exc) == "original failure"
assert faulthandler.is_enabled() == before
print("after capture")
print("after stderr", file=sys.stderr)
"""
            completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                       text=True, check=True, timeout=15)
            self.assertEqual(completed.stdout.splitlines(), ["before capture", "after capture"])
            self.assertEqual(completed.stderr.strip(), "after stderr")
            captured = Path(directory, "rank-0.log").read_text()
            for message in ("python stdout", "python stderr", "native stdout", "native stderr"):
                self.assertIn(message, captured)
            records = [json.loads(line) for line in captured.splitlines() if line.startswith("{")]
            self.assertEqual(records[-1]["phase"], "train_epoch.start")
            self.assertEqual((records[-1]["rank"], records[-1]["world"]), (0, 2))
            self.assertEqual(records[-1]["accumulation"], 4)
            self.assertTrue(records[-1]["enabled"])
            self.assertNotIn("worker.complete", captured)

    def test_failure_report_limits_tail_and_handles_rank_without_log(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "rank-0.log").write_text("start\n" + "x" * 20000 + "\nnative abort\n")
            output = io.StringIO()
            processes = [SimpleNamespace(pid=100, exitcode=-6), SimpleNamespace(pid=101, exitcode=None)]
            print_rank_failures(processes, directory, tail_bytes=64, stream=output)
            report = output.getvalue()
            self.assertIn("rank 0: pid=100, exitcode=-6", report)
            self.assertIn("native abort", report)
            self.assertNotIn("start\n", report)
            self.assertIn("Rank log unavailable", report)
            self.assertLess(len(report), 1000)


if __name__ == "__main__":
    unittest.main()
