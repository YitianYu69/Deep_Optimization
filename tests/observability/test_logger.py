import contextlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deepopt.logger import init_logger, logging_session, log_filename, log_output


class LoggerTests(unittest.TestCase):
    def test_names_cover_full_config_and_are_safe_and_stable(self):
        config = {"name": "../test / name", "training": {"batch_size": 8, "epochs": 2}}
        name = log_filename(config)
        self.assertEqual(name, log_filename(dict(reversed(list(config.items())))))
        self.assertEqual(Path(name).name, name)
        self.assertIn("bs8", name)
        self.assertNotEqual(name, log_filename({**config, "seed": 2}))
        self.assertNotEqual(name, log_filename({**config, "training": {"batch_size": 4, "epochs": 2}}))
        with self.assertRaises(ValueError):
            log_filename({"lr": float("nan")})
        with self.assertRaises(TypeError):
            log_filename({"model": object()})
        @dataclass
        class Settings:
            name: str = "demo"
        self.assertEqual(log_filename(Settings()), log_filename({"name": "demo"}))

    def test_console_file_output_levels_and_no_root_mutation(self):
        root = logging.getLogger()
        before = (root.level, root.handlers[:])
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with logging_session({"name": "test"}, directory, level="WARNING") as session:
                    logger = init_logger(__name__)
                    self.assertIs(logger, init_logger(__name__))
                    logger.debug("file-only debug")
                    logger.info("file-only info")
                    logger.warning("visible warning")
                    log_output(json.dumps({"loss": 1.5}))
                    logger.error("first line\nsecond line")
                text = session.path.read_text()
            self.assertEqual(json.loads(stdout.getvalue()), {"loss": 1.5})
            self.assertIn("file-only debug", text)
            self.assertNotIn("file-only", stderr.getvalue())
            self.assertEqual(text.count("visible warning"), 1)
            self.assertEqual(stderr.getvalue().count("visible warning"), 1)
            self.assertIn('{"loss": 1.5}', text)
            self.assertIn("[test_logger.py:", text)
            self.assertNotIn("\x1b", text)
            self.assertTrue(session._file.closed)
        self.assertEqual(before, (root.level, root.handlers[:]))

    def test_nested_sessions_restore_outer_and_reopening_appends(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            with logging_session({"name": "outer"}, directory) as outer:
                init_logger(__name__).info("outer-first")
                with logging_session({"name": "inner"}, directory) as inner:
                    init_logger(__name__).info("inner-only")
                init_logger(__name__).info("outer-last")
            with logging_session({"name": "outer"}, directory):
                init_logger(__name__).info("outer-appended")
            self.assertNotIn("inner-only", outer.path.read_text())
            self.assertIn("inner-only", inner.path.read_text())
            self.assertIn("outer-first", outer.path.read_text())
            self.assertIn("outer-last", outer.path.read_text())
            self.assertIn("outer-appended", outer.path.read_text())

    def test_preflight_buffer_uses_resolved_name_and_failure_flushes(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            with logging_session({"training": {"batch_size": 8}}, directory=None) as session:
                init_logger(__name__).warning("planning-warning")
                session.bind(directory, config={"training": {"batch_size": 4}})
            self.assertIn("bs4", session.path.name)
            self.assertIn("planning-warning", session.path.read_text())
            with self.assertRaisesRegex(RuntimeError, "bad trace"):
                with logging_session({"name": "failure"}, directory=None, failure_directory=directory) as failed:
                    raise RuntimeError("bad trace")
            self.assertIn("RuntimeError: bad trace", failed.path.read_text())
            self.assertTrue(failed._file.closed)

    def test_threads_with_separate_sessions_do_not_cross_write(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            def worker(index):
                with logging_session({"name": f"job{index}"}, directory, level="CRITICAL") as session:
                    init_logger(__name__).info("unique-message-%s", index)
                    return session.path
            with ThreadPoolExecutor(max_workers=2) as pool:
                paths = list(pool.map(worker, (0, 1)))
            for index, path in enumerate(paths):
                self.assertIn(f"unique-message-{index}", path.read_text())
                self.assertNotIn(f"unique-message-{1-index}", path.read_text())

    def test_configured_auto_plan_names_file_after_selected_batch(self):
        from deepopt.config import config_from_dict
        from deepopt.run import train
        config = config_from_dict({"name": "sized-log", "seed": 1,
            "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.01},
            "application": {"synthetic": {"train_samples": 12, "validation_samples": 8}}})
        plan = {"requested_batch_size": 8, "new_batch_size": 4, "mode": "auto",
                "status": "planned", "reason": "test plan", "memory_budget_bytes": 1000,
                "candidates": [], "report": {"peak_bytes_by_device": {"cpu": 900}}}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            run = Path(directory) / "run"
            with patch("deepopt.run.plan_batch_size", return_value=plan):
                train(config, output=run, progress_bar=False)
            resolved = json.loads((run / "config.json").read_text())
            self.assertEqual(resolved["training"]["batch_size"], 4)
            logs = list((run / "logs").glob("*.log"))
            self.assertEqual([p.name for p in logs], [log_filename(resolved)])
            self.assertIn("Training batch size changed from requested 8 to 4", logs[0].read_text())
            self.assertIn("Epoch 1/1", logs[0].read_text())

    def test_failed_failure_logging_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "training failed") as raised:
                with logging_session({"name": "failure"}, directory) as session:
                    session.emit = lambda record: (_ for _ in ()).throw(OSError("disk full"))
                    raise RuntimeError("training failed")
            self.assertIn("disk full", raised.exception.__notes__[0])

    def test_environment_console_level_color_and_prefix(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(stderr):
            with patch.dict(os.environ, {"DEEPOPT_LOGGING_LEVEL": "ERROR", "DEEPOPT_LOGGING_COLOR": "1",
                                         "DEEPOPT_LOGGING_PREFIX": "[recipe] ", "NO_COLOR": ""}):
                with logging_session({"name": "env"}, directory) as session:
                    init_logger(__name__).warning("hidden warning")
                    init_logger(__name__).error("visible error")
            self.assertNotIn("hidden warning", stderr.getvalue())
            self.assertIn("[recipe]", stderr.getvalue())
            self.assertIn("\x1b[31m", stderr.getvalue())
            self.assertNotIn("\x1b", session.path.read_text())


if __name__ == "__main__":
    unittest.main()
