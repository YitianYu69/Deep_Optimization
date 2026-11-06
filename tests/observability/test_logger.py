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

from deepopt.config import ConfigError, _logging_config_from_dict, get_config, load_config, reset_config
from deepopt.logger import init_logger, logging_session, log_filename, log_output


class LoggerTests(unittest.TestCase):
    def setUp(self):
        previous = get_config()
        reset_config()
        self.addCleanup(lambda: load_config(previous) if previous is not None else reset_config())

    def _configure_color(self, color):
        return load_config({"name": "logger-color", "seed": 1,
                            "training": {"epochs": 1, "batch_size": 2, "learning_rate": .01},
                            "logging": {"color": color}})

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
            self.assertIn("\x1b[34m", stderr.getvalue())
            self.assertNotIn("\x1b", session.path.read_text())

    def test_default_blue_is_uniform_across_console_levels(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), patch.dict(
                os.environ, {"DEEPOPT_LOGGING_COLOR": "1", "NO_COLOR": ""}):
            with logging_session(level="DEBUG") as session:
                logger = init_logger(__name__)
                for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
                    logger.log(level, "same color at level %s", level)
                self.assertEqual(session.color, "blue")
        lines = stderr.getvalue().splitlines()
        self.assertEqual(sum("same color at level" in line for line in lines), 5)
        self.assertTrue(all(line.startswith("\x1b[34m") and line.endswith("\x1b[0m") for line in lines))

    def test_configured_named_colors_are_case_normalized(self):
        for color, code in (("black", 30), ("red", 31), ("green", 32), ("yellow", 33),
                            ("blue", 34), ("magenta", 35), ("cyan", 36), ("white", 37)):
            with self.subTest(color=color):
                configured = self._configure_color(color.upper())
                self.assertEqual(configured.logging["color"], color)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), patch.dict(
                        os.environ, {"DEEPOPT_LOGGING_COLOR": "1", "NO_COLOR": ""}):
                    with logging_session() as session:
                        init_logger(__name__).error("configured color")
                        self.assertEqual(session.color, color)
                self.assertIn(f"\x1b[{code}m", stderr.getvalue())

    def test_session_color_override_and_snapshot_are_independent_of_global_changes(self):
        self._configure_color("red")
        metadata = {"logging": {"color": "GrEeN"}}
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), patch.dict(
                os.environ, {"DEEPOPT_LOGGING_COLOR": "1", "NO_COLOR": ""}):
            with logging_session(metadata) as inherited:
                self.assertEqual(inherited.color, "green")
                self._configure_color("yellow")
                init_logger(__name__).warning("snapshot stays green")
                with logging_session(metadata, color="CyAn") as override:
                    self.assertEqual(override.color, "cyan")
                    self._configure_color("white")
                    init_logger(__name__).error("explicit cyan")
                init_logger(__name__).warning("outer restored green")
        lines = stderr.getvalue().splitlines()
        self.assertTrue(next(line for line in lines if "snapshot stays green" in line).startswith("\x1b[32m"))
        self.assertTrue(next(line for line in lines if "explicit cyan" in line).startswith("\x1b[36m"))
        self.assertTrue(next(line for line in lines if "outer restored green" in line).startswith("\x1b[32m"))
        self.assertEqual(metadata, {"logging": {"color": "GrEeN"}})
        self.assertEqual(get_config().logging["color"], "white")

    def test_standalone_logging_resolves_current_global_color(self):
        stderr = io.StringIO()
        logger = init_logger(__name__)
        with contextlib.redirect_stderr(stderr), patch.dict(
                os.environ, {"DEEPOPT_LOGGING_COLOR": "1", "NO_COLOR": ""}):
            logger.error("default blue")
            self._configure_color("red")
            logger.error("global red")
            self._configure_color("green")
            logger.error("global green")
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        for line, code in zip(lines, (34, 31, 32)):
            self.assertTrue(line.startswith(f"\x1b[{code}m"))

    def test_named_color_respects_terminal_and_no_color_controls(self):
        cases = (("blue", "auto", True, "", True),
                 ("yellow", "auto", False, "", False),
                 ("magenta", "1", False, "", True),
                 ("red", "0", True, "", False),
                 ("red", "1", True, "1", False),
                 ("NoNe", "1", True, "", False))
        for color, mode, is_tty, no_color, enabled in cases:
            with self.subTest(color=color, mode=mode, tty=is_tty, no_color=no_color):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), patch.object(stderr, "isatty", return_value=is_tty), \
                        patch.dict(os.environ, {"DEEPOPT_LOGGING_COLOR": mode, "NO_COLOR": no_color}):
                    with logging_session({"logging": {"color": color}}):
                        init_logger(__name__).error("terminal policy")
                self.assertEqual("\x1b[" in stderr.getvalue(), enabled)

    def test_invalid_colors_are_rejected_by_config_and_session(self):
        for color in (None, True, 34, "orange", "", ["blue"]):
            with self.subTest(color=color):
                with self.assertRaises(ConfigError):
                    _logging_config_from_dict({"color": color})
                with self.assertRaises(ConfigError):
                    with logging_session({"logging": {"color": color}}):
                        self.fail("Invalid metadata color was accepted")
                with self.assertRaises(ConfigError):
                    with logging_session(color=color):
                        self.fail("Invalid explicit color was accepted")


if __name__ == "__main__":
    unittest.main()
