"""Worker-owned cycles must be released before the process group is destroyed."""

from contextlib import redirect_stderr
import gc
import io
from pathlib import Path
import tempfile
import traceback
import unittest
from unittest.mock import patch
import weakref

from tests.scheduling import test_ddp_black_box as worker_module


class _CyclicResource:
    def __init__(self):
        self.owner = self


class DDPWorkerLifetimeWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        # Require the worker's explicit collection, not an incidental GC cycle.
        was_enabled = gc.isenabled()
        gc.disable()
        if was_enabled:
            self.addCleanup(gc.enable)

    def _run_lifetime_case(self, *, fail):
        resources = []
        events = []
        caught = None
        reported_traceback = io.StringIO()

        def record(phase, **details):
            events.append((phase, details))

        def exercise(rank, world, log):
            resource = _CyclicResource()
            resources.append(weakref.ref(resource))
            log("exercise.complete", rank=rank, world=world)
            if fail:
                raise AssertionError("original assertion")

        def destroy():
            self.assertEqual(len(resources), 1)
            self.assertIsNone(resources[0](), "Exercise resource survived until group destruction")
            events.append(("destroy.called", {}))

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(worker_module.torch, "set_num_threads"), \
                patch.object(worker_module.dist, "init_process_group") as initialize, \
                patch.object(worker_module.dist, "destroy_process_group", side_effect=destroy) as cleanup, \
                patch.object(worker_module, "_exercise_schedules", new=exercise), \
                redirect_stderr(reported_traceback):
            try:
                worker_module._run_worker(0, 4, directory, record)
            except AssertionError as exc:
                caught = exc
            initialize.assert_called_once()
            self.assertEqual(initialize.call_args.kwargs["init_method"], Path(directory, "group").as_uri())
            cleanup.assert_called_once()

        phases = [phase for phase, _ in events]
        self.assertLess(phases.index("exercise.complete"), phases.index("resources.release.start"))
        self.assertLess(phases.index("resources.release.start"), phases.index("resources.release.complete"))
        self.assertLess(phases.index("resources.release.complete"), phases.index("process_group.destroy.start"))
        self.assertLess(phases.index("process_group.destroy.start"), phases.index("destroy.called"))
        self.assertEqual(phases[-1], "process_group.destroy.complete")
        if fail:
            self.assertIsNotNone(caught)
            self.assertEqual(str(caught), "original assertion")
            self.assertIn("original assertion", reported_traceback.getvalue())
            self.assertIn("exercise", reported_traceback.getvalue())
            self.assertLess(phases.index("worker.error"), phases.index("resources.release.start"))
            frames = traceback.extract_tb(caught.__traceback__)
            self.assertIn("exercise", [frame.name for frame in frames])
        else:
            self.assertIsNone(caught)
            self.assertNotIn("worker.error", phases)
            self.assertEqual(reported_traceback.getvalue(), "")

    def test_success_collects_exercise_cycles_before_destroying_group(self):
        self._run_lifetime_case(fail=False)

    def test_failure_collects_traceback_owned_cycles_and_preserves_original_assertion(self):
        self._run_lifetime_case(fail=True)


if __name__ == "__main__":
    unittest.main()
