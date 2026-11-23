"""Scheduler failure isolation and owned-process-group cleanup regressions."""

from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from analysis import scheduler
from analysis.matrix import DEFAULT_SETTINGS


class SchedulerFailureTests(unittest.TestCase):
    def plan(self, directory, count=2):
        tasks = [{"id": f"task-{index}", "variant_id": "baseline", "case": "do_bf16",
                  "workload": {"name": f"model-{index}", "factory": "custom:factory"},
                  "settings": deepcopy(DEFAULT_SETTINGS)} for index in range(count)]
        args = SimpleNamespace(devices=None, device="cuda:0", output=Path(directory) / "results",
            lease_db=Path(directory) / "broker.sqlite", device_affinity_file=None, profile_policy="all",
            deduplicate_effective_settings=False, resume=False, timing_isolation="exclusive",
            gpu_concurrency=1, memory_headroom_gib=0, timeout=2, scheduler="local", cpu_workers=2)
        plan = {"tasks": tasks, "options": {"profile_steps": 0, "profile_replay_steps": 0}, "code_sha256": "code"}
        return args, plan

    def job(self, directory):
        args, plan = self.plan(directory, count=1)
        return {"task": plan["tasks"][0], "options": plan["options"], "folder": str(args.output / "case"),
            "run_id": "run", "resume": False, "build": "build", "demand_key": "demand",
            "device": {"uuid": "GPU-test", "index": "0", "total": 80 * scheduler.GiB},
            "broker": str(args.lease_db), "isolation": "exclusive", "concurrency": 1, "headroom": 0,
            "timeout": 1, "cancel": str(args.output / "cancel"), "script": "worker.py"}

    def test_telemetry_timeout_is_persisted_and_independent_case_finishes(self):
        device = scheduler.Device("GPU-test", "0", 80 * scheduler.GiB)
        def worker(command, **kwargs):
            request = Path(command[-1])
            value = json.loads(request.read_text())
            scheduler.write_json(request.parent / "result.json", {"status": "ok", "task": value["task"],
                                                                  "run_id": value["run_id"]})
            return 0
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.plan(directory)
            with patch.object(scheduler, "discover_devices", return_value=[device]), \
                    patch("analysis.engine.installed_build_fingerprint", return_value="build"), \
                    patch("analysis.engine.resume_environment_fingerprint", return_value="environment"), \
                    patch.object(scheduler.Broker, "telemetry", side_effect=[subprocess.TimeoutExpired("nvidia-smi", 10), (device.total, False)]), \
                    patch.object(scheduler, "run_command", side_effect=worker) as launch:
                rows, run_id = scheduler.run_plan(args, plan)
            self.assertEqual(sorted(row["status"] for row in rows), ["error", "ok"])
            self.assertEqual(launch.call_count, 1)
            failure = next(row for row in rows if row["status"] == "error")
            self.assertIn("TimeoutExpired", failure["reason"])
            self.assertEqual(failure["run_id"], run_id)
            self.assertNotIn("admitted_utc_unix", failure["scheduling"])
            self.assertIn("nvidia-smi", Path(failure["scheduler_exception"]).read_text())
            saved = args.output / "cases" / failure["task"]["id"] / "result.json"
            self.assertEqual(json.loads(saved.read_text())["status"], "error")
            self.assertEqual(json.loads((args.output / "progress.json").read_text())["completed_unique_and_anchors"], 2)

    def test_unexpected_future_failure_keeps_the_failed_job_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.plan(directory)
            def result(job):
                if job["task"]["id"] == "task-0":
                    raise RuntimeError("supervisor transport failed")
                return {"status": "ok", "task": job["task"], "run_id": job["run_id"]}
            with patch.object(scheduler, "discover_devices", return_value=[scheduler.Device("cpu", "cpu", 0)]), \
                    patch("analysis.engine.installed_build_fingerprint", return_value="build"), \
                    patch("analysis.engine.resume_environment_fingerprint", return_value="environment"), \
                    patch.object(scheduler, "run_isolated", side_effect=result):
                rows, _ = scheduler.run_plan(args, plan)
            self.assertEqual({row["task"]["id"]: row["status"] for row in rows}, {"task-0": "error", "task-1": "ok"})
            self.assertIn("supervisor transport failed", rows[0]["reason"])

    def test_cancellation_before_admission_is_not_an_ordinary_case_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.job(directory)
            Path(job["cancel"]).parent.mkdir(parents=True)
            Path(job["cancel"]).touch()
            with patch.object(scheduler, "run_command") as launch, self.assertRaises(InterruptedError):
                scheduler.run_isolated(job)
            launch.assert_not_called()
            self.assertFalse((Path(job["folder"]) / "result.json").exists())

    def test_nonzero_exit_invalidates_fresh_success_json(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.job(directory)
            def worker(command, **kwargs):
                request = Path(command[-1])
                value = json.loads(request.read_text())
                scheduler.write_json(request.parent / "result.json", {
                    "status": "ok", "task": value["task"], "run_id": value["run_id"]})
                return 7
            with patch.object(scheduler.Broker, "telemetry", return_value=(80 * scheduler.GiB, False)), \
                    patch.object(scheduler, "run_command", side_effect=worker):
                row = scheduler._run_isolated(job)
            self.assertEqual(row["status"], "error")
            self.assertIn("exited 7", row["reason"])
            self.assertEqual(json.loads((Path(job["folder"]) / "result.json").read_text())["status"], "error")

    def test_keyboard_interrupt_cancels_the_coordinator(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.plan(directory, count=1)
            with patch.object(scheduler, "discover_devices", return_value=[scheduler.Device("cpu", "cpu", 0)]), \
                    patch("analysis.engine.installed_build_fingerprint", return_value="build"), \
                    patch("analysis.engine.resume_environment_fingerprint", return_value="environment"), \
                    patch.object(scheduler, "run_isolated", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                scheduler.run_plan(args, plan)
            self.assertEqual(len(list(args.output.glob("cancel-*"))), 1)

    def test_unverified_cleanup_keeps_the_device_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.job(directory)
            with patch.object(scheduler.Broker, "telemetry", return_value=(80 * scheduler.GiB, False)), \
                    patch.object(scheduler, "run_command", side_effect=scheduler.ProcessCleanupError("group remains active")):
                row = scheduler.run_isolated(job)
            self.assertEqual(row["status"], "error")
            self.assertTrue(row["scheduling"]["reservation_retained"])
            broker = scheduler.Broker(job["broker"])
            device = scheduler.Device(**job["device"])
            token, = broker.members(device)
            # Losing both recorded PIDs must not erase a reservation whose
            # untracked descendants were never confirmed to have exited.
            with broker.connect() as db:
                db.execute("UPDATE leases SET owner=?,child=? WHERE token=?", (999999999, 999999999, token))
            broker.release(token)
            self.assertFalse(broker.may_admit(device, True, 1))
            self.assertIsNone(broker.acquire(device, None, True, 1, 0))
            self.assertEqual(broker.members(device), [token])

    def test_interrupt_during_cleanup_propagates_and_quarantines_the_reservation(self):
        interruption = KeyboardInterrupt("second interrupt during cleanup")
        child = SimpleNamespace(pid=999999999)
        with patch.object(scheduler.subprocess, "Popen", return_value=child), \
                patch.object(scheduler, "child_exited", return_value=True), \
                patch.object(scheduler, "terminate_child", side_effect=interruption), self.assertRaises(KeyboardInterrupt) as raised:
            scheduler.run_command(["unused"], env={}, log=None, timeout=1, cancel_file="unused")
        self.assertIs(raised.exception, interruption)
        self.assertTrue(interruption._deepopt_cleanup_failed)
        with tempfile.TemporaryDirectory() as directory:
            device = scheduler.Device("GPU-test", "0", 80 * scheduler.GiB)
            broker_path = Path(directory) / "broker.sqlite"
            with patch.object(scheduler.Broker, "telemetry", return_value=(device.total, False)), \
                    patch.object(scheduler, "run_command", side_effect=interruption), self.assertRaises(KeyboardInterrupt):
                scheduler.run_external(["unused"], device=device, broker_path=broker_path, timeout=1,
                                       cancel=Path(directory) / "cancel")
            self.assertFalse(scheduler.Broker(broker_path).may_admit(device, True, 1))

    def test_cleanup_confirmation_after_sigkill_is_bounded(self):
        child = SimpleNamespace(pid=999999999, returncode=None, wait=Mock())
        with patch.object(scheduler, "process_group_active", return_value=True), \
                patch.object(scheduler.os, "killpg") as kill, \
                patch.object(scheduler.time, "monotonic", side_effect=itertools.count()), \
                patch.object(scheduler.time, "sleep"), self.assertRaises(scheduler.ProcessCleanupError):
            scheduler.terminate_child(child, grace=0)
        self.assertEqual([call.args[1] for call in kill.call_args_list], [signal.SIGTERM, signal.SIGKILL])
        child.wait.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "Benchmark workers own POSIX process groups")
    def test_exited_and_timed_out_leaders_leave_no_live_descendants_before_release(self):
        cleanup = scheduler.terminate_child
        for mode in ("exit", "timeout"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                marker = root / "descendant.pid"
                descendant = ("import os,signal,time; from pathlib import Path; "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)")
                leader = ("import subprocess,sys,time; from pathlib import Path; "
                    f"subprocess.Popen([sys.executable,'-c',{descendant!r}]); "
                    f"marker=Path({str(marker)!r})\n"
                    "while not marker.exists(): time.sleep(.01)\n" + ("sys.exit(0)" if mode == "exit" else "time.sleep(30)"))
                released_while_live = []
                def live():
                    if not marker.exists():return False
                    result = subprocess.run(["ps", "-p", marker.read_text(), "-o", "stat="], capture_output=True, text=True)
                    state = result.stdout.strip()
                    return bool(state) and not state.startswith("Z")
                release = scheduler.Broker.release
                def checked_release(broker, token):
                    released_while_live.append(live())
                    return release(broker, token)
                try:
                    with patch.object(scheduler.Broker, "telemetry", return_value=(80 * scheduler.GiB, False)), \
                            patch.object(scheduler.Broker, "release", new=checked_release), \
                            patch.object(scheduler, "terminate_child", side_effect=lambda child: cleanup(child, grace=.1)):
                        arguments = dict(device=scheduler.Device("GPU-test", "0", 80 * scheduler.GiB),
                            broker_path=root / "broker.sqlite", timeout=1, cancel=root / "cancel", log_path=root / "worker.log")
                        if mode == "timeout":
                            with self.assertRaises(TimeoutError):
                                scheduler.run_external([sys.executable, "-c", leader], **arguments)
                        else:
                            self.assertEqual(scheduler.run_external([sys.executable, "-c", leader], **arguments), 0)
                    self.assertTrue(marker.exists())
                    self.assertFalse(live())
                    self.assertEqual(released_while_live, [False])
                finally:
                    if live():os.kill(int(marker.read_text()), signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
