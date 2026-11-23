"""Transport-only cohort validation; scalar fixtures are not CUDA evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from examples.vae import benchmark_packed_offload as offload
from examples.vae import execution_memory, native_execution
from tests.benchmarks.test_vae_execution_benchmark import _successful_worker
from tests.benchmarks.test_vae_memory_comparison import audited_row
from tests.benchmarks.test_vae_native_execution import _receipt


def manifest(directory):
    root = Path(directory)
    target = root / "installed"
    package = {"__init__.py": b"# metadata fixture, never imported\n"}
    digest = hashlib.sha256()
    for name, data in package.items():
        path = target / "deepopt" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest.update(name.encode() + b"\0" + data + b"\0")
    wheel = root / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in package.items():
            archive.writestr("deepopt/" + name, data)
        archive.writestr("deep_optimization-1.0.dist-info/METADATA", "Name: deep-optimization\nVersion: 1.0+offload.fixture\n")
    policies = {name: {**offload.POLICY_DEFAULTS, "enabled": name != "resident",
        "policy": "demand" if name == "demand_offload" else "memory_first_overlap",
        "host_pinned_bytes": 2**20, "device_staging_bytes": 2**20, "d2h_inflight_bytes": 2**20}
        for name in offload.ARMS}
    value = {"schema_version": 1, "experiment_profile": offload.PROFILE,
        "candidate": {"wheel": {"path": str(wheel), "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()},
            "install_root": str(target), "expected_installed": {"version": "1.0+offload.fixture",
                "source_sha256": digest.hexdigest(), "files_hashed": 1}}, "policies": policies}
    path = root / "offload-experiment.json"
    path.write_text(json.dumps(value))
    return path, value


def arguments(directory, *, extra=()):
    source, _ = _receipt(directory)
    pin, _ = manifest(directory)
    return offload.parser().parse_args(["--from-result", str(source), "--experiment-manifest", str(pin),
        "--warmup", "1", "--updates", "2", "--trials", "1", "--output", str(Path(directory) / "result.json"), *extra])


def runtime_metadata(policy, *, cleanup=False, updates=3):
    plans = [{"region_id": "fwd:bwd", "selected_bytes": 256, "selected_bundle_ids": ["affine_0"]}] if policy["enabled"] else []
    host = {kind + "_" + field: (256 if policy["enabled"] and kind == "pinned" and field == "peak_bytes" else 0)
        for kind in ("pinned", "pageable") for field in ("active_bytes", "cached_bytes", "peak_bytes")}
    if policy.get("version", 1) == 2:
        host.update(pinned_physical_capacity_bytes=256 if policy["enabled"] else 0,
            pinned_physical_peak_bytes=256 if policy["enabled"] else 0,
            pinned_accounting_verified=True, pinned_allocator_domain_id=123)
    device = {"staging_bytes": 0, "staging_peak_bytes": 256 if policy["enabled"] else 0,
        "pending_retire_bytes": 0, "d2h_inflight_bytes": 0, "peak_d2h_inflight_bytes": 256 if policy["enabled"] else 0}
    counters = dict.fromkeys(("d2h_bytes", "h2d_bytes", "d2h_copies", "h2d_copies"), 256 if policy["enabled"] else 0)
    lifecycle = dict.fromkeys(("live_forwards", "backward_tasks", "graph_leases", "task_leases", "dma_leases"), 0)
    lifecycle["quiescent"] = True
    managers = [] if cleanup else [{"policy": dict(policy), "plans": plans, "host": host, "device": device,
        "counters": counters, "lifecycle": lifecycle, "testing_backend": False, "error": None,
        "profile_frozen": True, "profile_learning": False, "calibrated": False, "cold": True,
        "admission_history_truncated": False, "admission_history": [
            {"region_id": "fwd:bwd", "selected_bundle_ids": ["affine_0"], "resident_fallback": False}
            for _ in range(updates)] if policy["enabled"] else []}]
    return {"schema_version": 1, "available": True, "snapshot": {"managers": managers,
        "frozen": True,
        "totals": {"host": host, "device": device, "counters": counters, "lifecycle": lifecycle}}}


def successful(plan, arm, *, block=1):
    worker = _successful_worker(plan["workers"][arm], "native", identity=plan["installed"])
    worker.update(block=block, offload_arm=arm, worker_exit_code=0)
    policy = plan["manifest"]["policies"][arm]
    if policy.get("version", 1) == 2:
        worker["host_capacity_preflight"] = execution_memory.host_capacity_admission(
            policy["host_pinned_bytes"] + policy["host_pageable_bytes"] if policy["enabled"] else 0,
            headroom_bytes=worker["host_capacity_policy"]["reserved_headroom_bytes"],
            snapshot={"status": "available", "available_bytes": 2**40})
    worker["packed_offload_runtime"] = runtime_metadata(policy)
    worker["packed_offload_profile"] = runtime_metadata(policy, updates=1)
    worker["packed_offload_cleanup"] = runtime_metadata(policy, cleanup=True)
    for name in ("rng_before_warmup_sha256", "rng_after_warmup_sha256", "rng_after_measurement_sha256"):
        worker[name] = "a" * 64
    audit = audited_row(pid=100 + 3 * block + offload.ARMS.index(arm))
    worker["memory_audit"] = audit["memory_audit"]
    worker["memory_audit"]["snapshots"]["after_postchecks"]["host_process"] = {
        "current_rss_bytes": 4096, "peak_rss_bytes": 8192}
    return worker


def configuration_successful(plan, arm):
    return {**deepcopy(plan["workers"][arm]), "status": "completed", "installed": deepcopy(plan["installed"]),
        "scope": "configuration_preflight_only", "execution_completed": False, "timing_eligible": False,
        "installed_stable": True, "helper_source_stable": True,
        "configuration_preflight": {"requested": True, "valid": True,
            "checks": dict.fromkeys(("installed_identity", "arm_policy_pins", "typed_config", "backend_config", "single_process_execution"), True),
            "no_model_allocated": True, "no_input_allocated": True, "cuda_execution": False}}


class PackedOffloadPlanningTests(unittest.TestCase):
    def test_six_permutation_orders_and_declared_demand_only_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = offload.benchmark_plan(arguments(directory, extra=("--trials", "6")))
            self.assertEqual(plan["order"], [list(row) for row in offload.BALANCED_ORDERS])
            for arm in offload.ARMS:
                self.assertEqual([sum(row[index] == arm for row in plan["order"]) for index in range(3)], [2, 2, 2])
        with tempfile.TemporaryDirectory() as directory:
            plan = offload.benchmark_plan(arguments(directory, extra=("--trials", "6", "--arms", "demand-only")))
            self.assertEqual(plan["active_arms"], list(offload.ARMS[:2]))
            self.assertEqual(plan["order"], [list(offload.ARMS[:2]), list(reversed(offload.ARMS[:2]))] * 3)
            rows = [successful(plan, arm, block=block) for block in range(1, 7) for arm in offload.ARMS[:2]]
            report = offload.aggregate(rows, plan)
            self.assertTrue(report["valid"], report["invalid_reasons"])
            self.assertTrue(report["statistical_design"]["balanced_matrix_complete"])
            self.assertNotIn("predictive_offload", report["arms"])

    def test_actual_native_options_are_byte_pinned_and_resolved_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            plan = offload.benchmark_plan(args)
            path, log, command = offload._pinned_command(args, plan, Path(directory), "pinned", "resident")
            worker_args = native_execution.parser().parse_args(command[3:])
            pin = json.loads(worker_args.arm_manifest.read_text())
            self.assertIsNotNone(pin["native_options_sha256"])
            self.assertEqual(pin["native_options_sha256"], hashlib.sha256(worker_args.native_options_json.read_bytes()).hexdigest())
            worker = native_execution.execution_plan(worker_args)
            self.assertEqual(worker["settings"], plan["workers"]["resident"]["settings"])
            self.assertEqual(worker["resolved_native_options_sha256"], pin["resolved_native_options_sha256"])
            pin["resolved_native_options_sha256"] = "f" * 64
            worker_args.arm_manifest.write_text(json.dumps(pin))
            with self.assertRaisesRegex(ValueError, "Resolved native compiler options"):
                native_execution.execution_plan(worker_args)

    def test_v2_physical_semantics_are_explicit_and_v1_wire_shape_is_unchanged(self):
        first = offload.normalize_policy(offload.POLICY_DEFAULTS)
        second = offload.normalize_policy({**first, "version": 2})
        self.assertNotIn("version", first)
        self.assertEqual(second["version"], 2)
        with self.assertRaises(ValueError):
            offload.normalize_policy({"version": 3})

    def test_three_arms_same_candidate_and_math_with_strict_separate_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            original = args.from_result.read_bytes()
            plan = offload.benchmark_plan(args)
            self.assertEqual(args.from_result.read_bytes(), original)
        identities = []
        for arm in offload.ARMS:
            row = successful(plan, arm)
            self.assertEqual(offload.worker_errors(row, plan["manifest"]["policies"][arm], row["settings"], plan["installed"]), [])
            identities.append(offload._identity(row))
            settings = plan["workers"][arm]["settings"]
            self.assertIsNone(settings["runtime"]["compile"]["mode"])
            self.assertEqual(settings["runtime"]["activation_compression"]["group_size"], 1024)
            self.assertEqual(row["requested_input_shape"], [128, 3, 224, 224])
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[0], identities[2])
        self.assertFalse(plan["performance_policy_promoted"])

    def test_stdlib_dry_run_and_policy_file_pin_preserve_saved_mode_options_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            command = [sys.executable, "-S", str(Path(offload.__file__)), "--from-result", str(args.from_result),
                "--experiment-manifest", str(args.experiment_manifest), "--dry-run"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["warmup"], 5)
            self.assertEqual(plan["updates"], 10)
            self.assertEqual(plan["trials_requested"], 6)
            self.assertFalse(plan["CUDA_execution_proven"])
            policy = Path(directory) / "policy.json"
            policy.write_text(json.dumps(plan["manifest"]["policies"]["demand_offload"]))
            arm = Path(directory) / "arm.json"
            arm.write_text(json.dumps({"schema_version": 1, "arm": "demand_offload",
                "expected_installed": plan["manifest"]["candidate"]["expected_installed"],
                "wheel_sha256": plan["manifest"]["candidate"]["wheel"]["sha256"], "native_options_sha256": None,
                "packed_offload_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}))
            worker_args = native_execution.parser().parse_args(["--from-result", str(args.from_result),
                "--packed-offload-json", str(policy), "--arm-manifest", str(arm), "--benchmark"])
            worker_plan = native_execution.execution_plan(worker_args)
            self.assertIsNotNone(worker_plan["target_settings"]["runtime"]["compile"]["mode"])
            self.assertEqual(worker_plan["settings"]["runtime"]["packed_activation_offload"],
                worker_plan["settings"]["runtime"]["compile"]["options"]["packed_activation_offload"])
            policy.write_text(json.dumps({"enabled": False}))
            with self.assertRaisesRegex(ValueError, "pin"):
                native_execution.execution_plan(worker_args)

    def test_policy_budgets_and_manifest_arms_fail_closed(self):
        cases = ({"enabled": 1}, {"host_pinned_bytes": True}, {"policy": "fast"},
            {"unknown": 1}, {"latency_target_percent": float("nan")}, {"enabled": True},
            {"copy_chunk_bytes": 0}, {"host_pageable_bytes": -1})
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                offload.normalize_policy(value)
        with tempfile.TemporaryDirectory() as directory:
            path, value = manifest(directory)
            value["policies"]["predictive_offload"]["device_staging_bytes"] += 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "identical"):
                offload.load_manifest(path)


class PackedOffloadAggregationTests(unittest.TestCase):
    def cohort(self, directory):
        plan = offload.benchmark_plan(arguments(directory))
        return plan, [successful(plan, arm) for arm in offload.ARMS]

    def test_memory_minimum_keeps_slow_demand_point_on_descriptive_frontier(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, rows = self.cohort(directory)
        for row, peak, wall in zip(rows, (4000, 2000, 2500), (10, 30, 12)):
            row["memory"]["total"]["peak_allocated_bytes"] = peak
            row["measurement"]["wall_ms_per_update"] = wall
        summary = offload.aggregate(rows, plan)
        self.assertTrue(summary["valid"], summary["invalid_reasons"])
        self.assertEqual(summary["minimum_memory_arms"], ["demand_offload"])
        self.assertEqual(len(summary["frontier"]), 3)
        self.assertFalse(summary["frontier_qualified"])
        self.assertFalse(summary["performance_policy_promoted"])
        self.assertIn("host_pinned_peak_bytes_upper_sum", summary["arms"]["demand_offload"]["metrics"])
        self.assertEqual(summary["paired_comparisons"]["demand_offload"]["independent_pairs"], 1)
        self.assertEqual(summary["paired_comparisons"]["demand_offload"]["metrics"]["wall_ms_per_update_overhead_percent"]["median"], 200)
        self.assertEqual(summary["blocks"][0]["paired_vs_resident"]["demand_offload"]["total_peak_allocated_bytes_saved"], 2000)

    def test_failed_and_malformed_peers_do_not_erase_completed_resident(self):
        for bad in ("OOM", "nested"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                plan, rows = self.cohort(directory)
                if bad == "OOM":
                    rows[1].update(status="failed", error="CUDA out of memory", worker_exit_code=1)
                else:
                    rows[1]["measurement"] = []
                summary = offload.aggregate(rows, plan)
                self.assertFalse(summary["valid"])
                self.assertEqual(summary["arms"]["resident"]["successful_trials"], 1)
                self.assertEqual(summary["arms"]["predictive_offload"]["successful_trials"], 1)
                self.assertEqual(summary["arms"]["demand_offload"]["successful_trials"], 0)
                self.assertFalse(summary["paired_comparisons"])

    def test_all_eligible_requires_full_static_inventory_and_every_actual_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, rows = self.cohort(directory)
        row = rows[1]
        for location in (row["settings"], row["runtime_config"], plan["workers"]["demand_offload"]["settings"]):
            location["runtime"]["compile"]["options"]["packed_offload_selection_mode"] = "all_eligible"
        row["packed_offload_inventory"] = {"status": "available", "all_eligible_selected": True,
            "eligible_history_bytes": 256, "selected_history_bytes": 256,
            "eligible_history_count": 1, "selected_history_count": 1, "eligible_inventory_sha256": "1" * 64}
        def errors():
            return offload.worker_errors(row, plan["manifest"]["policies"]["demand_offload"],
                plan["workers"]["demand_offload"]["settings"], plan["installed"])
        self.assertNotIn("all_eligible_inventory_incomplete", errors())
        row["packed_offload_inventory"]["selected_history_bytes"] = 128
        self.assertIn("all_eligible_inventory_incomplete", errors())
        row["packed_offload_inventory"]["selected_history_bytes"] = 256
        row["packed_offload_runtime"]["snapshot"]["managers"][0]["admission_history"][-1].update(
            selected_bundle_ids=[], resident_fallback=True)
        self.assertIn("all_eligible_actual_admission_incomplete", errors())

    def test_missing_audits_name_failed_offload_arms_and_keep_resident_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, rows = self.cohort(directory)
        for row in rows[1:]:
            row.update(status="failed", worker_exit_code=1, error="ConfigError: configuration rejected",
                failure_context={"stage": "setup", "phase": "config_validation"})
            row["memory_audit"]["complete"] = False
            row["memory_audit"]["snapshots"] = {}
        summary = offload.aggregate(rows, plan)
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["arms"]["resident"]["successful_trials"], 1)
        self.assertIn("wall_ms_per_update", summary["arms"]["resident"]["metrics"])
        errors = summary["memory_isolation"]["invalid_reasons"]
        self.assertTrue(any(error.startswith("1:demand_offload:memory_audit_") for error in errors))
        self.assertTrue(any(error.startswith("1:predictive_offload:memory_audit_") for error in errors))
        self.assertFalse(any(error.startswith(("1:native:", "1:resident:")) for error in errors))
        self.assertTrue(all(row["execution_mode"] == "native" for row in rows))
        self.assertNotIn("arm", rows[0], "aggregation must not mutate a raw worker receipt")

    def test_cleanup_budget_selection_and_RNG_drift_refuse_comparison(self):
        for key in ("cleanup", "budget", "selection", "rng", "math", "transport", "unfrozen", "actual_admission", "truncated"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                plan, rows = self.cohort(directory)
                row = rows[2]
                manager = row["packed_offload_runtime"]["snapshot"]["managers"][0]
                if key == "cleanup":
                    row["packed_offload_cleanup"]["snapshot"]["totals"]["lifecycle"]["dma_leases"] = 1
                elif key == "budget":
                    manager["device"]["staging_peak_bytes"] = 2**30
                elif key == "selection":
                    manager["plans"][0]["selected_bundle_ids"] = ["affine_other"]
                elif key == "rng":
                    row["rng_after_measurement_sha256"] = "b" * 64
                elif key == "math":
                    row["settings"]["runtime"]["compile"]["options"]["require_full_aot"] = False
                elif key == "transport":
                    row["packed_offload_runtime"]["snapshot"]["totals"]["counters"]["h2d_bytes"] = 0
                elif key == "unfrozen":
                    row["packed_offload_profile"]["snapshot"]["frozen"] = False
                elif key == "actual_admission":
                    manager["admission_history"][-1].update(selected_bundle_ids=[], resident_fallback=True)
                else:
                    manager["admission_history_truncated"] = True
                summary = offload.aggregate(rows, plan)
                self.assertFalse(summary["valid"], key)
                self.assertFalse(summary["frontier_qualified"])

    def test_recovered_retirement_error_refuses_comparison_even_when_quiescent(self):
        for phase in ("measured", "cleanup"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                plan, rows = self.cohort(directory)
                row = rows[1]
                manager = row["packed_offload_runtime"]["snapshot"]["managers"][0]
                if phase == "cleanup":
                    manager = dict(manager)
                    row["packed_offload_cleanup"]["snapshot"]["managers"] = [manager]
                manager["cleanup_errors"] = [{"role": "consumer_retirement_event", "exception_class": "RuntimeError",
                    "message": "recovered by owned stream drain"}]
                self.assertTrue(manager["lifecycle"]["quiescent"])
                reasons = offload.worker_errors(row, plan["manifest"]["policies"]["demand_offload"],
                    plan["workers"]["demand_offload"]["settings"], plan["installed"])
                self.assertIn("runtime_cleanup_errors", reasons)
                self.assertFalse(offload.aggregate(rows, plan)["valid"])

    def test_profiles_are_separate_workers_and_failures_preserve_clean_raw_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory, extra=("--profile-after-measurement", "--profile-memory"))
            plan = offload.benchmark_plan(args)
            seen = []
            def run(command, **kwargs):
                mode = ("configuration" if "--configuration-preflight" in command else
                    "diagnostic" if "--profile-after-measurement" in command else "clean")
                path = Path(command[command.index("--output") + 1])
                arm_pin = json.loads(Path(command[command.index("--arm-manifest") + 1]).read_text())
                arm = arm_pin["arm"]
                seen.append((mode, arm))
                row = configuration_successful(plan, arm) if mode == "configuration" else successful(plan, arm)
                path.write_text(json.dumps(row))
                return SimpleNamespace(returncode=0)
            with patch.object(offload.subprocess, "run", side_effect=run), \
                    patch("benchmarks.compiler.installed_runtime.child_environment", return_value={}), \
                    redirect_stdout(io.StringIO()):
                code = offload.run_benchmark(args)
            receipt = json.loads(args.output.read_text())
            self.assertEqual(code, 0, receipt.get("error"))
            self.assertEqual(seen, [(kind, arm) for kind in ("configuration", "clean", "diagnostic") for arm in offload.ARMS])
            self.assertTrue(receipt["configuration_preflight"]["valid"])
            self.assertEqual(len(receipt["results"]), 3)
            self.assertEqual(len(receipt["diagnostics"]), 3)
            self.assertTrue(receipt["summary"]["valid"])
            self.assertTrue(all("result" in row for row in receipt["results"]))
            self.assertFalse([path for path in Path(directory).rglob("*") if path.suffix in (".pt", ".pkl", ".npy")])

    def test_malformed_middle_worker_keeps_raw_receipt_and_continues_final_peer(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            plan = offload.benchmark_plan(args)
            seen = []
            def run(command, **kwargs):
                path = Path(command[command.index("--output") + 1])
                arm = json.loads(Path(command[command.index("--arm-manifest") + 1]).read_text())["arm"]
                if "--configuration-preflight" in command:
                    row = configuration_successful(plan, arm)
                else:
                    seen.append(arm)
                    row = successful(plan, arm)
                    if arm == "demand_offload":
                        row.pop("status")
                        row["runtime_config"] = None
                path.write_text(json.dumps(row))
                return SimpleNamespace(returncode=0)
            with patch.object(offload.subprocess, "run", side_effect=run), \
                    patch("benchmarks.compiler.installed_runtime.child_environment", return_value={}), \
                    redirect_stdout(io.StringIO()):
                code = offload.run_benchmark(args)
            receipt = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(seen, list(offload.ARMS))
            self.assertFalse(receipt["summary"]["valid"])
            self.assertEqual(receipt["summary"]["arms"]["resident"]["successful_trials"], 1)
            self.assertEqual(receipt["summary"]["arms"]["predictive_offload"]["successful_trials"], 1)
            self.assertEqual(receipt["summary"]["frontier"], [])
            reference = receipt["results"][1]["result"]
            raw = (args.output.parent / reference["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), reference["sha256"])
            self.assertNotIn("status", json.loads(raw))

    def test_first_worker_exception_is_printed_before_cascading_evidence_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            plan = offload.benchmark_plan(args)
            def run(command, **kwargs):
                path = Path(command[command.index("--output") + 1])
                arm = json.loads(Path(command[command.index("--arm-manifest") + 1]).read_text())["arm"]
                configuration = "--configuration-preflight" in command
                row = configuration_successful(plan, arm) if configuration else successful(plan, arm)
                failed = not configuration and arm == "demand_offload"
                if failed:
                    row.update(status="failed", error="ConfigError: first actual failure",
                        failure_context={"stage": "setup", "phase": "config_validation"}, traceback="full original traceback")
                    row["memory_audit"]["snapshots"] = {}
                    row["memory_audit"]["complete"] = False
                path.write_text(json.dumps(row))
                return SimpleNamespace(returncode=1 if failed else 0)
            stdout = io.StringIO()
            with patch.object(offload.subprocess, "run", side_effect=run), \
                    patch("benchmarks.compiler.installed_runtime.child_environment", return_value={}), redirect_stdout(stdout):
                code = offload.run_benchmark(args)
            receipt = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            first_line = stdout.getvalue().splitlines()[0]
            self.assertIn("demand_offload: setup/config_validation: ConfigError: first actual failure", first_line)
            self.assertNotIn("memory_audit", first_line)
            terminal = offload._terminal_report(receipt, args.output)
            self.assertEqual(terminal["summary"]["worker_failures"][0]["message"], "ConfigError: first actual failure")
            self.assertLessEqual(len(terminal["summary"]["first_invalid_reasons"]), 8)
            self.assertGreater(len(receipt["summary"]["invalid_reasons"]), 8)
            self.assertEqual(receipt["summary"]["arms"]["resident"]["successful_trials"], 1)
            reference = receipt["results"][1]["result"]
            raw = json.loads((args.output.parent / reference["path"]).read_text())
            self.assertEqual(raw["traceback"], "full original traceback")


class PackedOffloadUnusedManagerTests(unittest.TestCase):
    def worker(self, directory):
        plan = offload.benchmark_plan(arguments(directory))
        arm = "demand_offload"
        policy = offload.normalize_policy({**plan["manifest"]["policies"][arm], "version": 2})
        plan["manifest"]["policies"][arm] = policy
        plan["workers"][arm]["settings"]["runtime"]["packed_activation_offload"] = deepcopy(policy)
        return successful(plan, arm), policy

    def unused(self, policy):
        # Real facade-owned runtime, not a hand-authored approximation of its
        # absent-allocator schema. This initializes no transport or CUDA state.
        from deepopt._packed_offload import resolve_packed_offload
        from deepopt.compiler.runtime.manager import compiled_runtime_scope

        class Owner:
            pass

        owner = Owner()
        with compiled_runtime_scope(resolve_packed_offload(policy), owner) as manager:
            try:
                manager.freeze_profiles()
                result = json.loads(json.dumps(manager.snapshot()))
                self.assertIsNone(manager.backend)
                self.assertIsNone(manager.pinned)
                self.assertIsNone(manager.pageable)
                self.assertIsNone(manager._thread)
            finally:
                manager.close()
        return result

    def test_real_unused_facade_in_both_orders_preserves_validation_and_exact_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            row, policy = self.worker(directory)
        expected = offload._metrics(row)
        idle = self.unused(policy)
        self.assertEqual(idle["plans"], [])
        self.assertNotIn("pinned_accounting_verified", idle["host"])
        for first in (False, True):
            with self.subTest(idle_first=first):
                received = deepcopy(row)
                for key in ("packed_offload_runtime", "packed_offload_profile"):
                    managers = received[key]["snapshot"]["managers"]
                    managers.insert(0 if first else len(managers), deepcopy(idle))
                before = deepcopy(received)
                self.assertEqual(offload.worker_errors(received, policy, received["settings"], received["installed"]), [])
                self.assertEqual(offload.runtime_budget_errors(received["packed_offload_profile"], policy), [])
                actual = offload._metrics(received)
                for key in ("physical_pinned_peak_bytes", "physical_pinned_peak_bytes_upper_sum",
                            "host_pinned_peak_bytes", "managed_staging_peak_bytes",
                            "actual_admitted_history_bytes", "actual_admitted_history_bytes_min",
                            "actual_admitted_history_bytes_max"):
                    self.assertEqual(actual[key], expected[key], key)
                self.assertEqual(received, before)
        stdlib = subprocess.run([sys.executable, "-S", "-c", """
import json, sys
from examples.vae.benchmark_packed_offload import runtime_budget_errors
receipt, policy = json.load(sys.stdin)
assert runtime_budget_errors(receipt, policy) == []
assert 'torch' not in sys.modules and 'triton' not in sys.modules
"""], input=json.dumps([received["packed_offload_profile"], policy]),
            capture_output=True, text=True, cwd=native_execution.ROOT, timeout=30)
        self.assertEqual(stdlib.returncode, 0, stdlib.stderr)

    def test_budget_preflight_has_no_measured_window_or_rng_requirement_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            row, policy = self.worker(directory)
        frozen = row["packed_offload_profile"]
        self.assertEqual(offload.runtime_budget_errors(frozen, policy), [])
        self.assertEqual(len(frozen["snapshot"]["managers"][0]["admission_history"]), 1)
        bad = deepcopy(frozen)
        bad["snapshot"]["managers"][0]["host"]["pinned_accounting_verified"] = False
        self.assertIn("physical_pinned_budget_unverified_or_exceeded", offload.runtime_budget_errors(bad, policy))
        bad = deepcopy(frozen)
        bad["snapshot"]["frozen"] = False
        self.assertIn("runtime_profile_learning_during_measurement", offload.runtime_budget_errors(bad, policy))
        self.assertEqual(offload.runtime_budget_errors({"available": False}, policy),
                         ["runtime_metadata_unavailable_or_malformed"])

    def test_budget_preflight_proves_automatic_limits_against_registered_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            row, policy = self.worker(directory)
        policy = dict(policy, device_staging_bytes=0, d2h_inflight_bytes=0, max_backward_tasks=2)
        frozen = row["packed_offload_profile"]
        manager, = frozen["snapshot"]["managers"]
        manager["policy"] = policy
        manager["transport_limits"] = dict(device_staging_bytes=512, d2h_inflight_bytes=256)
        manager["plans"][0].update(required_staging_bytes=256, required_d2h_inflight_bytes=256)
        frozen["snapshot"]["managers"].append(self.unused(policy))
        self.assertEqual(offload.runtime_budget_errors(frozen, policy), [])

        manager["device"]["peak_d2h_inflight_bytes"] = 257
        self.assertIn("missing_or_exceeded_managed_budget:peak_d2h_inflight_bytes",
                      offload.runtime_budget_errors(frozen, policy))
        del manager["transport_limits"]["device_staging_bytes"]
        self.assertIn("automatic_managed_budget_unverified:device_staging_bytes",
                      offload.runtime_budget_errors(frozen, policy))

    def test_unused_exemption_rejects_activity_malformed_fields_and_common_guard_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            row, policy = self.worker(directory)
        idle = self.unused(policy)
        mutations = {
            "host_activity": lambda value: value["host"].update(pinned_peak_bytes=1),
            "host_missing": lambda value: value["host"].pop("pinned_cached_bytes"),
            "unreported_host_claim": lambda value: value["host"].update(other_bytes=1),
            "device_activity": lambda value: value["device"].update(staging_bytes=1),
            "counter_activity": lambda value: value["counters"].update(h2d_bytes=1),
            "counter_boolean": lambda value: value["counters"].update(h2d_bytes=False),
            "lifecycle_activity": lambda value: value["lifecycle"].update(graph_leases=1),
            "not_quiescent": lambda value: value["lifecycle"].update(quiescent=False),
            "events": lambda value: value["events"].append({"kind": "producer"}),
            "profiles": lambda value: value["profiles"].update(example={}),
            "admissions": lambda value: value["admission_counts"].append({"offloaded_forwards": 1}),
            "allocator_domain_without_proof": lambda value: value["host"].update(pinned_allocator_domain_id=99),
            "runtime_error": lambda value: value.update(error="RuntimeError"),
            "testing_backend": lambda value: value.update(testing_backend=True),
            "learning": lambda value: value.update(profile_learning=True),
            "unfrozen": lambda value: value.update(profile_frozen=False),
            "cleanup_error": lambda value: value["cleanup_errors"].append({"role": "failed drain"}),
            "policy": lambda value: value["policy"].update(policy="memory_first_overlap"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                received = deepcopy(row)
                value = deepcopy(idle)
                mutate(value)
                received["packed_offload_runtime"]["snapshot"]["managers"].append(value)
                self.assertTrue(offload.worker_errors(received, policy, received["settings"], received["installed"]))

    def test_active_owner_requires_verified_domain_and_physical_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            row, policy = self.worker(directory)
        for name in ("unverified", "domain_missing", "domain_boolean", "capacity_exceeded", "peak_exceeded"):
            with self.subTest(name=name):
                received = deepcopy(row)
                host = received["packed_offload_runtime"]["snapshot"]["managers"][0]["host"]
                if name == "unverified":
                    host["pinned_accounting_verified"] = False
                elif name == "domain_missing":
                    host.pop("pinned_allocator_domain_id")
                elif name == "domain_boolean":
                    host["pinned_allocator_domain_id"] = True
                else:
                    host["pinned_physical_" + ("capacity_bytes" if name == "capacity_exceeded" else "peak_bytes")] = policy["host_pinned_bytes"] + 1
                self.assertIn("physical_pinned_budget_unverified_or_exceeded",
                    offload.worker_errors(received, policy, received["settings"], received["installed"]))

    def test_shared_domain_is_deduplicated_and_uncertain_owners_have_no_exact_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            row, _ = self.worker(directory)
        owners = row["packed_offload_runtime"]["snapshot"]["managers"]
        owners.append(deepcopy(owners[0]))
        self.assertEqual(offload._metrics(row)["physical_pinned_peak_bytes_upper_sum"], 256)
        self.assertIsNone(offload._metrics(row)["actual_admitted_history_bytes"])
        owners[1]["host"]["pinned_physical_peak_bytes"] = 512
        metrics = offload._metrics(row)
        self.assertEqual(metrics["physical_pinned_peak_bytes_upper_sum"], 512)
        self.assertIsNone(metrics["physical_pinned_peak_bytes"])
        owners[1]["host"].pop("pinned_allocator_domain_id")
        self.assertIsNone(offload._metrics(row)["physical_pinned_peak_bytes"])

    def test_distinct_physical_domains_share_the_declared_capacity_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            row, policy = self.worker(directory)
        owners = row["packed_offload_runtime"]["snapshot"]["managers"]
        owners.append(deepcopy(owners[0]))
        for value in owners:
            value["host"].update(pinned_physical_capacity_bytes=policy["host_pinned_bytes"],
                                 pinned_physical_peak_bytes=policy["host_pinned_bytes"])
        self.assertEqual(offload.runtime_budget_errors(row["packed_offload_runtime"], policy), [])
        owners[1]["host"]["pinned_allocator_domain_id"] += 1
        self.assertIn("physical_pinned_budget_unverified_or_exceeded",
            offload.runtime_budget_errors(row["packed_offload_runtime"], policy))


class PackedOffloadConfigurationPreflightTests(unittest.TestCase):
    def test_configuration_only_runs_all_three_pinned_workers_without_benchmark_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory, extra=("--configuration-preflight-only", "--profile-after-measurement"))
            plan = offload.benchmark_plan(args)
            seen = []
            def run(command, **kwargs):
                self.assertIn("--configuration-preflight", command)
                self.assertNotIn("--profile-after-measurement", command)
                arm = json.loads(Path(command[command.index("--arm-manifest") + 1]).read_text())["arm"]
                seen.append(arm)
                Path(command[command.index("--output") + 1]).write_text(json.dumps(configuration_successful(plan, arm)))
                return SimpleNamespace(returncode=0)
            with patch.object(offload.subprocess, "run", side_effect=run), \
                    patch("benchmarks.compiler.installed_runtime.child_environment", return_value={}), \
                    redirect_stdout(io.StringIO()):
                code = offload.run_benchmark(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(seen, list(offload.ARMS))
            self.assertEqual(report["scope"], "configuration_preflight_only")
            self.assertTrue(report["configuration_preflight"]["valid"])
            self.assertFalse(report["execution_completed"])
            self.assertFalse(report["timing_eligible"])
            self.assertFalse(report["CUDA_execution_proven"])
            self.assertEqual(report["results"], [])
            self.assertEqual(report["diagnostics"], [])
            self.assertEqual(report["summary"]["arms"], {})
            self.assertNotIn("measurement", report)
            for worker in report["configuration_preflight"]["workers"]:
                reference = worker["result"]
                raw = (args.output.parent / reference["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), reference["sha256"])

    def test_failed_configuration_preserves_all_raw_peers_and_stops_before_any_model_job(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            plan = offload.benchmark_plan(args)
            seen = []
            def run(command, **kwargs):
                self.assertIn("--configuration-preflight", command, "model work began despite invalid configuration")
                arm = json.loads(Path(command[command.index("--arm-manifest") + 1]).read_text())["arm"]
                seen.append(arm)
                row = configuration_successful(plan, arm)
                failed = arm == "demand_offload"
                if failed:
                    row.update(status="failed", error="ConfigError: runtime.ddp.bucket_cap_mb: invalid",
                        failure_context={"stage": "configuration_preflight", "phase": "config_validation"})
                    row["configuration_preflight"]["valid"] = False
                    row["configuration_preflight"]["checks"]["typed_config"] = False
                Path(command[command.index("--output") + 1]).write_text(json.dumps(row))
                return SimpleNamespace(returncode=1 if failed else 0)
            stdout = io.StringIO()
            with patch.object(offload.subprocess, "run", side_effect=run), \
                    patch("benchmarks.compiler.installed_runtime.child_environment", return_value={}), redirect_stdout(stdout):
                code = offload.run_benchmark(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(seen, list(offload.ARMS))
            self.assertEqual(len(report["configuration_preflight"]["workers"]), 3)
            self.assertFalse(report["configuration_preflight"]["valid"])
            self.assertEqual(report["results"], [])
            self.assertEqual(report["diagnostics"], [])
            self.assertFalse(report["execution_completed"])
            self.assertIn("demand_offload: configuration_preflight/config_validation: ConfigError:", stdout.getvalue().splitlines()[0])
            self.assertNotIn("memory_audit", stdout.getvalue())
            terminal = offload._terminal_report(report, args.output)
            self.assertEqual(terminal["configuration_preflight"]["worker_failures"][0]["arm"], "demand_offload")
            self.assertTrue(all("result" in row for row in report["configuration_preflight"]["workers"]))

    def test_configuration_schema_and_ownership_guards_fail_closed_on_malformed_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = offload.benchmark_plan(arguments(directory))
        for case in ("checks", "identity", "origin", "stability", "owner", "settings", "scope", "missing", "exit"):
            with self.subTest(case=case):
                row = configuration_successful(plan, "demand_offload")
                row["worker_exit_code"] = 0
                if case == "checks":
                    row["configuration_preflight"]["checks"]["backend_config"] = False
                elif case == "identity":
                    row["installed"]["source_sha256"] = "c" * 64
                elif case == "origin":
                    row["origin"] = {}
                elif case == "stability":
                    row["helper_source_stable"] = False
                elif case == "owner":
                    row["settings"]["runtime"]["compile"]["options"]["require_full_aot"] = False
                elif case == "settings":
                    row["settings"] = None
                elif case == "scope":
                    row["configuration_preflight"]["cuda_execution"] = True
                elif case == "missing":
                    row.pop("configuration_preflight")
                else:
                    row["worker_exit_code"] = 1
                self.assertTrue(offload._configuration_errors(row, plan, "demand_offload"))


class PackedOffloadHostMetadataTests(unittest.TestCase):
    def test_host_metadata_is_scalar_and_runtime_callback_failure_is_explicit(self):
        metadata = execution_memory.host_memory_snapshot()
        self.assertGreater(metadata["peak_rss_bytes"], 0)
        json.dumps(metadata, allow_nan=False)
        def broken():
            raise RuntimeError("offload diagnostic query failed")
        audit = execution_memory.ExecutionMemoryAudit(runtime_snapshot=broken)
        audit.bind(SimpleNamespace(), SimpleNamespace(type="cpu", index=None))
        row = audit.capture("after_first_update", update=1)
        self.assertFalse(row["complete"])
        self.assertEqual(audit.report["errors"][0]["action"], "packed_offload_metadata")
        self.assertIn("host_process", row)

    def test_pinned_allocator_cache_and_rounding_are_distinct_bounded_optional_metadata(self):
        counters = {"allocated_bytes.current": 2048, "allocated_bytes.peak": 4096,
            "active_bytes.current": 1024, "num_host_alloc": 4, "ignored_tensor": object()}
        torch = SimpleNamespace(cuda=SimpleNamespace(memory=SimpleNamespace(host_memory_stats=lambda: counters)))
        value = execution_memory.pinned_host_memory_snapshot(torch)
        self.assertEqual(value["status"], "available")
        self.assertEqual(value["statistics"]["allocated_bytes.current"], 2048)
        self.assertNotIn("ignored_tensor", value["statistics"])
        self.assertTrue(value["peak_is_approximate"])
        self.assertFalse(value["managed_offload_budget_is_process_allocator_cap"])
        missing = SimpleNamespace(cuda=SimpleNamespace(memory=SimpleNamespace()))
        self.assertEqual(execution_memory.pinned_host_memory_snapshot(missing)["status"], "unavailable")
        json.dumps(value, allow_nan=False)


class PackedOffloadInstalledCPUTests(unittest.TestCase):
    """Actual installed-wheel handoff; enabled CUDA transport is never simulated."""

    def setUp(self):
        from benchmarks.compiler.installed_runtime import installed_identity
        from deepopt import reset_config

        reset_config()
        self.addCleanup(reset_config)
        self.identity = installed_identity(native_execution.ROOT, full=True)
        if importlib.util.find_spec("deepopt._packed_offload") is None:
            self.skipTest("requires the newly installed packed-offload candidate wheel")
        import torch

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def arguments(self, directory, *, enabled=False):
        source, raw = _receipt(directory, tiny=True)
        self.assertIsNotNone(raw["runtime"]["compile"]["mode"])
        self.assertTrue(raw["runtime"]["compile"]["options"]["use_native_autodiff"])
        policy = Path(directory) / "policy.json"
        policy.write_text(json.dumps({**offload.POLICY_DEFAULTS, "enabled": enabled,
            "host_pinned_bytes": 2**20, "device_staging_bytes": 2**20, "d2h_inflight_bytes": 2**20}))
        pin = Path(directory) / "arm.json"
        pin.write_text(json.dumps({"schema_version": 1, "arm": "resident_cpu_diagnostic",
            "expected_installed": {key: self.identity[key] for key in ("version", "source_sha256", "files_hashed")},
            "native_options_sha256": None, "packed_offload_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}))
        return native_execution.parser().parse_args(["--from-result", str(source), "--device", "cpu",
            "--batch-size", "1", "--optimizer", "sgd", "--synthetic", "--benchmark", "--warmup", "1", "--updates", "2",
            "--packed-offload-json", str(policy), "--arm-manifest", str(pin), "--output", str(Path(directory) / "worker.json")])

    def test_real_resident_native_cpu_envelope_forward_backward_update_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with patch("examples.vae.qualification.TensorOracle.__init__", side_effect=AssertionError("no oracle")), \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            row = json.loads(args.output.read_text())
            self.assertEqual(code, 0, row.get("traceback"))
            self.assertTrue(row["native_ownership_verified"])
            self.assertTrue(row["measurement"]["valid"])
            self.assertEqual(row["warmup_completed"], 1)
            self.assertEqual(row["updates_completed"], 2)
            self.assertEqual(row["installed"], self.identity)
            self.assertIsNotNone(row["target_settings"]["runtime"]["compile"]["mode"])
            self.assertFalse(row["settings"]["runtime"]["packed_activation_offload"]["enabled"])
            self.assertFalse(row["input"]["requires_grad"])
            self.assertTrue(row["memory_audit"]["complete"], row["memory_audit"])
            self.assertTrue(row["packed_offload_runtime"]["available"])
            self.assertTrue(row["packed_offload_cleanup"]["available"])
            self.assertFalse(row["memory_audit"]["cuda_execution_evidence"])
            self.assertFalse([path for path in Path(directory).rglob("*") if path.suffix in (".pt", ".pkl", ".npy", ".ckpt")])

    def test_enabled_cpu_transport_fails_before_model_allocation_with_original_context(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory, enabled=True)
            with patch("examples.vae.benchmark_vae.make_optimizer", side_effect=AssertionError("must reject before allocation")), \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            row = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertFalse(row["execution_completed"])
            self.assertFalse(row["measurement"]["valid"])
            self.assertIn("CUDA", row["error"])
            self.assertIn("traceback", row)
            self.assertIn("phase", row["failure_context"])

    def test_inventory_export_failure_is_separate_from_completed_native_training(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            args.benchmark, args.warmup, args.updates = False, 0, 1
            args.inventory_output = Path(directory) / "requested-inventory.json"
            with redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            row = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(row["status"], "failed")
            self.assertTrue(row["execution_completed"])
            self.assertTrue(row["native_ownership_verified"])
            self.assertEqual(row["updates_completed"], 1)
            self.assertTrue(row["steps"][0]["optimizer_completed"])
            self.assertFalse(row["timing_eligible"])
            self.assertEqual(row["failure_context"]["stage"], "finalization")
            self.assertEqual(row["failure_context"]["phase"], "compiled_inventory_export")
            self.assertIn("distinct inventories=0", row["error"])
            self.assertEqual(row["packed_offload_inventory"]["status"], "unavailable")
            self.assertFalse(args.inventory_output.exists())
            self.assertFalse([path for path in Path(directory).rglob("*")
                if path.suffix in (".pt", ".pth", ".pkl", ".npy", ".ckpt")])

    def test_actual_CPU_worker_process_receipts_keep_resident_audit_and_failed_arm_labels(self):
        from benchmarks.compiler.installed_runtime import child_environment
        from examples.vae.benchmark_execution import worker_command
        from examples.vae.memory_comparison import worker_memory_errors

        with tempfile.TemporaryDirectory() as directory:
            rows, policies, workers = [], {}, {}
            environment = child_environment(native_execution.ROOT, identity=self.identity)
            environment["OMP_NUM_THREADS"] = "1"
            environment["MKL_NUM_THREADS"] = "1"
            for arm in offload.ARMS:
                worker_root = Path(directory) / arm
                worker_root.mkdir()
                args = self.arguments(worker_root, enabled=arm != "resident")
                policy = json.loads(args.packed_offload_json.read_text())
                policy["policy"] = "demand" if arm == "demand_offload" else "memory_first_overlap"
                args.packed_offload_json.write_text(json.dumps(policy))
                pin = json.loads(args.arm_manifest.read_text())
                pin.update(arm=arm, packed_offload_sha256=hashlib.sha256(args.packed_offload_json.read_bytes()).hexdigest())
                args.arm_manifest.write_text(json.dumps(pin))
                command = worker_command(args, "native", args.output) + ["--packed-offload-json", str(args.packed_offload_json)]
                result = subprocess.run(command, cwd=native_execution.ROOT, env=environment,
                    capture_output=True, text=True, check=False, timeout=120)
                self.assertTrue(args.output.is_file(), result.stderr)
                row = json.loads(args.output.read_text())
                row.update(block=1, offload_arm=arm, worker_exit_code=result.returncode)
                rows.append(row)
                policies[arm] = offload.normalize_policy(policy)
                workers[arm] = {"settings": row["settings"]}
                if arm == "resident":
                    self.assertEqual(result.returncode, 0, row.get("traceback", result.stderr))
                    self.assertTrue(row["native_ownership_verified"])
                    self.assertEqual(row["updates_completed"], 2)
                    self.assertTrue(row["measurement"]["valid"])
                    self.assertFalse(row["memory_audit"]["cuda_execution_evidence"])
                    self.assertEqual(worker_memory_errors(row), [])
                    self.assertEqual(offload.worker_errors(row, policies[arm], row["settings"], self.identity), [])
                else:
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("requires native compiled CUDA execution", row["error"])
                    self.assertEqual(row["failure_context"]["phase"], "packed_offload_device_validation")
                    self.assertNotIn("model", row)
            plan = {"objective": "CPU_process_handoff_diagnostic_only", "order": [list(offload.ARMS)],
                "baseline_tolerance_bytes": 0, "manifest": {"policies": policies}, "workers": workers,
                "installed": self.identity}
            summary = offload.aggregate(rows, plan)
            self.assertFalse(summary["valid"])
            self.assertFalse(summary["CUDA_execution_proven"])
            self.assertEqual(summary["arms"]["resident"]["successful_trials"], 1)
            self.assertIsNotNone(summary["arms"]["resident"]["metrics"]["wall_ms_per_update"]["median"])
            self.assertEqual(summary["arms"]["demand_offload"]["successful_trials"], 0)
            self.assertEqual(summary["arms"]["predictive_offload"]["successful_trials"], 0)
            errors = summary["memory_isolation"]["invalid_reasons"]
            self.assertTrue(any(error.startswith("1:demand_offload:") for error in errors))
            self.assertTrue(any(error.startswith("1:predictive_offload:") for error in errors))
            self.assertFalse(any(error.startswith(("1:resident:", "1:native:")) for error in errors))
            self.assertFalse([path for path in Path(directory).rglob("*") if path.suffix in (".pt", ".pkl", ".npy", ".ckpt")])


if __name__ == "__main__":
    unittest.main()
