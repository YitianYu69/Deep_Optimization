"""Capacity planning/schema fixtures; simulated children prove no CUDA facts."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from examples.vae import benchmark_packed_offload as offload
from examples.vae import benchmark_packed_offload_frontier as frontier
from examples.vae.execution_inventory import comparison_options, extract_inventory
from examples.vae import execution_memory
from tests.benchmarks.test_vae_packed_offload_benchmark import arguments, successful


def setup(directory, *, ceiling=16, budgets=(8,), eligible=12):
    args = arguments(directory)
    args.max_host_pinned_mib = ceiling
    args.host_pinned_mib = list(budgets)
    args.inventory_json = Path(directory) / "inventory-source.json"
    args.inventory_output = None
    args.output = Path(directory) / "frontier.json"
    manifest = json.loads(args.experiment_manifest.read_text())
    for policy in manifest["policies"].values():
        policy["version"] = 2
    args.experiment_manifest.write_text(json.dumps(manifest))
    plan = frontier._base(args)
    inventory = {"schema_version": 1, "status": "available", "candidate_metadata_only": True,
        "installed": {key: plan["installed"][key] for key in ("version", "source_sha256", "files_hashed")},
        "origin": deepcopy(plan["origin"]), "requested_input_shape": plan["workers"]["demand_offload"]["requested_input_shape"],
        "comparison_native_options_sha256": frontier._digest(comparison_options(plan["workers"]["demand_offload"]["settings"])),
        "eligible_history_bytes": eligible * 2**20, "eligible_history_count": 12,
        "all_eligible_host_physical_capacity_bytes": 32 * 2**20,
        "all_eligible_capacity_is_feasible_upper_bound_not_minimum": True,
        "required_staging_bytes": 1024, "required_d2h_inflight_bytes": 1024, "rejected_by_reason": {}}
    args.inventory_json.write_text(json.dumps(inventory))
    return args, plan, inventory


def comparison_receipt(command, *, demand_oom=False, capacity_status="cuda_oom"):
    """Simulate validated child metadata, never execute a CUDA worker."""
    path = Path(command[command.index("--output") + 1])
    plan = offload.benchmark_plan(offload.parser().parse_args(command[3:]))
    rows, references = [], []
    for arm in plan["active_arms"]:
        row = successful(plan, arm)
        if plan["manifest"]["policies"][arm]["enabled"]:
            row["packed_offload_inventory"] = {"status": "available", "all_eligible_selected": True,
                "eligible_history_bytes": 256, "selected_history_bytes": 256,
                "eligible_history_count": 1, "selected_history_count": 1,
                "eligible_inventory_sha256": "a" * 64}
        if demand_oom and arm == "demand_offload":
            row.update(status="failed", execution_completed=False, worker_exit_code=1,
                capacity_status=capacity_status, error="OutOfMemoryError: CUDA allocation failed")
            row["measurement"]["valid"] = False
        validation = offload.worker_errors(row, plan["manifest"]["policies"][arm],
            plan["workers"][arm]["settings"], plan["installed"])
        source = path.parent / (arm + ".json")
        source.write_text(json.dumps(row))
        references.append({"arm": arm, "block": 1, "status": row["status"],
            "worker_exit_code": row["worker_exit_code"], "validation_errors": validation,
            "result": frontier._ref(source)})
        rows.append(row)
    summary = offload.aggregate(rows, plan)
    code = 0 if summary["valid"] else 1
    path.write_text(json.dumps({"status": "completed" if code == 0 else "failed",
        "summary": summary, "results": references, "configuration_preflight": {"valid": True}}))
    return subprocess.CompletedProcess(command, code)


class PackedOffloadFrontierTests(unittest.TestCase):
    def test_explicit_ceiling_intermediates_and_below_upperbound_all_eligible_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, inventory = setup(directory)
            points = frontier.capacity_points(plan, inventory)
            self.assertEqual([row["host_pinned_bytes"] for row in points], [8 * 2**20, 16 * 2**20, 16 * 2**20])
            self.assertEqual(points[-1]["selection_mode"], "all_eligible")
            self.assertEqual(points[-1]["preflight_status"], "attempt")
            self.assertGreater(points[-1]["feasible_arena_upper_bound_bytes"], plan["ceiling_bytes"])
            self.assertTrue(points[-1]["upper_bound_is_not_minimum"])
            args.max_host_pinned_mib = 0
            with self.assertRaisesRegex(ValueError, "ceiling"):
                frontier._base(args)

    def test_frontier_requires_explicit_version2_without_migrating_legacy_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            args.max_host_pinned_mib, args.host_pinned_mib = 16, [8]
            args.inventory_json = args.inventory_output = None
            with self.assertRaisesRegex(ValueError, "version2"):
                frontier._base(args)

    def test_logical_floor_and_consumer_floors_report_resource_limited_without_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, inventory = setup(directory, eligible=20)
            point = frontier.capacity_points(plan, inventory)[-1]
            self.assertEqual(point["preflight_status"], "resource_limited")
            self.assertIn("logical_history_exceeds_explicit_physical_ceiling", point["resource_limits"])
            inventory["eligible_history_bytes"] = 10
            inventory["required_staging_bytes"] = 2**30
            point = frontier.capacity_points(plan, inventory)[-1]
            self.assertIn("all_eligible_consumer_staging_floor_exceeds_declared_budget", point["resource_limits"])
            inventory["required_d2h_inflight_bytes"] = 2**30
            policy = plan["manifest"]["policies"]["demand_offload"]
            policy.update(device_staging_bytes=0, d2h_inflight_bytes=0)
            point = frontier.capacity_points(plan, inventory)[-1]
            self.assertEqual(point["preflight_status"], "attempt")
            self.assertFalse(point["resource_limits"])

    def test_pageable_tier_uses_fixed_canonical_budget_and_reserves_both_host_capacities(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan, inventory = setup(directory, eligible=20)
            plan["manifest"]["policies"]["demand_offload"]["host_pageable_bytes"] = 32 * 2**20
            point = frontier.capacity_points(plan, inventory)[-1]
            self.assertEqual(point["preflight_status"], "attempt")
            self.assertEqual(point["fixed_host_pageable_bytes"], 32 * 2**20)
            with patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 1}):
                value, receipt = frontier._bootstrap(args, plan, Path(directory), {})
            self.assertIsNone(value)
            self.assertFalse(receipt["worker_started"])
            self.assertEqual(receipt["host_capacity_preflight"]["required_physical_capacity_bytes"], 48 * 2**20)

    def test_inventory_identity_and_upperbound_contract_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, source = setup(directory)
            for field in ("installed", "origin", "comparison_native_options_sha256", "all_eligible_capacity_is_feasible_upper_bound_not_minimum"):
                inventory = deepcopy(source)
                inventory[field] = False
                with self.subTest(field=field), self.assertRaises(ValueError):
                    frontier.capacity_points(plan, inventory)

    def test_compact_inventory_preserves_counts_floors_and_candidate_only_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, inventory = setup(directory)
            metadata = {key: value for key, value in inventory.items() if key not in
                ("schema_version", "status", "installed", "origin", "requested_input_shape", "comparison_native_options_sha256")}
            metadata.update(selected_history_bytes=100, selected_history_count=2,
                selected_physical_capacity_bytes=128, eligible_inventory_sha256="a" * 64)
            raw = {"native_autodiff_graphs": [{"packed_activation_offload": {"model_scope": {"inventory": metadata}}}]}
            result = extract_inventory(raw, plan["workers"]["demand_offload"], plan["installed"])
            self.assertEqual(result["status"], "available")
            self.assertTrue(result["candidate_metadata_only"])
            self.assertEqual(result["eligible_history_count"], 12)
            self.assertEqual(result["eligible_inventory_sha256"], "a" * 64)
            self.assertEqual(result["selected_history_count"], 2)
            self.assertTrue(result["all_eligible_capacity_is_feasible_upper_bound_not_minimum"])

    def test_large_schedule_does_not_truncate_declared_model_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, inventory = setup(directory)
            region = {"packed_activation_offload": {"transport_envelope_version": 2,
                "model_scope": {"inventory": deepcopy(inventory)}}}
            raw = {"graphs": [{"native_autodiff_graphs": [region, deepcopy(region)],
                "final_schedule": [{} for _ in range(500001)]}]}
            result = extract_inventory(raw, plan["workers"]["demand_offload"], plan["installed"])
            self.assertEqual(result["status"], "available")
            self.assertFalse(result["search_truncated"])
            self.assertEqual(result["eligible_history_bytes"], inventory["eligible_history_bytes"])

    def test_declared_graphs_reject_conflicting_or_incomplete_scope_inventories(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, inventory = setup(directory)
            region = {"packed_activation_offload": {"transport_envelope_version": 2,
                "model_scope": {"inventory": deepcopy(inventory)}}}
            conflict = deepcopy(region)
            conflict["packed_activation_offload"]["model_scope"]["inventory"]["eligible_history_bytes"] += 1
            incomplete = deepcopy(region)
            del incomplete["packed_activation_offload"]["model_scope"]["inventory"]
            missing_scope = {"packed_activation_offload": {"transport_envelope_version": 2}}
            for other in (conflict, incomplete, missing_scope, None):
                with self.subTest(other=other):
                    raw = {"graphs": [{"native_autodiff_graphs": [deepcopy(region)]},
                        {"native_autodiff_graphs": [other]}]}
                    result = extract_inventory(raw, plan["workers"]["demand_offload"], plan["installed"])
                    self.assertEqual(result["status"], "unavailable")

    def test_only_declared_model_scope_inventory_can_supply_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, inventory = setup(directory)
            decoy = deepcopy(inventory)
            decoy["eligible_history_bytes"] += 100
            raw = {"graphs": [{"native_autodiff_graphs": [{"packed_activation_offload": {
                "model_scope": {"inventory": deepcopy(inventory)}}}], "unrelated_diagnostic": decoy}]}
            result = extract_inventory(raw, plan["workers"]["demand_offload"], plan["installed"])
            self.assertEqual(result["status"], "available")
            self.assertEqual(result["eligible_history_bytes"], inventory["eligible_history_bytes"])
            raw["graphs"][0]["native_autodiff_graphs"] = []
            result = extract_inventory(raw, plan["workers"]["demand_offload"], plan["installed"])
            self.assertEqual(result["status"], "unavailable")

    def test_declared_owner_counts_and_enabled_v2_blocks_must_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, inventory = setup(directory)
            region = {"packed_activation_offload": {"transport_envelope_version": 2,
                "model_scope": {"inventory": deepcopy(inventory)}}}
            graph = {"native_autodiff_regions": 1, "native_autodiff_graphs": [region],
                "effective_compiler_options": {"packed_activation_offload": {"enabled": True, "version": 2}}}
            missing_regions = deepcopy(graph)
            del missing_regions["native_autodiff_graphs"]
            missing_block = deepcopy(graph)
            missing_block.update(native_autodiff_regions=2, native_autodiff_graphs=[deepcopy(region), {}])
            empty_block = deepcopy(graph)
            empty_block.update(native_autodiff_regions=2, native_autodiff_graphs=[deepcopy(region),
                {"packed_activation_offload": {}}])
            legacy_block = deepcopy(graph)
            legacy_block.update(native_autodiff_regions=2, native_autodiff_graphs=[deepcopy(region),
                {"packed_activation_offload": {"transport_envelope_version": 1}}])
            wrong_count = {**deepcopy(graph), "native_autodiff_regions": 2}
            cases = [{"captured_graphs": 2, "graphs": [deepcopy(graph), value]}
                for value in (missing_regions, missing_block, empty_block, legacy_block, wrong_count)]
            cases.append({"captured_graphs": 2, "graphs": [deepcopy(graph)]})
            cases.append({"captured_graphs": 1, "native_autodiff_regions": 2, "graphs": [deepcopy(graph)]})
            for index, raw in enumerate(cases):
                with self.subTest(case=index):
                    result = extract_inventory(raw, plan["workers"]["demand_offload"], plan["installed"])
                    self.assertEqual(result["status"], "unavailable")
            for policy in ({"enabled": False, "version": 2}, {"enabled": True, "version": 1}):
                other = {"native_autodiff_regions": 1, "native_autodiff_graphs": [{}],
                    "effective_compiler_options": {"packed_activation_offload": policy}}
                raw = {"captured_graphs": 2, "graphs": [deepcopy(graph), other]}
                result = extract_inventory(raw, plan["workers"]["demand_offload"], plan["installed"])
                self.assertEqual(result["status"], "available")

    def test_failed_maximum_keeps_completed_points_and_no_failed_timing_point(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = setup(directory)
            expected_source = offload.source_identity()
            requests = []
            def fake(command, **kwargs):
                requests.append(command)
                result_path = Path(command[command.index("--output") + 1])
                options_path = Path(command[command.index("--native-options-json") + 1])
                options = json.loads(options_path.read_text())
                child_args = offload.parser().parse_args(command[3:])
                child_plan = offload.benchmark_plan(child_args)
                if options["packed_offload_selection_mode"] == "all_eligible":
                    result_path.write_text(json.dumps({"status": "failed", "summary": {"valid": False},
                        "error": "OffloadAdmissionError: all eligible history does not fit", "results": []}))
                    return subprocess.CompletedProcess(command, 1)
                rows, refs = [], []
                for arm in offload.ARMS:
                    row = successful(child_plan, arm)
                    budget_mib = child_plan["manifest"]["policies"][arm]["host_pinned_bytes"] // 2**20
                    wall = 10 if arm == "resident" else 12 + budget_mib / 4
                    row["measurement"].update(wall_ms_per_update=wall, wall_seconds=wall * row["updates"] / 1000,
                        cuda_event_ms_per_update=wall - .1, images_per_second=128000 / wall)
                    row_path = result_path.parent / (arm + ".json")
                    row_path.write_text(json.dumps(row))
                    refs.append({"arm": arm, "result": {"path": row_path.name,
                        "bytes": row_path.stat().st_size, "sha256": hashlib.sha256(row_path.read_bytes()).hexdigest()}})
                    rows.append(row)
                summary = offload.aggregate(rows, child_plan)
                result_path.write_text(json.dumps({"status": "completed", "summary": summary, "results": refs}))
                return subprocess.CompletedProcess(command, 0)
            with patch.object(frontier.subprocess, "run", side_effect=fake), \
                    patch.object(offload, "source_identity", return_value=expected_source), \
                    patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 2**40}), \
                    redirect_stdout(io.StringIO()):
                code = frontier.run_frontier(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["successful_capacity_points"], 2)
            self.assertEqual(len(requests), 3)
            self.assertFalse(report["maximum_all_eligible_measured"])
            self.assertFalse(report["points"][-1]["timing_point_created"])
            self.assertTrue(report["points"][0]["valid"])
            self.assertIn("metrics", report["points"][0])
            first, second = (point["metrics"]["demand_offload"] for point in report["points"][:2])
            self.assertEqual(first["extra_wall_overhead_percentage_points_vs_first_successful_budget"], 0)
            self.assertAlmostEqual(second["extra_wall_overhead_percentage_points_vs_first_successful_budget"], 20)
            self.assertEqual(second["wall_ms_per_update_delta_vs_first_successful_budget"], 2)
            self.assertGreater(second["actual_admitted_history_bytes"], 0)
            self.assertIn("physical_pinned_peak_bytes", second)
            self.assertGreater(second["transport_warmup_and_measured_h2d_bytes"], 0)
            self.assertFalse([path for path in Path(directory).rglob("*") if path.suffix in (".pt", ".pth", ".pkl", ".npy")])

    def test_insufficient_job_RAM_preserves_unreached_ceiling_without_any_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = setup(directory)
            with patch.object(frontier.subprocess, "run") as children, \
                    patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 1}), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(frontier.run_frontier(args), 1)
            report = json.loads(args.output.read_text())
            children.assert_not_called()
            self.assertEqual(report["successful_capacity_points"], 0)
            self.assertEqual(report["maximum_point"]["host_pinned_bytes"], 16 * 2**20)
            self.assertEqual(report["maximum_point"]["status"], "resource_limited")
            self.assertFalse(report["maximum_all_eligible_measured"])
            self.assertTrue(all(not point["timing_point_created"] and not point["worker_started"] for point in report["points"]))

    def test_first_evidence_failure_stops_later_points_and_marks_maximum_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = setup(directory)
            reason = "1:demand_offload:physical_pinned_budget_unverified_or_exceeded"

            def failed(command, **kwargs):
                path = Path(command[command.index("--output") + 1])
                path.write_text(json.dumps({"status": "failed", "summary": {
                    "valid": False, "invalid_reasons": [reason]}, "results": []}))
                return subprocess.CompletedProcess(command, 1)

            terminal = io.StringIO()
            with patch.object(frontier.subprocess, "run", side_effect=failed) as children, \
                    patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 2**40}), \
                    redirect_stdout(terminal):
                self.assertEqual(frontier.run_frontier(args), 1)
            report = json.loads(args.output.read_text())
            self.assertEqual(children.call_count, 1)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["first_validation_error"], reason)
            self.assertIn(reason, report["error"])
            self.assertTrue(report["stopped_early"])
            self.assertEqual([point["status"] for point in report["points"]],
                ["failed", "not_executed", "not_executed"])
            for point in report["points"][1:]:
                self.assertFalse(point["worker_started"])
                self.assertFalse(point["timing_point_created"])
                self.assertEqual(point["stopped_after_point"], report["points"][0]["name"])
                self.assertIn(reason, point["not_executed_reason"])
            self.assertEqual(report["maximum_point"]["status"], "not_executed")
            self.assertFalse(report["maximum_all_eligible_measured"])
            self.assertIn("START peak-8MiB", terminal.getvalue())
            self.assertIn("COMPLETE peak-8MiB", terminal.getvalue())
            self.assertIn(str(args.output.resolve()), terminal.getvalue())
            self.assertGreaterEqual(report["points"][0]["elapsed_seconds"], 0)

    def test_resource_preflight_skip_continues_to_later_points(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = setup(directory)
            snapshots = iter(({"status": "available", "available_bytes": 1},
                {"status": "available", "available_bytes": 2**40},
                {"status": "available", "available_bytes": 2**40}))
            with patch.object(frontier.subprocess, "run", side_effect=lambda command, **kwargs: comparison_receipt(command)) as children, \
                    patch.object(execution_memory, "host_capacity_snapshot", side_effect=lambda: next(snapshots)), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(frontier.run_frontier(args), 1)
            report = json.loads(args.output.read_text())
            self.assertEqual(children.call_count, 2)
            self.assertEqual([point["status"] for point in report["points"]],
                ["resource_limited", "completed", "completed"])
            self.assertEqual(report["successful_capacity_points"], 2)
            self.assertFalse(report["stopped_early"])
            self.assertTrue(report["maximum_all_eligible_measured"])

    def test_completed_prefix_keeps_metrics_when_next_point_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = setup(directory)
            requests = []

            def child(command, **kwargs):
                requests.append(command)
                if len(requests) == 1:
                    return comparison_receipt(command)
                path = Path(command[command.index("--output") + 1])
                path.write_text(json.dumps({"status": "failed", "summary": {"valid": False,
                    "invalid_reasons": ["source_identity_changed"]}, "results": []}))
                return subprocess.CompletedProcess(command, 1)

            with patch.object(frontier.subprocess, "run", side_effect=child), \
                    patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 2**40}), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(frontier.run_frontier(args), 1)
            report = json.loads(args.output.read_text())
            self.assertEqual(len(requests), 2)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["successful_capacity_points"], 1)
            self.assertTrue(report["points"][0]["timing_point_created"])
            self.assertGreater(report["points"][0]["metrics"]["resident"]["wall_ms_per_update"], 0)
            self.assertEqual(report["maximum_point"]["status"], "not_executed")
            self.assertEqual(report["unexecuted_capacity_points"], 1)

    def test_only_explicit_verified_demand_cuda_oom_allows_later_budget_attempt(self):
        for capacity_status, expected_calls in (("cuda_oom", 3), ("not_established", 1)):
            with self.subTest(capacity_status=capacity_status), tempfile.TemporaryDirectory() as directory:
                args, _, _ = setup(directory)
                requests = []

                def child(command, **kwargs):
                    requests.append(command)
                    return comparison_receipt(command, demand_oom=len(requests) == 1,
                        capacity_status=capacity_status)

                with patch.object(frontier.subprocess, "run", side_effect=child), \
                        patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 2**40}), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(frontier.run_frontier(args), 1)
                report = json.loads(args.output.read_text())
                self.assertEqual(len(requests), expected_calls)
                self.assertFalse(report["points"][0]["valid"])
                self.assertFalse(report["points"][0]["timing_point_created"])
                if capacity_status == "cuda_oom":
                    self.assertEqual(report["points"][0]["failure_classification"], "demand_cuda_oom")
                    self.assertEqual(report["successful_capacity_points"], 2)
                    self.assertTrue(report["maximum_all_eligible_measured"])
                else:
                    self.assertEqual(report["maximum_point"]["status"], "not_executed")

    def test_malformed_point_receipt_stops_without_losing_raw_reference(self):
        malformed_values = ([1], {"status": "failed", "summary": {"valid": False},
            "results": [], "configuration_preflight": None},
            {"status": "failed", "summary": {"valid": False}, "results": [{"block": [], "arm": "demand_offload"}],
                "configuration_preflight": {"valid": True}})
        for value in malformed_values:
            with self.subTest(receipt=value), tempfile.TemporaryDirectory() as directory:
                args, _, _ = setup(directory)
                data = json.dumps(value)

                def malformed(command, **kwargs):
                    Path(command[command.index("--output") + 1]).write_text(data)
                    return subprocess.CompletedProcess(command, 0)

                with patch.object(frontier.subprocess, "run", side_effect=malformed) as children, \
                        patch.object(execution_memory, "host_capacity_snapshot", return_value={"status": "available", "available_bytes": 2**40}), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(frontier.run_frontier(args), 1)
                report = json.loads(args.output.read_text())
                self.assertEqual(children.call_count, 1)
                self.assertEqual(report["maximum_point"]["status"], "not_executed")
                self.assertEqual(report["points"][0]["result"]["bytes"], len(data.encode()))
                self.assertTrue(report["error"])


if __name__ == "__main__":
    unittest.main()
