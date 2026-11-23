"""Pinned paired-cohort rules; scalar fixtures are never CUDA evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.vae import benchmark_optimization as experiment, benchmark_packed_offload as offload, native_execution
from tests.benchmarks.test_vae_execution_benchmark import _successful_worker
from tests.benchmarks.test_vae_memory_comparison import audited_row
from tests.benchmarks.test_vae_native_execution import _receipt
from tests.benchmarks.test_vae_packed_offload_benchmark import manifest as candidate_manifest, runtime_metadata, configuration_successful


def _manifest(directory, *, combined=False, selection_mode="peak", best_workspace="tiled_dx"):
    _, candidate = candidate_manifest(directory)
    names = ("R0", "BEST") if combined else ("D0", "D1")
    arms = {name: {**deepcopy(candidate["candidate"]), "native_options": {
        **experiment.WORKSPACE_PROFILE_DEFAULTS, "packed_offload_selection_mode": selection_mode,
        "native_conv_workspace_reuse": "off" if index == 0 else best_workspace}}
        for index, name in enumerate(names)}
    policy = {**deepcopy(candidate["policies"]["demand_offload"]), "version": 2}
    value = {"schema_version": 1, "experiment_profile": experiment.COMBINED_PROFILE if combined else experiment.WORKSPACE_PROFILE,
        "arms": arms}
    if combined:
        value["policies"] = {"R0": {**policy, "enabled": False}, "BEST": policy}
    else:
        value["packed_activation_offload"] = policy
    path = Path(directory) / "experiment.json"
    path.write_text(json.dumps(value))
    return path, value


def _profile_manifest(directory, profile, *, selection_mode="peak"):
    path, value = _manifest(directory, selection_mode=selection_mode)
    names = experiment.MEMORY_ARMS[profile]
    value["arms"] = {name: deepcopy(value["arms"][source]) for name, source in zip(names, ("D0", "D1"))}
    for arm in value["arms"].values():
        arm["native_options"]["native_conv_workspace_reuse"] = "off"
    value["experiment_profile"] = profile
    if profile == experiment.SELECTION_PROFILE:
        value["arms"]["S0"]["native_options"]["packed_offload_selection_mode"] = "regional_order"
        value["arms"]["S1"]["native_options"]["packed_offload_selection_mode"] = "peak"
    elif profile == experiment.ARENA_PROFILE:
        policy = value.pop("packed_activation_offload")
        value["policies"] = {"H0": {**policy, "version": 1}, "H1": deepcopy(policy)}
    else:
        raise ValueError("Unsupported isolated fixture profile")
    path.write_text(json.dumps(value))
    return path, value


def _arguments(directory, *, combined=False, profile=None, selection_mode="peak", extra=()):
    source, _ = _receipt(directory)
    path, _ = (_profile_manifest(directory, profile, selection_mode=selection_mode) if profile else
               _manifest(directory, combined=combined, selection_mode=selection_mode))
    return experiment.parser().parse_args(["--from-result", str(source), "--experiment-manifest", str(path),
        "--warmup", "1", "--updates", "2", "--output", str(Path(directory) / "result.json"), *extra])


def _compiler_artifact(path, *, enabled=True, mode="off", version=2):
    components = [
        {"name": "saved_data", "saved_index": 0, "shape": [1, 64], "dtype": "torch.int32", "stride": [64, 1],
         "device": "cuda:0", "nbytes": 256, "role": "packed_data"},
        {"name": "saved_step", "saved_index": 1, "shape": [1], "dtype": "torch.float32", "stride": [1],
         "device": "cuda:0", "nbytes": 4, "role": "step"},
        {"name": "saved_minimum", "saved_index": 2, "shape": [1], "dtype": "torch.float32", "stride": [1],
         "device": "cuda:0", "nbytes": 4, "role": "minimum"}]
    inventory = {"eligible_inventory_sha256": "e" * 64}
    region = {"packed_activation_offload": {"transport_envelope_version": version,
        "plan": {"forward_signature": "forward_encoder",
        "backward_signature": "backward_" + mode, "selected_bytes": 264,
        "bundles": [{"bundle_id": "affine_0", "components": components, "bytes": 264}]},
        "model_scope": {"inventory": inventory}}}
    if version == 1:
        region["packed_activation_offload"].pop("model_scope")
    count = int(enabled)
    value = {"captured_graphs": 1, "native_autodiff_regions": count, "graphs": [{
        "native_autodiff_regions": count, "native_autodiff_graphs": [region] if enabled else [],
        "effective_compiler_options": {"packed_activation_offload": {"version": version, "enabled": enabled}}}]}
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _successful(plan, arm, directory, *, block=1, command=None):
    if command is None:
        args = SimpleNamespace(from_result=Path(plan["origin"]["path"]), device="cuda:0", batch_size=128,
            warmup=1, updates=2, project_root=None, train_dir=None, optimizer=None, synthetic=False,
            deterministic=None, memory_audit=True, profile_after_measurement=False, profile_updates=1,
            profile_memory=False, profile_record_shapes=False, host_headroom_mib=plan["host_headroom_mib"])
        _, _, command = offload._pinned_command(args, plan, Path(directory), f"fixture-{block}-{arm}", arm)
    worker_args = native_execution.parser().parse_args(command[3:])
    worker = native_execution.execution_plan(worker_args)
    row = _successful_worker(worker, "native", identity=plan["installed"])
    policy = plan["manifest"]["policies"][arm]
    mode = plan["arms"][arm]["native_options"]["native_conv_workspace_reuse"]
    region = "forward_encoder:backward_" + mode
    for field, updates in (("packed_offload_runtime", 3), ("packed_offload_profile", 1)):
        row[field] = runtime_metadata(policy, updates=updates)
        managers = row[field]["snapshot"]["managers"]
        if policy.get("version", 1) == 2:
            for manager in managers:
                manager["host"].update(pinned_physical_capacity_bytes=256, pinned_physical_peak_bytes=256,
                    pinned_accounting_verified=True)
        if policy["enabled"]:
            managers[0]["plans"][0].update(region_id=region, forward_signature="forward_encoder",
                backward_signature="backward_" + mode, selected_bytes=264, bundle_bytes={"affine_0": 264})
            for event in managers[0]["admission_history"]:
                event["region_id"] = region
    row["packed_offload_cleanup"] = runtime_metadata(policy, cleanup=True)
    if policy.get("version", 1) == 2:
        capacity = policy["host_pinned_bytes"] + policy["host_pageable_bytes"] if policy["enabled"] else 0
        required = capacity + plan["host_headroom_bytes"]
        row["host_capacity_preflight"] = {"admitted": True,
            "status": "admitted", "snapshot": {"status": "available", "available_bytes": required + 2**30},
            "required_plus_headroom_bytes": required,
            "required_physical_capacity_bytes": capacity,
            "reserved_headroom_bytes": plan["host_headroom_bytes"], "requested_capacity_clamped": False}
    for name in ("rng_before_warmup_sha256", "rng_after_warmup_sha256", "rng_after_measurement_sha256"):
        row[name] = "a" * 64
    row["steps"] = [{"stage": "warmup", **dict.fromkeys(("forward_completed", "backward_completed", "optimizer_completed",
        "all_trainable_gradients_present", "fp32_gradients", "finite_gradients"), True)}]
    row["compiler_report_summary"]["effective_compiler_options"] = [deepcopy(plan["arms"][arm]["native_options"])]
    row["compiler_report_summary"]["fusion_candidates"] = ([{"pattern": experiment.WORKSPACE_PATTERN,
        "selected": True, "execution_phase": "backward"}] if mode == "tiled_dx" else [])
    row["_expected_arm_pin"] = json.loads(worker_args.arm_manifest.read_text())
    row.update(block=block, offload_arm=arm, arm=arm, worker_exit_code=0)
    row["memory_audit"] = audited_row(arm=arm, pid=100 + block * 2 + plan["active_arms"].index(arm))["memory_audit"]
    path = worker_args.output.parent / f"compiler-{block}-{arm}.json"
    row["compiler_report_artifact"] = _compiler_artifact(path, enabled=policy["enabled"], mode=mode, version=policy.get("version", 1))
    row["packed_offload_inventory"] = {"status": "available", "all_eligible_selected": True,
        "selection_mode": plan["arms"][arm]["native_options"]["packed_offload_selection_mode"],
        "all_eligible_host_physical_capacity_bytes": 264, "eligible_history_bytes": 264,
        "eligible_history_count": 1, "selected_history_count": 1 if policy["enabled"] else 0,
        "eligible_inventory_sha256": "e" * 64,
        "selected_history_bytes": 264 if policy["enabled"] else 0}
    if policy.get("version", 1) == 1:
        row["packed_offload_inventory"] = {"status": "unavailable", "reason": "legacy_has_no_model_scope_inventory"}
    row["memory_audit"]["snapshots"]["after_postchecks"]["pinned_host_allocator"] = {
        "status": "available", "statistics": {"allocated_bytes.peak": 4096, "allocated_bytes.current": 4096},
        "peak_is_approximate": True}
    if policy["enabled"]:
        row["_workspace_history"] = experiment._workspace_read_history(row, worker_args.output)
    return row


def _cohort(directory, *, combined=False, profile=None, trials=6, selection_mode="peak"):
    args = _arguments(directory, combined=combined, profile=profile, selection_mode=selection_mode, extra=("--trials", str(trials)))
    plan = experiment.experiment_plan(args)
    rows = [_successful(plan, arm, directory, block=block) for block, order in enumerate(plan["order"], 1) for arm in order]
    return args, plan, rows


def _rename_selected(row, name):
    """A different independently verified static/admitted bundle fixture."""
    artifact = row["compiler_report_artifact"]
    path = Path(artifact["path"])
    value = json.loads(path.read_text())
    bundle = value["graphs"][0]["native_autodiff_graphs"][0]["packed_activation_offload"]["plan"]["bundles"][0]
    bundle["bundle_id"] = name
    for component in bundle["components"]:
        component["name"] += "_other"
    data = json.dumps(value).encode()
    path.write_bytes(data)
    artifact.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    row["_workspace_history"] = experiment._workspace_read_history(row, path.parent / "result.json")
    for field in ("packed_offload_runtime", "packed_offload_profile"):
        manager = row[field]["snapshot"]["managers"][0]
        manager["plans"][0].update(selected_bundle_ids=[name], bundle_bytes={name: 264})
        for event in manager["admission_history"]:
            event["selected_bundle_ids"] = [name]


class WorkspaceAblationPlanningTests(unittest.TestCase):
    def test_exact_same_candidate_options_policy_and_balanced_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(directory)
            original = args.from_result.read_bytes()
            plan = experiment.experiment_plan(args)
            self.assertEqual(args.from_result.read_bytes(), original)
            self.assertEqual(plan["active_arms"], ["D0", "D1"])
            self.assertEqual(plan["order"], [["D0", "D1"], ["D1", "D0"]] * 3)
            self.assertEqual(plan["installed_identities"]["D0"], plan["installed_identities"]["D1"])
            self.assertEqual(plan["requested_input_shape"], [128, 3, 224, 224])
            self.assertEqual(plan["configuration_preflight_workers"], 2)
            self.assertEqual(len(plan["arms"]["D0"]["native_options"]), 11)
            for arm in plan["active_arms"]:
                settings = plan["workers"][arm]["settings"]
                self.assertIsNone(settings["runtime"]["compile"]["mode"])
                self.assertTrue(settings["runtime"]["compile"]["options"]["require_full_aot"])
                self.assertEqual(settings["runtime"]["activation_compression"]["group_size"], 1024)
                self.assertEqual(settings["runtime"]["packed_activation_offload"]["version"], 2)
                self.assertEqual(settings["runtime"]["packed_activation_offload"]["policy"], "demand")

    def test_strict_whitelist_rejects_wrong_options_versions_and_unbalanced_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path, original = _manifest(directory)
            for kind in ("missing", "bool_batch", "zero_batch", "different_batch", "whole_batch", "legacy", "selection",
                         "different_candidate", "v1", "predictive"):
                value = deepcopy(original)
                options = value["arms"]["D1"]["native_options"]
                if kind == "missing":
                    options.pop("native_conv_dx_batch_size")
                elif kind == "bool_batch":
                    options["native_conv_dx_batch_size"] = True
                elif kind == "zero_batch":
                    options["native_conv_dx_batch_size"] = 0
                elif kind == "different_batch":
                    options["native_conv_dx_batch_size"] = 2
                elif kind == "whole_batch":
                    options["native_conv_workspace_reuse"] = "whole_batch"
                elif kind == "legacy":
                    options["fuse_norm_cat_body"] = True
                elif kind == "selection":
                    options["packed_offload_selection_mode"] = "all_eligible"
                elif kind == "different_candidate":
                    value["arms"]["D1"]["wheel"]["sha256"] = "c" * 64
                elif kind == "v1":
                    value["packed_activation_offload"]["version"] = 1
                else:
                    value["packed_activation_offload"]["policy"] = "memory_first_overlap"
                path.write_text(json.dumps(value))
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    experiment.load_manifest(path)
        for trials in (2, 3, 5, 7):
            with self.subTest(trials=trials), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "balanced"):
                    experiment.experiment_plan(_arguments(directory, extra=("--trials", str(trials))))

    def test_standard_library_only_plan_and_actual_resolved_pin_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(directory, selection_mode="all_eligible")
            completed = subprocess.run([sys.executable, "-S", str(Path(experiment.__file__)), "--from-result", str(args.from_result),
                "--experiment-manifest", str(args.experiment_manifest), "--dry-run"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads(completed.stdout)
            self.assertEqual(plan["trials_requested"], 6)
            self.assertEqual(plan["warmup"], 5)
            self.assertEqual(plan["updates"], 10)
            self.assertFalse(plan["target_CUDA_evidence_complete"])
            self.assertFalse(args.output.exists())
            plan = experiment.experiment_plan(args)
            path, _, command = offload._pinned_command(args, plan, Path(directory), "pin", "D1", configuration=True)
            worker_args = native_execution.parser().parse_args(command[3:])
            actual = native_execution.execution_plan(worker_args)
            self.assertEqual(actual["settings"], plan["workers"]["D1"]["settings"])
            pin = json.loads(worker_args.arm_manifest.read_text())
            self.assertEqual(actual["resolved_native_options_sha256"], pin["resolved_native_options_sha256"])
            self.assertEqual(pin["native_options_sha256"], hashlib.sha256(worker_args.native_options_json.read_bytes()).hexdigest())
            self.assertNotEqual(path, args.output)
            pin["resolved_native_options_sha256"] = "a" * 64
            worker_args.arm_manifest.write_text(json.dumps(pin))
            with self.assertRaisesRegex(ValueError, "Resolved"):
                native_execution.execution_plan(worker_args)

    def test_cpu_transport_cannot_become_a_performance_substitute(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(directory, extra=("--device", "cpu", "--trials", "1"))
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                experiment.experiment_plan(args)
            args.dry_run = True
            self.assertFalse(experiment.experiment_plan(args)["target_CUDA_evidence_complete"])

    def test_host_headroom_is_strict_pinned_and_forwarded_to_configuration_and_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(directory, extra=("--host-headroom-mib", "17", "--trials", "1"))
            plan = experiment.experiment_plan(args)
            self.assertEqual(plan["host_headroom_bytes"], 17 * 2**20)
            for configuration in (False, True):
                _, _, command = offload._pinned_command(args, plan, Path(directory), "head-" + str(configuration), "D1", configuration=configuration)
                worker = native_execution.execution_plan(native_execution.parser().parse_args(command[3:]))
                self.assertEqual(worker["host_capacity_policy"], plan["workers"]["D1"]["host_capacity_policy"])
                self.assertEqual(worker["host_capacity_policy"]["reserved_headroom_bytes"], 17 * 2**20)
            for invalid in (-1, True):
                args.host_headroom_mib = invalid
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "headroom"):
                    experiment.experiment_plan(args)


class WorkspaceAblationAggregationTests(unittest.TestCase):
    def test_same_components_and_admissions_allow_different_backward_schedule_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, rows = _cohort(directory)
            self.assertNotEqual(rows[0]["packed_offload_runtime"], rows[1]["packed_offload_runtime"])
            summary = experiment.aggregate_workspace(rows, plan)
        self.assertTrue(summary["valid"], summary["invalid_reasons"])
        self.assertTrue(summary["independent_dense_reuse_attribution"])
        self.assertTrue(summary["statistical_design"]["balanced_matrix_complete"])
        self.assertEqual(summary["ratios"]["runtime_speedup_D0_over_D1"]["samples"], 6)
        self.assertFalse(summary["performance_policy_promoted"])

    def test_candidate_geometry_or_actual_admission_drift_retains_points_and_clears_ratios(self):
        for kind in ("eligible", "geometry", "admission"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                _, plan, rows = _cohort(directory, trials=1)
                row = rows[1]
                if kind == "eligible":
                    row["_workspace_history"]["eligible_inventory_sha256"] = "f" * 64
                elif kind == "geometry":
                    part = row["_workspace_history"]["selected_regions"][0]["bundles"][0]["components"][0]
                    part.update(shape=[2, 32], stride=[32, 1])
                else:
                    event = row["packed_offload_runtime"]["snapshot"]["managers"][0]["admission_history"][-1]
                    event.update(selected_bundle_ids=[], resident_fallback=True)
                summary = experiment.aggregate_workspace(rows, plan)
                self.assertFalse(summary["valid"])
                self.assertTrue(summary["combined_point_exploratory"])
                self.assertFalse(summary["ratios"])
                self.assertTrue(all(not block["ratios"] for block in summary["blocks"]))
                self.assertEqual([summary["arms"][arm]["successful_trials"] for arm in ("D0", "D1")], [1, 1])

    def test_effective_flags_pattern_phase_grads_recompile_hash_and_capacity_fail_closed(self):
        for kind in ("effective", "phase", "gradient", "recompile", "hash", "capacity", "headroom", "physical", "nested"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                _, plan, rows = _cohort(directory, trials=1, selection_mode="all_eligible")
                self.assertEqual(experiment._workspace_worker_errors(rows[0], plan, "D0"), [], "the control fixture must satisfy the current full receipt contract")
                row = rows[1]
                if kind == "effective":
                    row["compiler_report_summary"]["effective_compiler_options"][0].pop("native_conv_dx_batch_size")
                elif kind == "phase":
                    row["compiler_report_summary"]["fusion_candidates"][0]["execution_phase"] = "forward"
                elif kind == "gradient":
                    row["steps"][0]["fp32_gradients"] = False
                elif kind == "recompile":
                    row["measurement"]["native_graph_count_after"] = 2
                elif kind == "hash":
                    row["resolved_native_options_sha256"] = "0" * 64
                elif kind == "capacity":
                    row["packed_offload_inventory"]["all_eligible_selected"] = False
                elif kind == "headroom":
                    row["host_capacity_preflight"]["reserved_headroom_bytes"] = 0
                elif kind == "physical":
                    row["packed_offload_runtime"]["snapshot"]["managers"][0]["host"]["pinned_accounting_verified"] = False
                else:
                    row["measurement"] = []
                summary = experiment.aggregate_workspace(rows, plan)
                self.assertFalse(summary["valid"])
                self.assertFalse(summary["ratios"])
                self.assertEqual(summary["arms"]["D0"]["successful_trials"], 1)
                self.assertEqual(summary["arms"]["D1"]["successful_trials"], 0)

    def test_pageable_and_pinned_capacity_are_both_required_before_the_paired_run(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(directory, extra=("--trials", "1"))
            manifest = json.loads(args.experiment_manifest.read_text())
            manifest["packed_activation_offload"]["host_pageable_bytes"] = 4096
            args.experiment_manifest.write_text(json.dumps(manifest))
            plan = experiment.experiment_plan(args)
            rows = [_successful(plan, arm, directory) for arm in plan["active_arms"]]
            summary = experiment.aggregate_workspace(rows, plan)
            self.assertTrue(summary["valid"], summary["invalid_reasons"])
            policy = plan["manifest"]["policies"]["D1"]
            capacity = rows[1]["host_capacity_preflight"]
            self.assertEqual(capacity["required_physical_capacity_bytes"], policy["host_pinned_bytes"] + 4096)
            capacity["required_physical_capacity_bytes"] = policy["host_pinned_bytes"]
            capacity["required_plus_headroom_bytes"] = policy["host_pinned_bytes"] + plan["host_headroom_bytes"]
            failed = experiment.aggregate_workspace(rows, plan)
            self.assertFalse(failed["valid"])
            self.assertFalse(failed["ratios"])
            self.assertEqual(failed["arms"]["D0"]["successful_trials"], 1)
            self.assertEqual(failed["arms"]["D1"]["successful_trials"], 0)

    def test_saved_report_bytes_and_component_byte_counts_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, rows = _cohort(directory, trials=1)
            row = rows[0]
            source = Path(row["compiler_report_artifact"]["path"])
            source.write_text(source.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                experiment._workspace_read_history(row, Path(directory) / "result.json")
            value = {"graphs": [{"native_autodiff_graphs": [{"packed_activation_offload": {"plan": {
                "forward_signature": "f", "selected_bytes": 3,
                "bundles": [{"bundle_id": "b", "components": [], "bytes": 3}]}}}]}]}
            with self.assertRaisesRegex(ValueError, "three distinct"):
                experiment._workspace_history_from_report(value)

    def test_history_uses_declared_owners_despite_large_or_decoy_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "compiler.json"
            _compiler_artifact(source)
            raw = json.loads(source.read_text())
            expected = experiment._workspace_history_from_report(raw)
            for kind in ("large_schedule", "unrelated_offload_decoy"):
                report = deepcopy(raw)
                if kind == "large_schedule":
                    report["graphs"][0]["final_schedule"] = [{} for _ in range(500001)]
                else:
                    decoy = deepcopy(raw["graphs"][0]["native_autodiff_graphs"][0])
                    decoy["packed_activation_offload"]["plan"]["forward_signature"] = "not_a_declared_owner"
                    decoy["packed_activation_offload"]["model_scope"]["inventory"]["eligible_inventory_sha256"] = "f" * 64
                    report["unrelated_diagnostic"] = decoy
                with self.subTest(kind=kind):
                    self.assertEqual(experiment._workspace_history_from_report(report), expected)

    def test_every_declared_history_owner_and_canonical_candidate_must_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "compiler.json"
            _compiler_artifact(source)
            raw = json.loads(source.read_text())
            for kind in ("missing_list", "missing_block", "empty_block", "legacy_block", "missing_plan",
                         "graph_count", "region_count", "aggregate_count", "candidate_conflict", "duplicate_forward"):
                report = {"captured_graphs": 2, "native_autodiff_regions": 2,
                    "graphs": [deepcopy(raw["graphs"][0]), deepcopy(raw["graphs"][0])]}
                graph = report["graphs"][1]
                owner = graph["native_autodiff_graphs"][0]
                owner["packed_activation_offload"]["plan"]["forward_signature"] = "second_native_owner"
                if kind == "missing_list":
                    del graph["native_autodiff_graphs"]
                elif kind == "missing_block":
                    del owner["packed_activation_offload"]
                elif kind == "empty_block":
                    owner["packed_activation_offload"] = {}
                elif kind == "legacy_block":
                    owner["packed_activation_offload"] = {"transport_envelope_version": 1}
                elif kind == "missing_plan":
                    del owner["packed_activation_offload"]["plan"]
                elif kind == "graph_count":
                    report["captured_graphs"] = 3
                elif kind == "region_count":
                    graph["native_autodiff_regions"] = 2
                elif kind == "aggregate_count":
                    report["native_autodiff_regions"] = 3
                elif kind == "candidate_conflict":
                    owner["packed_activation_offload"]["model_scope"]["inventory"]["eligible_inventory_sha256"] = "f" * 64
                else:
                    owner["packed_activation_offload"]["plan"]["forward_signature"] = "forward_encoder"
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    experiment._workspace_history_from_report(report)

    def test_history_keeps_standalone_graph_and_legacy_disabled_nonowner_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "compiler.json"
            _compiler_artifact(source)
            report = json.loads(source.read_text())
            graph = report["graphs"][0]
            expected = experiment._workspace_history_from_report(report)
            self.assertEqual(experiment._workspace_history_from_report(graph), expected)
            for policy in ({"enabled": False, "version": 2}, {"enabled": True, "version": 1}):
                report["graphs"].append({"native_autodiff_regions": 1, "native_autodiff_graphs": [{}],
                    "effective_compiler_options": {"packed_activation_offload": policy}})
            report.update(captured_graphs=3, native_autodiff_regions=3)
            self.assertEqual(experiment._workspace_history_from_report(report), expected)

    def test_state_RNG_or_starting_memory_drift_blocks_independent_attribution(self):
        for kind in ("state", "rng", "starting_memory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                _, plan, rows = _cohort(directory, trials=1)
                row = rows[1]
                if kind == "state":
                    row["optimizer"]["initial_state_sha256"] = "d" * 64
                elif kind == "rng":
                    row["rng_after_measurement_sha256"] = "c" * 64
                else:
                    row["memory_audit"]["snapshots"]["before_model"]["device_free_bytes"] -= 2**30
                summary = experiment.aggregate_workspace(rows, plan)
                self.assertFalse(summary["valid"])
                self.assertFalse(summary["ratios"])
                self.assertFalse(summary["independent_dense_reuse_attribution"])


class WorkspaceAblationOrchestrationTests(unittest.TestCase):
    def _fake_worker(self, plan, directory, seen, *, fail=None):
        def run(command, **kwargs):
            args = native_execution.parser().parse_args(command[3:])
            pin = json.loads(args.arm_manifest.read_text())
            arm = pin["arm"]
            phase = "configuration" if args.configuration_preflight else "clean"
            seen.append((phase, arm))
            row = (configuration_successful(plan, arm) if phase == "configuration"
                   else _successful(plan, arm, directory, command=command))
            if fail == (phase, arm):
                row.update(status="failed", error="ValueError: first actual failure", failure_context={"stage": "setup", "phase": "config_validation"})
            args.output.write_text(json.dumps(row))
            return SimpleNamespace(returncode=1 if fail == (phase, arm) else 0)
        return run

    def test_two_configuration_workers_before_both_clean_peers_and_raw_failure_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(directory, extra=("--trials", "1"))
            plan = experiment.experiment_plan(args)
            seen, terminal = [], io.StringIO()
            with patch.object(experiment.subprocess, "run", side_effect=self._fake_worker(plan, directory, seen, fail=("clean", "D0"))), redirect_stdout(terminal):
                code = experiment.run_experiment(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(seen, [("configuration", "D0"), ("configuration", "D1"), ("clean", "D0"), ("clean", "D1")])
            self.assertEqual(report["summary"]["arms"]["D1"]["successful_trials"], 1)
            self.assertEqual(len(report["results"]), 2)
            self.assertFalse(report["summary"]["ratios"])
            self.assertIn("config_validation: ValueError: first actual failure", terminal.getvalue())
            raw = args.output.parent / report["results"][0]["result"]["path"]
            self.assertEqual(json.loads(raw.read_text())["error"], "ValueError: first actual failure")

    def test_configuration_only_has_no_model_workers_and_failure_never_launches_clean(self):
        for fail in (None, ("configuration", "D0")):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                args = _arguments(directory, extra=("--trials", "1", "--configuration-preflight-only"))
                plan, seen = experiment.experiment_plan(args), []
                with patch.object(experiment.subprocess, "run", side_effect=self._fake_worker(plan, directory, seen, fail=fail)), redirect_stdout(io.StringIO()):
                    code = experiment.run_experiment(args)
                report = json.loads(args.output.read_text())
                self.assertEqual(code, 0 if fail is None else 1)
                self.assertEqual(seen, [("configuration", "D0"), ("configuration", "D1")])
                self.assertFalse(report["execution_completed"])
                self.assertFalse(report["timing_eligible"])
                self.assertEqual(report["results"], [])
                self.assertFalse(report["summary"]["target_CUDA_evidence_complete"])

    def test_legacy_conservative_host_envelope_is_checked_before_any_H0_model_process(self):
        from examples.vae import execution_memory

        for admitted in (False, True):
            with self.subTest(admitted=admitted), tempfile.TemporaryDirectory() as directory:
                args = _arguments(directory, profile=experiment.ARENA_PROFILE, extra=("--trials", "1"))
                plan, seen = experiment.experiment_plan(args), []
                admission = {"admitted": admitted, "required_physical_capacity_bytes": 2 * plan["manifest"]["policies"]["H0"]["host_pinned_bytes"],
                    "reserved_headroom_bytes": plan["host_headroom_bytes"], "requested_capacity_clamped": False}
                fake = WorkspaceAblationOrchestrationTests()._fake_worker(plan, directory, seen)
                with patch.object(execution_memory, "host_capacity_admission", return_value=deepcopy(admission)) as guard, \
                        patch.object(experiment.subprocess, "run", side_effect=fake), redirect_stdout(io.StringIO()):
                    code = experiment.run_experiment(args)
                guard.assert_called_once_with(2 * plan["manifest"]["policies"]["H0"]["host_pinned_bytes"],
                    headroom_bytes=plan["host_headroom_bytes"])
                report = json.loads(args.output.read_text())
                self.assertEqual(code, 0 if admitted else 1)
                self.assertIn(("clean", "H1"), seen)
                self.assertEqual(("clean", "H0") in seen, admitted)
                self.assertEqual(report["summary"]["arms"]["H1"]["successful_trials"], 1)
                if not admitted:
                    record = next(row for row in report["results"] if row["arm"] == "H0")
                    self.assertFalse(record["worker_launched"])
                    self.assertEqual(record["primary_failure"]["kind"], "resource_preflight")
                    raw = json.loads((args.output.parent / record["result"]["path"]).read_text())
                    self.assertTrue(raw["resource_limited"])
                    self.assertEqual(raw["receipt_producer"], "paired_supervisor_resource_preflight")
                    self.assertNotIn("measurement", raw)

    def test_malformed_first_receipt_does_not_prevent_the_second_clean_process(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(directory, extra=("--trials", "1"))
            plan, seen = experiment.experiment_plan(args), []
            ordinary = self._fake_worker(plan, directory, seen)
            def run(command, **kwargs):
                result = ordinary(command, **kwargs)
                options = native_execution.parser().parse_args(command[3:])
                if not options.configuration_preflight and json.loads(options.arm_manifest.read_text())["arm"] == "D0":
                    raw = json.loads(options.output.read_text())
                    raw["measurement"] = []
                    options.output.write_text(json.dumps(raw))
                return result
            with patch.object(experiment.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                code = experiment.run_experiment(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(seen[-2:], [("clean", "D0"), ("clean", "D1")])
            self.assertEqual(report["summary"]["arms"]["D1"]["successful_trials"], 1)
            raw_path = args.output.parent / report["results"][0]["result"]["path"]
            self.assertEqual(json.loads(raw_path.read_text())["measurement"], [])

    def test_combined_public_runner_uses_only_R0_and_declared_BEST_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(directory, combined=True, extra=("--trials", "1"))
            plan, seen = experiment.experiment_plan(args), []
            with patch.object(experiment.subprocess, "run", side_effect=self._fake_worker(plan, directory, seen)), redirect_stdout(io.StringIO()):
                code = experiment.run_experiment(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 0, report.get("error"))
            self.assertEqual(seen, [("configuration", "R0"), ("configuration", "BEST"), ("clean", "R0"), ("clean", "BEST")])
            self.assertTrue(report["summary"]["total_combined_improvement_attribution"])
            self.assertFalse(report["summary"]["independent_dense_reuse_attribution"])
            self.assertFalse(report["summary"]["milestone"]["assessed"])


class CombinedMemoryPlanningAndStatisticsTests(unittest.TestCase):
    def test_combined_stdlib_dry_run_is_fresh_paired_and_python311_compatible(self):
        ast.parse(Path(experiment.__file__).read_text(), feature_version=(3, 11))
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(directory, combined=True)
            result = subprocess.run([sys.executable, "-S", str(Path(experiment.__file__)), "--from-result", str(args.from_result),
                "--experiment-manifest", str(args.experiment_manifest), "--dry-run"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["active_arms"], ["R0", "BEST"])
            self.assertEqual(plan["order"], [["R0", "BEST"], ["BEST", "R0"]] * 3)
            self.assertFalse(plan["manifest"]["policies"]["R0"]["enabled"])
            self.assertTrue(plan["manifest"]["policies"]["BEST"]["enabled"])
            self.assertFalse(plan["residency_control"]["candidate_selected_and_actual_admission_must_match"])
            self.assertFalse(args.output.exists())

    def test_combined_same_candidate_demand_point_keeps_residency_difference_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, rows = _cohort(directory, combined=True)
            for row in rows:
                if row["offload_arm"] == "BEST":
                    row["measurement"]["wall_ms_per_update"] = 9.99
                    for name in ("startup", "total"):
                        row["memory"][name]["peak_allocated_bytes"] = 3200
            summary = experiment.aggregate_workspace(rows, plan)
        self.assertTrue(summary["valid"], summary["invalid_reasons"])
        self.assertFalse(summary["independent_dense_reuse_attribution"])
        self.assertTrue(summary["total_combined_improvement_attribution"])
        self.assertTrue(summary["milestone"]["memory_threshold_observed"])
        self.assertTrue(summary["milestone"]["timing_upper_bound_within_one_percent"])
        self.assertAlmostEqual(summary["milestone"]["one_sided_95_percent_runtime_ratio_upper_bound"], .999)
        self.assertEqual(summary["deltas_right_minus_left"]["total_peak_allocated_bytes"]["median"], -800)
        self.assertFalse(summary["performance_policy_promoted"])

    def test_combined_rejects_predictive_undeclared_math_and_budget_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, original = _manifest(directory, combined=True)
            for kind in ("predictive", "budget", "legacy", "batch", "whole_batch", "resident_enabled"):
                value = deepcopy(original)
                if kind == "predictive":
                    for policy in value["policies"].values():
                        policy["policy"] = "memory_first_overlap"
                elif kind == "budget":
                    value["policies"]["BEST"]["host_pinned_bytes"] *= 2
                elif kind == "legacy":
                    value["arms"]["BEST"]["native_options"]["cuda_conv_dx_geometry"] = True
                elif kind == "batch":
                    value["arms"]["BEST"]["native_options"]["native_conv_dx_batch_size"] = 3
                elif kind == "whole_batch":
                    value["arms"]["BEST"]["native_options"]["native_conv_workspace_reuse"] = "whole_batch"
                else:
                    value["policies"]["R0"]["enabled"] = True
                path.write_text(json.dumps(value))
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    experiment.load_manifest(path)
            path, value = _manifest(directory, combined=True, best_workspace="off")
            self.assertEqual(experiment.load_manifest(path)[0], value)

    def test_one_pair_or_uncertain_runtime_keeps_low_memory_visible_without_milestone(self):
        for trials in (1, 6):
            with self.subTest(trials=trials), tempfile.TemporaryDirectory() as directory:
                _, plan, rows = _cohort(directory, combined=True, trials=trials)
                for row in rows:
                    if row["offload_arm"] == "BEST":
                        row["memory"]["total"]["peak_allocated_bytes"] = 3000
                        row["measurement"]["wall_ms_per_update"] = (9 if row["block"] % 2 else 11)
                summary = experiment.aggregate_workspace(rows, plan)
                self.assertTrue(summary["valid"], summary["invalid_reasons"])
                self.assertTrue(summary["milestone"]["memory_threshold_observed"])
                self.assertFalse(summary["milestone"]["preferred_milestone_estimate_met"])
                self.assertFalse(summary["performance_policy_promoted"])
                self.assertEqual(summary["arms"]["BEST"]["successful_trials"], trials)
                if trials == 6:
                    self.assertTrue(summary["milestone"]["assessed"])
                    self.assertGreater(summary["milestone"]["one_sided_95_percent_runtime_ratio_upper_bound"], 1.01)
                else:
                    self.assertFalse(summary["milestone"]["assessed"])


class IndependentSelectionAndArenaTests(unittest.TestCase):
    def test_selection_is_a_v2_only_axis_and_arena_is_a_version_only_axis(self):
        for profile in (experiment.SELECTION_PROFILE, experiment.ARENA_PROFILE):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                args = _arguments(directory, profile=profile)
                plan = experiment.experiment_plan(args)
                names = experiment.MEMORY_ARMS[profile]
                self.assertEqual(plan["order"], [list(names), list(reversed(names))] * 3)
                self.assertEqual(plan["installed_identities"][names[0]], plan["installed_identities"][names[1]])
                self.assertEqual(plan["configuration_preflight_workers"], 2)
                if profile == experiment.SELECTION_PROFILE:
                    self.assertIn("not an exact historical .3 replay", plan["selection_reference_scope"])
                    self.assertTrue(plan["residency_control"]["eligible_inventory_must_match"])
                    self.assertFalse(plan["residency_control"]["selected_and_actual_admission_must_match"])
                else:
                    self.assertNotIn("version", plan["manifest"]["policies"]["H0"], "legacy wire policy stays unversioned")
                    self.assertEqual(plan["manifest"]["policies"]["H1"]["version"], 2)
                    self.assertTrue(plan["residency_control"]["selected_and_actual_admission_must_match"])

    def test_strict_profiles_reject_unrelated_native_policy_and_pinned_tier_changes(self):
        for profile in (experiment.SELECTION_PROFILE, experiment.ARENA_PROFILE):
            with tempfile.TemporaryDirectory() as directory:
                path, original = _profile_manifest(directory, profile)
                names = experiment.MEMORY_ARMS[profile]
                for kind in ("batch", "workspace", "mode", "version", "budget"):
                    value = deepcopy(original)
                    after = value["arms"][names[1]]["native_options"]
                    if kind == "batch":
                        after["native_conv_dx_batch_size"] = 2
                    elif kind == "workspace":
                        after["native_conv_workspace_reuse"] = "tiled_dx"
                    elif kind == "mode":
                        after["packed_offload_selection_mode"] = "all_eligible" if profile == experiment.SELECTION_PROFILE else "regional_order"
                    elif profile == experiment.SELECTION_PROFILE:
                        value["packed_activation_offload"]["version" if kind == "version" else "enabled"] = 1 if kind == "version" else False
                    else:
                        value["policies"]["H1"]["version" if kind == "version" else "host_pinned_bytes"] = 1
                    path.write_text(json.dumps(value))
                    with self.subTest(profile=profile, kind=kind), self.assertRaises(ValueError):
                        experiment.load_manifest(path)
                if profile == experiment.ARENA_PROFILE:
                    for part in original["policies"].values():
                        part["host_pageable_bytes"] = 1000
                    path.write_text(json.dumps(original))
                    with self.assertRaisesRegex(ValueError, "pinned budgets"):
                        experiment.load_manifest(path)
        for combined in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                path, value = _manifest(directory, combined=combined, selection_mode="regional_order")
                with self.assertRaises(ValueError):
                    experiment.load_manifest(path)

    def test_selection_allows_changed_admissions_only_with_same_eligible_components(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, rows = _cohort(directory, profile=experiment.SELECTION_PROFILE)
            for row in rows:
                if row["offload_arm"] == "S1":
                    _rename_selected(row, "other_eligible_bundle")
            summary = experiment.aggregate_workspace(rows, plan)
            self.assertTrue(summary["valid"], summary["invalid_reasons"])
            self.assertTrue(summary["independent_selection_attribution"])
            self.assertFalse(summary["independent_dense_reuse_attribution"])
            self.assertIn("runtime_speedup_S0_over_S1", summary["ratios"])
            rows[-1]["_workspace_history"]["eligible_inventory_sha256"] = "f" * 64
            invalid = experiment.aggregate_workspace(rows, plan)
            self.assertFalse(invalid["valid"])
            self.assertFalse(invalid["ratios"])
            self.assertTrue(any("eligible_component_inventory_drift" in reason for reason in invalid["invalid_reasons"]))

    def test_arena_compares_actual_same_geometry_and_conservative_process_pinned_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, rows = _cohort(directory, profile=experiment.ARENA_PROFILE)
            self.assertIsNone(rows[0]["_workspace_history"]["eligible_inventory_sha256"])
            summary = experiment.aggregate_workspace(rows, plan)
        self.assertTrue(summary["valid"], summary["invalid_reasons"])
        self.assertTrue(summary["independent_pinned_pool_layout_attribution"])
        self.assertFalse(summary["total_combined_improvement_attribution"])
        self.assertEqual(len(summary["arena_physical_checks"]), 6)
        self.assertTrue(all(check["valid"] and check["input_and_allocator_cache_included"] for check in summary["arena_physical_checks"]))
        self.assertTrue(all(check["legacy_allocator_enforces_cap"] is False for check in summary["arena_physical_checks"]))

    def test_arena_missing_or_exceeded_legacy_physical_cost_or_inventory_drift_suppresses_ratio(self):
        for kind in ("missing", "exceeded", "selection", "admission"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                _, plan, rows = _cohort(directory, profile=experiment.ARENA_PROFILE, trials=1)
                legacy, arena = rows
                pinned = legacy["memory_audit"]["snapshots"]["after_postchecks"]["pinned_host_allocator"]
                if kind == "missing":
                    pinned["statistics"].pop("allocated_bytes.peak")
                elif kind == "exceeded":
                    pinned["statistics"]["allocated_bytes.peak"] = plan["manifest"]["policies"]["H0"]["host_pinned_bytes"] + 1
                elif kind == "selection":
                    _rename_selected(arena, "different_bundle")
                else:
                    event = arena["packed_offload_runtime"]["snapshot"]["managers"][0]["admission_history"][-1]
                    event.update(selected_bundle_ids=[], resident_fallback=True)
                summary = experiment.aggregate_workspace(rows, plan)
                self.assertFalse(summary["valid"])
                self.assertFalse(summary["ratios"])
                self.assertFalse(summary["independent_pinned_pool_layout_attribution"])
                self.assertEqual(summary["arms"]["H1"]["successful_trials"], 1)

    def test_both_isolated_profiles_have_stdlib_dry_runs_and_configuration_only_processes(self):
        for profile in (experiment.SELECTION_PROFILE, experiment.ARENA_PROFILE):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                args = _arguments(directory, profile=profile, extra=("--configuration-preflight-only", "--trials", "1"))
                dry = subprocess.run([sys.executable, "-S", str(Path(experiment.__file__)), "--from-result", str(args.from_result),
                    "--experiment-manifest", str(args.experiment_manifest), "--dry-run"], capture_output=True, text=True)
                self.assertEqual(dry.returncode, 0, dry.stderr)
                self.assertEqual(json.loads(dry.stdout)["experiment_profile"], profile)
                plan, seen = experiment.experiment_plan(args), []
                fake = WorkspaceAblationOrchestrationTests()._fake_worker(plan, directory, seen)
                with patch.object(experiment.subprocess, "run", side_effect=fake), redirect_stdout(io.StringIO()):
                    code = experiment.run_experiment(args)
                self.assertEqual(code, 0)
                self.assertEqual(seen, [("configuration", name) for name in experiment.MEMORY_ARMS[profile]])
                report = json.loads(args.output.read_text())
                self.assertFalse(report["execution_completed"])
                self.assertFalse(report["timing_eligible"])


if __name__ == "__main__":
    unittest.main()
