"""Pinned DDP launcher contracts; simulated rank receipts are not GPU evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from unittest.mock import patch
import zipfile

from benchmarks.compiler import qualify_packed_offload_ddp as qualification


ROOT = Path(__file__).resolve().parents[2]


def fixture(root):
    for name in qualification.REQUIRED_SOURCES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original = ROOT / name
        if original.is_file():
            shutil.copyfile(original, target)
        else:
            target.write_text("# scalar worker wiring fixture\n")
    package = {name: b"# release pin fixture; never imported\n" for name in qualification.REQUIRED_PACKAGE}
    digest = hashlib.sha256()
    for name in sorted(package, key=lambda value: Path(value).parts):
        digest.update(name.encode() + b"\0" + package[name] + b"\0")
    wheel = root / "packages/candidate.whl"
    wheel.parent.mkdir()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in package.items():
            archive.writestr("deepopt/" + name, data)
        archive.writestr("deep_optimization-1.0.dist-info/METADATA",
            "Name: deep-optimization\nVersion: 1.0+packedddp.fixture\n")
    manifest = {"schema_version": 1, "profile": qualification.PROFILE,
        "candidate": {"wheel": {"path": "packages/candidate.whl", "sha256": qualification._sha(wheel)},
            "expected_installed": {"version": "1.0+packedddp.fixture", "source_sha256": digest.hexdigest(), "files_hashed": len(package)}},
        "sources": {name: {key: value for key, value in qualification._reference(root / name).items() if key != "path"}
            for name in qualification.REQUIRED_SOURCES}}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    args = qualification.parser().parse_args(["--manifest", str(path), "--installed-target", str(root / "installed"),
        "--output", str(root / "work/result.json")])
    return args, manifest


def identity(plan):
    target = Path(plan["installed_target"])
    return {**plan["expected_installed"], "distribution_name": "deep-optimization", "editable": False,
        "install_root": str(target), "package_path": str(target / "deepopt"), "import_path": str(target / "deepopt/__init__.py")}


def rank_receipt(rank, plan, installed):
    cases = ("preflight", "forward_error", "native_demand", "checkpoint_demand",
        "native_memory_first_overlap", "checkpoint_memory_first_overlap", "native_demand_torch_amp",
        "checkpoint_demand_torch_amp") if plan["device"] == "cuda" else (
        "preflight", "forward_error", "native_resident", "checkpoint_resident", "generated_testing_transport")
    required = (*qualification.COMMON_CHECKS, "actual_cuda_startup_preflight", "whole_model_offload_plan_preserved",
        "one_training_runtime_transport_owner_per_rank") if plan["device"] == "cuda" else (
        *qualification.COMMON_CHECKS, "cpu_offload_rejected", "generated_joint_saved_abi", "graph_leases_released", "ddp_optimizer_graph_split")
    return {"schema_version": 1, "status": "completed", "rank": rank, "local_rank": rank,
        "world_size": plan["processes"], "backend": plan["backend"], "device": f"cuda:{rank}" if plan["device"] == "cuda" else "cpu",
        "installed": installed, "installed_after": installed, "group_initialized_after_shutdown": False, "tensor_files_saved": 0,
        "scope": "CUDA_NCCL_real_packed_offload" if plan["device"] == "cuda" else "CPU_Gloo_native_and_explicit_test_transport",
        "cuda_execution_verified": plan["device"] == "cuda", "test_transport": plan["device"] == "cpu",
        "checks": dict.fromkeys(required, True), "cases": [{"name": name, "checks": dict.fromkeys(required, True),
            "compiler_scope": {"ddp_optimizer_partitioning": "disabled", "communication_overlap_qualified": False},
            "compiler": {"captured_graphs": 1, "native_autodiff_operations": 3, "native_autodiff_regions": 1,
                "runtime_fallbacks": 0, "legacy_backward_operations": []}} for name in cases]}


class PackedOffloadDDPQualificationTests(unittest.TestCase):
    def test_dry_run_is_stdlib_only_pins_sources_and_preserves_explicit_four_ranks(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(qualification, "ROOT", Path(directory)):
            args, _ = fixture(Path(directory))
            args.processes, args.dry_run = 4, True
            output = io.StringIO()
            with patch.object(qualification.importlib, "import_module", side_effect=AssertionError("No runtime import in dry-run")), redirect_stdout(output):
                self.assertEqual(qualification.run(args), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["processes"], 4)
            self.assertFalse(plan["CUDA_execution_proven"])
            self.assertFalse(plan["source_substitution_allowed"])
            self.assertEqual(plan["expected_installed"]["files_hashed"], 3)
            self.assertFalse(args.output.exists())

    def test_version_wheel_and_source_drift_are_rejected_before_runtime(self):
        for mismatch in ("version", "wheel", "source"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory, patch.object(qualification, "ROOT", Path(directory)):
                args, manifest = fixture(Path(directory))
                if mismatch == "version":
                    manifest["candidate"]["expected_installed"]["version"] = "incorrect"
                    args.manifest.write_text(json.dumps(manifest))
                elif mismatch == "wheel":
                    (Path(directory) / "packages/candidate.whl").write_bytes(b"changed")
                else:
                    (Path(directory) / qualification.WORKER).write_text("# changed worker\n")
                with patch.object(qualification, "_device_preflight") as runtime, self.assertRaises(ValueError):
                    qualification.qualification_plan(args)
                runtime.assert_not_called()

    def test_single_rank_cpu_substitution_and_unbounded_timeout_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(qualification, "ROOT", Path(directory)):
            args, _ = fixture(Path(directory))
            for values in ({"processes": 1}, {"timeout_seconds": 1801}, {"device": "cpu", "require_h200": True}):
                options = deepcopy(args)
                for key, value in values.items():
                    setattr(options, key, value)
                with self.subTest(values=values), self.assertRaises(ValueError):
                    qualification.qualification_plan(options)

    def test_collects_every_rank_and_rejects_missing_identity_or_fake_cuda(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(qualification, "ROOT", Path(directory)):
            root = Path(directory).resolve()
            args, _ = fixture(root)
            plan = qualification.qualification_plan(args)
            installed = identity(plan)
            ranks = root / "ranks"
            ranks.mkdir()
            first = rank_receipt(0, plan, installed)
            (ranks / "rank-0.json").write_text(json.dumps(first))
            rows, errors = qualification._rank_results(ranks, plan, installed)
            self.assertTrue(rows[0]["valid"])
            self.assertFalse(rows[1]["valid"])
            self.assertTrue(errors)
            second = rank_receipt(1, plan, installed)
            for field, value in (("installed", {}), ("test_transport", True), ("cuda_execution_verified", False),
                    ("checks", {"checkpoint_next_update": True}), ("cases", second["cases"][:-1]), ("device", "cpu")):
                bad = {**second, field: value}
                (ranks / "rank-1.json").write_text(json.dumps(bad))
                with self.subTest(field=field):
                    rows, errors = qualification._rank_results(ranks, plan, installed)
                    self.assertTrue(rows[0]["valid"])
                    self.assertTrue(errors)
            (ranks / "rank-1.json").write_text(json.dumps(second))
            self.assertFalse(qualification._rank_results(ranks, plan, installed)[1])

    def test_failed_peer_retains_primary_receipt_and_only_metadata_is_archived(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(qualification, "ROOT", Path(directory)):
            root = Path(directory).resolve()
            args, manifest = fixture(root)
            documentation = root / "docs/extra-pinned-source.md"
            documentation.parent.mkdir()
            documentation.write_text("Pinned qualification scope.\n")
            manifest["sources"]["docs/extra-pinned-source.md"] = {key: value for key, value in qualification._reference(documentation).items() if key != "path"}
            args.manifest.write_text(json.dumps(manifest))
            plan = qualification.qualification_plan(args)
            installed = identity(plan)

            def simulate(command, log, environment, timeout_seconds, *, stage):
                self.assertNotIn("--allow-source", command)
                self.assertIn("--require-cuda", command)
                self.assertEqual(environment["PYTHONPATH"].split(os.pathsep), [plan["installed_target"], str(root / "benchmarks"), str(root)])
                self.assertNotIn("TORCHDYNAMO_REPRO_AFTER", environment)
                rank_path = Path(command[command.index("--output-dir") + 1])
                for rank in range(2):
                    row = rank_receipt(rank, plan, installed)
                    if rank == 1:
                        row.update(status="failed", error="ValueError: rank failure", failure_context={"phase": "backward"})
                    (rank_path / f"rank-{rank}.json").write_text(json.dumps(row))
                (rank_path / "temporary-checkpoint.pt").write_bytes(b"test checkpoint must never be archived")
                Path(log).write_text("simulated rank progress\n")
                return {"exit_code": 1, "timed_out": False, "command": command, "log": qualification._reference(log)}

            with patch.object(qualification, "_device_preflight", return_value={"devices": []}), \
                    patch.object(qualification, "_installed", return_value=installed), \
                    patch.object(qualification, "_stream_process", side_effect=simulate), redirect_stdout(io.StringIO()):
                self.assertEqual(qualification.run(args), 1)
            report = json.loads(args.output.read_text())
            self.assertTrue(report["ranks"][0]["valid"])
            self.assertEqual(report["ranks"][1]["receipt"]["failure_context"]["phase"], "backward")
            self.assertFalse(report["CUDA_execution_proven"])
            self.assertTrue(report["error"])
            with zipfile.ZipFile(report["archive"]["path"]) as archive:
                self.assertFalse(any(name.endswith((".pt", ".pkl", ".pickle")) or "-install/" in name for name in archive.namelist()))
                self.assertIn("sources/docs/extra-pinned-source.md", archive.namelist())
                metadata = json.loads(archive.read("archive-manifest.json"))
                for name, pin in metadata["files"].items():
                    data = archive.read(name)
                    self.assertEqual(len(data), pin["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), pin["sha256"])

    def test_success_requires_complete_suite_and_cpu_remains_diagnostic(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory, patch.object(qualification, "ROOT", Path(directory)):
                root = Path(directory).resolve()
                args, _ = fixture(root)
                args.device = device
                plan = qualification.qualification_plan(args)
                installed = identity(plan)

                def simulate(command, log, environment, timeout_seconds, *, stage):
                    rank_path = Path(command[command.index("--output-dir") + 1])
                    for rank in range(2):
                        (rank_path / f"rank-{rank}.json").write_text(json.dumps(rank_receipt(rank, plan, installed)))
                    Path(log).write_text("simulated complete suite\n")
                    return {"exit_code": 0, "timed_out": False, "log": qualification._reference(log)}

                with patch.object(qualification, "_device_preflight", return_value={"device": device}), \
                        patch.object(qualification, "_installed", return_value=installed), \
                        patch.object(qualification, "_stream_process", side_effect=simulate), redirect_stdout(io.StringIO()):
                    self.assertEqual(qualification.run(args), 0)
                report = json.loads(args.output.read_text())
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["CUDA_execution_proven"], device == "cuda")
                self.assertFalse(report["performance_qualified"])
                self.assertTrue(all(row["valid"] for row in report["ranks"]))

    def test_guide_preserves_user_geometry_AMP_optimizer_and_global_batch(self):
        guide = (ROOT / "docs/execution/packed_activation_offload_ddp.md").read_text()
        raw = tomllib.loads((ROOT / "examples/vae/native_ddp_packed_offload.toml").read_text())
        snippet = tomllib.loads(guide.split("```toml\n", 1)[1].split("```", 1)[0])
        self.assertEqual(snippet["runtime"]["packed_activation_offload"], raw["runtime"]["packed_activation_offload"])
        self.assertEqual(raw["training"]["batch_size"] * 4 * raw["training"]["accumulation_steps"], 768)
        self.assertEqual((raw["application"]["data"]["image_size"], raw["application"]["data"]["resize_size"]), (158, 158))
        self.assertEqual(raw["runtime"]["bf16_policy"], "torch_amp")
        self.assertNotIn("cuda_graph", raw["runtime"])
        policy = raw["runtime"]["packed_activation_offload"]
        self.assertEqual(policy["host_pinned_gb"], 80)
        self.assertNotIn("device_staging_gb", policy)
        self.assertNotIn("d2h_inflight_gb", policy)
        self.assertEqual(policy["host_pageable_gb"], 0)
        self.assertNotIn("host_pinned_bytes", policy)
        self.assertNotIn("copy_chunk_bytes", policy)
        self.assertNotIn("reduction_scratch_bytes", raw["runtime"]["activation_memory"])
        self.assertEqual(raw["runtime"]["compile"]["options"]["packed_offload_selection_mode"], "peak")
        self.assertEqual(raw["runtime"]["activation_compression"]["group_size"], 1024)
        self.assertEqual(raw["training"]["optimizer"], "custom")
        self.assertEqual(raw["training"]["checkpoint_every_steps"], 200)
        self.assertEqual(raw["training"]["learning_rate"], .001)

    @unittest.skipUnless(os.name == "posix", "whole-process-group termination uses POSIX")
    def test_actual_timeout_stops_parent_and_descendant_and_streams_progress(self):
        for leader_wait in (10, 0):
            with self.subTest(leader_wait=leader_wait), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                marker = root / "descendant-finished.txt"
                leaf = f"import pathlib,time;time.sleep(2);pathlib.Path({str(marker)!r}).write_text('not stopped')"
                command = [sys.executable, "-u", "-c",
                    f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{leaf!r}]);print('worker started',flush=True);time.sleep({leader_wait})"]
                terminal = io.StringIO()
                with redirect_stdout(terminal):
                    result = qualification._stream_process(command, root / "worker.log", dict(os.environ), .3, stage="timeout_test")
                self.assertTrue(result["timed_out"])
                self.assertLess(result["elapsed_seconds"], 7)
                self.assertIn("worker started", (root / "worker.log").read_text())
                self.assertIn("START timeout_test", terminal.getvalue())
                time.sleep(2)
                self.assertFalse(marker.exists(), "descendant survived whole-group timeout")

    def test_shell_help_works_without_install_or_worker_execution(self):
        environment = dict(os.environ, PYTHON=sys.executable)
        result = subprocess.run(["bash", str(ROOT / "scripts/run_h200_packed_offload_ddp.sh"), "--help"],
            env=environment, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--processes", result.stdout)
        self.assertIn("--timeout-seconds", result.stdout)

    def test_actual_cpu_cuda_preflight_failure_precedes_pip_and_is_archived(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("CPU-only negative CUDA preflight smoke")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(qualification, "ROOT", root):
                args, _ = fixture(root)
            environment = dict(os.environ, PYTHON=sys.executable)
            result = subprocess.run(["bash", str(root / "scripts/run_h200_packed_offload_ddp.sh"),
                "--manifest", str(args.manifest), "--output", str(args.output)],
                env=environment, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["stage"], "device_preflight")
            self.assertFalse(report["commands"])
            self.assertFalse(Path(report["installed_target"]).exists())
            self.assertTrue(Path(report["archive"]["path"]).is_file())
            self.assertIn("no CPU substitution", report["error"])


if __name__ == "__main__":
    unittest.main()
