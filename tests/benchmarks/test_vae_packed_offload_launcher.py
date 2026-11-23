"""Failure archives and launcher wiring; simulated children prove no CUDA facts."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from tests.benchmarks.test_vae_optimization_launcher import SHIM
from tests.benchmarks.test_vae_native_execution import _receipt


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/run_h200_vae_packed_offload.sh"
SIMULATOR = SHIM.replace("VAE_OPTIMIZATION_LAUNCHER:", "VAE_PACKED_OFFLOAD_LAUNCHER:").replace(
    "examples/vae/benchmark_optimization.py", "examples/vae/benchmark_packed_offload.py").replace(
    "simulated A/B/C experiment", "simulated three-arm transport wiring").replace(
    'record("experiment")', 'record("configuration" if "--configuration-preflight-only" in args else "experiment")').replace(
    'code=int(os.environ.get("SIMULATED_EXPERIMENT_CODE","0"))',
    'code=int(os.environ.get("SIMULATED_CONFIGURATION_CODE" if "--configuration-preflight-only" in args else "SIMULATED_EXPERIMENT_CODE","0"))').replace(
    'if "examples/vae/benchmark_packed_offload.py" in args:',
    'if any(name in args for name in ("examples/vae/benchmark_packed_offload.py", "examples/vae/benchmark_packed_offload_frontier.py")):').replace(
    '"TORCHDYNAMO_REPRO_AFTER")', '"DEEPOPT_TEST_CUDA_CONV_DX_BATCH_SIZE","TORCHDYNAMO_REPRO_AFTER")')


class PackedOffloadLauncherTests(unittest.TestCase):
    def fixture(self, root, *, include_runtime=True):
        for name in (SCRIPT, "examples/vae/benchmark_packed_offload.py", "examples/vae/benchmark_optimization.py",
                     "examples/vae/native_execution.py", "examples/vae/benchmark_packed_offload_frontier.py"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        (root / "packages").mkdir()
        wheel = root / "packages/candidate.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("deepopt/__init__.py", "# scalar launcher fixture\n")
            archive.writestr("deepopt/_packed_offload.py", "class PackedActivationOffloadPolicy:\n    enabled: bool = False\n")
            for name in ("compiler/offload_plan.py", "compiler/runtime/manager.py", "compiler/runtime/packed_offload.py"):
                if include_runtime or name != "compiler/runtime/packed_offload.py":
                    archive.writestr("deepopt/" + name, "# fixture code only\n")
            archive.writestr("deep_optimization-1.0.dist-info/METADATA", "Name: deep-optimization\nVersion: 1.0+transport.fixture\n")
        (root / "manifest.json").write_text(json.dumps({"version": "1.0+transport.fixture", "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "experiment_profile": "native_training_fusion_v3", "packed_offload_profile": "packed_offload_v1"}))
        previous, _ = _receipt(root)
        binary = root / "bin"
        binary.mkdir()
        shim = binary / "python"
        shim.write_text(SIMULATOR.replace("@PYTHON@", sys.executable).replace("@PYTHON_LITERAL@", repr(sys.executable)))
        shim.chmod(0o755)
        return previous, dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
            SHIM_EXECUTABLE=str(shim), TORCHDYNAMO_REPRO_AFTER="invalid")

    def run_launcher(self, *, preflight=0, experiment=0, configuration=0, include_runtime=True, extra=(),
                     native_options=None, auto_transport=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous, environment = self.fixture(root, include_runtime=include_runtime)
            environment.update(SIMULATED_PREFLIGHT_CODE=str(preflight), SIMULATED_EXPERIMENT_CODE=str(experiment),
                               SIMULATED_CONFIGURATION_CODE=str(configuration))
            if native_options is not None:
                options_path = root / "native-options.json"
                options_path.write_text(json.dumps(native_options))
                extra = (*extra, "--native-options-json", str(options_path))
            transport = () if auto_transport else ("--device-staging-mib", "4096", "--d2h-inflight-mib", "2048")
            args = ["bash", SCRIPT, str(previous), "--host-pinned-mib", "16384", *transport, *extra]
            result = subprocess.run(args, cwd=root, env=environment, capture_output=True, text=True, timeout=40)
            expected = configuration or preflight or experiment or (0 if include_runtime else 1)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            requests = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()] if (root / "requests.jsonl").exists() else []
            installs = [row for row in requests if row["kind"] == "install"]
            self.assertEqual(len(installs), int(include_runtime))
            for row in installs:
                self.assertIn("--target", row["args"])
                self.assertNotIn("--force-reinstall", row["args"])
            contracts = [row for row in requests if row["kind"] == "preflight"]
            config = [row for row in requests if row["kind"] == "configuration"]
            self.assertEqual(len(config), int(include_runtime))
            if include_runtime:
                self.assertIn("--configuration-preflight-only", config[0]["args"])
                kinds = [row["kind"] for row in requests]
                if not configuration:
                    self.assertLess(kinds.index("configuration"), kinds.index("preflight"))
            if include_runtime and not configuration:
                tiled = native_options is not None and native_options.get("native_conv_workspace_reuse") == "tiled_dx"
                self.assertEqual(len(contracts), 1 if preflight else 6 if tiled else 4)
                for flag in ("--require-cuda", "--require-triton", "--fail-on-skip"):
                    self.assertIn(flag, contracts[0]["args"])
                if tiled and not preflight:
                    original, = [row for row in contracts if "tests.compiler.test_native_convolution_reuse.NativeConvolutionReuseOriginalShapeCudaTests" in row["args"]]
                    self.assertEqual(original["environment"]["DEEPOPT_TEST_CUDA_CONV_DX_BATCH_SIZE"], str(native_options["native_conv_dx_batch_size"]))
                    self.assertIs(original, contracts[0])
                elif not preflight:
                    self.assertFalse(any("test_native_convolution_reuse" in " ".join(row["args"]) for row in contracts))
                legacy = [row for row in contracts if "tests.compiler.test_compiled_packed_offload.CompiledPackedOffloadCudaTests" in row["args"]]
                self.assertEqual(len(legacy), int(not (tiled and preflight)))
                for selector in (() if not legacy else ("tests.compiler.test_compiled_packed_offload.CompiledPackedOffloadCudaTests",
                        "tests.compiler.test_packed_offload_cuda.PackedOffloadCudaTests",
                        "tests.compiler.test_packed_offload_cuda.PackedOffloadOriginalShapeCudaTests",
                        "tests.compiler.test_packed_offload_panels_cuda.PackedOffloadPanelCudaTests",
                        "tests.compiler.test_packed_offload_panels_cuda.PackedOffloadPanelOriginalShapeCudaTests")):
                    self.assertIn(selector, legacy[0]["args"])
            executions = [row for row in requests if row["kind"] == "experiment"]
            self.assertEqual(bool(executions), include_runtime and not preflight and not configuration)
            for row in executions:
                for flag, value in (("--batch-size", "128"), ("--warmup", "5"), ("--updates", "10"), ("--trials", "6")):
                    self.assertEqual(row["args"][row["args"].index(flag) + 1], value)
                self.assertEqual(row["args"][row["args"].index("--host-headroom-mib") + 1], "8192")
                self.assertNotIn("--optimizer", row["args"])
                self.assertNotIn("--synthetic", row["args"])
                self.assertIsNone(row["environment"]["TORCHDYNAMO_REPRO_AFTER"])
            if executions:
                value = json.loads((root / "observed-experiment.json").read_text())
                self.assertEqual(set(value["policies"]), {"resident", "demand_offload", "predictive_offload"})
                self.assertFalse(value["policies"]["resident"]["enabled"])
                self.assertEqual(value["policies"]["demand_offload"]["host_pinned_bytes"], 16 * 2**30)
                self.assertEqual(value["policies"]["predictive_offload"]["host_pinned_bytes"], 16 * 2**30)
                self.assertEqual(value["policies"]["demand_offload"]["admission"], "resident_fallback")
                self.assertEqual(value["policies"]["demand_offload"]["device_staging_bytes"], 0 if auto_transport else 4 * 2**30)
                self.assertEqual(value["policies"]["demand_offload"]["d2h_inflight_bytes"], 0 if auto_transport else 2 * 2**30)
            archives = list((root / "work").glob("*.zip"))
            self.assertEqual(len(archives), 1)
            with zipfile.ZipFile(archives[0]) as archive:
                names = archive.namelist()
                self.assertFalse(any("-installs/" in name or name.endswith((".pt", ".pkl")) for name in names))
                path, = [name for name in names if name.endswith("/launcher.json")]
                receipt = json.loads(archive.read(path))
                self.assertEqual(receipt["exit_code"], expected)
                self.assertFalse(receipt["performance_policy_promoted"])
                if (include_runtime and not configuration and native_options is not None
                        and native_options.get("native_conv_workspace_reuse") == "tiled_dx"):
                    self.assertEqual(receipt["requested_convolution_dx_batch_size"], native_options["native_conv_dx_batch_size"])
                    if preflight:
                        self.assertNotIn("qualified_convolution_dx_batch_size", receipt)
                    else:
                        self.assertEqual(receipt["qualified_convolution_dx_batch_size"], native_options["native_conv_dx_batch_size"])
                inventory_path, = [name for name in names if name.endswith("/archive-manifest.json")]
                for name, metadata in json.loads(archive.read(inventory_path))["files"].items():
                    data = archive.read(name)
                    self.assertEqual(len(data), metadata["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), metadata["sha256"])
                if preflight:
                    self.assertEqual(receipt["stage"], "focused_CUDA_convolution-reuse-original" if
                        native_options is not None and native_options.get("native_conv_workspace_reuse") == "tiled_dx"
                        else "focused_CUDA_contracts")
                if configuration:
                    self.assertEqual(receipt["stage"], "all_arm_configuration_preflight")
                    self.assertFalse(contracts)
                    self.assertFalse(executions)
                    self.assertTrue(any(name.endswith("/configuration-preflight.log") for name in names))

    def test_same_candidate_fresh_target_and_explicit_budgets(self):
        self.run_launcher(extra=("--profile-after-measurement", "--profile-memory"))

    def test_default_transport_limits_follow_the_compiled_plan(self):
        self.run_launcher(auto_transport=True)

    def test_explicit_maximum_routes_to_frontier_without_changing_training_batch(self):
        self.run_launcher(extra=("--host-pinned-mib", "16384", "32768", "65536", "--max-host-pinned-mib", "131072",
            "--cold-diagnostics", "--arms", "demand-only"))

    def test_original_convolution_gate_uses_the_pinned_actual_dX_bound(self):
        self.run_launcher(native_options={"native_conv_workspace_reuse": "tiled_dx", "native_conv_dx_batch_size": 16})

    def test_cuda_contract_failure_archives_and_blocks_all_model_workers(self):
        self.run_launcher(preflight=23)

    def test_failed_cuda_contract_does_not_qualify_requested_dx_bound(self):
        self.run_launcher(preflight=23,
            native_options={"native_conv_workspace_reuse": "tiled_dx", "native_conv_dx_batch_size": 16})

    def test_all_arm_configuration_failure_is_archived_before_CUDA_contracts_or_models(self):
        self.run_launcher(configuration=17)

    def test_worker_failure_retains_logs_and_source_only_artifacts(self):
        self.run_launcher(experiment=29)

    def test_missing_runtime_is_rejected_before_installation(self):
        self.run_launcher(include_runtime=False)

    def test_target_overrides_are_rejected_and_host_budget_is_required(self):
        for extra in (("--batch-size", "16"), ("--synthetic",), ("--no-memory-audit",)):
            result = subprocess.run(["bash", str(ROOT / SCRIPT), "missing.json", *extra], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous, environment = self.fixture(root)
            result = subprocess.run(["bash", SCRIPT, str(previous)], cwd=root, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--host-pinned-mib", result.stderr)
            self.assertFalse((root / "work").exists())

    def test_actual_cpu_cuda_preflight_failure_is_archived_before_any_install(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("negative environment smoke is CPU-only; do not run a model")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous, environment = self.fixture(root)
            shim = root / "bin/python"
            shim.write_text("#!" + sys.executable + "\nimport os,sys\nos.execv(" + repr(sys.executable)
                + ", [" + repr(sys.executable) + ", *sys.argv[1:]])\n")
            result = subprocess.run(["bash", SCRIPT, str(previous), "--host-pinned-mib", "16",
                "--device-staging-mib", "16", "--d2h-inflight-mib", "16"], cwd=root, env=environment,
                capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("no CPU substitution", result.stdout)
            self.assertFalse(list((root / "work").glob("*-installs")))
            archive, = list((root / "work").glob("*.zip"))
            with zipfile.ZipFile(archive) as evidence:
                name, = [part for part in evidence.namelist() if part.endswith("/launcher.json")]
                receipt = json.loads(evidence.read(name))
                self.assertEqual(receipt["stage"], "CUDA_environment_preflight")
                self.assertEqual(len(receipt["commands"]), 1)
                self.assertFalse(receipt["install_trees_archived"])


if __name__ == "__main__":
    unittest.main()
