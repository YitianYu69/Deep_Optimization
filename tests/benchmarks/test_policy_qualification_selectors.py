"""Portable policy plans preserve every case; device evidence stays separate.

Only standard-library planning and source ASTs run here. Selecting a CUDA class
does not execute it or establish numerical, memory or performance qualification.
"""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from compiler import qualify_backend as suite


CPU_SELECTORS = {
    "autodiff": (
        "tests.opt.test_bf16_policy.BF16PolicyTests",
        "tests.compiler.test_bf16_storage_native.BF16StorageActivationTests",
    ),
    "fusion": (
        "tests.opt.test_bf16_storage_normalization.BF16StorageNormalizationTests",
        "tests.opt.test_native_group_staged.NativeGroupStagedTests",
    ),
    "codegen": (
        "tests.opt.test_bf16_dtype_diagnostics.DTypeDiagnosticsTests",
        "tests.opt.test_storage_exp_kernels.StorageExpKernelTests",
        "tests.opt.test_storage_exp_policy_guards.StorageExpPolicyGuardTests",
    ),
    "memory": (
        "tests.compiler.test_policy_routes_physical_memory.PolicyRouteTests",
        "tests.compiler.test_policy_routes_physical_memory.PhysicalObservationTests",
    ),
    "integration": (
        "tests.cuda.test_raw_bf16_policy_guard.RawBF16PolicyGuardTests",
        "tests.benchmarks.test_bf16_policy_matrix.BF16PolicyMatrixTests",
    ),
}
CUDA_SELECTORS = {
    "bf16-storage-native-cuda": (
        "tests.compiler.test_bf16_storage_native.BF16StorageNativeCUDATests",
    ),
    "bf16-storage-normalization-cuda": (
        "tests.opt.test_bf16_storage_normalization.BF16StorageNormalizationCudaTests",
        "tests.opt.test_native_group_staged.NativeGroupStagedCudaTests",
    ),
    "storage-exp-cuda": (
        "tests.opt.test_storage_exp_kernels.StorageExpCudaTests",
        "tests.opt.test_storage_exp_policy_guards.StorageExpIndependentOracleCUDA",
    ),
    "physical-memory-cuda": (
        "tests.compiler.test_policy_routes_physical_memory.PhysicalMemoryCUDATests",
    ),
    "raw-policy-replay-cuda": (
        "tests.cuda.test_raw_bf16_policy_guard.RawBF16PolicyGuardCUDATests",
    ),
}
MULTI_GPU_SELECTORS = {
    "physical-memory-multi-gpu-cuda": (
        "tests.compiler.test_policy_routes_physical_memory.PhysicalMemoryMultiGPUCUDATests",
    ),
    "codec-multi-gpu-cuda": (
        "tests.opt.test_bf16_codegen_safety.BF16CodecMultiGPUCUDA",
    ),
}
ORIGINAL_FULL_STAGES = (
    "autodiff-contracts", "fusion-contracts", "codegen-contracts", "memory-contracts", "integration-contracts",
    "semantic-followup-cuda", "vae-cuda", "attention-cuda", "native-bf16-cuda", "fusion-cuda",
    "configuration-cuda", "pointwise-replay-cuda", "fp8-cuda", "role-kernel-benchmarks",
    "fusion-dense", "fusion-dense-gelu", "fusion-dense-gelu-tanh", "fusion-compressed",
    "fusion-gemm-compressed-silu", "fusion-gemm-compressed-gelu", "fusion-gemm-compressed-gelu-tanh",
    "fusion-compressed-gelu", "fusion-compressed-gelu-tanh", "fusion-packed-multiply", "fusion-shared-packs",
    "fusion-residual", "fusion-groupnorm-amp", "fusion-dense-bf16-storage", "fusion-dense-gelu-bf16-storage",
    "fusion-dense-gelu-tanh-bf16-storage", "fusion-compressed-bf16-storage",
    "fusion-gemm-compressed-silu-bf16-storage", "fusion-gemm-compressed-gelu-bf16-storage",
    "fusion-gemm-compressed-gelu-tanh-bf16-storage", "fusion-compressed-gelu-bf16-storage",
    "fusion-compressed-gelu-tanh-bf16-storage", "fusion-packed-multiply-bf16-storage",
    "fusion-shared-packs-bf16-storage", "fusion-residual-bf16-storage", "fusion-groupnorm-amp-bf16-storage",
    "linear-mlp-vit",
)


def modules(stage):
    command = stage["command"]
    return tuple(command[command.index("--modules") + 1:command.index("--device")])


class PolicyQualificationSelectorTests(unittest.TestCase):
    def plan(self, *arguments):
        return suite.make_plan(suite.parser().parse_args(list(arguments)))

    def test_cpu_policy_selectors_are_explicit_host_classes(self):
        plan = self.plan("--profile", "cpu")
        stages = {row["name"]: row for row in plan["stages"]}
        for area, selectors in CPU_SELECTORS.items():
            stage = stages[area + "-contracts"]
            self.assertEqual(stage["validator"], "tests")
            self.assertEqual(stage["command"][stage["command"].index("--device") + 1], "cpu")
            self.assertTrue(set(selectors).issubset(modules(stage)))
        all_cpu = {selector for row in plan["stages"] if row["validator"] == "tests" for selector in modules(row)}
        for selectors in (*CUDA_SELECTORS.values(), *MULTI_GPU_SELECTORS.values()):
            self.assertTrue(all_cpu.isdisjoint(selectors))
        self.assertFalse(any(row["require_observed"] for row in plan["stages"]))

    def test_single_gpu_policy_stages_require_actual_execution_and_no_skips(self):
        for profile in ("gpu-smoke", "gpu-full"):
            with self.subTest(profile=profile):
                plan = self.plan("--profile", profile)
                stages = {row["name"]: row for row in plan["stages"]}
                self.assertFalse(plan["execution_policy"]["multi_gpu_checks_requested"])
                self.assertEqual(plan["execution_policy"]["multi_gpu_stages"], ())
                for name, selectors in CUDA_SELECTORS.items():
                    stage = stages[name]
                    self.assertEqual(modules(stage), selectors)
                    self.assertEqual(stage["validator"], "tests_strict")
                    for flag in ("--require-cuda", "--require-triton", "--fail-on-skip"):
                        self.assertIn(flag, stage["command"])
                selected = {selector for row in plan["stages"] if row["validator"] == "tests_strict"
                            for selector in modules(row)}
                for selectors in MULTI_GPU_SELECTORS.values():
                    self.assertTrue(selected.isdisjoint(selectors))

    def test_two_gpu_checks_are_additive_explicit_and_never_cpu(self):
        default = self.plan("--profile", "gpu-full")
        opted = self.plan("--profile", "gpu-full", "--multi-gpu")
        defaults = {row["name"]: row for row in default["stages"]}
        opteds = {row["name"]: row for row in opted["stages"]}
        self.assertEqual(set(opteds) - set(defaults), set(MULTI_GPU_SELECTORS))
        self.assertEqual(defaults, {name: opteds[name] for name in defaults})
        self.assertTrue(opted["execution_policy"]["multi_gpu_checks_requested"])
        self.assertEqual(set(opted["execution_policy"]["multi_gpu_stages"]), set(MULTI_GPU_SELECTORS))
        for name, selectors in MULTI_GPU_SELECTORS.items():
            self.assertEqual(modules(opteds[name]), selectors)
            self.assertIn("--fail-on-skip", opteds[name]["command"])
            self.assertEqual(opteds[name]["validator"], "tests_strict")
        for profile in ("cpu", "metadata"):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "GPU profile"):
                self.plan("--profile", profile, "--multi-gpu")

    def test_every_original_full_stage_and_sample_setting_is_preserved(self):
        plan = self.plan("--profile", "gpu-full")
        names = [row["name"] for row in plan["stages"]]
        self.assertEqual([name for name in names if name in ORIGINAL_FULL_STAGES], list(ORIGINAL_FULL_STAGES))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names) - set(ORIGINAL_FULL_STAGES),
                         {*CUDA_SELECTORS, "linear-mlp-vit-bf16-storage"})
        self.assertEqual(plan["timing"], {"warmup": 10, "iterations": 50, "repeats": 5, "trials": 3})
        self.assertEqual(plan["execution_policy"]["gpu_timing_concurrency"], 1)
        self.assertFalse(plan["performance_claims_automatic"])
        for stage in plan["stages"]:
            command = stage["command"]
            if "--group-size" in command:
                self.assertEqual(command[command.index("--group-size") + 1], "1024")
            if command[0].endswith("isolated_fusion.py") or stage["validator"] in ("models", "kernels"):
                for key, expected in plan["timing"].items():
                    self.assertEqual(command[command.index("--" + key) + 1], str(expected))

    def test_fusion_recipe_axes_keep_all_915_main_and_1116_diagnostic_jobs(self):
        from compiler import isolated_fusion

        totals = {policy: [0, 0] for policy in ("torch_amp", "bf16_storage")}
        identities = {}
        for stage in self.plan("--profile", "gpu-full")["stages"]:
            command = stage["command"]
            if not command[0].endswith("isolated_fusion.py"):
                continue
            args = isolated_fusion.parser().parse_args(command[1:])
            frozen = isolated_fusion.make_plan(args)
            policy = args.bf16_policy
            self.assertIn("--qualify-native-group-fusion", command)
            totals[policy][0] += len(frozen["jobs"])
            totals[policy][1] += len(frozen["diagnostic_jobs"])
            signatures = []
            for job in (*frozen["jobs"], *frozen["diagnostic_jobs"]):
                self.assertEqual(job["fixture"]["bf16_policy"], policy)
                self.assertEqual(job["fixture"]["group_size"], 1024)
                self.assertTrue(job["strategy"]["options"]["qualify_native_group_fusion"])
                self.assertEqual(job["measurement"]["iterations"], 50)
                self.assertEqual(job["measurement"]["repeats"], 5)
                fixture = {name: value for name, value in job["fixture"].items() if name != "bf16_policy"}
                signatures.append({"fixture": fixture, "strategy": job["strategy"], "seed": job["seed"],
                                   "trial": job["trial"], "measurement": job["measurement"],
                                   "diagnostic_order": job.get("diagnostic_order")})
            name = stage["name"].removesuffix("-bf16-storage")
            identities.setdefault(name, {})[policy] = signatures
        self.assertEqual(totals, {"torch_amp": [915, 1116], "bf16_storage": [915, 1116]})
        for pair in identities.values():
            self.assertEqual(pair["torch_amp"], pair["bf16_storage"])

    def test_model_policy_stages_keep_270_plus_18_and_native_controls_amp(self):
        from analysis.matrix import bf16_policy_identity
        from compiler import benchmark_native_autodiff

        model_stages = [row for row in self.plan("--profile", "gpu-full")["stages"] if row["validator"] == "models"]
        self.assertEqual([row["name"] for row in model_stages], ["linear-mlp-vit", "linear-mlp-vit-bf16-storage"])
        controls = []
        with tempfile.TemporaryDirectory() as directory:
            for stage in model_stages:
                command = list(stage["command"][1:])
                policy = command[command.index("--bf16-policy") + 1]
                output = Path(directory) / policy
                command[command.index("--output") + 1] = str(output)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(benchmark_native_autodiff.main([*command, "--dry-run"]), 0)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["bf16_policy"], {"name": policy, "version": 1})
                original, native, control = [], [], []
                for job in manifest["jobs"]:
                    for task in job["expected_tasks"]:
                        settings = task["settings"]
                        self.assertEqual(settings["compression"]["group_size"], 1024)
                        expected = "torch_amp" if task["case"] == "torch_bf16" else policy
                        self.assertEqual(bf16_policy_identity(settings, task["case"]), {"name": expected, "version": 1})
                        self.assertIsNone(task["skip_reason"])
                        if job["strategy"] == "torch-inductor":
                            native.append(task)
                            self.assertEqual(task["case"], "torch_bf16")
                            self.assertEqual(settings["runtime"]["bf16_policy"], "torch_amp")
                            self.assertEqual(settings["runtime"]["compile"]["backend"], "inductor")
                            control.append(settings)
                        else:
                            original.append(task)
                self.assertEqual((len(original), len(native)), (270, 18))
                self.assertTrue(manifest["inductor_reference"]["native_torch_ops_only"])
                controls.append(control)
        self.assertEqual(controls[0], controls[1])

    def test_policy_stage_validation_rejects_stale_or_forged_model_identity(self):
        stage = {"validator": "models", "command": ["worker", "--bf16-policy", "bf16_storage"]}
        def row(strategy, case, policy):
            receipt = {"passed": True, "policy": {"name": policy, "version": 1}}
            return {"passed": True, "strategy": strategy, "key": [0, "linear-b8", case],
                    "same_bf16_policy": True, "bf16_policy": deepcopy(receipt),
                    "reference_bf16_policy": deepcopy(receipt)}

        data = {"passed": True, "comparisons": [row("triton-native-control", "compression", "bf16_storage"),
                    row("torch-inductor", "torch_bf16", "torch_amp")]}
        self.assertTrue(suite.validate_report(stage, data)[0])
        for path, invalid in (("name", "torch_amp"), ("version", True), ("version", "1")):
            bad = deepcopy(data)
            bad["comparisons"][0]["bf16_policy"]["policy"][path] = invalid
            self.assertFalse(suite.validate_report(stage, bad)[0])
        for field in ("same_bf16_policy", "bf16_policy", "reference_bf16_policy"):
            bad = deepcopy(data)
            bad["comparisons"][0].pop(field)
            self.assertFalse(suite.validate_report(stage, bad)[0])
        bad = deepcopy(data)
        bad["comparisons"][1]["reference_bf16_policy"]["policy"]["name"] = "bf16_storage"
        self.assertFalse(suite.validate_report(stage, bad)[0])

    def test_all_selected_classes_exist_without_importing_device_test_modules(self):
        for option in ((), ("--multi-gpu",)):
            for stage in self.plan("--profile", "gpu-full", *option)["stages"]:
                if "--modules" not in stage["command"]:
                    continue
                for selector in modules(stage):
                    with self.subTest(selector=selector):
                        parts = selector.split(".")
                        for end in range(len(parts), 0, -1):
                            source = ROOT.joinpath(*parts[:end]).with_suffix(".py")
                            if source.is_file():
                                nodes = ast.parse(source.read_text()).body
                                break
                        else:
                            self.fail(f"Missing selector source: {selector}")
                        for attribute in parts[end:]:
                            node = next((node for node in nodes if getattr(node, "name", None) == attribute), None)
                            self.assertIsNotNone(node, selector)
                            nodes = getattr(node, "body", ())

    def test_planning_is_standard_library_only_and_does_not_claim_execution(self):
        code = """
import json,sys
sys.path.insert(0,'benchmarks')
from compiler import qualify_backend as suite
plan=suite.make_plan(suite.parser().parse_args(['--profile','gpu-full','--multi-gpu']))
assert 'torch' not in sys.modules and 'deepopt' not in sys.modules
assert not plan['performance_claims_automatic']
print(json.dumps({'stages':len(plan['stages']),'planning_only':True}))
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), {"stages": 49, "planning_only": True})
        limits = " ".join(self.plan("--profile", "gpu-full")["limitations"])
        self.assertIn("inference-only", limits)
        self.assertIn("Bounded panel decoding", limits)
        self.assertIn("separate strict target stages", limits)


if __name__ == "__main__":
    unittest.main()
