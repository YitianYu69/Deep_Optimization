"""Launcher process/archival tests; simulated workers establish no CUDA evidence."""

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


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "dist/vae-diagnostics-20261119/deep_optimization-0.1.0+nativeautodiff.20261119.vaememory.4-py3-none-any.whl"
if not CONTROL.is_file() and (ROOT / "manifest.json").is_file():
    bundled = json.loads((ROOT / "manifest.json").read_text()).get("control", {}).get("bundle_wheel")
    if bundled:
        CONTROL = ROOT / bundled
CONTROL_SHA = "d2b7c417237211cbf8853c47d72028f1aa3cdf9d7782de715787dc345195ab94"
CONTROL_SOURCE = "fb5a619fad46e77831fd97ba00818af870db46feb6dc39bc3b32c6392cf09775"
SHIM = '''#!@PYTHON@
from pathlib import Path
import json, os, subprocess, sys, zipfile
args = sys.argv[1:]
def record(kind):
    with Path("requests.jsonl").open("a") as stream:
        stream.write(json.dumps({"kind":kind,"args":args,"environment":{name:os.environ.get(name) for name in
            ("PYTHONPATH","DEEPOPT_RUN_H200_NORM_CAT","DEEPOPT_TEST_LARGE_CUDA_DONATION",
             "DEEPOPT_RUN_H200_CONV_ORDER","DEEPOPT_TEST_LARGE_CUDA_CONV_GEOMETRY",
             "DEEPOPT_RUN_H200_NORM_ACTIVATION","DEEPOPT_TEST_LARGE_CUDA_SWIGLU",
             "DEEPOPT_FUSION_DIAGNOSTIC_DIR",
             "DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY","TORCHDYNAMO_REPRO_AFTER")}})+"\\n")
if args[:2] == ["-m","pip"]:
    record("install")
    target=Path(args[args.index("--target")+1])
    if target.name == os.environ.get("SIMULATED_INSTALL_FAILURE"):
        print("simulated package installation failure",flush=True)
        raise SystemExit(31)
    with zipfile.ZipFile(args[-1]) as wheel:
        wheel.extractall(target)
    raise SystemExit(0)
if args and args[0] == "-c" and "torch.cuda.is_available" in args[1]:
    record("environment")
    print("simulated CUDA availability; wiring test only",flush=True)
    raise SystemExit(0)
if "benchmarks/compiler/qualification_tests.py" in args:
    record("preflight")
    code=int(os.environ.get("SIMULATED_PREFLIGHT_CODE","0"))
    output=Path(args[args.index("--output")+1])
    if os.environ.get("DEEPOPT_FUSION_DIAGNOSTIC_DIR"):
        diagnostic=Path(os.environ["DEEPOPT_FUSION_DIAGNOSTIC_DIR"])/"layout-case"/"partial-affine"
        diagnostic.mkdir(parents=True)
        for extension in ("ttir","ttgir","llir","ptx","sass"):
            (diagnostic/("kernel."+extension)).write_text("// simulated compiled kernel code only\\n")
        (diagnostic/"metadata.json").write_text(json.dumps({"tensor_payloads":False,"cuda_evidence":False}))
        (diagnostic/"excluded.pt").write_text("suffix exclusion sentinel; not a tensor")
    output.write_text(json.dumps({"scope":"simulated_launcher_test","passed":code==0,"cuda_evidence":False}))
    print("simulated focused contract tests",flush=True)
    raise SystemExit(code)
if "examples/vae/benchmark_optimization.py" in args:
    record("experiment")
    code=int(os.environ.get("SIMULATED_EXPERIMENT_CODE","0"))
    output=Path(args[args.index("--output")+1])
    experiment=json.loads(Path(args[args.index("--experiment-manifest")+1]).read_text())
    Path("observed-experiment.json").write_text(json.dumps(experiment))
    output.write_text(json.dumps({"status":"completed" if code==0 else "failed","execution_completed":code==0,"cuda_evidence":False}))
    artifacts=output.parent/(output.stem+"-artifacts")
    artifacts.mkdir()
    (artifacts/"compiled_forward.py").write_text("# fixture code only\\n")
    (artifacts/"kernel.ttir").write_text("// fixture IR only\\n")
    (artifacts/"trace.json").write_text('{"traceEvents":[]}')
    (artifacts/"forbidden.pt").write_text("fixture suffix sentinel, never a tensor")
    print("simulated A/B/C experiment",flush=True)
    raise SystemExit(code)
if args and args[0] == "-":
    source=sys.stdin.read()
    if "VAE_OPTIMIZATION_LAUNCHER:" in source:
        bootstrap="import os,sys;sys.executable=os.environ['SHIM_EXECUTABLE'];exec(compile(sys.stdin.read(),'<launcher>','exec'))"
        raise SystemExit(subprocess.run([@PYTHON_LITERAL@,"-c",bootstrap,*args[1:]],input=source,text=True).returncode)
    raise SystemExit(subprocess.run([@PYTHON_LITERAL@,*args],input=source,text=True).returncode)
os.execv(@PYTHON_LITERAL@,[@PYTHON_LITERAL@,*args])
'''


@unittest.skipUnless(CONTROL.is_file(),
                     "Historical control wheel fixture is required; keep dist/vae-diagnostics-20261119")
class VAEOptimizationLauncherTests(unittest.TestCase):
    def check_launcher(self, *, preflight=0, experiment=0, install_failure=None, extra=(),
                       profile=None, omit_convolution_module=False, invalid_fusion_defaults=False,
                       expected_stage=None, expected_code=None):
        self.assertTrue(CONTROL.is_file(), "Historical control wheel fixture is required")
        self.assertEqual(hashlib.sha256(CONTROL.read_bytes()).hexdigest(), CONTROL_SHA)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/run_h200_vae_optimization.sh", root / "scripts/run_h200_vae_optimization.sh")
            module = root / "examples/vae/benchmark_optimization.py"
            module.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / "examples/vae/benchmark_optimization.py", module)
            (root / "packages").mkdir()
            control = root / "packages" / CONTROL.name
            shutil.copyfile(CONTROL, control)
            candidate = root / "packages/candidate.whl"
            with zipfile.ZipFile(candidate, "w") as wheel:
                wheel.writestr("deepopt/__init__.py", "# code-only candidate fixture\n")
                if profile == "native_convolution_wave2":
                    wheel.writestr("deepopt/compiler/autodiff/convolution_order.py", "# code-only order fixture\n")
                    if not omit_convolution_module:
                        wheel.writestr("deepopt/compiler/autodiff/convolution_geometry.py", "# code-only geometry fixture\n")
                if profile == "native_training_fusion_v3":
                    wheel.writestr("deepopt/compiler/autodiff/norm_activation_residuals.py", "# code-only norm fixture\n")
                    if not omit_convolution_module:
                        wheel.writestr("deepopt/compiler/autodiff/swiglu_fusion.py", "# code-only swiglu fixture\n")
                    from examples.vae.benchmark_optimization import OPTION_DEFAULTS
                    defaults = dict(OPTION_DEFAULTS)
                    if invalid_fusion_defaults:
                        defaults["norm_activation_fusion_stage"] = False
                    wheel.writestr("deepopt/compiler/backend.py", "class BackendConfig:\n" + "".join(
                        f"    {name}: {type(value).__name__} = {value!r}\n" for name, value in defaults.items()))
                wheel.writestr("deep_optimization-0.1.1.dist-info/METADATA",
                    "Name: deep-optimization\nVersion: 0.1.1+launcher.fixture\n")
            (root / "manifest.json").write_text(json.dumps({"version": "0.1.1+launcher.fixture",
                "wheel": candidate.name, "wheel_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                **({"experiment_profile": profile} if profile is not None else {}),
                "control": {"version": "0.1.0+nativeautodiff.20261119.vaememory.4",
                    "bundle_wheel": "packages/" + control.name, "wheel_sha256": CONTROL_SHA,
                    "package_source_sha256": CONTROL_SOURCE, "files_hashed": 457}}))
            previous = root / "original.json"
            previous.write_text("{}")
            binary = root / "bin"
            binary.mkdir()
            shim = binary / "python"
            shim.write_text(SHIM.replace("@PYTHON@", sys.executable).replace("@PYTHON_LITERAL@", repr(sys.executable)))
            shim.chmod(0o755)
            environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                SHIM_EXECUTABLE=str(shim), SIMULATED_PREFLIGHT_CODE=str(preflight),
                SIMULATED_EXPERIMENT_CODE=str(experiment), TORCHDYNAMO_REPRO_AFTER="invalid")
            environment.pop("DEEPOPT_FUSION_DIAGNOSTIC_DIR", None)
            if install_failure is not None:
                environment["SIMULATED_INSTALL_FAILURE"] = install_failure
            completed = subprocess.run(["bash", "scripts/run_h200_vae_optimization.sh", str(previous), *extra],
                cwd=root, env=environment, capture_output=True, text=True, timeout=45)
            expected = expected_code if expected_code is not None else 31 if install_failure else preflight or experiment
            self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
            request_file = root / "requests.jsonl"
            requests = [json.loads(line) for line in request_file.read_text().splitlines()] if request_file.exists() else []
            installs = [value for value in requests if value["kind"] == "install"]
            for value in installs:
                self.assertIn("--target", value["args"])
                self.assertNotIn("--force-reinstall", value["args"])
            if not install_failure and expected_stage is None:
                targets = [value["args"][value["args"].index("--target") + 1] for value in installs]
                self.assertEqual(len(set(targets)), 2)
                self.assertTrue(all(Path(value).is_dir() for value in targets))
                preflight_request, = [value for value in requests if value["kind"] == "preflight"]
                for flag in ("--require-cuda", "--require-triton", "--fail-on-skip"):
                    self.assertIn(flag, preflight_request["args"])
                selectors = (("tests.compiler.test_native_norm_activation_residuals.NativeNormActivationResidualCudaTests",
                    "tests.compiler.test_native_norm_activation_residuals.NativeNormActivationResidualOriginalShapeCudaTests",
                    "tests.compiler.test_native_swiglu_fusion.NativeSwiGLUFusionCudaTests",
                    "tests.compiler.test_native_swiglu_fusion.NativeSwiGLUFusionOriginalShapeCudaTests",
                    "tests.compiler.test_native_training_fusion_combined.NativeTrainingFusionCombinedCudaTests",
                    "tests.compiler.test_native_norm_fusion_layout.NativeNormFusionLayoutCudaTests")
                    if profile == "native_training_fusion_v3" else
                    ("tests.compiler.test_native_convolution_order.NativeConvolutionOrderCudaTests",
                    "tests.compiler.test_native_convolution_geometry.NativeConvolutionGeometryCudaTests",
                    "tests.compiler.test_native_convolution_geometry.NativeConvolutionGeometryOriginalShapeCudaTests",
                    "tests.compiler.test_native_convolution_memory_combined.NativeConvolutionMemoryCombinedCudaTests.test_actual_cuda_combined_native_vjp_and_retained_backward")
                    if profile == "native_convolution_wave2" else
                    ("tests.compiler.test_norm_cat.NormCatBodyCUDATests.test_original_b128_encoder1_groupnorm_cat_geometry",
                     "tests.compiler.test_backward_add_donation.BackwardAddDonationCudaTests",
                     "tests.compiler.test_backward_add_donation.BackwardAddDonationOriginalShapeCudaTests"))
                for selector in selectors:
                    self.assertIn(selector, preflight_request["args"])
                for name in ("DEEPOPT_RUN_H200_NORM_CAT", "DEEPOPT_TEST_LARGE_CUDA_DONATION"):
                    self.assertEqual(preflight_request["environment"][name], None if profile else "1")
                for name in ("DEEPOPT_RUN_H200_CONV_ORDER", "DEEPOPT_TEST_LARGE_CUDA_CONV_GEOMETRY"):
                    self.assertEqual(preflight_request["environment"][name], "1" if profile == "native_convolution_wave2" else None)
                for name in ("DEEPOPT_RUN_H200_NORM_ACTIVATION", "DEEPOPT_TEST_LARGE_CUDA_SWIGLU"):
                    self.assertEqual(preflight_request["environment"][name], "1" if profile == "native_training_fusion_v3" else None)
                if profile == "native_training_fusion_v3":
                    diagnostics = Path(preflight_request["environment"]["DEEPOPT_FUSION_DIAGNOSTIC_DIR"])
                    self.assertEqual(diagnostics, Path(preflight_request["args"][preflight_request["args"].index("--output") + 1]).parent / "fusion-diagnostics")
                pin = json.loads(preflight_request["environment"]["DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY"])
                self.assertEqual(pin["version"], "0.1.1+launcher.fixture")
            executed = [value for value in requests if value["kind"] == "experiment"]
            self.assertEqual(bool(executed), not preflight and not install_failure and expected_stage is None)
            for value in executed:
                arguments = value["args"]
                self.assertEqual(arguments[arguments.index("--batch-size") + 1], "128")
                self.assertEqual(arguments[arguments.index("--warmup") + 1], "5")
                self.assertEqual(arguments[arguments.index("--updates") + 1], "10")
                self.assertEqual(arguments[arguments.index("--trials") + 1], "3" if "--trials" in extra else "1")
                self.assertEqual(arguments[arguments.index("--start-free-tolerance-mib") + 1],
                    extra[extra.index("--start-free-tolerance-mib") + 1] if "--start-free-tolerance-mib" in extra else "256")
                self.assertNotIn("--synthetic", arguments)
                self.assertNotIn("--optimizer", arguments)
                self.assertIsNone(value["environment"]["TORCHDYNAMO_REPRO_AFTER"])
            arms = None
            if executed:
                arms = json.loads((root / "observed-experiment.json").read_text())["arms"]
                self.assertEqual(arms["A"]["native_options"], {})
                defaults = {"fuse_norm_cat_body": False, "donate_fresh_backward_add": False}
                if profile == "native_convolution_wave2":
                    defaults.update(native_conv_backward_order="source", cuda_conv_dx_geometry=False)
                elif profile == "native_training_fusion_v3":
                    from examples.vae.benchmark_optimization import OPTION_DEFAULTS
                    defaults = dict(OPTION_DEFAULTS)
                self.assertEqual(arms["B"]["native_options"], defaults)
                self.assertEqual(set(arms["C"]["native_options"]), set(arms["B"]["native_options"]))
                self.assertEqual(arms["B"]["install_root"], arms["C"]["install_root"])
                self.assertEqual(arms["B"]["wheel"], arms["C"]["wheel"])
                from examples.vae.benchmark_optimization import load_manifest
                # Exercise the real supervisor's typed boundary, rather than
                # accepting only what this test's subprocess shim records.
                self.assertEqual(load_manifest(root / "observed-experiment.json")[0]["arms"], arms)
            archives = list((root / "work").glob("*.zip"))
            self.assertEqual(len(archives), 1)
            with zipfile.ZipFile(archives[0]) as archive:
                names = archive.namelist()
                self.assertFalse(any("-installs/" in name or name.endswith(".pt") for name in names))
                launcher_path, = [name for name in names if name.endswith("/launcher.json")]
                launcher = json.loads(archive.read(launcher_path))
                self.assertEqual(launcher["exit_code"], expected)
                self.assertEqual(launcher["status"], "failed" if expected else "completed")
                self.assertFalse(launcher["performance_policy_promoted"])
                if profile == "native_training_fusion_v3" and not install_failure and expected_stage is None:
                    self.assertEqual(launcher["fusion_diagnostics"]["directory"], str(diagnostics))
                    for extension in ("ttir", "ttgir", "llir", "ptx", "sass"):
                        self.assertTrue(any(name.endswith("/fusion-diagnostics/layout-case/partial-affine/kernel." + extension) for name in names))
                    self.assertTrue(any(name.endswith("/fusion-diagnostics/layout-case/partial-affine/metadata.json") for name in names))
                if expected_stage is not None:
                    self.assertEqual(launcher["stage"], expected_stage)
                    self.assertFalse(requests)
                    self.assertIn("ValueError", launcher["error"])
                inventory_path, = [name for name in names if name.endswith("/archive-manifest.json")]
                for name, value in json.loads(archive.read(inventory_path))["files"].items():
                    data = archive.read(name)
                    self.assertEqual(len(data), value["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), value["sha256"])
                if not install_failure and expected_stage is None:
                    self.assertTrue(any(name.endswith("focused-CUDA-tests.json") for name in names))
                    self.assertTrue(any(name.endswith("focused-CUDA-tests.log") for name in names))
                if executed:
                    self.assertTrue(any(name.endswith("kernel.ttir") for name in names))
                    self.assertTrue(any(name.endswith("trace.json") for name in names))
            return arms, executed

    def test_original_recipe_uses_two_isolated_install_targets_and_one_default_block(self):
        arms, _ = self.check_launcher()
        self.assertTrue(all(arms["C"]["native_options"].values()))

    def test_three_blocks_profile_options_and_single_pass_ablation_are_forwarded(self):
        arms, requests = self.check_launcher(extra=("--trials", "3", "--passes", "norm-cat",
            "--profile-after-measurement", "--profile-memory", "--profile-updates", "2"))
        self.assertEqual(arms["C"]["native_options"], {"fuse_norm_cat_body": True, "donate_fresh_backward_add": False})
        self.assertIn("--profile-after-measurement", requests[0]["args"])
        self.assertIn("--profile-memory", requests[0]["args"])

    def test_new_profile_defaults_to_typed_convolution_pair_with_old_passes_disabled(self):
        arms, _ = self.check_launcher(profile="native_convolution_wave2")
        self.assertEqual(arms["C"]["native_options"], {"fuse_norm_cat_body": False,
            "donate_fresh_backward_add": False, "native_conv_backward_order": "dw_first",
            "cuda_conv_dx_geometry": True})

    def test_convolution_ablations_and_start_memory_tolerance_are_exactly_forwarded(self):
        for passes in ("conv-order", "conv-geometry"):
            with self.subTest(passes=passes):
                arms, requests = self.check_launcher(profile="native_convolution_wave2", extra=(
                    "--passes", passes, "--start-free-tolerance-mib", "192",
                    "--profile-after-measurement", "--profile-memory", "--profile-record-shapes"))
                self.assertEqual(arms["C"]["native_options"], {"fuse_norm_cat_body": False,
                    "donate_fresh_backward_add": False,
                    "native_conv_backward_order": "dw_first" if passes == "conv-order" else "source",
                    "cuda_conv_dx_geometry": passes == "conv-geometry"})
                self.assertIn("--profile-record-shapes", requests[0]["args"])

    def test_profile_mismatch_unknown_profile_and_incomplete_candidate_fail_before_install(self):
        for profile, options in ((None, ("--passes", "conv-both")),
                ("native_convolution_wave2", ("--passes", "both")), ("unknown_profile", ())):
            with self.subTest(profile=profile):
                self.check_launcher(profile=profile, extra=options, expected_code=1, expected_stage="experiment_profile")
        self.check_launcher(profile="native_convolution_wave2", omit_convolution_module=True,
            expected_code=1, expected_stage="experiment_profile")

    def test_new_profile_focused_cuda_failure_is_archived_and_prevents_workers(self):
        self.check_launcher(profile="native_convolution_wave2", preflight=19)

    def test_training_fusion_profile_defaults_to_both_forward_backward_with_legacy_controls_off(self):
        from examples.vae.benchmark_optimization import OPTION_DEFAULTS

        arms, _ = self.check_launcher(profile="native_training_fusion_v3")
        self.assertEqual(arms["C"]["native_options"], {**OPTION_DEFAULTS,
            "fuse_norm_activation_residuals": True, "norm_activation_fusion_stage": "forward_backward",
            "fuse_swiglu_residuals": True, "swiglu_fusion_stage": "forward_backward"})

    def test_training_fusion_feature_and_stage_ablations_are_exact(self):
        from examples.vae.benchmark_optimization import OPTION_DEFAULTS

        for passes, stage in (("fusion-both", "forward"), ("norm-activation", "forward_backward"),
                              ("swiglu", "forward"), ("swiglu", "activation_pack")):
            with self.subTest(passes=passes, stage=stage):
                arms, _ = self.check_launcher(profile="native_training_fusion_v3", extra=(
                    "--passes", passes, "--fusion-stage", stage))
                norm, swiglu = passes != "swiglu", passes != "norm-activation"
                self.assertEqual(arms["C"]["native_options"], {**OPTION_DEFAULTS,
                    "fuse_norm_activation_residuals": norm, "norm_activation_fusion_stage": stage if norm else "off",
                    "fuse_swiglu_residuals": swiglu, "swiglu_fusion_stage": stage if swiglu else "off"})

    def test_training_profile_bad_combinations_and_wheel_default_contract_fail_before_install(self):
        for profile, options in ((None, ("--passes", "fusion-both")),
                ("native_convolution_wave2", ("--fusion-stage", "forward")),
                ("native_training_fusion_v3", ("--passes", "conv-both")),
                ("native_training_fusion_v3", ("--fusion-stage", "activation_pack")),
                ("native_training_fusion_v3", ("--passes", "norm-activation", "--fusion-stage", "activation_pack"))):
            with self.subTest(profile=profile, options=options):
                self.check_launcher(profile=profile, extra=options, expected_code=1, expected_stage="experiment_profile")
        for bad_default in (False, True):
            self.check_launcher(profile="native_training_fusion_v3", invalid_fusion_defaults=bad_default,
                omit_convolution_module=not bad_default, expected_code=1, expected_stage="experiment_profile")

    def test_training_profile_focused_cuda_and_worker_failures_remain_failed_archives(self):
        self.check_launcher(profile="native_training_fusion_v3", preflight=29)
        self.check_launcher(profile="native_training_fusion_v3", experiment=37)

    def test_preflight_failure_is_archived_without_starting_full_model_workers(self):
        self.check_launcher(preflight=17)

    def test_experiment_failure_keeps_original_exit_code_and_code_IR(self):
        self.check_launcher(experiment=23)

    def test_installation_failure_is_archived_and_does_not_start_preflight(self):
        self.check_launcher(install_failure="candidate")

    def test_fixed_recipe_and_archive_overrides_are_rejected(self):
        for option in ("--output=/tmp/x.json", "--batch-size=16", "--optimizer=sgd", "--synthetic"):
            with self.subTest(option=option):
                result = subprocess.run(["bash", str(ROOT / "scripts/run_h200_vae_optimization.sh"),
                    "/nonexistent.json", option], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("original CUDA B128", result.stderr)


if __name__ == "__main__":
    unittest.main()
