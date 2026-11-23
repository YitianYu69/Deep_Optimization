"""Standard-library orchestration tests; fake captures never qualify real GPUs."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/compiler/save_inductor_references.py"
SPEC = importlib.util.spec_from_file_location("reference_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

FAKE_WORKER = '''
import json, os, sys, time
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
job, scratch = config['job'], Path(config['scratch'])
mode = os.environ.get('REFERENCE_TEST_MODE', 'complete')
if mode != 'missing-code':
    (scratch/'output_code.py').write_text("# generated code fixture, not real GPU evidence\\n")
    (scratch/'forward.py').write_text("# AOT ID: ['0_forward']\\n")
    (scratch/'backward.py').write_text("# AOT ID: ['0_backward']\\n")
(scratch/'ir_pre_fusion.txt').write_text('fixture before fusion')
(scratch/'snapshot.pt.tmp').write_bytes(b'forbidden fake payload')
(scratch/'pid.json').write_text(json.dumps({'pid': os.getpid()}))
controls = {'backward_pass_autocast':'off','fallback_random':True,
            'emulate_precision_casts':job['arithmetic_preset']=='cast-preserving'}
result = {'status': 'complete', 'request': job, 'gpu_capture_observed': True,
          'executed_updates': 2+job['warmup']+job['steps'],
          'numerical_controls':dict(preset=job['arithmetic_preset'],requested=controls,applied=controls,effective=controls),
          'later_update_numerics':[{'update':2,'matching_eager':{'passed':True},'finite':True,'rng_post_state_equal':True}],
          'matching_eager': {'passed': True,'dtype_and_None_equality_required':True},
          'rng_post_state_parity': {'passed':True}, 'final_state_finite':True,
          'native_ops_provenance': {'native_ops_only':True,'graph_breaks':{},
              'graphs':[{'native_ops_only':True,'nodes':[{'target':'aten.mul.Tensor'}], 'opaque_custom_boundaries':[]}]},
          'optimized_baseline_eligible':True,
          'finite_checks':[{'finite':True,'outside_measurement':True,'phase':'timed','step':i} for i in range(job['steps'])],
          'compiled_phase_artifacts': {'forward':['forward.py'], 'backward':['backward.py']},
          'timing': {'eager_wall_ms':[2.]*job['steps'],'wall_ms':[1.]*job['steps']},
          'fixture_only': True}
body_names = ['network.conv_proj.weight'] + [f'network.encoder.layers.encoder_layer_{i}.mlp.0.weight' for i in range(12)]
result['gradient_coverage'] = [{'update':2, **{strategy:{'parameters':{
    name:{'present':True,'nonzero':True,'elements':4,'nonzero_elements':4} for name in body_names}}
    for strategy in ('eager','inductor')}}]
result['vit_body_gradient_coverage'] = {'passed':True,'update':2,'declared_parameters':body_names,
    'scope':'patch projection and MLP input weight in each of the original 12 transformer blocks','failures':[]}
if mode == 'vit-coverage-missing':
    del result['gradient_coverage']
if mode == 'vit-coverage-zero':
    result['gradient_coverage'][0]['inductor']['parameters'][body_names[0]].update(nonzero=False,nonzero_elements=0)
if mode == 'vit-coverage-mismatch':
    result['gradient_coverage'][0]['inductor']['parameters'][body_names[0]]['nonzero_elements'] = 3
if mode == 'vit-coverage-body-missing':
    del result['gradient_coverage'][0]['eager']['parameters'][body_names[-1]]
if mode == 'blocked':
    result.update(status='blocked',gpu_capture_observed=False,reason='No CUDA fixture')
if mode == 'mismatch':
    result['request'] = dict(job, recipe='foreign')
if mode == 'no-parity':
    result['matching_eager'] = {'passed':False}
if mode == 'malformed':
    result['matching_eager'] = []
if mode == 'opaque':
    result['native_ops_provenance']['native_ops_only'] = False
if mode == 'graph-break':
    result['native_ops_provenance']['graph_breaks'] = {'opaque DeepOpt call':1}
if mode == 'empty-graphs':
    result['native_ops_provenance']['graphs'] = []
if mode == 'missing-break-counter':
    del result['native_ops_provenance']['graph_breaks']
if mode == 'native-int':
    result['native_ops_provenance']['native_ops_only'] = 1
if mode == 'eligible-int':
    result['optimized_baseline_eligible'] = 1
if mode == 'bad-controls':
    result['numerical_controls']['effective']['backward_pass_autocast'] = 'same_as_forward'
if mode == 'later-gradient-failure':
    result['later_update_numerics'][0]['matching_eager']['passed'] = False
if mode == 'later-RNG-failure':
    result['later_update_numerics'][0]['rng_post_state_equal'] = False
if mode == 'oversized-nonphase':
    (scratch/'extra-diagnostics.log').write_text('x'*32768)
if mode == 'phase-not-copied':
    result['compiled_phase_artifacts']['backward'] = ['missing-backward.py']
if mode == 'rng-false':
    result['rng_post_state_parity']['passed'] = False
if mode == 'rng-int':
    result['rng_post_state_parity']['passed'] = 1
if mode == 'finite-false':
    result['final_state_finite'] = False
if mode == 'finite-int':
    result['final_state_finite'] = 1
Path(config['result']).write_text(json.dumps(result))
if mode == 'timeout':
    time.sleep(20)
sys.exit(7 if mode == 'nonzero' else 0)
'''


class InductorReferenceRunnerTests(unittest.TestCase):
    def plan(self, root, *options):
        return runner.make_plan(runner.parser().parse_args([
            "--output", str(root / "durable"), "--models", "resnet18", *options]))

    def run_fixture(self, root, mode="complete", *options):
        root = root.resolve()
        worker = root / "fake-worker.py"
        worker.write_text(FAKE_WORKER)
        plan = self.plan(root, *options)
        with patch.dict("os.environ", {"REFERENCE_TEST_MODE": mode}):
            report = runner.run_plan(plan, worker_command=[sys.executable, "-B", str(worker)])
        return report, plan

    def test_help_and_dry_run_import_no_torch_and_create_no_artifacts(self):
        original = __import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "deepopt", "torchvision", "triton"):
                self.fail(f"Planning imported {name}")
            return original(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            with patch("builtins.__import__", guarded), patch("sys.stdout", new=io.StringIO()) as stdout:
                self.assertEqual(runner.main(["--dry-run", "--output", str(output)]), 0)
            plan = json.loads(stdout.getvalue())
            self.assertEqual([job["model"] for job in plan["jobs"]], list(runner.MODELS))
            self.assertEqual({job["recipe"] for job in plan["jobs"]}, {"torch-bf16"})
            self.assertFalse(output.exists())
        process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_full_architectures_and_recipe_matrix_are_explicit(self):
        plan = runner.make_plan(runner.parser().parse_args(["--recipes", *runner.RECIPES]))
        self.assertEqual(len(plan["jobs"]), 4)
        self.assertEqual(runner.RECIPES, ("torch-bf16",))
        self.assertEqual({row["group_size"] for row in plan["jobs"]}, {1024})
        self.assertEqual({row["image_size"] for row in plan["jobs"]}, {224})
        self.assertEqual({(row["warmup"], row["steps"]) for row in plan["jobs"]}, {(10, 50)})
        self.assertFalse(plan["scope"]["deepopt_native_autodiff"])
        self.assertEqual(plan["scope"]["optimizer"], "eager diagnostic update")
        for options in (["--image-size", "32"], ["--steps", "0"], ["--timeout", "nan"],
                        ["--warmup", "-1"], ["--models", "resnet18", "resnet18"], ["--device", "cpu"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runner.make_plan(runner.parser().parse_args(options))

    def test_worker_environment_disables_tensor_dumps_and_isolates_caches(self):
        with patch.dict("os.environ", {"TORCH_TRACE": "/unsafe", "TORCH_COMPILE_DEBUG_SAVE_REAL": "1",
                                      "TORCHINDUCTOR_SAVE_COMPILE_ARGS": "1"}):
            environment = runner.worker_environment(Path("/scratch"))
        self.assertNotIn("TORCH_TRACE", environment)
        self.assertEqual(environment["TORCH_COMPILE_DEBUG_SAVE_REAL"], "0")
        self.assertEqual(environment["TORCHINDUCTOR_SAVE_COMPILE_ARGS"], "0")
        for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TRITON_DUMP_DIR", "TORCH_COMPILE_DEBUG_DIR"):
            self.assertTrue(Path(environment[name]).is_relative_to("/scratch"))

    def test_callable_backend_translates_mode_kwargs_to_inductor_options(self):
        modes = unittest.mock.Mock(return_value={"max_autotune": True})
        torch = SimpleNamespace(_inductor=SimpleNamespace(list_mode_options=modes))
        self.assertEqual(runner.inductor_compile_options(torch, "max-autotune",
            {"mode": "max-autotune", "options": {"coordinate_descent_tuning": True}}),
            {"max_autotune": True, "coordinate_descent_tuning": True})
        modes.assert_called_once_with("max-autotune")
        for kwargs in ({"mode": "default"}, {"foreign": True}, {"options": [1]}):
            with self.assertRaises(ValueError):
                runner.inductor_compile_options(torch, "max-autotune", kwargs)
        unsupported = SimpleNamespace(_inductor=SimpleNamespace())
        self.assertEqual(runner.inductor_compile_options(unsupported, "default", {}), {})
        with self.assertRaises(RuntimeError):
            runner.inductor_compile_options(unsupported, "max-autotune", {})

    def test_repro_after_is_removed_instead_of_setting_an_invalid_empty_value(self):
        for inherited in ("", "aot", "dynamo", "invalid"):
            with self.subTest(inherited=inherited), patch.dict("os.environ", {"TORCHDYNAMO_REPRO_AFTER": inherited}):
                environment = runner.worker_environment(Path("/scratch"))
                self.assertNotIn("TORCHDYNAMO_REPRO_AFTER", environment)
                self.assertEqual(environment["TORCHDYNAMO_REPRO_LEVEL"], "0")

    def test_fresh_inductor_compilation_accepts_the_actual_worker_environment(self):
        # Exercise the actual backend and AOT backward, rather than only a fake
        # worker. Native CPU matmul uses the external library route and needs
        # no GPU or separate C++ toolchain. This is not full-model GPU evidence.
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary).resolve()
            code = '''import importlib.util, os
from pathlib import Path
import torch
import torch._dynamo.config as config
assert config.repro_after is None and config.repro_level == 0
spec = importlib.util.spec_from_file_location("capture_check", RUNNER_FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
config.repro_after, config.repro_level = "", 4
runner.configure_diagnostics(torch, os.environ["TORCH_COMPILE_DEBUG_DIR"])
assert config.repro_after is None and config.repro_level == 0
from benchmarks.compiler.inductor_benchmark_artifacts import inductor_reference_context
import torch._functorch.config as functorch
previous = functorch.backward_pass_autocast
scope = inductor_reference_context('cast-preserving')
controls = scope.__enter__()
assert controls['effective']['backward_pass_autocast'] == 'off'
def backend(graph, inputs):
    assert functorch.backward_pass_autocast == 'off'
    assert runner.native_graph_provenance(graph)["native_ops_only"]
    return torch._inductor.compile(graph, inputs)
def native_mm(left, right):
    with torch.autocast('cpu', enabled=False):
        return left @ right
compiled = torch.compile(native_mm, backend=backend, fullgraph=True)
left, right = torch.randn(8,16,requires_grad=True), torch.randn(16,12,requires_grad=True)
upstream = torch.randn(8,12)
expected_left, expected_right = left.detach().clone().requires_grad_(), right.detach().clone().requires_grad_()
expected = expected_left @ expected_right
expected.backward(upstream)
with torch.autocast('cpu', dtype=torch.bfloat16):
    actual = compiled(left, right)
actual.backward(upstream)
for result, reference in ((actual,expected),(left.grad,expected_left.grad),(right.grad,expected_right.grad)):
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
assert "CompiledFunctionBackward" in type(actual.grad_fn).__name__
# A changed shape forces another lazy compilation and AOT backward while the
# same contract is active; profiler execution remains a separate diagnostic.
other = torch.randn(9,16,requires_grad=True)
other_right = torch.randn(16,12,requires_grad=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
    with torch.autocast('cpu', dtype=torch.bfloat16):
        again = compiled(other, other_right)
    again.backward(torch.ones_like(again))
assert functorch.backward_pass_autocast == 'off'
torch.testing.assert_close(other.grad, torch.ones(9,12) @ other_right.detach().T)
scope.__exit__(None, None, None)
assert functorch.backward_pass_autocast == previous
root = Path(os.environ["TORCH_COMPILE_DEBUG_DIR"])
assert len(list(root.rglob("output_code.py"))) >= 2
assert not any(set(path.suffixes) & {".pt", ".pth", ".pkl", ".pickle", ".npy", ".npz", ".safetensors"} for path in root.rglob("*"))
print("real_inductor_forward_backward_passed")
'''
            code = code.replace("RUNNER_FILE", repr(str(SCRIPT)))
            with patch.dict("os.environ", {"TORCHDYNAMO_REPRO_AFTER": "", "TORCHDYNAMO_REPRO_LEVEL": "4"}):
                environment = runner.worker_environment(scratch)
            result = subprocess.run([sys.executable, "-B", "-c", code], env=environment,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("real_inductor_forward_backward_passed", result.stdout)

    def test_already_imported_dynamo_config_is_disabled_using_python_None(self):
        import torch._dynamo.config as config

        with config.patch(repro_after="", repro_level=4):
            report = runner.disable_dynamo_reproduction()
            self.assertIsNone(config.repro_after)
            self.assertEqual(config.repro_level, 0)
            self.assertEqual(report["applied"], {"repro_after": None, "repro_level": 0})

    def test_unique_fresh_workers_durable_artifacts_and_no_tensor_files(self):
        from benchmarks.compiler.inductor_reference_store import verify_run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, plan = self.run_fixture(root, "complete", "--models", "resnet18", "vit-base")
            rerun, _ = self.run_fixture(root)
            runs = [*report["runs"], *rerun["runs"]]
            self.assertEqual(len({row["run_dir"] for row in runs}), 3)
            pids = []
            for row in runs:
                run = Path(row["run_dir"])
                self.assertTrue(verify_run(run)["valid"])
                self.assertEqual(row["manifest"]["status"], "complete")
                self.assertTrue((run / "artifacts/worker/output_code.py").exists())
                self.assertFalse(list(run.rglob("*.pt*")))
                pids.append(json.loads((run / "artifacts/worker/pid.json").read_text())["pid"])
                config = json.loads((run / "artifacts/worker/worker-request.json").read_text())
                self.assertFalse(Path(config["scratch"]).exists())
                omitted = row["manifest"]["metadata"]["artifact_copy_report"]["omitted"]
                self.assertTrue(any(item["reason"] == "tensor_or_pickle_forbidden" for item in omitted))
            self.assertEqual(len(set(pids)), 3)
            self.assertTrue((Path(plan["output"]) / "index.json").exists())

    def test_failure_modes_retain_partial_dumps_and_never_report_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("nonzero", "mismatch", "no-parity", "malformed", "opaque", "graph-break",
                         "empty-graphs", "missing-break-counter", "native-int", "eligible-int",
                         "bad-controls", "later-gradient-failure", "later-RNG-failure",
                         "phase-not-copied", "rng-false", "rng-int", "finite-false", "finite-int", "missing-code", "timeout"):
                with self.subTest(mode=mode):
                    report, _ = self.run_fixture(root, mode, "--timeout", ".25")
                    row = report["runs"][0]
                    self.assertEqual(row["manifest"]["status"], "partial")
                    self.assertFalse(row["manifest"]["reference_eligible"])
                    self.assertTrue((Path(row["run_dir"]) / "artifacts/worker/ir_pre_fusion.txt").exists())

    def test_benchmark_output_has_paired_medians_code_and_verified_phase_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, _ = self.run_fixture(Path(temporary))
            row = report["runs"][0]
            results = json.loads((Path(row["run_dir"]) / "benchmark-results.json").read_text())
            self.assertEqual(results["timing"]["eager_median_ms"], 2.)
            self.assertEqual(results["timing"]["inductor_median_ms"], 1.)
            self.assertEqual(results["timing"]["paired_speedup_median"], 2.)
            self.assertTrue(results["copied_phase_artifacts"]["passed"])
            self.assertTrue((Path(row["run_dir"]) / "artifacts/worker/output_code.py").exists())
            for phase in ("forward", "backward"):
                entry = results["copied_phase_artifacts"][phase][0]
                self.assertEqual(runner.sha256_file(Path(row["run_dir"]) / entry["path"]), entry["sha256"])

    def test_secondary_corpus_is_optional_primary_still_keeps_code_and_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            report, _ = self.run_fixture(root, "complete", "--reference-root", str(root / "secondary"))
            row = report["runs"][0]
            self.assertEqual(row["secondary_reference"]["status"], "complete")
            self.assertTrue((Path(row["run_dir"]) / "benchmark-results.json").exists())
            secondary = Path(row["secondary_reference"]["run_dir"])
            self.assertTrue((secondary / "artifacts/benchmark/worker/output_code.py").exists())

    def test_vit_parent_rejects_missing_zero_or_mismatched_second_update_body_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good, _ = self.run_fixture(root, "complete", "--models", "vit-base")
            self.assertEqual(good["runs"][0]["manifest"]["status"], "complete")
            for mode in ("vit-coverage-missing", "vit-coverage-zero", "vit-coverage-mismatch", "vit-coverage-body-missing"):
                with self.subTest(mode=mode):
                    report, _ = self.run_fixture(root, mode, "--models", "vit-base")
                    row = report["runs"][0]
                    self.assertEqual(row["manifest"]["status"], "partial")
                    self.assertFalse(row["manifest"]["reference_eligible"])
                    result = json.loads((Path(row["run_dir"]) / "benchmark-results.json").read_text())
                    self.assertFalse(result["optimized_baseline_eligible"])
                    self.assertIn("ViT body gradients", result["reason"])

    def test_nonphase_copy_limit_downgrades_primary_results_and_manifest(self):
        from benchmarks.compiler import inductor_reference_store as store

        original = store.save_artifacts

        def bounded(*args, **kwargs):
            return original(*args, **kwargs, max_file_bytes=16384)

        with tempfile.TemporaryDirectory() as temporary, patch.object(store, "save_artifacts", side_effect=bounded):
            report, _ = self.run_fixture(Path(temporary), "oversized-nonphase")
            row = report["runs"][0]
            result = json.loads((Path(row["run_dir"]) / "benchmark-results.json").read_text())
            self.assertTrue(result["copied_phase_artifacts"]["passed"])
            self.assertTrue(any(error["path"] == "extra-diagnostics.log"
                                and error["reason"] == "artifact_byte_limit"
                                for error in result["artifact_copy_report"]["errors"]))
            self.assertEqual(result["status"], "partial")
            self.assertFalse(result["optimized_baseline_eligible"])
            self.assertEqual(row["manifest"]["status"], "partial")
            self.assertFalse(row["manifest"]["metadata"]["optimized_baseline_eligible"])

    def test_secondary_copy_error_downgrades_its_metadata(self):
        from benchmarks.compiler import inductor_reference_store as store

        original = store.save_artifacts

        def incomplete_secondary(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get("namespace") == "benchmark":
                result["errors"].append({"path": "diagnostics.log", "reason": "fixture_copy_error"})
                result["status"] = "partial"
            return result

        with tempfile.TemporaryDirectory() as temporary, patch.object(store, "save_artifacts", side_effect=incomplete_secondary):
            root = Path(temporary).resolve()
            report, _ = self.run_fixture(root, "complete", "--reference-root", str(root / "secondary"))
            row = report["runs"][0]
            self.assertEqual(row["manifest"]["status"], "complete")
            secondary = json.loads((Path(row["secondary_reference"]["run_dir"]) / "manifest.json").read_text())
            self.assertEqual(secondary["status"], "partial")
            self.assertEqual(secondary["metadata"]["status"], "partial")
            self.assertFalse(secondary["metadata"]["optimized_baseline_eligible"])

    def test_paired_summary_rejects_noncomparable_samples_and_source_hashes_cover_owners(self):
        for eager, compiled in (([1.], []), ([1.], [0.]), ([float("nan")], [1.])):
            with self.assertRaises(ValueError):
                runner.paired_timing_summary(eager, compiled)
        identities = runner.source_identity()
        self.assertEqual(set(identities), {"runner", "models", "store", "original_vae", "inductor_context", "measurement"})
        self.assertTrue(all(len(value["sha256"]) == 64 for value in identities.values()))

    def test_blocked_attempt_is_retained_and_missing_cuda_does_not_compile(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, _ = self.run_fixture(Path(temporary), "blocked")
            self.assertEqual(report["runs"][0]["manifest"]["status"], "blocked")
        job = self.plan(Path("unused"))["jobs"][0]
        with patch.object(runner, "runtime_preflight", return_value={"blockers": ["CUDA unavailable"]}):
            result = runner.execute_capture(job, "unused")
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["gpu_capture_observed"])

    def test_phase_evidence_requires_actual_compiler_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "plain.py").write_text("def forward(): pass\n")
            self.assertFalse(any(runner.phase_evidence(root).values()))
            (root / "compiled.py").write_text("# AOT ID: ['0_forward']\n")
            (root / "backward.py").write_text("# AOT ID: ['0_backward']\n")
            self.assertTrue(all(runner.phase_evidence(root).values()))

    def test_launch_error_finalizes_partial_and_continues_next_job(self):
        original = subprocess.Popen
        calls = []

        def launch(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise FileNotFoundError("Missing worker executable fixture")
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch.object(runner.subprocess, "Popen", launch):
            report, _ = self.run_fixture(Path(temporary), "complete", "--models", "resnet18", "vit-base")
            self.assertEqual([row["manifest"]["status"] for row in report["runs"]], ["partial", "complete"])
            run = Path(report["runs"][0]["run_dir"])
            self.assertIn("Missing worker", (run / "artifacts/worker/runner-error.log").read_text())

    def test_copy_exception_recovers_partial_artifacts_before_cleanup(self):
        from benchmarks.compiler import inductor_reference_store as store
        original = store.save_artifacts
        calls = []

        def copy(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("Transient copy error fixture")
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch.object(store, "save_artifacts", copy):
            report, _ = self.run_fixture(Path(temporary))
            row = report["runs"][0]
            self.assertEqual(row["manifest"]["status"], "partial")
            self.assertTrue((Path(row["run_dir"]) / "artifacts/recovery/output_code.py").exists())
            self.assertTrue(row["manifest"]["metadata"]["artifact_copy_exceptions"])

    def test_requested_environment_does_not_persist_inherited_auth_fields(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {"TRITON_REMOTE_TOKEN": "test-secret"}):
            report, _ = self.run_fixture(Path(temporary))
        environment = report["runs"][0]["manifest"]["metadata"]["requested_compiler_environment"]
        self.assertEqual(set(environment), set(runner.COMPILER_ENV_KEYS))
        self.assertNotIn("test-secret", json.dumps(environment))

    def test_interrupt_stops_child_and_finalizes_before_scratch_cleanup(self):
        original = subprocess.Popen

        def launch(*args, **kwargs):
            child = original(*args, **kwargs)
            wait = child.wait
            calls = []

            def interrupted_wait(*args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    raise KeyboardInterrupt()
                return wait(*args, **kwargs)

            child.wait = interrupted_wait
            return child

        with tempfile.TemporaryDirectory() as temporary, patch.object(runner.subprocess, "Popen", launch):
            report, _ = self.run_fixture(Path(temporary), "complete", "--models", "resnet18", "vit-base")
            self.assertTrue(report["interrupted"])
            self.assertEqual(len(report["runs"]), 1)
            row = report["runs"][0]
            self.assertEqual(row["manifest"]["status"], "partial")
            self.assertTrue((Path(row["run_dir"]) / "artifacts/worker/runner-error.log").exists())


class InductorReferenceNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        if str(ROOT / "benchmarks") not in sys.path:
            sys.path.insert(0, str(ROOT / "benchmarks"))

    def test_333_to_334_fp32_state_gradient_and_update_are_rejected(self):
        torch = self.torch
        for key in ("parameter_grad.weight", "parameter_update.weight", "buffer.running_mean", "output.loss"):
            with self.subTest(key=key):
                expected = {key: torch.tensor([333.], dtype=torch.float32)}
                actual = {key: torch.tensor([334.], dtype=torch.float32)}
                self.assertFalse(runner.compare_snapshots(actual, expected)["passed"])
                self.assertTrue(runner.compare_snapshots(expected, expected)["passed"])

    def test_dtype_None_and_missing_keys_are_never_numerical_matches(self):
        torch = self.torch
        for left, right in (({"x": torch.ones(2)}, {"x": torch.ones(2, dtype=torch.bfloat16)}),
                            ({"x": None}, {}), ({"x": None}, {"x": torch.ones(2)})):
            self.assertFalse(runner.compare_snapshots(left, right)["passed"])
        self.assertTrue(runner.compare_snapshots({"x": None}, {"x": None})["passed"])

    def test_bf16_rounding_budget_is_separate_from_fp32_state_policy(self):
        torch = self.torch
        expected = {"output.logits": torch.tensor([1.], dtype=torch.bfloat16)}
        actual = {"output.logits": torch.tensor([1.0078125], dtype=torch.bfloat16)}
        report = runner.compare_snapshots(actual, expected)
        self.assertTrue(report["passed"])
        self.assertEqual(report["tensors"]["output.logits"]["policy"], "bf16_output")
        self.assertFalse(runner.compare_snapshots({"buffer.x": actual["output.logits"]},
                                                {"buffer.x": expected["output.logits"]})["passed"])

    def test_nearzero_updates_use_postparameter_rounding_without_relaxing_gradients(self):
        torch = self.torch
        post = torch.ones(7, dtype=torch.float32)
        expected = {"parameter.weight": post, "parameter_update.weight": torch.zeros(7)}
        actual = {"parameter.weight": post.clone(), "parameter_update.weight": torch.full((7,), 5.96e-8)}
        report = runner.compare_snapshots(actual, expected)
        self.assertTrue(report["passed"])
        metric = report["tensors"]["parameter_update.weight"]
        self.assertFalse(metric["direction_defined"])
        self.assertIsNone(metric["cosine"])
        self.assertGreater(metric["relative_l2"], 1.)
        actual["parameter_update.weight"].fill_(1e-3)
        self.assertFalse(runner.compare_snapshots(actual, expected)["passed"])
        actual["parameter_update.weight"].zero_()
        actual["parameter_grad.weight"], expected["parameter_grad.weight"] = torch.tensor([334.]), torch.tensor([333.])
        self.assertFalse(runner.compare_snapshots(actual, expected)["passed"])

    def test_update_metrics_aggregate_whole_tensor_independent_of_chunk_size(self):
        from analysis.measurement import errors

        torch = self.torch
        expected = {"parameter.x": torch.ones(97), "parameter_update.x": torch.linspace(-1e-9, 1e-9, 97)}
        actual = {"parameter.x": torch.ones(97), "parameter_update.x": expected["parameter_update.x"] + 5e-8}
        reports = [errors(actual, expected, rtol=1e-4, atol=1e-8, max_relative_l2=1e-3,
                          update_policy={"parameter_rounding_ulps": 2}, chunk_elements=count)
                   for count in (1, 13, 65536)]
        self.assertTrue(all(report["passed"] for report in reports))
        for report in reports[1:]:
            for key in ("absolute_l2", "reference_l2", "relative_l2"):
                self.assertAlmostEqual(report["tensors"]["parameter_update.x"][key],
                                       reports[0]["tensors"]["parameter_update.x"][key], delta=1e-10)

    def test_original_zero_head_later_update_exposes_body_gradients(self):
        # Focused native control for the full ViT's zero-head phenomenon; this
        # fixture is never substituted into full-model benchmark execution.
        torch = self.torch
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3))
        torch.nn.init.zeros_(model[2].weight)
        torch.nn.init.zeros_(model[2].bias)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        inputs = torch.randn(2, 4)
        reports = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1])).backward()
            reports.append(runner.gradient_coverage(model))
            optimizer.step()
        self.assertFalse(reports[0]["parameters"]["0.weight"]["nonzero"])
        self.assertTrue(reports[1]["parameters"]["0.weight"]["nonzero"])

    def test_stochastic_branch_observer_is_untimed_rng_neutral_and_cleans_hooks(self):
        from benchmarks.compiler.inductor_reference_models import observe_stochastic_branches

        torch = self.torch

        class StochasticDepth(torch.nn.Module):
            __module__ = "torchvision.ops.stochastic_depth"
            p, mode = .5, "row"

            def forward(self, value):
                return value * (torch.rand(value.shape[0], 1) >= self.p) * 2

        model = StochasticDepth()
        inputs = torch.ones(8, 3)
        inputs[0].zero_()
        state = torch.get_rng_state()
        with observe_stochastic_branches(model) as observations:
            output = model(inputs)
        observed_rng = torch.get_rng_state()
        torch.set_rng_state(state)
        torch.testing.assert_close(model(inputs), output)
        self.assertTrue(torch.equal(torch.get_rng_state(), observed_rng))
        self.assertEqual(observations[0]["unobservable_zero_inputs"], 1)
        self.assertEqual(observations[0]["observed_kept"] + observations[0]["observed_dropped"], 7)
        self.assertFalse(model._forward_hooks)
        with self.assertRaises(ValueError), observe_stochastic_branches(model):
            raise ValueError("forced observer cleanup")
        self.assertFalse(model._forward_hooks)

    def test_inductor_numerical_context_restores_nested_and_exception_scopes(self):
        from benchmarks.compiler.inductor_benchmark_artifacts import inductor_reference_context
        import torch._functorch.config as functorch
        import torch._inductor.config as inductor

        old = (functorch.backward_pass_autocast, inductor.emulate_precision_casts)
        with self.assertRaisesRegex(ValueError, "forced"):
            with inductor_reference_context("default") as default:
                self.assertEqual(default["effective"]["backward_pass_autocast"], "off")
                self.assertFalse(inductor.emulate_precision_casts)
                with inductor_reference_context("cast-preserving"):
                    self.assertTrue(inductor.emulate_precision_casts)
                self.assertFalse(inductor.emulate_precision_casts)
                raise ValueError("forced")
        self.assertEqual((functorch.backward_pass_autocast, inductor.emulate_precision_casts), old)

    def test_rng_post_state_hash_detects_consumption_without_persisting_state(self):
        torch = self.torch
        generator = torch.Generator().manual_seed(1024)
        before = runner.rng_fingerprint({"torch": generator.get_state(), "python": (1, None)})
        torch.rand(1, generator=generator)
        after = runner.rng_fingerprint({"torch": generator.get_state(), "python": (1, None)})
        self.assertNotEqual(before, after)

    def test_final_finite_check_covers_gradient_and_optimizer_state(self):
        torch = self.torch
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        inputs = (torch.ones(1, 2, requires_grad=True),)
        outputs = {"loss": model(*inputs).sum()}
        outputs["loss"].backward()
        optimizer.step()
        self.assertTrue(runner.finite_state(model, inputs, outputs, optimizer)["finite"])
        model.weight.grad[0, 0] = float("nan")
        optimizer.state[model.weight]["momentum_buffer"][0, 0] = float("inf")
        optimizer.state[model.weight]["scalar_counter"] = float("nan")
        report = runner.finite_state(model, inputs, outputs, optimizer)
        self.assertFalse(report["finite"])
        self.assertTrue(any(name.startswith("parameter_grad.") for name in report["nonfinite"]))
        self.assertTrue(any(name.startswith("optimizer.") for name in report["nonfinite"]))
        self.assertTrue(any(name.endswith("scalar_counter") for name in report["nonfinite"]))

    def test_real_fx_native_graph_passes_and_opaque_namespace_rejects(self):
        torch = self.torch

        def graph_with(target):
            graph = torch.fx.Graph()
            value = graph.placeholder("value")
            graph.output(graph.call_function(target, (value, value)))
            return torch.fx.GraphModule({}, graph)

        native = runner.native_graph_provenance(graph_with(torch.ops.aten.mul.Tensor))
        self.assertTrue(native["native_ops_only"])
        library = torch.library.Library("inductor_reference_unit", "DEF")
        library.define("opaque(Tensor left, Tensor right) -> Tensor")
        opaque = runner.native_graph_provenance(graph_with(torch.ops.inductor_reference_unit.opaque.default))
        self.assertFalse(opaque["native_ops_only"])
        self.assertEqual(opaque["opaque_custom_boundaries"][0]["namespace"], "inductor_reference_unit")

    def test_raw_vae_preflight_requires_installed_deepopt(self):
        from benchmarks.compiler import installed_runtime

        job = runner.make_plan(runner.parser().parse_args(["--models", "vae-v2", "--preflight"]))["jobs"][0]
        with patch.object(installed_runtime, "installed_identity", side_effect=RuntimeError("source checkout rejected")):
            probe = runner.runtime_preflight(job)
        self.assertTrue(probe["requires_installed_deepopt"])
        self.assertTrue(any("source checkout rejected" in value for value in probe["blockers"]))


if __name__ == "__main__":
    unittest.main()
