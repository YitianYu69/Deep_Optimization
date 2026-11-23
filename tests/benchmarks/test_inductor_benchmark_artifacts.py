"""Benchmark-local code dumps; fake generated code never proves CUDA execution."""

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.compiler import inductor_benchmark_artifacts as artifacts


ROOT = Path(__file__).resolve().parents[2]


class FakeConfig(SimpleNamespace):
    def __init__(self):
        super().__init__(trace=SimpleNamespace(enabled=False, save_real_tensors=True, debug_dir=None,
            fx_graph=False, fx_graph_transformed=False, ir_pre_fusion=False, ir_post_fusion=False,
            output_code=False, debug_log=False, info_log=False, compile_profile=False),
            fx_graph_cache=True, fx_graph_remote_cache=True, autotune_remote_cache=True,
            fallback_random=False, emulate_precision_casts=False)

    @contextmanager
    def patch(self, values):
        previous = {}
        for key, value in values.items():
            owner = self
            for name in key.split(".")[:-1]:
                owner = getattr(owner, name)
            name = key.split(".")[-1]
            previous[key] = (owner, name, getattr(owner, name))
            setattr(owner, name, value)
        try:
            yield
        finally:
            for owner, name, value in previous.values():
                setattr(owner, name, value)


class FakeDynamoConfig(SimpleNamespace):
    def __init__(self, *, repro_after="", repro_level=4):
        super().__init__(repro_after=repro_after, repro_level=repro_level)

    patch = FakeConfig.patch


def task(*, case="torch_bf16", gradient="all"):
    return {"case": case, "workload": {"op": "model", "name": "mlp", "gradient_mode": gradient},
            "settings": {"runtime": {"execution": "compile", "compile": {"backend": "inductor"}}}}


def row(device="cpu"):
    return {"status": "ok", "timing": {"median_ms": 1.5, "samples": [1.5]}, "finite_after_timing": True,
            "environment": {"device": device, "torch": "fixture"},
            "execution": {"compile_counters": {"frames": {"total": 1}}}}


def write_code(scratch, *, backward=True, triton=True, custom=False):
    for phase in ["forward", *(["backward"] if backward else [])]:
        directory = scratch / "debug" / "torchinductor" / f"model__0_{phase}_1.0"
        directory.mkdir(parents=True)
        source = "# AOT ID: ['0_" + phase + "']\nextern_kernels.mm(a,b)\n"
        if triton:
            source += "async_compile.triton('kernel', '''@triton.jit\ndef kernel(): pass''')\n"
        (directory / "output_code.py").write_text(source)
        (directory / "fx_graph_readable.py").write_text(
            "torch.ops.deepopt.bad.default(x)\n" if custom else "torch.ops.aten.mm.default(x, weight)\n")
        (directory / "ir_pre_fusion.txt").write_text("fixture IR")
    (scratch / "compiler-cache.pkl").write_bytes(b"not retained")
    (scratch / "snapshot.pt.tmp").write_bytes(b"not retained")


class InductorBenchmarkArtifactTests(unittest.TestCase):
    def config_scope(self, config=None, dynamo=None):
        configs = {"torch._inductor.config": config or FakeConfig(),
                   "torch._dynamo.config": dynamo or FakeDynamoConfig(),
                   "torch._functorch.config": FakeDynamoConfig()}
        configs["torch._functorch.config"].backward_pass_autocast = "same_as_forward"
        configs["torch"] = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(
            matmul=SimpleNamespace(allow_tf32=False, allow_bf16_reduced_precision_reduction=True)),
            cudnn=SimpleNamespace(allow_tf32=True)))
        return patch.object(artifacts.importlib, "import_module", side_effect=configs.__getitem__)

    def test_inherited_repro_target_is_absent_during_capture_and_restored(self):
        for inherited in ("", "None", "invalid", "aot"):
            with self.subTest(inherited=inherited), tempfile.TemporaryDirectory() as temporary, self.config_scope(), patch.dict(os.environ, {
                    "TORCHDYNAMO_REPRO_AFTER": inherited, "TORCHDYNAMO_REPRO_LEVEL": "4",
                    "TORCH_COMPILE_DEBUG_SAVE_REAL": "1", "TORCHINDUCTOR_SAVE_COMPILE_ARGS": "1"}):
                with artifacts.InductorBenchmarkArtifacts(temporary, task(), enabled=True) as capture:
                    self.assertNotIn("TORCHDYNAMO_REPRO_AFTER", os.environ)
                    self.assertEqual(os.environ["TORCHDYNAMO_REPRO_LEVEL"], "0")
                    self.assertEqual(os.environ["TORCH_COMPILE_DEBUG_SAVE_REAL"], "0")
                    self.assertEqual(os.environ["TORCHINDUCTOR_SAVE_COMPILE_ARGS"], "0")
                    write_code(capture.scratch)
                    report = capture.finish(row=row())
                self.assertEqual(os.environ["TORCHDYNAMO_REPRO_AFTER"], inherited)
                self.assertEqual(os.environ["TORCHDYNAMO_REPRO_LEVEL"], "4")
                self.assertEqual(os.environ["TORCH_COMPILE_DEBUG_SAVE_REAL"], "1")
                self.assertEqual(os.environ["TORCHINDUCTOR_SAVE_COMPILE_ARGS"], "1")
                manifest = json.loads((Path(temporary) / report["manifest"]).read_text())
                self.assertIn("TORCHDYNAMO_REPRO_AFTER", manifest["metadata"]["configuration"]["environment"]["removed"])
                self.assertFalse(list(Path(temporary).rglob("*.pt*")))

    def test_already_imported_invalid_dynamo_config_is_patched_and_restored(self):
        for fail in (False, True):
            dynamo = FakeDynamoConfig(repro_after="", repro_level=4)
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary, self.config_scope(dynamo=dynamo):
                failure = RuntimeError("fixture failure") if fail else None
                try:
                    with artifacts.InductorBenchmarkArtifacts(temporary, task(), enabled=True) as capture:
                        self.assertIsNone(dynamo.repro_after)
                        self.assertEqual(dynamo.repro_level, 0)
                        self.assertEqual(capture.configuration["dynamo"]["applied"],
                                         {"repro_after": None, "repro_level": 0})
                        if failure is not None:
                            raise failure
                        write_code(capture.scratch)
                        capture.finish(row=row())
                except RuntimeError as error:
                    self.assertIs(error, failure)
                self.assertEqual(dynamo.repro_after, "")
                self.assertEqual(dynamo.repro_level, 4)
                manifest = json.loads(next(Path(temporary).rglob("manifest.json")).read_text())
                self.assertEqual(manifest["metadata"]["configuration"]["dynamo"]["applied"],
                                 {"repro_after": None, "repro_level": 0})

    def test_native_only_and_disabled_paths_create_no_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "case"
            with artifacts.InductorBenchmarkArtifacts(directory, task(case="bf16_compression"), enabled=False) as capture:
                self.assertIsNone(capture.finish(row=row()))
            self.assertFalse(directory.exists())
            with self.assertRaisesRegex(ValueError, "raw torch_bf16"):
                with artifacts.InductorBenchmarkArtifacts(directory, task(case="bf16_compression"), enabled=True):
                    self.fail("custom recipe was accepted")
            self.assertFalse(directory.exists())

    def test_dump_is_permanent_local_verified_and_outside_measurement(self):
        with tempfile.TemporaryDirectory() as temporary, self.config_scope():
            directory = Path(temporary) / "case"
            with artifacts.InductorBenchmarkArtifacts(directory, task(), enabled=True) as capture:
                scratch = capture.scratch
                write_code(scratch)
                result = capture.finish(row=row())
                self.assertEqual(result["status"], "complete")
                self.assertFalse(result["cuda_execution_observed"])
                self.assertEqual(len(result["compiled_phase_artifacts"]["backward"]), 1)
            self.assertFalse(scratch.exists())
            self.assertTrue((directory / result["manifest"]).exists())
            self.assertTrue(all((directory / name).is_file() for name in result["triton_source_paths"]))
            self.assertFalse(list(directory.rglob("*.pt*")))
            self.assertFalse(list(directory.rglob("*.pkl")))
            verified = artifacts.validate_inductor_artifacts(result, directory)
            self.assertTrue(verified["passed"], verified)
            self.assertTrue(verified["native_ops_only"])
            self.assertEqual(len(verified["triton_code_files"]), 2)
            self.assertFalse(verified["cuda_execution_observed"])
            self.assertEqual(verified["graph_breaks_and_fallbacks"]["frames"], {"total": 1})
            original = (directory / result["output_code_paths"][0])
            original.write_text(original.read_text() + "# modified")
            self.assertFalse(artifacts.validate_inductor_artifacts(result, directory)["passed"])

    def test_missing_backward_and_custom_captured_target_fail_closed(self):
        for options in ({"backward": False}, {"custom": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary, self.config_scope():
                directory = Path(temporary)
                with artifacts.InductorBenchmarkArtifacts(directory, task(), enabled=True) as capture:
                    write_code(capture.scratch, **options)
                    result = capture.finish(row=row())
                self.assertEqual(result["status"], "partial")
                self.assertFalse(result["reference_valid"])
                self.assertFalse(artifacts.validate_inductor_artifacts(result, directory)["passed"])

    def test_graph_break_and_opaque_autograd_boundary_do_not_qualify(self):
        for opaque in (False, True):
            with self.subTest(opaque=opaque), tempfile.TemporaryDirectory() as temporary, self.config_scope():
                directory = Path(temporary)
                measured = row()
                with artifacts.InductorBenchmarkArtifacts(directory, task(), enabled=True) as capture:
                    write_code(capture.scratch)
                    if opaque:
                        fx = next(capture.scratch.rglob("fx_graph_readable.py"))
                        fx.write_text("torch.ops.higher_order.autograd_function_apply(x)\n")
                    else:
                        measured["execution"]["compile_counters"] = {"graph_break": {"untraced fixture": 1}}
                    result = capture.finish(row=measured)
                self.assertEqual(result["status"], "partial")
                self.assertFalse(result["reference_valid"])
                self.assertFalse(artifacts.validate_inductor_artifacts(result, directory)["passed"])

    def test_runnable_repro_header_is_not_an_operator_namespace(self):
        header = ("from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional\n"
                  "torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()\n")
        for target, accepted in (("torch.ops.aten.mm.default(x, weight)", True),
                                 ("torch.ops.deepopt.custom.default(x)", False),
                                 ("torch.ops.higher_order.triton_kernel_wrapper_functional(x)", False),
                                 ("triton_kernel_wrapper_functional(x)", False)):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary, self.config_scope():
                with artifacts.InductorBenchmarkArtifacts(temporary, task(), enabled=True) as capture:
                    write_code(capture.scratch)
                    (capture.scratch / "fx_graph_runnable.py").write_text(header +
                        "class Repro:\n    def forward(self, x, weight):\n        return " + target + "\n")
                    result = capture.finish(row=row())
                audit = result["native_operator_audit"]
                self.assertEqual(audit["status"], "passed" if accepted else "failed")
                self.assertNotIn("triton_kernel_wrap", audit["operator_namespaces"])
                self.assertEqual(result["reference_valid"], accepted)

    def test_timing_scope_uses_row_then_task_and_preserves_vae_implementation(self):
        for field, expected in (({}, "VAE wall and CUDA events"),
                                ({"timing_scope": {"clock": "wall", "outside": ["compile"]}},
                                 {"clock": "wall", "outside": ["compile"]})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary, self.config_scope():
                request = task()
                request["timing_implementation"] = "VAE wall and CUDA events"
                measured = row()
                measured.update(field)
                with artifacts.InductorBenchmarkArtifacts(temporary, request, enabled=True) as capture:
                    write_code(capture.scratch)
                    result = capture.finish(row=measured)
                self.assertEqual(result["timing_scope"], expected)
                manifest = json.loads((Path(temporary) / result["manifest"]).read_text())
                self.assertEqual(manifest["metadata"]["timing_scope"], expected)
        with tempfile.TemporaryDirectory() as temporary, self.config_scope():
            request = task()
            request["timing_scope"] = "Task-specific full-update clock"
            with artifacts.InductorBenchmarkArtifacts(temporary, request, enabled=True) as capture:
                write_code(capture.scratch)
                result = capture.finish(row=row())
            self.assertEqual(result["timing_scope"], request["timing_scope"])

    def test_copy_failure_retains_unqualified_working_code_for_recovery(self):
        with tempfile.TemporaryDirectory() as temporary, self.config_scope():
            directory = Path(temporary)
            with artifacts.InductorBenchmarkArtifacts(directory, task(), enabled=True) as capture:
                write_code(capture.scratch)
                scratch = capture.scratch
                with patch.object(artifacts, "save_artifacts", side_effect=OSError("fixture disk failure")):
                    result = capture.finish(row=row())
            try:
                self.assertFalse(result["reference_valid"])
                self.assertEqual(result["uncopied_working_directory"], str(scratch))
                self.assertTrue(list(scratch.rglob("output_code.py")))
            finally:
                import shutil
                shutil.rmtree(scratch)

    def test_library_only_forward_case_is_labeled_without_fake_triton(self):
        with tempfile.TemporaryDirectory() as temporary, self.config_scope():
            directory = Path(temporary)
            with artifacts.InductorBenchmarkArtifacts(directory, task(gradient="none"), enabled=True) as capture:
                write_code(capture.scratch, backward=False, triton=False)
                result = capture.finish(row=row())
            checked = artifacts.validate_inductor_artifacts(result, directory)
            self.assertTrue(checked["passed"], checked)
            self.assertFalse(checked["triton_generated"])
            self.assertTrue(checked["external_library_calls"])
            self.assertIn("no_generated_triton_source", result["triton_generation"])

    def test_failure_preserves_partial_code_and_restores_all_settings(self):
        config = FakeConfig()
        with tempfile.TemporaryDirectory() as temporary, self.config_scope(config), patch.dict(os.environ, {
                "TORCH_COMPILE_DEBUG_SAVE_REAL": "1", "TORCH_TRACE": "/unsafe", "TRITON_OVERRIDE_DIR": "/unsafe"}):
            directory = Path(temporary)
            failure = RuntimeError("execution failed")
            with self.assertRaisesRegex(RuntimeError, "execution failed"):
                with artifacts.InductorBenchmarkArtifacts(directory, task(), enabled=True) as capture:
                    self.assertFalse(config.trace.save_real_tensors)
                    self.assertNotIn("TORCH_TRACE", os.environ)
                    self.assertNotIn("TRITON_OVERRIDE_DIR", os.environ)
                    write_code(capture.scratch)
                    raise failure
            self.assertEqual(failure.inductor_artifacts["status"], "partial")
            self.assertFalse(failure.inductor_artifacts["reference_valid"])
            self.assertTrue(config.trace.save_real_tensors)
            self.assertEqual(os.environ["TORCH_TRACE"], "/unsafe")
            self.assertEqual(os.environ["TORCH_COMPILE_DEBUG_SAVE_REAL"], "1")
            self.assertTrue((directory / "inductor-artifacts.json").exists())
            self.assertTrue(list(directory.rglob("output_code.py")))

    def test_unsupported_tensor_dump_safety_refuses_compilation(self):
        config = FakeConfig()
        del config.trace.save_real_tensors
        with tempfile.TemporaryDirectory() as temporary, self.config_scope(config):
            with self.assertRaisesRegex(RuntimeError, "real-tensor"):
                with artifacts.InductorBenchmarkArtifacts(temporary, task(), enabled=True):
                    self.fail("unsafe compiler was entered")
            report = json.loads((Path(temporary) / "inductor-artifacts.json").read_text())
            self.assertFalse(report["reference_valid"])

    def test_validator_rejects_manifest_outside_case_and_forged_success(self):
        for report in (None, {}, {"reference_valid": True, "manifest": "../other/manifest.json"},
                       {"reference_valid": True, "manifest": "/other/manifest.json"}):
            checked = artifacts.validate_inductor_artifacts(report, "/case")
            self.assertFalse(checked["passed"])

    def test_engine_wrapper_retains_measurements_and_attaches_local_code(self):
        sys.path.insert(0, str(ROOT / "benchmarks"))
        self.addCleanup(lambda: sys.path.remove(str(ROOT / "benchmarks")))
        from analysis import engine

        def execute(actual_task, options, directory):
            self.assertEqual(actual_task, task())
            scratch = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).parent
            write_code(scratch)
            return deepcopy(expected)

        expected = row()
        with tempfile.TemporaryDirectory() as temporary, self.config_scope(), patch.dict(os.environ, {
                artifacts.ENVIRONMENT_FLAG: "1"}), patch.object(engine, "_run_task", side_effect=execute):
            # engine imports the helper through its exported sibling namespace.
            with patch.dict(sys.modules, {"compiler.inductor_benchmark_artifacts": artifacts}):
                result = engine.run_task(task(), {}, Path(temporary))
            self.assertEqual(result["timing"], expected["timing"])
            self.assertTrue(result["inductor_artifacts"]["reference_valid"])
            self.assertEqual(result["execution"]["inductor_artifacts"]["manifest"], result["inductor_artifacts"]["manifest"])
            core = json.loads((Path(temporary) / "core-result.json").read_text())
            self.assertEqual(core["inductor_artifacts"], result["inductor_artifacts"])


if __name__ == "__main__":
    unittest.main()
