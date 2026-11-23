"""Real CUDA same-recipe qualification of native FP8 derivatives and updates.

Run through qualification_tests.py with --require-fp8 --require-triton and
--fail-on-skip. These tests compare against matching prepared DeepOpt eager
execution; they do not measure compression error against uncompressed BF16.
Each case prints a NATIVE_FP8_RECIPE JSON diagnostic, including failed cases.
"""

from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn


CURRENT_BLOCKWISE_SHAPE = (2, 4)


class NativeFP8RecipeFixtureTests(unittest.TestCase):
    def test_blockwise_fixture_supports_cuda_tail_blocks_with_both_compression_routes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
        from deepopt.opt.quantization.quantizer import quantize_fp8

        # Fake CUDA keeps the actual GPU structural checks active on CPU CI.
        # Both dimensions retain masked tails rather than making the fixture
        # divisible by its legal power-of-two scaling rectangles.
        with FakeTensorMode():
            for compressed in (False, True):
                with self.subTest(compressed=compressed):
                    value = torch.empty((81, 13), device="cuda", dtype=torch.bfloat16)
                    if compressed:
                        quantized, packed = quantize_pack_fp8_activation(value,
                            block_shape=CURRENT_BLOCKWISE_SHAPE, group_size=1024, bits=4, check_finite=False)
                        self.assertEqual(packed.shape, value.shape)
                        self.assertEqual(packed.block_shape, CURRENT_BLOCKWISE_SHAPE)
                    else:
                        quantized = quantize_fp8(value, block_shape=CURRENT_BLOCKWISE_SHAPE, check_finite=False)
                    self.assertEqual(quantized.data.shape, value.shape)
                    self.assertEqual(quantized.scale_inv.shape, (41, 4))
                    self.assertEqual(quantized.block_shape, CURRENT_BLOCKWISE_SHAPE)
                    with self.assertRaisesRegex(ValueError, "powers of two"):
                        if compressed:
                            quantize_pack_fp8_activation(value, block_shape=(2, 3), group_size=1024, check_finite=False)
                        else:
                            quantize_fp8(value, block_shape=(2, 3), check_finite=False)


def _source_identity():
    import deepopt

    package = Path(deepopt.__file__).resolve().parent
    paths = [path for path in package.rglob("*") if path.is_file() and path.suffix in (".py", ".toml")]
    paths.append(Path(__file__).resolve())
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    return {"package": str(package), "test": str(Path(__file__).resolve()), "files": len(hashes),
            "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


@unittest.skipUnless(torch.cuda.is_available(), "Native FP8 recipe qualification requires CUDA")
class NativeFP8RecipesCUDATests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("Native Triton FP8 qualification requires compute capability >= 9")
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Native FP8 recipe qualification requires Triton")
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def _close(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(bool(torch.isfinite(expected).all()))
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)
        return float((actual.detach().float() - expected.detach().float()).abs().max())

    def _caches(self, model, previous_versions):
        records = {}
        for name, module in model.named_modules():
            cache = getattr(module, "_fp8_weight_cache", None)
            weight = getattr(module, "weight", None)
            if cache is None or weight is None:
                continue
            owner = weight._base if weight._base is not None else weight
            # Inspect only version metadata, never retain cached payloads in
            # diagnostics. A later forward must evict the preceding SGD version.
            versions = [key[1] for key, entry in cache._entries.items() if entry[0]() is owner]
            self.assertTrue(versions, f"{name}: no actual weight-cache entry was exercised")
            self.assertEqual(set(versions), {weight._version}, f"{name}: stale weight quantization")
            if name in previous_versions:
                self.assertGreater(weight._version, previous_versions[name], f"{name}: SGD did not mutate weight")
                self.assertNotIn(previous_versions[name], versions, f"{name}: previous SGD generation survived")
            records[name] = {"weight_version": weight._version, "cached_versions": versions,
                             "entry_count": len(versions)}
            previous_versions[name] = weight._version
        self.assertTrue(records, "The fixture did not exercise a prepared weight cache")
        return records

    def _case(self, label, recipe, blocks, compressed, *, transport=False):
        from deepopt import FP8Config, autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.kernel_policy import kernel_policy_scope
        from deepopt.opt.policy import prepare_fp8_model

        source_before = _source_identity()
        diagnostic = {"case": label, "recipe": recipe, "block_shapes": dict.fromkeys(
            ("activation", "weight", "grad"), blocks), "compressed": compressed,
            "transport": transport, "seed": 1828, "steps": [], "source_before": source_before,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "device": str(torch.device("cuda", torch.cuda.current_device())), "status": "running"}
        for package in ("deep-optimization", "triton"):
            try:
                diagnostic[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                diagnostic[package] = None
        try:
            torch.compiler.reset()
            torch.manual_seed(1828)
            torch.cuda.manual_seed_all(1828)
            compression = {"enabled": compressed, "bits": 4, "group_size": 1024}
            with kernel_policy_scope("triton"):
                settings = FP8Config(enabled=True, backend="triton", out_dtype="bf16", scaling_recipe=recipe,
                    block_shapes=dict.fromkeys(("activation", "weight", "grad"), blocks),
                    backward_precision="auto" if transport else "native")
                base = nn.Sequential(nn.Linear(7, 11), nn.Linear(11, 3)) if transport else nn.Linear(13, 7)
                model = prepare_fp8_model(base.cuda(), settings,
                    activation_compression=compression, config={})[0]
                resolved = {}
                for name, module in model.named_modules():
                    if not hasattr(module, "_fp8_weight_cache"):
                        continue
                    plan = module.plan
                    options = plan["fp8_training"]
                    self.assertEqual(options["backend"], "triton")
                    self.assertEqual(options["scaling_recipe"], recipe)
                    self.assertEqual(plan["block_shapes"], dict.fromkeys(("activation", "weight", "grad"), blocks))
                    self.assertEqual(options["activation_compression"]["enabled"], compressed)
                    resolved[name] = {"backend": options["backend"], "scaling_recipe": options["scaling_recipe"],
                                      "block_shapes": plan["block_shapes"],
                                      "activation_compression": options["activation_compression"]}
                self.assertTrue(resolved)
                diagnostic["resolved_plans"] = resolved
                reference = deepcopy(model)
                original_parameters = tuple(id(value) for value in model.parameters())
                optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9)
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.03, momentum=.9)
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                    dynamic=False, options={"require_full_aot": True, "use_native_autodiff": True,
                                            "kernel_backend": "triton"}))
                previous_versions = {}
                initial_updates = {name: value.detach().cpu().clone() for name, value in model.named_buffers()
                                   if name.endswith("fp8_scaling.updates")}
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    # Distinct magnitudes exercise the delayed history, not just
                    # identical cached quantizations on two unchanged forwards.
                    x = (torch.randn(147, 7 if transport else 13, device="cuda") * (.5 if step == 0 else 2.)).requires_grad_()
                    xr = x.detach().clone().requires_grad_()
                    with autocast("cuda", enabled=True, dtype=torch.bfloat16, fp8=settings,
                                  activation_compression=compression, check_finite=False, config={}):
                        expected = reference(xr)
                        actual = model(x)
                    gradient = torch.randn_like(expected) * (.25 if step == 0 else .5)
                    expected.backward(gradient)
                    # A native region must never delegate to the original
                    # per-operation eager backward implementation.
                    with patch("deepopt.opt.nn.autograd.linear._DOLinearFunction._backward",
                               side_effect=AssertionError("legacy FP8 derivative escape")):
                        actual.backward(gradient)
                    torch.cuda.synchronize()
                    metrics = {"output": self._close(actual, expected), "input_gradient": self._close(x.grad, xr.grad)}
                    actual_parameters = dict(model.named_parameters())
                    expected_parameters = dict(reference.named_parameters())
                    self.assertEqual(actual_parameters.keys(), expected_parameters.keys())
                    for name, parameter in actual_parameters.items():
                        metrics["parameter_gradient." + name] = self._close(parameter.grad, expected_parameters[name].grad)
                    # Weight quantization must reflect the current parameter
                    # version before this step's optimizer mutates it again.
                    caches = self._caches(model, previous_versions)
                    buffers = dict(model.named_buffers())
                    expected_buffers = dict(reference.named_buffers())
                    self.assertEqual(buffers.keys(), expected_buffers.keys())
                    for name, value in buffers.items():
                        torch.testing.assert_close(value, expected_buffers[name], rtol=0, atol=0)
                    state = {name: value.detach().cpu().tolist() for name, value in buffers.items()
                             if name.endswith((".scales", ".history", ".updates", ".generations"))}
                    slots = getattr(model, "_deepopt_gradient_edges", {})
                    transport_records = {name: deepcopy(slot.last_input_execution) for name, slot in slots.items()
                                         if slot.last_input_execution is not None}
                    if transport:
                        self.assertTrue(any(info["reused_fp8"] for info in transport_records.values()),
                                        "Prepared FP8 gradient transport did not reuse an incoming FP8 gradient")
                    optimizer.step()
                    reference_optimizer.step()
                    for name, parameter in actual_parameters.items():
                        expected_parameter = expected_parameters[name]
                        metrics["parameter_after_sgd." + name] = self._close(parameter, expected_parameter)
                        actual_state, expected_state = optimizer.state[parameter], reference_optimizer.state[expected_parameter]
                        self.assertEqual(actual_state.keys(), expected_state.keys())
                        for key in actual_state:
                            metrics["optimizer." + name + "." + key] = self._close(actual_state[key], expected_state[key])
                    self.assertEqual(tuple(id(value) for value in model.parameters()), original_parameters)
                    diagnostic["steps"].append({"index": step, "max_abs_errors": metrics,
                        "weight_cache": caches, "scaling_state": state, "gradient_transport": transport_records})
                if recipe == "delayed":
                    self.assertTrue(initial_updates, "No owned delayed scaling buffers were exercised")
                    final_buffers = dict(model.named_buffers())
                    for name, before in initial_updates.items():
                        after = final_buffers[name].detach().cpu()
                        self.assertTrue(bool((after >= before).all()))
                        if not transport:
                            self.assertTrue(bool((after >= before + 2).all()), f"{name}: two observations per role required")
                    self.assertTrue(any(bool((final_buffers[name].detach().cpu() > before).any())
                                        for name, before in initial_updates.items()))
                report = compilation_report(model).as_dict()
                self.assertGreater(report["captured_graphs"], 0)
                self.assertGreater(report["native_autodiff_operations"], 0)
                self.assertEqual(report["runtime_fallbacks"], 0)
                self.assertFalse(report["legacy_backward_operations"])
                self.assertTrue(all(not graph["eager_autograd_operations"] and not graph["legacy_backward_operations"]
                                    for graph in report["graphs"]))
                operations = {rule["operation"] for graph in report["graphs"] for rule in graph["differentiation_rules"]}
                self.assertIn("autodiff_fp8_linear", operations)
                diagnostic["compiler"] = {name: report[name] for name in (
                    "captured_graphs", "native_autodiff_operations", "runtime_fallbacks", "legacy_backward_operations")}
                diagnostic["compiler"]["derivative_operations"] = sorted(operations)
                diagnostic["status"] = "completed"
        except BaseException as error:
            diagnostic.update(status="failed", reason=f"{type(error).__name__}: {error}")
            raise
        finally:
            after = _source_identity()
            diagnostic.update(source_after=after, source_unchanged=source_before == after)
            print("NATIVE_FP8_RECIPE " + json.dumps(diagnostic, sort_keys=True))
            self.assertEqual(source_before, after, "Source changed during native FP8 GPU qualification")

    def _recipes(self, label, recipe, blocks):
        for compressed in (False, True):
            with self.subTest(recipe=label, compressed=compressed):
                self._case(label, recipe, blocks, compressed)

    def test_current_tensorwise_recipe_two_sgd_updates(self):
        self._recipes("current-tensorwise", "current", None)

    def test_delayed_tensorwise_recipe_two_sgd_updates(self):
        self._recipes("delayed-tensorwise", "delayed", None)

    def test_rowwise_recipe_two_sgd_updates(self):
        self._recipes("current-rowwise", "current", (1, 4))

    def test_blockwise_recipe_two_sgd_updates(self):
        self._recipes("current-blockwise", "current", CURRENT_BLOCKWISE_SHAPE)

    def test_prepared_gradient_transport_current_and_delayed(self):
        for recipe in ("current", "delayed"):
            for compressed in (False, True):
                with self.subTest(recipe=recipe, compressed=compressed):
                    self._case("prepared-transport-" + recipe, recipe, None, compressed, transport=True)


if __name__ == "__main__":
    unittest.main()
