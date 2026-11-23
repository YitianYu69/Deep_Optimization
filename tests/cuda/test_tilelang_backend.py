"""Real-GPU backend qualification; CPU references are numerical oracles only.

No CUDA/Triton/TileLang implementation is substituted for the actual model.
Import and launch blockers only turn accidental Triton execution into failure.
"""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.abc
import importlib.util
import inspect
import os
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import CUDAGraphCore, FP8Config, autocast, fp8_scaling_scope
from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.scaling.delayed import delayed_states


def _cuda_reason():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return "TileLang end-to-end qualification requires a real NVIDIA CUDA GPU"
    if torch.cuda.get_device_capability()[0] < 9:
        return "TileLang FP8 qualification requires NVIDIA compute capability >= 9"
    if importlib.util.find_spec("tilelang") is None:
        return "TileLang is not installed"
    return ""


_CUDA_REASON = _cuda_reason()
if _CUDA_REASON and os.environ.get("DEEPOPT_REQUIRE_TILELANG") == "1":
    raise RuntimeError(_CUDA_REASON)


@contextmanager
def _forbid_deepopt_backend(backend="triton"):
    """Block an unselected family's imports and already-loaded launchers."""
    prefixes = (f"deepopt.opt.backends.{backend}", f"deepopt.compiler.codegen.{backend}")
    if backend == "triton":
        prefixes += ("do_compiler_triton_module_",)

    def forbidden(name):
        return name.startswith(prefixes)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if forbidden(fullname):
                raise AssertionError(f"Attempted an unselected {backend} import: {fullname}")
            return None

    existing = {name: module for name, module in tuple(sys.modules.items())
                if forbidden(name) and module is not None}
    blocked = {name: None for name in existing}
    blocked.update({name: None for name in prefixes})
    finder, seen = Finder(), set()
    with ExitStack() as stack:
        # Captured references to imported Triton JIT objects must fail too.
        # Patches never replace the selected TileLang executor or its output.
        for module in existing.values():
            for name, value in tuple(vars(module).items()):
                if inspect.isfunction(value) and forbidden(value.__module__):
                    stack.enter_context(patch.object(
                        module, name, side_effect=AssertionError(f"Entered an unselected {backend} launcher"),
                    ))
                if id(value) in seen or isinstance(value, type):
                    continue
                seen.add(id(value))
                if callable(getattr(value, "run", None)) and (
                    type(value).__module__.startswith(backend + ".")
                ):
                    stack.enter_context(patch.object(
                        value, "run", side_effect=AssertionError(f"Launched an unselected {backend} JIT kernel"),
                    ))
        stack.enter_context(patch.dict(sys.modules, blocked))
        sys.meta_path.insert(0, finder)
        try:
            yield
        finally:
            sys.meta_path.remove(finder)


def _compression(enabled):
    return {"enabled": enabled, "bits": 2, "group_size": 32}


def _settings(recipe="current", backward="inherit"):
    return FP8Config(enabled=True, backend="inherit", trace_mode="modules",
                     scaling_recipe=recipe, backward_backend=backward,
                     out_dtype="fp32", autotune=False, intercept_linear=False)


def _prepare_pair(source, settings, compression, policy):
    config = {"runtime": {"kernels": policy}}
    actual, report = prepare_fp8_model(deepcopy(source).cuda(), settings,
                                      config=config, activation_compression=compression)
    reference_settings = replace(settings, backend="reference", backward_backend="inherit")
    reference, _ = prepare_fp8_model(deepcopy(source).cpu(), reference_settings,
                                     activation_compression=compression)
    return actual, reference, reference_settings, config, report


def _forward(model, x, settings, compression, config=None):
    with autocast(x.device.type, enabled=False, fp8=settings,
                  activation_compression=compression, config=config,
                  cache_enabled=False, check_finite=False):
        return model(x)


def _relative_error(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    return float(torch.linalg.vector_norm(actual - expected) /
                 torch.linalg.vector_norm(expected).clamp_min(1e-12))


@unittest.skipIf(_CUDA_REASON, _CUDA_REASON)
class TileLangBackendCUDATests(unittest.TestCase):
    def setUp(self):
        from deepopt.config import reset_config

        reset_config()
        self.addCleanup(reset_config)
        torch.manual_seed(2601)

    def _assert_gradients(self, actual, reference, x, xr, tolerance=.05):
        self.assertIsNotNone(x.grad)
        self.assertLessEqual(_relative_error(x.grad, xr.grad), tolerance)
        actual_gradients, reference_gradients = [], []
        expected_parameters = dict(reference.named_parameters())
        self.assertEqual(dict(actual.named_parameters()).keys(), expected_parameters.keys())
        for name, parameter in actual.named_parameters():
            expected = expected_parameters[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(expected.grad, name)
            self.assertEqual(parameter.dtype, torch.float32, name)
            self.assertEqual(parameter.grad.dtype, torch.float32, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            actual_gradients.append(parameter.grad.flatten())
            reference_gradients.append(expected.grad.flatten())
        self.assertLessEqual(_relative_error(torch.cat(actual_gradients), torch.cat(reference_gradients)), tolerance)

    def _pipeline(self, recipe, compressed, *, compiled=False):
        source = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 65))
        settings, compression = _settings(recipe), _compression(compressed)
        actual, reference, reference_settings, config, report = _prepare_pair(
            source, settings, compression, {"backend": "tilelang"},
        )
        self.assertTrue(report.converted_modules)
        self.assertTrue(all(type(actual[index]).__module__.startswith("deepopt.opt.nn.")
                            for index in (0, 1, 2, 3)))
        if compiled:
            from deepopt.config import CompileConfig
            from deepopt.engine.execution.compilation import compile_model

            torch.compiler.reset()
            self.addCleanup(torch.compiler.reset)
            compile_model(actual, CompileConfig(enabled=True, backend="deepopt", fullgraph=False), config=config)
        optimizer = torch.optim.SGD(actual.parameters(), lr=.01)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        parameter_ids = tuple(id(parameter) for parameter in actual.parameters())
        actual_states, expected_states = delayed_states(actual), delayed_states(reference)
        saved = []

        def retain(tensor):
            saved.append((tensor.dtype, tuple(tensor.shape), tensor.device.type))
            return tensor

        with _forbid_deepopt_backend():
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                expected_optimizer.zero_grad(set_to_none=True)
                raw = torch.randn(2, 17, 128) * .2
                x, xr = raw.cuda().requires_grad_(), raw.clone().requires_grad_()
                upstream = torch.randn(2, 17, 65) * .01
                before = [value.detach().clone() for value in actual.parameters()]
                expected_before = [value.detach().clone() for value in reference.parameters()]
                scales = [state.scales.clone() for state in actual_states]
                with fp8_scaling_scope(states=actual_states), record_gemms() as records:
                    with torch.autograd.graph.saved_tensors_hooks(retain, lambda tensor: tensor):
                        output = _forward(actual, x, settings, compression, config)
                    # Deliberately leave autocast before worker-thread backward.
                    output.backward(upstream.cuda())
                    for state, original in zip(actual_states, scales):
                        torch.testing.assert_close(state.scales, original, rtol=0, atol=0)
                with fp8_scaling_scope(states=expected_states):
                    expected = _forward(reference, xr, reference_settings, compression)
                    expected.backward(upstream)
                self.assertEqual(output.device.type, "cuda")
                torch.testing.assert_close(output.detach().cpu(), expected.detach(), rtol=.02, atol=.002)
                self._assert_gradients(actual, reference, x, xr)
                self.assertTrue(records)
                self.assertEqual({record["effective_backend"] for record in records}, {"tilelang"})
                self.assertTrue({"forward", "dx", "dw"}.issubset({record["role"] for record in records}))
                optimizer.step()
                expected_optimizer.step()
                self.assertEqual(tuple(id(parameter) for parameter in actual.parameters()), parameter_ids)
                updates, expected_updates = [], []
                for parameter, old, target, target_old in zip(actual.parameters(), before, reference.parameters(), expected_before):
                    self.assertTrue(torch.isfinite(parameter).all())
                    updates.append((parameter.detach() - old).flatten())
                    expected_updates.append((target.detach() - target_old).flatten())
                self.assertLessEqual(_relative_error(torch.cat(updates), torch.cat(expected_updates)), .06)
                self.assertEqual(len(actual_states), len(expected_states))
                for state, target in zip(actual_states, expected_states):
                    torch.testing.assert_close(state.updates.cpu(), target.updates, rtol=0, atol=0)
                    torch.testing.assert_close(state.scales.cpu(), target.scales, rtol=.02, atol=.002)
                    if int(target.updates.max().item()) > 0:
                        self.assertGreaterEqual(int(state.updates.max().item()), step + 1)
                if recipe == "delayed":
                    self.assertTrue(actual_states)
                    self.assertTrue(any(int(state.updates.max().item()) >= step + 1 for state in actual_states))
            if compressed:
                self.assertTrue(any(dtype == torch.int32 and device == "cuda" for dtype, _, device in saved))
            if compiled:
                from deepopt.compiler import compilation_report

                evidence = compilation_report(actual)
                # Wrapper-valued later inputs retain their existing backward
                # owner; at least the first dense Linear is compiler-owned.
                self.assertGreaterEqual(evidence.lowered_deepopt_operations, 1)
                self.assertGreaterEqual(evidence.compiled_deepopt_backward_regions, 1)
                self.assertEqual(evidence.runtime_fallbacks, 0)
                self.assertTrue(all(graph.kernel_backend == "tilelang" for graph in evidence.graphs))
        torch.cuda.synchronize()

    def test_current_pipeline_values_gradients_saved_state_and_sgd(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                self._pipeline("current", compressed)
        if importlib.util.find_spec("triton") is not None:
            # Legacy recipes remain the Triton baseline even with TileLang
            # installed or deliberately unavailable to this execution.
            source = nn.Linear(65, 37)
            compression = _compression(True)
            legacy_settings = replace(_settings(), backend="triton")
            legacy, _ = prepare_fp8_model(deepcopy(source).cuda(), legacy_settings,
                                           activation_compression=compression)
            explicit_settings = _settings()
            config = {"runtime": {"kernels": {"backend": "triton"}}}
            explicit, _ = prepare_fp8_model(deepcopy(source).cuda(), explicit_settings,
                                             config=config, activation_compression=compression)
            raw = torch.randn(17, 65, device="cuda") * .2
            x, other = raw.clone().requires_grad_(), raw.clone().requires_grad_()
            upstream = torch.randn(17, 37, device="cuda") * .01
            with _forbid_deepopt_backend("tilelang"):
                left = _forward(legacy, x, legacy_settings, compression)
                right = _forward(explicit, other, explicit_settings, compression, config)
                left.backward(upstream)
                right.backward(upstream)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
            for a, b in zip(legacy.parameters(), explicit.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_delayed_pipeline_updates_once_per_step_with_and_without_compression(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                self._pipeline("delayed", compressed)

    def test_native_backward_unsupported_layout_falls_back_to_tilelang(self):
        # Odd K/N in both derivative orientations force the declared metadata
        # fallback; native execution failures are never caught by this test.
        for backend in ("native_fast", "native_precise"):
            with self.subTest(backend=backend):
                settings, compression = _settings(backward=backend), _compression(True)
                actual, reference, reference_settings, config, _ = _prepare_pair(
                    nn.Linear(65, 37), settings, compression, {"backend": "tilelang"},
                )
                raw, upstream = torch.randn(17, 65) * .2, torch.randn(17, 37) * .01
                x, xr = raw.cuda().requires_grad_(), raw.clone().requires_grad_()
                with _forbid_deepopt_backend(), record_gemms() as records:
                    output = _forward(actual, x, settings, compression, config)
                    output.backward(upstream.cuda())
                expected = _forward(reference, xr, reference_settings, compression)
                expected.backward(upstream)
                torch.testing.assert_close(output.cpu(), expected, rtol=4e-4, atol=5e-4)
                self._assert_gradients(actual, reference, x, xr, tolerance=.005)
                backwards = [row for row in records if row["role"] in ("dx", "dw")]
                self.assertEqual({row["role"] for row in backwards}, {"dx", "dw"})
                self.assertTrue(all(row["requested_backend"] == backend for row in backwards))
                self.assertTrue(all(row["effective_backend"] == "tilelang" and row["fallback_reason"] for row in backwards))

    def test_mixed_products_and_saved_codec_cross_both_backend_directions(self):
        if importlib.util.find_spec("triton") is None:
            if os.environ.get("DEEPOPT_REQUIRE_TILELANG") == "1":
                self.fail("Mixed GPU qualification requires Triton as well as TileLang")
            self.skipTest("Mixed GPU qualification requires Triton as well as TileLang")
        for forward, dw in (("triton", "tilelang"), ("tilelang", "triton")):
            with self.subTest(forward=forward, dw=dw):
                policy = {"backend": "mixed", "preferred": forward,
                          "families": {"gemm": forward, "quantization": forward, "activation_codec": forward},
                          "roles": {"linear.dw": dw}}
                settings, compression = _settings(), _compression(True)
                actual, reference, reference_settings, config, _ = _prepare_pair(
                    nn.Linear(65, 37), settings, compression, policy,
                )
                decoder = importlib.import_module(f"deepopt.opt.backends.{dw}.quantization.activation.affine")
                raw, upstream = torch.randn(17, 65) * .2, torch.randn(17, 37) * .01
                x, xr = raw.cuda().requires_grad_(), raw.clone().requires_grad_()
                with record_gemms() as records, patch.object(
                    decoder, "dequantize_unpack_affine", wraps=decoder.dequantize_unpack_affine,
                ) as decode:
                    output = _forward(actual, x, settings, compression, config)
                    output.backward(upstream.cuda())
                expected = _forward(reference, xr, reference_settings, compression)
                expected.backward(upstream)
                torch.testing.assert_close(output.cpu(), expected, rtol=4e-4, atol=5e-4)
                self._assert_gradients(actual, reference, x, xr, tolerance=.005)
                self.assertGreater(decode.call_count, 0)
                self.assertTrue(all(row["effective_backend"] == forward for row in records if row["role"] == "forward"))
                self.assertTrue(any(row["effective_backend"] == dw and row["role"] == "dw" for row in records))
                self.assertTrue(all(row["kernel_mode"] == "mixed" for row in records))

    def test_compiled_current_pipeline_keeps_tilelang_training_ownership(self):
        self._pipeline("current", True, compiled=True)

    def test_current_pipeline_cuda_graph_replays_changed_inputs_and_weights(self):
        settings, compression = _settings(), _compression(True)
        source = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 65))
        actual, reference, reference_settings, config, _ = _prepare_pair(
            source, settings, compression, {"backend": "tilelang"},
        )
        sample = torch.randn(2, 17, 128, device="cuda", requires_grad=True)
        handle = None
        with _forbid_deepopt_backend():
            try:
                with autocast("cuda", enabled=False, fp8=settings, config=config,
                              activation_compression=compression, cache_enabled=False, check_finite=False):
                    handle = CUDAGraphCore("cuda").capture_model(actual, (sample,), warmup=3, autograd=True)
                optimizer = torch.optim.SGD(actual.parameters(), lr=.01)
                expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    expected_optimizer.zero_grad(set_to_none=True)
                    raw = torch.randn(2, 17, 128) * .2
                    x, xr = raw.cuda().requires_grad_(), raw.clone().requires_grad_()
                    output = _forward(actual, x, settings, compression, config)
                    upstream = torch.randn(2, 17, 65) * .01
                    output.backward(upstream.cuda())
                    expected = _forward(reference, xr, reference_settings, compression)
                    expected.backward(upstream)
                    torch.testing.assert_close(output.cpu(), expected, rtol=.02, atol=.002)
                    self._assert_gradients(actual, reference, x, xr)
                    optimizer.step()
                    expected_optimizer.step()
                self.assertEqual(handle.statistics["forward_replays"], 2)
            finally:
                if handle is not None:
                    handle.close()
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
