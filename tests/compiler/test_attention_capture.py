"""Functional attention retains its recipe through actual Dynamo fullgraph capture."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.overrides import TorchFunctionMode, _get_current_function_mode

from deepopt import autocast
from deepopt.compiler import compilation_report
from deepopt.compiler.capture import remove_capture
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))


class Attention(nn.Module):
    def __init__(self, *, dropout=0., causal=False, scale=None, gqa=False):
        super().__init__()
        self.dropout, self.causal, self.scale, self.gqa = dropout, causal, scale, gqa

    def forward(self, q, k, v, mask=None):
        return F.scaled_dot_product_attention(q, k, v, mask, self.dropout, self.causal,
                                             scale=self.scale, enable_gqa=self.gqa)


class AttentionCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def compile(self, model):
        return compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            dynamic=False, options={"require_full_aot": True}), config={})

    def assert_native(self, model):
        report = compilation_report(model).as_dict()
        self.assertEqual(report["captured_graphs"], 1)
        self.assertGreater(report["native_autodiff_operations"], 0)
        self.assertFalse(report["legacy_backward_operations"])
        self.assertEqual(report["runtime_fallbacks"], 0)
        self.assertTrue(all(not graph["eager_autograd_operations"] for graph in report["graphs"]))
        return report

    def compare_attention(self, *, math=False, dropout=0., causal=False, scale=None, gqa=False, mask=False):
        torch.manual_seed(311)
        model = Attention(dropout=dropout, causal=causal, scale=scale, gqa=gqa)
        reference = deepcopy(model)
        self.compile(model)
        inputs = (torch.randn(1, 4 if gqa else 2, 33, 16, dtype=torch.bfloat16, requires_grad=True),
                  torch.randn(1, 2, 35, 16, dtype=torch.bfloat16, requires_grad=True),
                  torch.randn(1, 2, 35, 16, dtype=torch.bfloat16, requires_grad=True))
        if mask:
            inputs += (torch.randn(1, 1, 33, 35, dtype=torch.bfloat16, requires_grad=True),)
        refs = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in inputs)
        compression = {"enabled": True, "bits": 2, "attention_bits": 4, "group_size": 1024}
        backends = [SDPBackend.MATH] if math else [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
        payloads = []

        def observe(value):
            if not isinstance(value, FakeTensor) and value.dtype == torch.int32 and value.numel():
                payloads.append(tuple(value.shape))
            return value

        with sdpa_kernel(backends), autocast("cpu", fp8=False, activation_compression=compression, config={}):
            torch.manual_seed(911)
            expected = reference(*refs)
            torch.manual_seed(911)
            with torch.autograd.graph.saved_tensors_hooks(observe, lambda value: value):
                actual = model(*inputs)
        self.assertGreaterEqual(len(payloads), 3, "Compiled SDPA must retain packed Q, K and V")
        torch.testing.assert_close(actual, expected)
        gradient = torch.randn_like(expected)
        expected.backward(gradient)
        with patch("torch.autograd.grad", side_effect=AssertionError("delegated derivative")):
            actual.backward(gradient)
        from analysis.measurement import errors

        parity = errors({str(index): value.grad for index, value in enumerate(inputs)},
                        {str(index): value.grad for index, value in enumerate(refs)}, .02, .002, max_relative_l2=.1)
        self.assertTrue(parity["passed"], parity)
        return model, self.assert_native(model)

    def test_native_functional_sdpa_captures_compressed_qkv(self):
        self.compare_attention()

    def test_mask_gradients_gqa_dropout_and_scale_keep_the_same_recipe(self):
        self.compare_attention(math=True, dropout=.2, scale=-.3, gqa=True, mask=True)

    def test_causal_attention_keeps_native_options(self):
        self.compare_attention(math=True, causal=True, scale=.25)

    def test_shared_vit_session_preserves_compressed_attention_and_gelu(self):
        from analysis.benchmark_do_ops import make_plan, parser
        from analysis.catalog import normalize_workload
        from analysis.engine import Session
        from analysis.matrix import DEFAULT_SETTINGS, merge
        from analysis.measurement import errors, snapshot

        snapshots = []
        for execution in ("eager", "compile"):
            torch.compiler.reset()
            settings = merge(deepcopy(DEFAULT_SETTINGS), {"compression": {"group_size": 1024}, "runtime": {"binding": "prepared",
                "execution": execution, "optimizer": "sgd", "optimizer_warmup_steps": 0,
                "compile": {"backend": "deepopt", "fullgraph": True, "dynamic": False,
                    "options": {"require_full_aot": True}}}})
            task = {"id": "attention-capture", "settings": settings, "case": "compression",
                "workload": normalize_workload({"op": "model", "name": "vit-block", "shape": [1, 17, 96],
                    "factory": "analysis.compiler_workloads:build", "factory_kwargs": {
                        "kind": "vit-block", "batch": 1, "tokens": 17, "width": 96, "hidden": 192, "output": 96}})}
            options = make_plan(parser().parse_args(["--device", "cpu", "--small", "--warmup", "0",
                "--iterations", "1", "--repeats", "1", "--phase-steps", "0"]))["options"]
            session = Session(task, options)
            try:
                snapshots.append(snapshot(session))
                if execution == "compile":
                    report = self.assert_native(session.model)
                    rules = str(report["graphs"])
                    self.assertIn("autodiff_sdpa", rules)
                    self.assertIn("native_gelu", rules)
                    saved = [value for graph in report["graphs"] for native in graph["native_autodiff_graphs"]
                             for value in native["saved_state"]]
                    self.assertTrue(any(value["name"].endswith("__coefficient") and value["representation"] == "packed"
                                        for value in saved))
            finally:
                session.close()
        parity = errors(snapshots[1], snapshots[0], .02, .002, max_relative_l2=.1)
        self.assertTrue(parity["passed"], parity)

    def test_capture_is_instance_owned_and_restores_eager_execution(self):
        native = F.scaled_dot_product_attention
        from deepopt.opt.nn.functional import do_scaled_dot_product_attention
        disabled = do_scaled_dot_product_attention
        model, _ = self.compare_attention()
        copied = deepcopy(model)
        self.assertIsNone(copied._compiled_call_impl)
        remove_capture(model)
        self.assertIsNone(model._compiled_call_impl)
        self.assertFalse(hasattr(model, "_deepopt_compiler_functional_capture"))
        self.assertIs(F.scaled_dot_product_attention, native)
        from deepopt.opt.nn import functional
        self.assertIs(functional.do_scaled_dot_product_attention, disabled)
        self.assertIsNone(_get_current_function_mode())
        q = torch.randn(1, 2, 33, 16, dtype=torch.bfloat16, requires_grad=True)
        with autocast("cpu", fp8=False,
                      activation_compression=dict(enabled=True, bits=2, group_size=1024), config={}):
            expected = F.scaled_dot_product_attention(q, q, q)
            torch.testing.assert_close(model(q, q, q), expected)
            torch.testing.assert_close(copied(q, q, q), expected)

    def test_user_mode_survives_compiled_calls_and_errors(self):
        class UserMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **({} if kwargs is None else kwargs))

        model = self.compile(Attention())
        q = torch.randn(1, 2, 33, 16, dtype=torch.bfloat16, requires_grad=True)
        with UserMode(), autocast("cpu", fp8=False,
                                  activation_compression=dict(enabled=True, bits=2, group_size=1024), config={}):
            active = _get_current_function_mode()
            model(q, q, q).sum().backward()
            self.assertIs(_get_current_function_mode(), active)
            with patch.object(model._compiled_call_impl, "original", side_effect=RuntimeError("failed invocation")):
                with self.assertRaisesRegex(RuntimeError, "failed invocation"):
                    model(q, q, q)
            self.assertIs(_get_current_function_mode(), active)
        self.assertIsNone(_get_current_function_mode())
        self.assert_native(model)

    def test_selected_operations_suspend_owned_modes_in_either_order(self):
        from deepopt.compiler.capture import _FunctionalCaptureMode
        from deepopt.opt._autocast import _FP8LinearMode, _suspend_fp8_interception

        class UserMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **({} if kwargs is None else kwargs))

        for mode_types in ((_FP8LinearMode, _FunctionalCaptureMode),
                           (_FunctionalCaptureMode, _FP8LinearMode)):
            with self.subTest(order=mode_types), UserMode() as user:
                modes = [kind({}) if kind is _FunctionalCaptureMode else kind() for kind in mode_types]
                with modes[0], modes[1]:
                    active = _get_current_function_mode()
                    with _suspend_fp8_interception():
                        self.assertIs(_get_current_function_mode(), user)
                    self.assertIs(_get_current_function_mode(), active)
                    with self.assertRaisesRegex(RuntimeError, "selected operation"):
                        with _suspend_fp8_interception():
                            raise RuntimeError("selected operation failed")
                    self.assertIs(_get_current_function_mode(), active)
        self.assertIsNone(_get_current_function_mode())


if __name__ == "__main__":
    unittest.main()
