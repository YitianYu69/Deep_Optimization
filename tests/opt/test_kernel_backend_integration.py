"""Host integration contracts for explicit custom-kernel backend ownership."""

import builtins
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt._config_resolution import resolve_config, ResolvedConfig
from deepopt.kernel_policy import current_kernel_policy, kernel_policy_scope, resolve_kernel_policy
from deepopt.opt.policy import prepare_fp8_model


POLICIES = (
    {"backend": "triton"},
    {"backend": "tilelang"},
    {"backend": "mixed", "preferred": "tilelang", "families": {"gemm": "triton", "activation_codec": "tilelang"}},
)


@contextmanager
def no_optional_compiler_imports():
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in ("triton", "tilelang"):
            raise AssertionError(f"Host execution imported optional compiler {name}")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", guarded):
        yield


def configuration(policy, *, fp8=True, compute="reference", compressed=True):
    return resolve_config({"runtime": {
        "device": "cpu", "precision": "fp32", "checks": False,
        "kernels": deepcopy(policy),
        "fp8": {"enabled": fp8, "backend": compute, "trace_mode": "modules"},
        "activation_compression": {"enabled": compressed, "bits": 4, "group_size": 17},
    }}, base=None)


def model():
    return nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.GELU(), nn.Linear(8, 4))


class KernelBackendIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2791)

    def test_prepared_explicit_policy_is_not_revalidated_during_forward_or_backward(self):
        prepared, _ = prepare_fp8_model(model(), config=configuration({"backend": "tilelang"}))
        with patch("deepopt.kernel_policy.kernel_config_from_dict", side_effect=AssertionError("policy revalidation")):
            output = prepared(torch.randn(3, 8, requires_grad=True))
            output.sum().backward()

    def test_delayed_state_retains_owner_after_scope_exit(self):
        from deepopt.opt.scaling.delayed import DelayedScalingState, fp8_scaling_scope
        dtypes = dict.fromkeys(("activation", "weight", "grad"), torch.float8_e4m3fn)
        with kernel_policy_scope("tilelang"):
            state = DelayedScalingState({}, dtypes, device="cpu")
        with fp8_scaling_scope():
            quantized = state.quantize(torch.randn(3, 8), "activation")[0]
        self.assertEqual(quantized._kernel_policy.backend, "tilelang")
        self.assertEqual(current_kernel_policy().backend, "triton")
        with kernel_policy_scope("triton"), self.assertRaisesRegex(ValueError, "conflicts"):
            state.quantize(torch.randn(3, 8), "activation")

    def test_reference_fp8_compressed_training_identical_across_three_kernel_modes(self):
        from deepopt.engine.execution.configuration import prepare_execution_model
        original = model()
        x = torch.randn(3, 8)
        upstream = torch.randn(3, 4)
        results = []
        with no_optional_compiler_imports():
            for policy in POLICIES:
                with self.subTest(backend=policy["backend"]):
                    source = deepcopy(original)
                    parameters = dict(source.named_parameters())
                    state_names = tuple(source.state_dict())
                    snapshot = configuration(policy)
                    prepared, fp8, report = prepare_execution_model(source, snapshot)
                    self.assertTrue(fp8.enabled)
                    self.assertEqual(fp8.backend, "reference")
                    self.assertIsNotNone(report)
                    self.assertEqual(prepared._deepopt_kernel_policy, resolve_kernel_policy(policy).as_dict())
                    self.assertEqual(tuple(prepared.state_dict()), state_names)
                    for name, parameter in prepared.named_parameters():
                        self.assertIs(parameter, parameters[name])
                    for module in prepared.modules():
                        plan = getattr(module, "plan", None)
                        if isinstance(plan, dict) and "fp8_training" in plan:
                            self.assertEqual(plan["fp8_training"]["backend"], "reference")
                            self.assertEqual(plan["fp8_training"]["kernel_policy"], resolve_kernel_policy(policy).as_dict())
                    value = x.clone().requires_grad_()
                    output = prepared(value)
                    output.backward(upstream)
                    results.append((output.detach(), value.grad, tuple(p.grad for p in prepared.parameters())))
        for actual in results[1:]:
            torch.testing.assert_close(actual, results[0], rtol=0, atol=0)
        self.assertEqual(current_kernel_policy().backend, "triton")

    def test_training_runtime_step_owns_kernel_policy_and_preserves_optimizer_parameters(self):
        from deepopt.engine import TrainingRuntime
        original = model()
        inputs = torch.randn(3, 8)
        states = []
        for policy in POLICIES:
            with self.subTest(backend=policy["backend"]), no_optional_compiler_imports():
                source = deepcopy(original)
                optimizer = torch.optim.SGD(source.parameters(), lr=0.01)
                owned = tuple(optimizer.param_groups[0]["params"])
                runtime = TrainingRuntime(source, optimizer, config=configuration(policy))
                self.assertEqual(tuple(runtime.model.parameters()), owned)
                observed = []
                hook = runtime.model.register_forward_pre_hook(lambda *args: observed.append(current_kernel_policy().backend))
                try:
                    output = runtime.compute(inputs, lambda batch: runtime.model(batch))
                    runtime.backward(output.square().mean())
                    runtime.step()
                finally:
                    hook.remove()
                self.assertEqual(observed, [policy["backend"]])
                self.assertTrue(all(p.grad is None for p in runtime.model.parameters()))
                states.append(deepcopy(runtime.model.state_dict()))
        for actual in states[1:]:
            torch.testing.assert_close(actual, states[0], rtol=0, atol=0)

    def test_tilelang_fp8_compute_rejects_cpu_but_cpu_reference_compute_is_allowed(self):
        with no_optional_compiler_imports():
            prepared, _ = prepare_fp8_model(nn.Linear(8, 4), config=configuration({"backend": "tilelang"}, compute="inherit"))
            self.assertEqual(prepared.plan["fp8_training"]["backend"], "tilelang")
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                prepared(torch.randn(3, 8))
            reference, _ = prepare_fp8_model(nn.Linear(8, 4), config=configuration({"backend": "tilelang"}))
            result = reference(torch.randn(3, 8, requires_grad=True))
            result.sum().backward()
            self.assertEqual(tuple(result.shape), (3, 4))
            self.assertTrue(torch.isfinite(reference.weight.grad).all())

    def test_native_cpu_compression_stays_reference_under_all_kernel_modes(self):
        original = nn.Sequential(nn.LayerNorm(8), nn.SiLU(), nn.AvgPool1d(2))
        x = torch.randn(2, 3, 8)
        results = []
        for policy in POLICIES:
            with self.subTest(backend=policy["backend"]), no_optional_compiler_imports():
                prepared, report = prepare_fp8_model(deepcopy(original), config=configuration(policy, fp8=False))
                self.assertIsNotNone(report)
                value = x.clone().requires_grad_()
                output = prepared(value)
                output.square().sum().backward()
                results.append((output.detach(), value.grad, tuple(p.grad for p in prepared.parameters())))
        for actual in results[1:]:
            torch.testing.assert_close(actual, results[0], rtol=0, atol=0)

    def test_fake_cuda_metadata_uses_no_compiler_device_query_or_pointer(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        from deepopt.opt.backends import gemm_nt
        from deepopt.opt.nn.functional import do_relu, do_layer_norm, do_avg_pool2d, do_conv2d, do_batch_norm, do_group_norm
        from deepopt.opt.policy import normalization_plan, conv_plan
        from deepopt.opt.nn.normalization.geometry import NormGeometry
        from deepopt.opt.backends.reference import normalization as normalization_reference
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
        with ExitStack() as stack:
            stack.enter_context(FakeTensorMode())
            stack.enter_context(no_optional_compiler_imports())
            for name in ("get_device_properties", "get_device_capability", "current_device"):
                stack.enter_context(patch.object(torch.cuda, name, side_effect=AssertionError("GPU query")))
            stack.enter_context(patch.object(torch.Tensor, "data_ptr", side_effect=AssertionError("GPU pointer")))
            for policy in POLICIES:
                with self.subTest(backend=policy["backend"]), kernel_policy_scope(policy):
                    x = torch.empty((3, 8), device="cuda:0")
                    q = quantize_fp8(x, block_shape=(2, 4))
                    y = gemm_nt(q, q, backend=policy["backend"])
                    self.assertIsInstance(y, FakeTensor)
                    self.assertEqual(tuple(y.shape), (3, 3))
                    self.assertEqual(y.stride(), (4, 1))
                    packed = quantize_pack_activation(x, group_size=17, bits=4)
                    decoded = dequantize_unpack_activation(packed, dtype=torch.float32)
                    self.assertIsInstance(decoded, FakeTensor)
                    self.assertEqual(tuple(decoded.shape), (3, 8))
                    plan = normalization_plan(False, config=configuration(policy, fp8=False))
                    self.assertEqual(tuple(do_relu(x, plan=plan).shape), (3, 8))
                    self.assertEqual(tuple(do_layer_norm(x, (8,), plan=plan).shape), (3, 8))
                    spatial = torch.empty((2, 4, 3, 5), device="cuda:0")
                    self.assertEqual(tuple(do_avg_pool2d(spatial, 2, plan=plan).shape), (2, 4, 1, 2))
                    self.assertEqual(tuple(do_batch_norm(spatial, None, None, training=True, plan=plan).shape), tuple(spatial.shape))
                    self.assertEqual(tuple(do_group_norm(spatial, 2, plan=plan).shape), tuple(spatial.shape))
                    affine = torch.empty(4, device="cuda:0")
                    for kind in ("batch", "group"):
                        geometry = NormGeometry(kind, tuple(spatial.shape), groups=2)
                        result, mean, inverse, saved = normalization_reference.forward(spatial, affine, affine,
                            geometry, dict(enabled=True, bits=4, group_size=17), out_dtype=torch.float32)
                        gradients = normalization_reference.backward(torch.empty_like(result), affine, geometry,
                            packed=saved, mean=mean, inverse=inverse, input_dtype=torch.float32)
                        for value, shape in zip(gradients, (tuple(spatial.shape), (4,), (4,))):
                            self.assertIsInstance(value, FakeTensor)
                            self.assertEqual(tuple(value.shape), shape)
                            self.assertEqual(value.device, spatial.device)
                    weights = torch.empty((6, 4, 2, 2), device="cuda:0")
                    convolution = conv_plan({"enabled": True, "backend": "inherit"},
                        config=configuration(policy, compute="inherit"))
                    self.assertEqual(tuple(do_conv2d(spatial, weights, plan=convolution).shape), (2, 6, 2, 4))

    def test_escaped_fp8_wrapper_owns_policy_across_detach_view_and_tensor_serialization(self):
        from deepopt.opt.quantization.activation_tensor import FP8Activation
        from deepopt.opt.quantization.tensor import QuantizedTensor
        for policy in POLICIES:
            owner = resolve_kernel_policy(policy)
            with self.subTest(backend=policy["backend"]), kernel_policy_scope(owner):
                raw = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.float8_e4m3fn)
                value = FP8Activation(raw, torch.full((1, 1), 0.5), (2, 3, 1, 1))
            with kernel_policy_scope("tilelang" if owner.backend == "triton" else "triton"):
                for result in (value.detach(), value.clone(), value.view(2, 3)):
                    self.assertEqual(result._kernel_policy, owner)
                    self.assertEqual(result.quantized()._kernel_policy, owner)
                    self.assertEqual(result.quantized().transpose(contiguous=False)._kernel_policy, owner)
                names, metadata = value.__tensor_flatten__()
                restored = FP8Activation.__tensor_unflatten__({name: getattr(value, name) for name in names}, metadata,
                    value.shape, value.stride())
                self.assertEqual(restored._kernel_policy, owner)
                torch.testing.assert_close(restored.dequantize(), value.dequantize())
            with kernel_policy_scope(owner):
                q = QuantizedTensor(raw, torch.ones(1, 1))
            self.assertEqual(q._kernel_policy, owner)
        self.assertEqual(current_kernel_policy().backend, "triton")

    def test_preparation_reuse_keeps_owned_policy_and_rejects_conflicting_reconfiguration(self):
        prepared, report = prepare_fp8_model(model(), config=configuration({"backend": "tilelang"}))
        reused, repeated = prepare_fp8_model(prepared, config=ResolvedConfig())
        self.assertIs(reused, prepared)
        self.assertIs(repeated, report)
        self.assertEqual(reused._deepopt_kernel_policy["backend"], "tilelang")
        with self.assertRaisesRegex(ValueError, "Kernel policy differs"):
            prepare_fp8_model(prepared, config=configuration({"backend": "triton"}))

    def test_fresh_process_pure_modes_never_import_opposite_backend_symbols(self):
        source = """
import builtins, sys, torch
import torch._dynamo
from torch import nn
from deepopt.opt.policy import prepare_fp8_model
from deepopt._config_resolution import ResolvedConfig
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in ('triton', 'tilelang'):
        raise AssertionError('optional compiler import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
selected = sys.argv[1]
foreign = 'triton' if selected == 'tilelang' else 'tilelang'
config = ResolvedConfig({'runtime': {'kernels': {'backend': selected},
    'fp8': {'enabled': True, 'backend': 'reference', 'trace_mode': 'modules'},
    'activation_compression': {'enabled': True, 'bits': 4, 'group_size': 17}}})
prepared, _ = prepare_fp8_model(nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 4)), config=config)
x = torch.randn(2, 8, requires_grad=True)
prepared(x).sum().backward()
assert torch.isfinite(x.grad).all()
assert not any(name.startswith('deepopt.opt.backends.' + foreign) for name in sys.modules)
assert 'triton' not in sys.modules and 'tilelang' not in sys.modules
"""
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
        for selected in ("triton", "tilelang"):
            with self.subTest(backend=selected):
                result = subprocess.run([sys.executable, "-c", source, selected], env=env,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
