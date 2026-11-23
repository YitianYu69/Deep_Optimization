"""Source-frozen forced-backend regression for the archived H100 SDPA geometry.

Run with the installed DeepOpt library using ``python -m unittest discover -s tests/compiler -p
test_attention_saved_state_cuda.py -v``. Each backend emits one JSON manifest;
unavailable backends remain explicit skips and never select another kernel.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn.attention import SDPBackend, sdpa_kernel


def _source_identity():
    import deepopt

    paths = [*Path(deepopt.__file__).parent.rglob("*.py"), Path(__file__)]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for actual SDPA backend qualification")
class AttentionSavedStateCUDATests(unittest.TestCase):
    def _qualify(self, backend):
        from deepopt import autocast
        from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
        from deepopt.compiler.autodiff.attention import SDPA
        from deepopt.opt.nn.functional import do_scaled_dot_product_attention

        before = _source_identity()
        diagnostic = dict(backend_requested=backend.name, seed=371, torch=torch.__version__,
            torch_path=torch.__file__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()), invocations=[],
            autocast_dtype="torch.bfloat16", restored_operands=[],
            declared_cases=[dict(batch=batch, compressed=compressed) for batch in (8, 32) for compressed in (False, True)])
        try:
            diagnostic["triton"] = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            diagnostic["triton"] = None
        original_forward = SDPA.forward
        original_restore = SDPA._restore

        def record_forward(specification, *args):
            outputs = original_forward(specification, *args)
            selected = int(outputs[-1].item())
            self.assertEqual(selected, int(backend), "Forced SDPA backend silently changed")
            self.assertEqual(specification.output_dtype, torch.bfloat16)
            self.assertEqual(outputs[0].dtype, torch.bfloat16)
            for value in args[:3]:
                self.assertEqual(value.dtype, torch.bfloat16)
            saved_dtype = torch.float32 if backend == SDPBackend.MATH else torch.bfloat16
            for index in (1, 4, 7):
                self.assertEqual(outputs[index].dtype,
                                 torch.int32 if specification.bits else saved_dtype)
            state = {name: {"type": type(value).__name__, "present": value is not None,
                **({"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                    "stride": list(value.stride()), "empty": value.numel() == 0}
                    if isinstance(value, torch.Tensor) else {})}
                for name, value in zip(specification.names, outputs[1:])}
            diagnostic["invocations"].append(dict(selected_implementation=selected,
                query_shape=list(args[0].shape), query_dtype=str(args[0].dtype),
                query_stride=list(args[0].stride()), mask=None,
                compression=specification.compression, saved_state=state))
            return outputs

        def record_restore(specification, values, shape, dtype):
            restored = original_restore(specification, values, shape, dtype)
            expected_dtype = torch.float32 if backend == SDPBackend.MATH else torch.bfloat16
            self.assertEqual(dtype, expected_dtype)
            self.assertEqual(restored.dtype, expected_dtype)
            self.assertEqual(tuple(restored.shape), tuple(shape))
            diagnostic["restored_operands"].append(dict(shape=list(shape), dtype=str(restored.dtype),
                compressed=bool(specification.bits)))
            return restored

        try:
            torch.manual_seed(371)
            probe = torch.randn(8, 197, 12, 64, dtype=torch.bfloat16, device="cuda").transpose(1, 2).requires_grad_()
            supported = {SDPBackend.FLASH_ATTENTION: torch.backends.cuda.can_use_flash_attention,
                SDPBackend.EFFICIENT_ATTENTION: torch.backends.cuda.can_use_efficient_attention,
                SDPBackend.CUDNN_ATTENTION: torch.backends.cuda.can_use_cudnn_attention}.get(backend)
            if supported is not None and not supported(torch.backends.cuda.SDPAParams(
                    probe, probe, probe, None, 0., False, False), debug=True):
                reason = f"PyTorch {supported.__name__} rejected the declared geometry (see backend diagnostics)"
                diagnostic.update(status="unavailable", reason=reason)
                self.skipTest(f"Forced {backend.name} is unavailable: {reason}")
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(probe, probe, probe)
                selected = int(torch.ops.aten._fused_sdp_choice.default(probe, probe, probe))
            self.assertEqual(selected, int(backend), "Preflight did not select the requested backend")
            del probe
            for batch in (8, 32):
                for compressed in (False, True):
                    # Keep failures visible to the manifest exception handler.
                    # subTest would swallow them before its status is recorded.
                    with sdpa_kernel(backend):
                        compression = dict(enabled=compressed, bits=2, group_size=1024)
                        graph = torch.fx.Graph()
                        args = tuple(graph.placeholder(name) for name in ("q", "k", "v"))
                        graph.output(graph.call_function(do_scaled_dot_product_attention, args,
                            {"plan": {"fp8": False, "activation_compression": compression}}))
                        gm = torch.fx.GraphModule(nn.Module(), graph)
                        inputs = tuple(torch.randn(batch, 197, 12, 64, dtype=torch.bfloat16,
                            device="cuda").transpose(1, 2).requires_grad_() for _ in range(3))
                        refs = tuple(value.detach().clone().requires_grad_() for value in inputs)
                        report = CompilationReport()
                        with sdpa_kernel(backend):
                            with autocast("cuda", dtype=torch.bfloat16, fp8=False,
                                          activation_compression=compression, config={}):
                                ShapeProp(gm).propagate(*inputs)
                                expected = gm(*refs)
                                with patch.object(SDPA, "forward", record_forward):
                                    compiled = compile_fx(gm, inputs,
                                        config=BackendConfig(require_full_aot=True), report=report)
                                    actual = compiled(*inputs)
                            # The saved ABI must survive leaving the precision scope.
                            gradient = torch.randn_like(actual)
                            expected.backward(gradient)
                            with patch("torch.autograd.grad", side_effect=AssertionError("eager derivative escape")), \
                                 patch.object(SDPA, "_restore", record_restore):
                                actual.backward(gradient)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        self.assertEqual(gradient.dtype, torch.bfloat16)
                        for value, ref in zip(inputs, refs):
                            self.assertEqual(value.grad.dtype, torch.bfloat16)
                            self.assertEqual(ref.grad.dtype, torch.bfloat16)
                            torch.testing.assert_close(value.grad, ref.grad, rtol=.02, atol=.002)
                        evidence = report.graphs[0]
                        self.assertGreater(evidence.native_autodiff_operations, 0)
                        self.assertFalse(evidence.legacy_backward_operations or evidence.eager_autograd_operations)
            self.assertEqual(len(diagnostic["restored_operands"]), 12)
            diagnostic["status"] = "completed"
        except unittest.SkipTest:
            raise
        except BaseException as exc:
            diagnostic.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            after = _source_identity()
            diagnostic["source_unchanged"] = before == after
            diagnostic["source_digest"] = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
            diagnostic["imported_sources"] = {str(Path(module.__file__).resolve()): before.get(
                str(Path(module.__file__).resolve())) for name, module in tuple(sys.modules.items())
                if name == "deepopt" or name.startswith("deepopt.")
                if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")}
            print("NATIVE_ATTENTION_STATE " + json.dumps(diagnostic, sort_keys=True))
            self.assertEqual(before, after, "Source changed during GPU qualification")

    def test_forced_math_saved_state(self):
        self._qualify(SDPBackend.MATH)

    def test_forced_flash_saved_state(self):
        self._qualify(SDPBackend.FLASH_ATTENTION)

    def test_forced_efficient_saved_state(self):
        self._qualify(SDPBackend.EFFICIENT_ATTENTION)

    def test_forced_cudnn_saved_state(self):
        self._qualify(SDPBackend.CUDNN_ATTENTION)


if __name__ == "__main__":
    unittest.main()
