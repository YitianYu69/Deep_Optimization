"""Real CUDA coverage for fused delayed conversion and bias preprocessing."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import FP8Config, DelayedScalingConfig, TrainingRuntime, DiagnosticPolicy, fp8_scaling_scope, fp8_checkpoint
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.scaling.delayed import DelayedScalingState, delayed_states
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


def state(device, dtype):
    return DelayedScalingState({"history_length": 3},
                                dict.fromkeys(("activation", "weight", "grad"), dtype), device=device)


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class DelayedScalingCUDATests(unittest.TestCase):
    def test_real_autograd_workers_keep_one_window_across_accumulation_and_checkpointing(self):
        options = FP8Config(enabled=True, backend="triton", scaling_recipe="delayed", intercept_linear=False,
                            delayed_scaling=DelayedScalingConfig(history_length=3))
        for mode in ("max_speed", "balanced", "analyze"):
            for checkpointed in (False, True):
                with self.subTest(mode=mode, checkpointed=checkpointed):
                    model = nn.Sequential(nn.Linear(65, 97), nn.GELU(), nn.Linear(97, 33)).cuda()
                    runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.001), fp8=options,
                                              precision="bf16", execution_mode=mode)
                    states = delayed_states(runtime.model)

                    def compute(x):
                        value = fp8_checkpoint(runtime.model, x) if checkpointed else runtime.model(x)
                        return value.float().square().mean()

                    for update in range(2):
                        scales = [state.scales.clone() for state in states]
                        for _ in range(2):
                            loss = runtime.compute(torch.randn(2, 17, 65, device="cuda", requires_grad=True), compute)
                            runtime.backward(loss)
                            for state, original in zip(states, scales):
                                self.assertEqual(state.updates.cpu().tolist(), [update] * 3)
                                torch.testing.assert_close(state.scales, original, rtol=0, atol=0)
                        runtime.step()
                        for state in states:
                            self.assertEqual(state.updates.cpu().tolist(), [update + 1] * 3)
                        for parameter in runtime.model.parameters():
                            self.assertTrue(torch.isfinite(parameter).all())
                        runtime.zero_grad()

    def test_unchecked_nonfinite_conversion_preserves_payload_and_previous_scale(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(dtype=dtype, nonfinite=nonfinite):
                    actual, expected = state("cuda", dtype), state("cpu", dtype)
                    actual.scales.fill_(.5)
                    expected.scales.fill_(.5)
                    source = torch.full((17, 33), 2.)
                    source[0, 0] = nonfinite
                    a, at, _ = actual.quantize(source.cuda(), "activation", transposed=True, check_finite=False)
                    b, bt, _ = expected.quantize(source, "activation", transposed=True, check_finite=False)
                    for value, oracle in ((a, b), (at, bt)):
                        torch.testing.assert_close(value.data.float().cpu(), oracle.data.float(),
                                                   rtol=0, atol=0, equal_nan=True)
                        torch.testing.assert_close(value.scale_inv.cpu(), oracle.scale_inv, rtol=0, atol=0)
                    torch.testing.assert_close(actual.scales.cpu(), expected.scales, rtol=0, atol=0)
                    torch.testing.assert_close(actual.history.cpu(), expected.history,
                                               rtol=0, atol=0, equal_nan=True)
                    self.assertEqual(actual.updates.cpu().tolist(), [1, 0, 0])

    def test_checked_nonfinite_conversion_raises_without_updating_history(self):
        actual = state("cuda", torch.float8_e4m3fn)
        for nonfinite in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(nonfinite=nonfinite):
                source = torch.ones((17, 33), device="cuda")
                source[0, 0] = nonfinite
                with self.assertRaisesRegex(ValueError, "finite inputs"):
                    actual.quantize(source, "activation", check_finite=True)
                self.assertEqual(actual.updates.cpu().tolist(), [0, 0, 0])
                self.assertEqual(torch.count_nonzero(actual.history).item(), 0)
                torch.testing.assert_close(actual.scales, torch.ones_like(actual.scales), rtol=0, atol=0)

    def test_flat_chunk_reduction_preserves_nonfinite_and_activity_flags(self):
        from deepopt.opt.backends.triton.quantization.delayed import reduce_observations
        from deepopt.opt.backends.triton.quantization.launch import MetadataCache

        observations = [(0, torch.tensor([2.], device="cuda")),
                        (0, torch.full((25001,), 7., device="cuda")),
                        (1, torch.tensor([float("nan")], device="cuda")),
                        (2, torch.tensor([float("inf")], device="cuda"))]
        cache = MetadataCache()
        for _ in range(2):
            result = reduce_observations(observations, 4, torch.device("cuda", torch.cuda.current_device()), cache)
            expected = torch.tensor([[7., 0., 0., 0.], [1., 1., 1., 0.], [0., 2., 1., 0.]], device="cuda")
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertEqual(cache.hits, 1)

    def test_warmed_converter_allocates_only_selected_tile_scratch(self):
        from deepopt.opt.backends import triton_delayed
        from deepopt.opt.backends.triton.quantization.launch import conversion_cache_key

        source = torch.randn(129, 257, device="cuda")
        scale = torch.ones(1, device="cuda")
        options = dict(normal=True, transposed=True, bias=True, observe=True, check_finite=False)
        triton_delayed.convert(source, scale, torch.float8_e4m3fn, **options)
        key = conversion_cache_key(source, scale, torch.float8_e4m3fn, **options)
        selected = triton_delayed._convert.cache[key]
        _, _, _, observations, _ = triton_delayed.convert(source, scale, torch.float8_e4m3fn, **options)
        bm, bn = (selected.kwargs[name] for name in ("BM", "BN"))
        expected = ((129 + bm - 1) // bm) * ((257 + bn - 1) // bn) * 4
        self.assertEqual(observations.untyped_storage().nbytes(), expected)

    def test_conversion_payload_scale_snapshot_padding_and_bias_match_cpu(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            actual, expected = state("cuda", dtype), state("cpu", dtype)
            for normal, transpose in ((True, True), (True, False), (False, True)):
                source = torch.randn(65, 129).T
                a, at, bias = actual.quantize(source.cuda(), "grad", normal=normal, transposed=transpose, bias=True)
                b, bt, expected_bias = expected.quantize(source, "grad", normal=normal, transposed=transpose, bias=True)
                for value, oracle in ((a, b), (at, bt)):
                    if value is None:
                        self.assertIsNone(oracle)
                        continue
                    torch.testing.assert_close(value.data.float().cpu(), oracle.data.float(), rtol=0, atol=0)
                    torch.testing.assert_close(value.scale_inv.cpu(), oracle.scale_inv, rtol=1e-6, atol=0)
                    rows, cols = value.data.shape
                    stride = value.data.stride(0)
                    padded = value.data.as_strided((rows, stride), (stride, 1))
                    self.assertTrue(torch.all(padded[:, cols:].float() == 0))
                torch.testing.assert_close(bias.cpu(), expected_bias, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(actual.scales.cpu(), expected.scales)
                torch.testing.assert_close(actual.history.cpu(), expected.history)

    def test_ordered_bias_reduction_covers_many_row_tiles(self):
        actual = state("cuda", torch.float8_e4m3fn)
        value = torch.randn(17001, 65, device="cuda", dtype=torch.bfloat16)
        _, _, bias = actual.quantize(value, "grad", normal=False, transposed=True, bias=True)
        torch.testing.assert_close(bias, value.float().sum(0), rtol=2e-5, atol=.001)

    def test_multiple_streams_keep_snapshots_and_live_outputs_owned(self):
        actual = state("cuda", torch.float8_e4m3fn)
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        with fp8_scaling_scope():
            with torch.cuda.stream(first):
                a, _, _ = actual.quantize(torch.ones(65, 129, device="cuda"), "activation")
            with torch.cuda.stream(second):
                b, _, _ = actual.quantize(torch.full((65, 129), 8., device="cuda"), "activation")
        torch.cuda.synchronize()
        self.assertNotEqual(a.data.data_ptr(), b.data.data_ptr())
        torch.testing.assert_close(a.data.float(), torch.ones_like(a.data, dtype=torch.float32))
        torch.testing.assert_close(b.data.float(), torch.full_like(b.data, 8., dtype=torch.float32))
        torch.testing.assert_close(a.scale_inv, torch.ones_like(a.scale_inv))
        self.assertEqual(actual.updates.cpu().tolist(), [1, 0, 0])

    def test_execution_mode_cadences_cover_delayed_forward_and_fused_backward(self):
        from deepopt.opt.backends import triton_delayed
        from deepopt.opt.nn.autograd import linear as linear_autograd_func
        from deepopt.opt.backends import gemm_nt

        model = nn.Linear(65, 97, device="cuda")
        options = FP8Config(enabled=True, backend="triton", scaling_recipe="delayed", intercept_linear=False,
                            delayed_scaling=DelayedScalingConfig(history_length=3))
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.001), fp8=options,
                                  diagnostics=DiagnosticPolicy(loss_finite_every=2, gradient_finite_every=3))
        original = triton_delayed.convert
        records, phase = [], None

        def convert(*args, **kwargs):
            records.append((runtime.mode_controller.completed_updates + 1, phase, kwargs["check_finite"]))
            return original(*args, **kwargs)

        def oracle(a, b, **kwargs):
            return gemm_nt(a, b, **{**kwargs, "backend": "reference"})

        with patch.object(triton_delayed, "convert", new=convert), patch.object(linear_autograd_func, "gemm_nt", new=oracle):
            for update in range(6):
                phase = "forward"
                loss = runtime.compute(torch.randn(129, 65, device="cuda", requires_grad=update % 2 == 0),
                                       lambda x: runtime.model(x).square().mean())
                phase = "backward"
                runtime.backward(loss)
                runtime.step()
        for update, phase, checked in records:
            self.assertEqual(checked, update % (2 if phase == "forward" else 3) == 0)

    def test_dtype_move_keeps_exact_fp32_metadata(self):
        model = nn.Linear(65, 97, device="cuda")
        model, _ = prepare_fp8_model(model, FP8Config(enabled=True, backend="triton", scaling_recipe="delayed"))
        model.fp8_scaling.scales.fill_(1.1234567)
        before = deepcopy(model.fp8_scaling.scales)
        model.bfloat16()
        torch.testing.assert_close(model.fp8_scaling.scales, before, rtol=0, atol=0)
