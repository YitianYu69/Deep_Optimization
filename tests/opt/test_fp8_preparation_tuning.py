"""Bounded preparation choices and persistent scheduling without a CUDA runtime."""

from itertools import product
import ast
import inspect
from importlib.resources import files
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from deepopt.opt.backends._shared.paired_cast import (
    MAX_PAIRED_CAST_CACHE, PAIRED_CAST_BASELINE, PAIRED_CAST_CANDIDATES,
    _paired_cast_cache, clear_paired_cast_cache, paired_cast_key, select_paired_cast,
)
from deepopt.opt.backends._shared.gemm_diagnostics import (
    record_gemms, record_gemm_launch, note_gemm_launch, observe_gemm,
    gemm_launch_is_recording,
)
from deepopt.opt.backends.triton.gemm.autotune import (
    MAX_TUNING_CONFIGS, MAX_PERSISTENT_CONFIGS, gemm_program_grid,
    grouped_tile_coordinates, prune_gemm_configs,
)
from deepopt.opt.quantization.quantizer import _quantize_tensorwise_pair


def key(**updates):
    values = dict(shape=(100864, 768), strides=(768, 1), source_dtype="bf16", target_dtype="e4m3fn",
                  device=0, capability=90, sms=132, transpose_only=False, shared_statistics=False)
    return paired_cast_key(**{**values, **updates})


class PreparationTuningTests(unittest.TestCase):
    def setUp(self):
        clear_paired_cast_cache()

    def test_disabled_and_cold_capture_keep_baseline_without_search(self):
        benchmark = Mock(side_effect=AssertionError("unexpected tuning"))
        for autotune, capturing in ((False, False), (False, True), (True, True)):
            self.assertEqual(select_paired_cast(key(), benchmark, autotune=autotune, capturing=capturing),
                             PAIRED_CAST_BASELINE)
        self.assertFalse(_paired_cast_cache)
        benchmark.assert_not_called()

    def test_winner_is_reused_in_capture_and_cache_has_no_storage(self):
        benchmark = Mock(side_effect=lambda config: 0.1 if config == (64, 4) else 0.3)
        self.assertEqual(select_paired_cast(key(), benchmark, autotune=True, capturing=False), (64, 4))
        self.assertEqual(benchmark.call_count, len(PAIRED_CAST_CANDIDATES))
        never = Mock(side_effect=AssertionError("warm tuning"))
        self.assertEqual(select_paired_cast(key(), never, autotune=True, capturing=True), (64, 4))
        self.assertEqual(select_paired_cast(key(), never, autotune=False, capturing=False), PAIRED_CAST_BASELINE)
        self.assertEqual(_paired_cast_cache, {key(): (64, 4)})

    def test_geometry_dtype_device_and_reduction_owner_separate_keys(self):
        for update in ({"shape": (768, 100864)}, {"strides": (1, 100864)},
                       {"source_dtype": "fp32"}, {"target_dtype": "e5m2"},
                       {"device": 1}, {"sms": 114}, {"capability": 100},
                       {"transpose_only": True}, {"shared_statistics": True}, {"input_alignment": 2}):
            self.assertNotEqual(key(**update), key())
        for index in range(MAX_PAIRED_CAST_CACHE + 2):
            select_paired_cast(key(shape=(index + 1024, 768)), lambda _: 1.0,
                               autotune=True, capturing=False)
        self.assertEqual(len(_paired_cast_cache), MAX_PAIRED_CAST_CACHE)
        self.assertNotIn(key(shape=(1024, 768)), _paired_cast_cache)

    def test_invalid_timings_are_not_promoted_and_failed_search_is_not_cached(self):
        times = iter((float("inf"), float("nan"), -1.0, 0.2, 0.3))
        self.assertEqual(select_paired_cast(key(), lambda _: next(times), autotune=True, capturing=False), (64, 8))
        clear_paired_cast_cache()
        with self.assertRaisesRegex(RuntimeError, "No tensorwise"):
            select_paired_cast(key(), lambda _: float("inf"), autotune=True, capturing=False)
        self.assertFalse(_paired_cast_cache)

    def test_cpu_reference_keeps_untuned_bytes_and_validates_policy(self):
        values = torch.randn(65, 97).T
        baseline = _quantize_tensorwise_pair(values, autotune=False)
        tuned = _quantize_tensorwise_pair(values, autotune=True)
        for actual, expected in zip(tuned, baseline):
            self.assertTrue(torch.equal(actual.data.view(torch.uint8), expected.data.view(torch.uint8)))
            torch.testing.assert_close(actual.scale_inv, expected.scale_inv, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "autotune must be a boolean"):
            _quantize_tensorwise_pair(values, autotune=1)


class PersistentScheduleTests(unittest.TestCase):
    def test_actual_persistent_jit_body_resets_accumulator_and_applies_bias_per_tile(self):
        from tests.opt.test_relu_kernels import CPUCast, ReLULanguage, TypedPointer

        class Language(ReLULanguage):
            range = staticmethod(lambda start, stop, step, **_: range(start, stop, step))
            assume = staticmethod(lambda _: None)
            static_assert = staticmethod(lambda condition, _: self.assertTrue(condition))

            @staticmethod
            def dot(a, b, accumulator=None, **_):
                result = a.float() @ b.float()
                return result if accumulator is None else accumulator + result

        class Descriptor:
            def __init__(self, value, block):
                self.value, self.block = value, block
                self.writes = []

            def load(self, offsets):
                row, col = offsets
                bm, bn = self.block
                result = torch.zeros(bm, bn)
                values = self.value[row:row + bm, col:col + bn]
                result[:values.shape[0], :values.shape[1]] = values.float()
                return result

            def store(self, offsets, value):
                row, col = offsets
                values = self.value[row:row + value.shape[0], col:col + value.shape[1]]
                values.copy_(value[:values.shape[0], :values.shape[1]])
                self.writes.append((row, col))

        language = Language()
        namespace = {"tl": language, "_grouped_tile_coordinates": grouped_tile_coordinates}
        tree = ast.parse(files("deepopt.opt.backends.triton").joinpath("gemm/fp8.py").read_text())
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_gemm_nt_kernel")
        kernel.decorator_list = []
        module = CPUCast().visit(ast.Module(body=[kernel], type_ignores=[]))
        exec(compile(ast.fix_missing_locations(module), "gemm/fp8.py", "exec"), namespace)
        function = namespace["_gemm_nt_kernel"]
        arguments = {name: 0 for name, parameter in inspect.signature(function).parameters.items()
                     if parameter.default is inspect.Parameter.empty}
        a = torch.randn(33, 19).to(torch.float8_e4m3fn)
        b = torch.randn(35, 19).to(torch.float8_e5m2)
        bias = torch.linspace(-1, 1, 35)
        expected = ((a.float() @ b.float().T) * 0.5) * 2.0 + bias
        for persistent, group in product((False, True), (1, 8)):
            output = torch.empty(33, 35)
            descriptor = Descriptor(output, (16, 16))
            arguments.update(a_desc=Descriptor(a, (16, 8)), b_desc=Descriptor(b, (16, 8)), c_desc=descriptor,
                             a_scale_inv_ptr=TypedPointer(torch.tensor([0.5])),
                             b_scale_inv_ptr=TypedPointer(torch.tensor([2.0])), bias_ptr=TypedPointer(bias),
                             M=33, N=35, K=19, TENSORWISE=True, NATIVE_FP8=True, HAS_BIAS=True,
                             OUTPUT_DTYPE=torch.float32, NUM_SMS=2, BLOCK_M=16, BLOCK_N=16, BLOCK_K=8,
                             NUM_STAGES=3, SPLIT_K=1, GROUP_SIZE_M=group, PERSISTENT=persistent)
            for pid in range(gemm_program_grid(33, 35, 16, 16, 1, 2, persistent)[0]):
                language.programs = (pid, 0, 0)
                function(**arguments)
            self.assertEqual(len(descriptor.writes), 9)
            self.assertEqual(len(set(descriptor.writes)), 9)
            torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)

    def test_every_tail_tile_is_owned_once_for_both_grouped_schedules(self):
        for m, n, bm, bn, sms, grouped in product((33, 257, 1009), (65, 513),
                                                (32, 128), (32, 128), (1, 7, 132), (1, 8)):
            rows, cols = (m + bm - 1) // bm, (n + bn - 1) // bn
            count = rows * cols
            for persistent in (False, True):
                grid, split = gemm_program_grid(m, n, bm, bn, 1, sms, persistent)
                owned = []
                for first in range(grid):
                    tiles = range(first, count, sms) if persistent else (first,)
                    owned.extend(grouped_tile_coordinates(tile, rows, cols, grouped) for tile in tiles)
                self.assertEqual(split, 1)
                self.assertEqual(len(owned), count)
                self.assertEqual(set(owned), set(product(range(rows), range(cols))))

    def test_only_tensorwise_unsplit_outputs_admit_persistent_candidates(self):
        from tests.opt.test_triton_autotune_policy import TritonAutotunePolicyTests, arguments

        fixture = TritonAutotunePolicyTests()
        fixture.setUp()
        configs = fixture.configs + [SimpleNamespace(
            kwargs={**config.kwargs, "PERSISTENT": True}, num_warps=config.num_warps,
            num_stages=config.num_stages) for config in fixture.configs if config.kwargs["SPLIT_K"] == 1]
        args = arguments(M=4096, N=4096, K=1024, TENSORWISE=True)
        selected = prune_gemm_configs(configs, args)
        persistent = [config for config in selected if config.kwargs.get("PERSISTENT", False)]
        self.assertTrue(persistent)
        self.assertLessEqual(len(persistent), MAX_PERSISTENT_CONFIGS)
        self.assertLessEqual(len(selected), MAX_TUNING_CONFIGS)
        self.assertTrue(all(config.kwargs["SPLIT_K"] == 1 for config in persistent))
        for updates in ({"TENSORWISE": False}, {"GRADIENT_FP8": True}, {"M": 16, "N": 16}):
            selected = prune_gemm_configs(configs, {**args, **updates})
            self.assertFalse(any(config.kwargs.get("PERSISTENT", False) for config in selected))

    def test_launch_observation_is_scoped_and_does_not_leak_to_the_next_call(self):
        operand = SimpleNamespace(data=torch.ones(2, 3), scale_inv=torch.ones(1, 1), block_shape=None)

        @record_gemm_launch
        def execute(note):
            if note:
                note_gemm_launch(kind="triton_tma", schedule="persistent", tile=(32, 64, 32))
            output = torch.ones(2, 2)
            observe_gemm(operand, operand, "triton", output, "forward")

        self.assertFalse(gemm_launch_is_recording())
        with record_gemms() as records:
            execute(True)
            execute(False)
        self.assertEqual(records[0]["kernel_launch"]["schedule"], "persistent")
        self.assertNotIn("kernel_launch", records[1])
        self.assertFalse(gemm_launch_is_recording())


if __name__ == "__main__":
    unittest.main()
