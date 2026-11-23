"""Native-derived coefficient fusion preserves the explicit codec oracle."""

from dataclasses import replace
from contextlib import nullcontext
import importlib.util
import inspect
import json
import os
from math import prod
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from deepopt.compiler.autodiff.activation_saving import SavedActivationPolicy
from deepopt.compiler.autodiff.aot_auto_diff import AOTAutoDiff
from deepopt.compiler.autodiff.packed_multiply import backward, _multiply as native_multiply
from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import IRGraph, InputNode, TensorMeta, register_uses
from deepopt.compiler.registry import KernelCodegenContext
from deepopt.compiler.scheduler import Scheduler
from deepopt.compiler.trace.builder import GraphBuilder
from deepopt.compiler.trace.metadata_utils import contiguous_stride
from deepopt.kernel_policy import KernelPolicy
from deepopt._activation_memory import ActivationMemoryPlan
from deepopt.opt.backends._shared.packed_multiply import plan_packed_multiply, multiply as shared_multiply
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation


MULTIPLY_GRADIENT_CASES = (
    ((3, 23, 17), (1, 1, 17)),
    ((3, 23, 17), (3, 23, 17)),
    # Cross the 256-element partial boundary, including a masked tail.
    # Scalar and per-channel outputs exercise distinct finish geometries.
    ((1, 1, 1025), ()),
    ((1, 1, 2049), ()),
    ((257, 1, 17), (1, 1, 17)),
    ((513, 1, 17), (1, 1, 17)),
    # Cross the capped shard grid; each shard must loop over reduction tiles.
    ((65537, 1, 3), (1, 1, 3)),
)


class PackedMultiplyFusionTests(unittest.TestCase):
    def fixture(self, *, bits=4, shape=(3, 23, 17), gate_shape=(1, 1, 17), device="cpu", compressed=True,
                gradient_stride=None):
        graph = IRGraph()
        builder = GraphBuilder(graph)
        def metadata(dimensions):
            return TensorMeta(dimensions, torch.bfloat16, torch.device(device), contiguous_stride(dimensions), True)
        x = builder.input(0, metadata(shape), name="features")
        gate = builder.input(1, metadata(gate_shape), name="gate")
        gate.value.provenance = "parameter"
        builder.output(x * gate)
        trace = AOTAutoDiff(graph, saved_activation_policy=SavedActivationPolicy(compressed, bits, 1024)).diff()
        if gradient_stride is not None:
            seed = next(node.value for node in trace.graph.nodes if isinstance(node, InputNode) and node.index == 0)
            seed.meta = replace(seed.meta, stride=tuple(gradient_stride))
        output = graph.nodes[-1]
        output.result = (output.result, trace.saved_tensors)
        register_uses(output)
        return graph, trace

    def executable(self, graph):
        return WrapperCodegen(torch.nn.Module(), Scheduler(graph).run(), enable_autodiff=False).run().fn

    def select_static(self, graph, *, enabled=True, backend="triton", scratch_bytes=8 * 1024 * 1024):
        for node in graph.nodes:
            for value in node.outputs():
                value.meta = replace(value.meta, device=torch.device("cuda:0"))
        records = []
        patterns = backward(graph, KernelCodegenContext(kernel_policy=KernelPolicy(backend=backend),
                            fuse_packed_multiply=enabled,
                            activation_memory=ActivationMemoryPlan(reduction_scratch_bytes=scratch_bytes)), records)
        for node in graph.nodes:
            for value in node.outputs():
                value.meta = replace(value.meta, device=torch.device("cpu"))
        graph.validate()
        return patterns, records

    def test_target_se_and_scalar_cost_gate_uses_bounded_launch_plan(self):
        shape = (128, 768, 224, 224)
        plan = plan_packed_multiply(shape, (128, 768, 1, 1))
        self.assertEqual((plan.parts, plan.output_tile, plan.output_tiles, plan.reduction_loops), (196, 8192, 12, 1))
        self.assertEqual(plan.scratch_bytes, 6422528)
        scalar = plan_packed_multiply(shape, ())
        self.assertEqual((scalar.parts, scalar.output_tile, scalar.output_tiles), (256, 1, 1))
        self.assertGreater(scalar.reduction_loops, 4096)
        self.assertEqual(scalar.scratch_bytes, 1024)
        _, trace = self.fixture(bits=2, shape=shape, gate_shape=(128, 768, 1, 1))
        patterns, records = self.select_static(trace.graph)
        self.assertIn("packed_multiply_reduce", patterns)
        selected = next(row for row in records if row["selected"])
        self.assertEqual(selected["replacement_contract"]["launch_plan"], plan.as_dict())
        _, trace = self.fixture(bits=2, shape=shape, gate_shape=(128, 768, 1, 1))
        patterns, records = self.select_static(trace.graph, scratch_bytes=4)
        self.assertFalse(patterns)
        self.assertEqual(records[0]["reason"], "bounded_reduction_cost_gate")

    def test_static_plans_cover_output_and_reduction_tails_without_triton(self):
        for outputs in (1, 15, 16, 17, 8191, 8192, 8193, 16385):
            for reduced in (1, 255, 256, 257, 65535, 65536, 65537):
                with self.subTest(outputs=outputs, reduced=reduced):
                    plan = plan_packed_multiply((reduced, outputs), (1, outputs), 8192)
                    self.assertLessEqual(plan.scratch_bytes, 8192)
                    if plan.direct:
                        self.assertEqual(plan.scratch_bytes, 0)
                        continue
                    self.assertLessEqual(plan.parts, 256)
                    self.assertEqual(plan.output_tile & (plan.output_tile - 1), 0)
                    self.assertEqual(plan.output_tiles, (outputs + plan.output_tile - 1) // plan.output_tile)
                    self.assertGreaterEqual(plan.parts * plan.reduction_loops * 256, reduced)
        for shape, target in (((0, 17), (1, 17)), ((0, 17), (0, 17)), ((), ())):
            plan = plan_packed_multiply(shape, target)
            self.assertEqual(plan.scratch_bytes, 0)
        with self.assertRaisesRegex(ValueError, "broadcast_or_sum_to_size_contract"):
            plan_packed_multiply((3, 17), (2, 17))

    def test_selected_native_cuda_route_rejects_runtime_contract_drift(self):
        gradient = SimpleNamespace(shape=(2, 512), dtype=torch.float64, device=torch.device("cuda"))
        options = dict(logical_shape=(2, 512), logical_dtype=torch.bfloat16, group_size=1024, bits=2)
        with patch("deepopt.opt.quantization.activation.dequantize_unpack_activation",
                   side_effect=AssertionError("selected CUDA replacement fell back to materialization")):
            with self.assertRaisesRegex(ValueError, "dtype_or_scalar_promotion"):
                native_multiply(gradient, None, None, None, options=options, shape=(2, 512),
                    product_dtype=torch.bfloat16, target_shape=())

    def test_same_shape_and_broadcast_gradients_match_codec_product_rounding(self):
        for bits in (1, 2, 4, 8):
            for gate_shape in ((3, 23, 17), (1, 1, 17), (17,), ()):
                with self.subTest(bits=bits, gate_shape=gate_shape):
                    graph, trace = self.fixture(bits=bits, gate_shape=gate_shape)
                    x = torch.randn(3, 23, 17, dtype=torch.bfloat16)
                    gate = torch.randn(gate_shape, dtype=torch.bfloat16)
                    _, saved = self.executable(graph)(x, gate)
                    seed = torch.randn_like(x)
                    expected = self.executable(trace.graph)(seed, *saved)
                    patterns, records = self.select_static(trace.graph)
                    self.assertTrue(patterns)
                    json.dumps(records, allow_nan=False)
                    actual = self.executable(trace.graph)(seed, *saved)
                    for value, control in zip(actual, expected):
                        torch.testing.assert_close(value, control, rtol=0, atol=0)
                    record = next(row for row in records if row["selected"])
                    self.assertTrue(record["profile_range"].startswith("deepopt.fusion::"))
                    self.assertLessEqual(record["replacement_contract"]["estimated_scratch_bytes"], 8 * 1024 * 1024)
                    self.assertFalse(record["replacement_contract"]["coefficient_restore_retained"])

    def test_disabled_and_incompatible_backend_retain_numerical_operations(self):
        for enabled, backend, reason in ((False, "triton", "disabled"), (True, "tilelang", "backend_has_no_packed_multiply")):
            _, trace = self.fixture()
            patterns, records = self.select_static(trace.graph, enabled=enabled, backend=backend)
            self.assertFalse(patterns)
            self.assertEqual(records[0]["reason"], reason)
            self.assertIn("codec.restore_activation", [getattr(node, "op", None) for node in trace.graph.nodes])

    def test_uncompressed_does_not_claim_packed_work(self):
        _, trace = self.fixture(compressed=False)
        patterns, records = self.select_static(trace.graph)
        self.assertFalse(patterns or records)

    def test_noncontiguous_gradient_fixture_declares_actual_stride(self):
        for shape, gate_shape in MULTIPLY_GRADIENT_CASES:
            with self.subTest(shape=shape, gate_shape=gate_shape):
                seed = torch.randn((*reversed(shape), 2), dtype=torch.bfloat16)[..., 0].permute(2, 1, 0)
                self.assertFalse(seed.is_contiguous())
                graph, trace = self.fixture(shape=shape, gate_shape=gate_shape, gradient_stride=seed.stride())
                x = torch.randn(shape, dtype=torch.bfloat16)
                gate = torch.randn(gate_shape, dtype=torch.bfloat16)
                _, saved = self.executable(graph)(x, gate)
                expected = self.executable(trace.graph)(seed, *saved)
                patterns, _ = self.select_static(trace.graph)
                self.assertTrue(patterns)
                actual = self.executable(trace.graph)(seed, *saved)
                for value, control in zip(actual, expected):
                    torch.testing.assert_close(value, control, rtol=0, atol=0)

    def test_packed_kernel_host_layout_covers_broadcast_reductions_and_strides(self):
        # Exercise launch metadata and the kernel's address arithmetic on CPU.
        # Identity JIT decoration deliberately does not claim Triton compilation.
        language = SimpleNamespace(constexpr=object(), int64=torch.int64, static_range=range,
            full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype))
        triton = SimpleNamespace(jit=lambda function: function, language=language,
            cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
            next_power_of_2=lambda value: 1 << (value - 1).bit_length())
        source = importlib.util.find_spec("deepopt.opt.backends.triton.packed_multiply")
        specification = importlib.util.spec_from_file_location(
            "deepopt.opt.backends.triton._packed_multiply_layout_test", source.origin)
        backend = importlib.util.module_from_spec(specification)
        with patch.dict(sys.modules, {"triton": triton, "triton.language": language,
                "deepopt.opt.backends.triton.quantization.activation.affine":
                    SimpleNamespace(_affine_decode=None)}):
            specification.loader.exec_module(backend)

        shape = (3, 23, 17)
        storage = torch.randn(17, 23, 3, dtype=torch.bfloat16)
        gradient = storage.permute(2, 1, 0)
        logical_strides = contiguous_stride(shape)
        addresses = backend._addresses(torch.arange(prod(shape)), shape, logical_strides, gradient.stride())
        torch.testing.assert_close(storage.flatten()[addresses], gradient.flatten(), rtol=0, atol=0)
        huge = (128, 768, 224, 224)
        indices = torch.tensor([0, 2**31 - 1, 2**31, 2**32 - 1, prod(huge) - 1], dtype=torch.int64)
        addresses = backend._addresses(indices, huge, contiguous_stride(huge), contiguous_stride(huge))
        torch.testing.assert_close(addresses, indices, rtol=0, atol=0)
        addresses = backend._addresses(torch.tensor([1, 2], dtype=torch.int32), (3,), (1,), (2**32,))
        torch.testing.assert_close(addresses, torch.tensor([2**32, 2**33]), rtol=0, atol=0)
        options = dict(group_size=1024, bits=4, logical_dtype=torch.bfloat16, logical_shape=shape)
        data = torch.zeros((20, 2), dtype=torch.int32)
        scale = torch.ones(20)
        for target_shape in (shape, (1, 1, 17), (3, 1, 17), (1, 23, 1), (17,), ()):
            with self.subTest(target_shape=target_shape):
                dense, partial, finish = MagicMock(), MagicMock(), MagicMock()
                kernel = backend._packed_multiply if target_shape == shape else backend._packed_multiply_partial
                with patch.object(backend, "_packed_multiply", dense), \
                        patch.object(backend, "_packed_multiply_partial", partial), \
                        patch.object(backend, "_packed_multiply_finish", finish), \
                        patch.object(torch.cuda, "device", return_value=nullcontext()):
                    result = backend.multiply(gradient, data, scale, scale, options=options, shape=shape,
                        product_dtype=torch.bfloat16, target_shape=target_shape)
                self.assertEqual(tuple(result.shape), target_shape)
                launch = (dense if target_shape == shape else partial).__getitem__.return_value.call_args
                arguments = inspect.signature(kernel).bind(*launch.args).arguments
                self.assertEqual(arguments["LOGICAL_STRIDES"], logical_strides)
                self.assertEqual(arguments["GS"], gradient.stride())
                if target_shape == shape:
                    finish.__getitem__.assert_not_called()
                    continue
                target, reduction = arguments["TARGET"], arguments["REDUCTION"]
                outputs, reduced = prod(target), prod(reduction)
                logical = (backend._addresses(torch.arange(outputs), target,
                    arguments["TARGET_STRIDES"], logical_strides)[None, :]
                    + backend._addresses(torch.arange(reduced), reduction,
                    arguments["REDUCTION_STRIDES"], logical_strides)[:, None])
                reduced_axes = [axis for axis in range(len(shape)) if target[axis] == 1]
                kept_axes = [axis for axis in range(len(shape)) if target[axis] != 1]
                expected = torch.arange(prod(shape)).reshape(shape).permute(
                    reduced_axes + kept_axes).reshape(reduced, outputs)
                torch.testing.assert_close(logical, expected, rtol=0, atol=0)
                finish.__getitem__.return_value.assert_called_once()

        # The actual SE host launch allocates a single bounded partial buffer,
        # then reuses its identity for all twelve output tiles.
        shape, target_shape = (128, 768, 224, 224), (128, 768, 1, 1)
        metadata_gradient = SimpleNamespace(shape=shape, stride=lambda: contiguous_stride(shape), device="cpu")
        options.update(logical_shape=shape, bits=2)
        partial, finish = MagicMock(), MagicMock()
        with patch.object(backend, "_packed_multiply_partial", partial), \
                patch.object(backend, "_packed_multiply_finish", finish), \
                patch.object(torch.cuda, "device", return_value=nullcontext()):
            backend.multiply(metadata_gradient, data, scale, scale, options=options, shape=shape,
                product_dtype=torch.bfloat16, target_shape=target_shape)
        partial_calls = partial.__getitem__.return_value.call_args_list
        self.assertEqual(len(partial_calls), 12)
        self.assertEqual(finish.__getitem__.return_value.call_count, 12)
        self.assertEqual(len({id(call.args[4]) for call in partial_calls}), 1)
        self.assertEqual(tuple(partial_calls[0].args[4].shape), (196, 8192))
        bindings = [inspect.signature(backend._packed_multiply_partial).bind(*call.args).arguments
                    for call in partial_calls]
        self.assertEqual([binding["OUTPUT_OFFSET"] for binding in bindings], list(range(0, 98304, 8192)))
        self.assertTrue(all(binding["TILE_OUTPUTS"] == 8192 and binding["PARTS"] == 196
                            and binding["LOOPS"] == 1 for binding in bindings))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA qualification requires a GPU")
    def test_cuda_output_tiles_and_distinct_streams_own_independent_scratch(self):
        from deepopt.opt.backends.triton import packed_multiply as backend

        shape, target = (1025, 33), (1, 33)
        sources = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        packed = [quantize_pack_activation(source, 1024, bits=2) for source in sources]
        seeds = [torch.randn(33, 1025, 2, device="cuda", dtype=torch.bfloat16)[..., 0].T for _ in range(2)]
        expected = [(seed * dequantize_unpack_activation(state, dtype=torch.bfloat16,
                    row_alignment=1)).sum_to_size(target) for seed, state in zip(seeds, packed)]
        streams = [torch.cuda.Stream() for _ in range(2)]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        options = dict(logical_shape=shape, logical_dtype=torch.bfloat16, group_size=1024, bits=2,
                       traversal="logical_row_major")
        # Warm compilation separately; the following pass measures invocations.
        shared_multiply(seeds[0], packed[0].data, packed[0].step, packed[0].minimum,
            options=options, shape=shape, product_dtype=torch.bfloat16, target_shape=target,
            scratch_bytes=512, allow_materialized=False)
        torch.cuda.synchronize()
        allocations, results, reports = [], [], []
        empty = torch.empty
        def allocate(*args, **kwargs):
            value = empty(*args, **kwargs)
            if kwargs.get("dtype") == torch.float32:
                allocations.append((value.data_ptr(), tuple(value.shape), value.numel() * value.element_size()))
            return value
        with patch.object(backend.torch, "empty", side_effect=allocate):
            for stream, seed, state in zip(streams, seeds, packed):
                info = {}
                with torch.cuda.stream(stream):
                    results.append(shared_multiply(seed, state.data, state.step, state.minimum,
                        options=options, shape=shape, product_dtype=torch.bfloat16, target_shape=target,
                        scratch_bytes=512, execution_info=info, allow_materialized=False))
                reports.append(info)
        for stream in streams:
            stream.synchronize()
        self.assertEqual(len(allocations), 2)
        self.assertNotEqual(allocations[0][0], allocations[1][0])
        self.assertTrue(all(allocation[2] <= 512 for allocation in allocations))
        for value, control, report in zip(results, expected, reports):
            torch.testing.assert_close(value, control, rtol=.015, atol=.015)
            self.assertEqual(report["launch_plan"]["output_tiles"], 3)
            self.assertEqual(report["route"], "packed_streaming")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA qualification requires a GPU")
    def test_cuda_cancellation_and_product_rounding_before_grid_stride_reduction(self):
        shape = (65537, 3)
        indices = torch.arange(prod(shape), device="cuda").reshape(shape)
        # Alternate signs and adjacent BF16 values so cancellation would expose
        # accumulation of unrounded FP32 products.
        source = (1 + (indices % 7).float() / 128).to(torch.bfloat16)
        seed = torch.where(indices % 2 == 0, 1.0078125, -.99609375).to(torch.bfloat16)
        for bits in (1, 2, 4, 8):
            state = quantize_pack_activation(source, 1024, bits=bits)
            coefficient = dequantize_unpack_activation(state, dtype=torch.bfloat16, row_alignment=1)
            options = dict(logical_shape=shape, logical_dtype=torch.bfloat16, group_size=1024, bits=bits,
                           traversal="logical_row_major")
            direct = shared_multiply(seed, state.data, state.step, state.minimum, options=options,
                shape=shape, product_dtype=torch.bfloat16, target_shape=shape, allow_materialized=False)
            torch.testing.assert_close(direct, seed * coefficient, rtol=0, atol=0)
            reduced = shared_multiply(seed, state.data, state.step, state.minimum, options=options,
                shape=shape, product_dtype=torch.bfloat16, target_shape=(1, 3), allow_materialized=False)
            expected = (seed * coefficient).sum_to_size((1, 3))
            torch.testing.assert_close(reduced, expected, rtol=.015, atol=.015)
            error, reference = torch.linalg.vector_norm(reduced.float() - expected.float()), torch.linalg.vector_norm(expected.float())
            self.assertLessEqual(float(error), .015 * float(reference) + .015)
            self.assertEqual(torch.count_nonzero(reduced), torch.count_nonzero(expected))

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("DEEPOPT_TEST_LARGE_CUDA") == "1",
                         "Full H200 SE operator requires DEEPOPT_TEST_LARGE_CUDA=1 and CUDA")
    def test_h200_actual_se_geometry_has_no_dense_decode_or_product(self):
        if "H200" not in torch.cuda.get_device_name():
            self.skipTest("The target geometry allocation test requires the H200 reference device")
        shape, target = (128, 768, 224, 224), (128, 768, 1, 1)
        groups = (prod(shape) + 1023) // 1024
        seed = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
        data = torch.zeros((groups, 64), dtype=torch.int32, device="cuda")
        step = torch.ones(groups, device="cuda")
        minimum = torch.full((groups,), .5, device="cuda")
        options = dict(logical_shape=shape, logical_dtype=torch.bfloat16, group_size=1024, bits=2,
                       traversal="logical_row_major")
        # No coefficient checkpoint is needed: every packed code decodes to .5.
        # A clean allocator window is local to this opt-in operator regression.
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        info = {}
        with patch("deepopt.opt.quantization.activation.dequantize_unpack_activation",
                   side_effect=AssertionError("target packed route materialized a coefficient")):
            result = shared_multiply(seed, data, step, minimum, options=options, shape=shape,
                product_dtype=torch.bfloat16, target_shape=target, execution_info=info, allow_materialized=False)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, torch.full(target, 25088, dtype=torch.bfloat16, device="cuda"), rtol=0, atol=0)
        self.assertEqual(info["scratch_bytes"], 6422528)
        self.assertEqual(info["launch_plan"]["output_tiles"], 12)
        self.assertLess(torch.cuda.max_memory_allocated() - baseline, 16 * 1024 * 1024)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA qualification requires a GPU")
    def test_cuda_broadcast_reduction_and_noncontiguous_gradient(self):
        for bits in (1, 2, 4, 8):
            for shape, gate_shape in MULTIPLY_GRADIENT_CASES:
                with self.subTest(bits=bits, shape=shape, gate_shape=gate_shape):
                    # The slice retains a nonunit stride even for scalar-output
                    # reductions where all but one logical axis are singleton.
                    seed = torch.randn((*reversed(shape), 2), device="cuda",
                        dtype=torch.bfloat16)[..., 0].permute(2, 1, 0)
                    self.assertFalse(seed.is_contiguous())
                    graph, trace = self.fixture(bits=bits, shape=shape, gate_shape=gate_shape,
                        device="cuda", gradient_stride=seed.stride())
                    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                    gate = torch.randn(gate_shape, device="cuda", dtype=torch.bfloat16)
                    _, saved = self.executable(graph)(x, gate)
                    expected = self.executable(trace.graph)(seed, *saved)
                    records = []
                    patterns = backward(trace.graph, KernelCodegenContext(kernel_policy=KernelPolicy(),
                        fuse_packed_multiply=True), records)
                    self.assertTrue(patterns)
                    if shape == gate_shape:
                        self.assertIn("packed_multiply", patterns)
                    else:
                        self.assertIn("packed_multiply_reduce", patterns)
                        scratch = plan_packed_multiply(shape, gate_shape).scratch_bytes
                        self.assertIn(scratch, [record["replacement_contract"]["estimated_scratch_bytes"]
                            for record in records if record["selected"]])
                    actual = self.executable(trace.graph)(seed, *saved)
                    for value, control in zip(actual, expected):
                        torch.testing.assert_close(value, control, rtol=.015, atol=.015)
                        error = torch.linalg.vector_norm(value.float() - control.float())
                        reference = torch.linalg.vector_norm(control.float())
                        self.assertLessEqual(float(error), .015 * float(reference) + .015)
                        self.assertGreaterEqual(torch.count_nonzero(value), .95 * torch.count_nonzero(control))


if __name__ == "__main__":
    unittest.main()
