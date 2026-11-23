"""Padding retains geometry only and shares its eager/native numerical adjoint."""

import unittest
import weakref
import json
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.nn import functional as F

from deepopt import autocast
from deepopt.opt.nn.functional import do_pad


@contextmanager
def _strict_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _zero_eager_vjp(test, *, device, dtype, channels_last):
    generator = torch.Generator().manual_seed(834)
    shape = (1, 3, 224, 224)
    source = torch.randn(shape, generator=generator, dtype=dtype).to(device)
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    source = source.contiguous(memory_format=memory_format).requires_grad_()
    gradient = torch.randn((1, 3, 224, 448), generator=generator, dtype=dtype).to(device)[..., ::2]
    for mode in ("reflect", "replicate"):
        with test.subTest(mode=mode, dtype=dtype, channels_last=channels_last):
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value,
                                                          lambda value: value):
                output = do_pad(source, (0, 0, 0, 0), mode)
            actual, = torch.autograd.grad(output, source, gradient)
            test.assertEqual(saved, [])
            torch.testing.assert_close(output, source, rtol=0, atol=0)
            torch.testing.assert_close(actual, gradient, rtol=0, atol=0)
            test.assertTrue(actual.is_contiguous(memory_format=memory_format))
            test.assertNotEqual(actual.data_ptr(), gradient.data_ptr())
            test.assertTrue(torch.are_deterministic_algorithms_enabled())
            test.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())


class MetadataPaddingTests(unittest.TestCase):
    def test_zero_width_adjoint_is_fresh_and_preserves_dtype_and_layout(self):
        from deepopt.opt.nn.padding import PaddingGeometry, padding_adjoint

        for rank in (1, 2, 3):
            shape = (2, 3, *([5] * rank))
            memory_format = (torch.contiguous_format, torch.channels_last, torch.channels_last_3d)[rank - 1]
            for dtype in (torch.float32, torch.bfloat16):
                source = torch.empty(shape, dtype=dtype).contiguous(memory_format=memory_format)
                gradient = torch.randn((*shape[:-1], 2 * shape[-1]))[..., ::2]
                original = gradient.clone()
                for mode in ("constant", "reflect", "replicate", "circular"):
                    with self.subTest(rank=rank, dtype=dtype, mode=mode):
                        geometry = PaddingGeometry.from_tensor(source, (0,) * (2 * rank), mode)
                        actual = padding_adjoint(gradient, geometry)
                        torch.testing.assert_close(actual, gradient.to(dtype), rtol=0, atol=0)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertTrue(actual.is_contiguous(memory_format=memory_format))
                        self.assertNotEqual(actual.data_ptr(), gradient.data_ptr())
                        actual.add_(1)
                        torch.testing.assert_close(gradient, original, rtol=0, atol=0)

    def test_zero_width_eager_vjp_avoids_native_rejection_under_strict_policy(self):
        rejection = RuntimeError("padding_backward_cuda does not have a deterministic implementation")
        with _strict_algorithms(), \
                patch.object(torch.ops.aten.reflection_pad2d_backward, "default", side_effect=rejection) as reflect, \
                patch.object(torch.ops.aten.replication_pad2d_backward, "default", side_effect=rejection) as replicate:
            for dtype in (torch.float32, torch.bfloat16):
                for channels_last in (False, True):
                    _zero_eager_vjp(self, device="cpu", dtype=dtype, channels_last=channels_last)
            self.assertEqual(reflect.call_count, 0)
            self.assertEqual(replicate.call_count, 0)

    def test_nonzero_pads_still_preserve_native_strict_rejections(self):
        from deepopt.opt.nn.padding import PaddingGeometry, padding_adjoint

        source = torch.randn(2, 3, 5, 7)
        for mode, operation in (("reflect", torch.ops.aten.reflection_pad2d_backward),
                                ("replicate", torch.ops.aten.replication_pad2d_backward)):
            # A zero *sum* still changes geometry and must use its true adjoint.
            for pads in ((-1, 1, 0, 0), (1, 2, 1, 2)):
                with self.subTest(mode=mode, pads=pads), _strict_algorithms():
                    gradient = torch.randn_like(F.pad(source, pads, mode))
                    geometry = PaddingGeometry.from_tensor(source, pads, mode)
                    with patch.object(operation, "default", side_effect=RuntimeError(
                            "fixture does not have a deterministic implementation")) as backward:
                        with self.assertRaisesRegex(RuntimeError, "does not have a deterministic implementation"):
                            padding_adjoint(gradient, geometry)
                    backward.assert_called_once()
                    self.assertEqual(backward.call_args.args[2], list(pads))
                    self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def _check(self, shape, pads, mode, *, dtype=torch.float32, channels_last=False, device="cpu"):
        torch.manual_seed(834)
        x = torch.randn(shape, dtype=dtype, device=device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x = x.detach().requires_grad_()
        reference = x.detach().clone().requires_grad_()
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
            actual = do_pad(x, pads, mode)
        expected = F.pad(reference, pads, mode)
        self.assertEqual(saved, [])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        if device == "cuda" and mode in ("reflect", "replicate"):
            # Native CUDA can atomically accumulate in different orders.
            # Independently check both routes against the same per-cell oracle;
            # direct adapter/native agreement cannot justify either route alone.
            from benchmarks.compiler.padding_validation import padding_error_report, padding_reference

            oracle = padding_reference(gradient, shape, pads, mode)
            for label, value in (("metadata", x.grad), ("native", reference.grad)):
                report = padding_error_report(value, oracle)
                self.assertTrue(report["passed"], f"{label}: {json.dumps(report)}")
        else:
            torch.testing.assert_close(x.grad, reference.grad, rtol=1e-6, atol=1e-6)
        if channels_last:
            self.assertTrue(x.grad.is_contiguous(memory_format=torch.channels_last))

    def test_all_spatial_ranks_and_modes(self):
        for rank in (1, 2, 3):
            for mode in ("constant", "reflect", "replicate", "circular"):
                with self.subTest(rank=rank, mode=mode):
                    self._check((2, 3, *([5] * rank)), (1, 2) * rank, mode)

    def test_negative_cropping_and_asymmetric_pads(self):
        for mode in ("constant", "reflect", "replicate", "circular"):
            for pads in ((-1, 2, 1, -2), (-1, -1, -2, 1)):
                with self.subTest(mode=mode, pads=pads):
                    self._check((2, 3, 6, 7), pads, mode)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_bf16_metadata_padding_matches_native(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("requires CUDA BF16 support")
        for mode in ("constant", "reflect", "replicate", "circular"):
            with self.subTest(mode=mode):
                self._check((2, 3, 6, 7), (2, 3, 2, 1), mode, dtype=torch.bfloat16,
                            channels_last=True, device="cuda")

    def test_circular_wrap_keeps_native_bf16_accumulation_boundaries(self):
        torch.manual_seed(9)
        for dtype in (torch.float32, torch.bfloat16):
            source = torch.randn(2, 3, 5, 7, dtype=dtype, requires_grad=True)
            reference = source.detach().clone().requires_grad_()
            actual = do_pad(source, (5, 4, 3, 2), "circular")
            expected = F.pad(reference, (5, 4, 3, 2), "circular")
            gradient = torch.randn_like(actual)
            actual.backward(gradient)
            expected.backward(gradient)
            torch.testing.assert_close(source.grad, reference.grad, rtol=0, atol=0)

    def test_padding_oracle_geometry_and_budget_reject_wrong_adjoint(self):
        from benchmarks.compiler.padding_validation import padding_error_report, padding_reference

        for rank in (1, 2, 3):
            for batched in (False, True):
                shape = (*( (2,) if batched else ()), 3, *([5] * rank))
                for pads in ((1, 2) * rank, (-1, 2) * rank):
                    for mode in ("reflect", "replicate"):
                        with self.subTest(rank=rank, batched=batched, pads=pads, mode=mode):
                            source = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                            expected = F.pad(source, pads, mode)
                            gradient = torch.randint(-2, 3, expected.shape).to(torch.float64)
                            oracle = padding_reference(gradient, shape, pads, mode)
                            expected.backward(gradient)
                            # Integer terms and bounded sums are exact in FP64.
                            torch.testing.assert_close(oracle["sum"], source.grad, rtol=0, atol=0)
                            self.assertTrue(padding_error_report(source.grad, oracle)["passed"])
                            wrong = source.grad.clone()
                            wrong.flatten()[0] += 1
                            self.assertFalse(padding_error_report(wrong, oracle)["passed"])
        with self.assertRaisesRegex(ValueError, "element bound"):
            padding_reference(torch.ones(2, 3, 8, 8), (2, 3, 6, 6), (1, 1, 1, 1),
                              "replicate", max_elements=10)

    def test_padding_fp32_and_bf16_rounding_budgets_have_independent_integer_controls(self):
        from benchmarks.compiler.padding_validation import padding_error_report, padding_reference

        generator = torch.Generator().manual_seed(834)
        shape, pads = (2, 3, 6, 7), (2, 3, 2, 1)
        for dtype in (torch.bfloat16, torch.float32):
            for mode, contributors in (("reflect", 4), ("replicate", 12)):
                source = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
                output = F.pad(source, pads, mode)
                gradient = torch.randn(output.shape, generator=generator, dtype=dtype)
                actual, = torch.autograd.grad(output, source, gradient)
                oracle = padding_reference(gradient, shape, pads, mode)
                report = padding_error_report(actual, oracle)
                self.assertEqual(report["max_contributors"], contributors)
                self.assertTrue(report["passed"], json.dumps(report))
                wrong = actual.clone()
                wrong.flatten()[0] += 32
                self.assertFalse(padding_error_report(wrong, oracle)["passed"])
                with self.assertRaisesRegex(ValueError, "matching upstream/output"):
                    padding_error_report(actual.double(), oracle)
                integer = torch.randint(-2, 3, output.shape, generator=generator).to(dtype)
                reference = padding_reference(integer, shape, pads, mode)
                expected, = torch.autograd.grad(F.pad(source, pads, mode), source, integer)
                torch.testing.assert_close(expected.double(), reference["sum"], rtol=0, atol=0)

    def test_padding_oracle_rejects_overflow_and_bounds_mixed_crop_pad_geometry(self):
        from benchmarks.compiler.padding_validation import padding_locations, padding_reference, padding_error_report

        with self.assertRaisesRegex(ValueError, "oracle overflow"):
            padding_reference(torch.full((1, 3), 1e308, dtype=torch.float64), (1, 1), (1, 1), "replicate")
        # Both original and final maps have1000 elements. Building an expanded
        # intermediate before cropping would allocate one million elements.
        locations = padding_locations((1, 1000, 1), (999, 0, -999, 0), "replicate", max_elements=1000)
        self.assertEqual(tuple(locations.shape), (1, 1, 1000))
        self.assertTrue(torch.equal(locations, torch.full_like(locations, 999)))
        oracle = padding_reference(torch.ones(1, 3), (1, 1), (1, 1), "replicate")
        bad = torch.full((1, 1), float("nan"))
        report = padding_error_report(bad, oracle)
        self.assertFalse(report["passed"])
        json.dumps(report, allow_nan=False)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_padding_geometry_and_rounding_oracle(self):
        from benchmarks.compiler.padding_validation import padding_error_report, padding_reference
        from deepopt.opt.nn.padding import PaddingGeometry, padding_adjoint

        if not torch.cuda.is_bf16_supported():
            self.skipTest("requires CUDA BF16 support")
        shape, pads = (2, 3, 6, 7), (2, 3, 2, 1)
        for dtype in (torch.bfloat16, torch.float32):
            for channels_last in (False, True):
                for mode in ("reflect", "replicate"):
                    with self.subTest(dtype=dtype, channels_last=channels_last, mode=mode):
                        generator = torch.Generator().manual_seed(834)
                        source = torch.randn(shape, generator=generator, dtype=dtype).cuda()
                        if channels_last:
                            source = source.contiguous(memory_format=torch.channels_last)
                        output_shape = F.pad(source, pads, mode).shape
                        integer = torch.randint(-2, 3, output_shape, generator=generator).to(dtype).cuda()
                        floating = torch.randn(output_shape, generator=generator, dtype=dtype).cuda()
                        if channels_last:
                            integer = integer.contiguous(memory_format=torch.channels_last)
                            floating = floating.contiguous(memory_format=torch.channels_last)
                        geometry = PaddingGeometry.from_tensor(source, pads, mode)
                        reports = []
                        observed = {"native": [], "metadata": []}
                        for label, gradient in (("integer_geometry", integer), ("floating_rounding", floating)):
                            oracle = padding_reference(gradient, shape, pads, mode)
                            for repeat in range(32):
                                leaf = source.detach().requires_grad_()
                                native, = torch.autograd.grad(F.pad(leaf, pads, mode), leaf, gradient)
                                metadata = padding_adjoint(gradient, geometry)
                                for route, result in (("native", native), ("metadata", metadata)):
                                    report = padding_error_report(result, oracle)
                                    report.update(route=route, repeat=repeat, control=label)
                                    reports.append(report)
                                    self.assertTrue(report["passed"], json.dumps(report))
                                    if label == "integer_geometry":
                                        # Exact integer sums isolate mapping from floating accumulation.
                                        torch.testing.assert_close(result.cpu().double(), oracle["sum"], rtol=0, atol=0)
                                    else:
                                        observed[route].append(result.detach().cpu().double())
                        # Repeat and cross-route spreads are diagnostics, not a
                        # blanket tolerance or replacement for per-cell checks.
                        evidence = {"dtype": str(dtype), "layout": "channels_last" if channels_last else "contiguous",
                            "mode": mode, "repeats_per_route": 32, "max_contributors": reports[-1]["max_contributors"],
                            "source_strides": list(source.stride()), "gradient_strides": list(floating.stride()),
                            "native_repeat_spread": float((torch.stack(observed["native"]).amax(0) -
                                                           torch.stack(observed["native"]).amin(0)).max()),
                            "metadata_repeat_spread": float((torch.stack(observed["metadata"]).amax(0) -
                                                             torch.stack(observed["metadata"]).amin(0)).max()),
                            "cross_route_max_abs": float((torch.stack(observed["native"]) -
                                                          torch.stack(observed["metadata"])).abs().max()),
                            "all_oracle_checks_passed": all(row["passed"] for row in reports)}
                        print("padding_oracle=" + json.dumps(evidence, sort_keys=True), flush=True)

    def test_unbatched_strided_and_channels_last_sources(self):
        for mode in ("reflect", "replicate", "circular"):
            self._check((3, 6, 7), (2, 1, 1, 2), mode)
            self._check((2, 3, 6, 7), (2, 1, 1, 2), mode, channels_last=True)
        source = torch.randn(2, 3, 6, 14, requires_grad=True)
        expected_source = source.detach().clone().requires_grad_()
        do_pad(source[..., ::2], (1, 2, 1, 2), "reflect").sum().backward()
        F.pad(expected_source[..., ::2], (1, 2, 1, 2), "reflect").sum().backward()
        torch.testing.assert_close(source.grad, expected_source.grad, rtol=0, atol=0)

    def test_bf16_scope_intercepts_native_pad_without_source_save(self):
        x = torch.randn(2, 3, 5, 7, dtype=torch.bfloat16, requires_grad=True)
        saved = []
        with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                activation_compression={"enabled": True, "bits": 2, "group_size": 1024}, config={}), \
                torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
            output = F.pad(x, (1, 2, 2, 1), "reflect")
        self.assertEqual(saved, [])
        self.assertEqual(type(output.grad_fn).__name__, "_PaddingFunctionBackward")
        output.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_interleaved_invocations_retained_graph_and_scope_exit(self):
        x = torch.randn(2, 3, 5, 7, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        first = do_pad(x, (1, 2, 2, 1), "reflect")
        second = do_pad(x, (2, 0, 0, 1), "replicate")
        expected = F.pad(reference, (1, 2, 2, 1), "reflect")
        first.sum().backward(retain_graph=True)
        expected.sum().backward(retain_graph=True)
        first.sum().backward()
        expected.sum().backward()
        second.sum().backward()
        F.pad(reference, (2, 0, 0, 1), "replicate").sum().backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)

    def test_source_mutation_uses_geometry_and_no_retained_value(self):
        # These adjoints are independent of input values and permit mutation.
        source = torch.randn(2, 3, 5, 7, requires_grad=True)
        intermediate = source * 2
        reference = weakref.ref(intermediate)
        output = do_pad(intermediate, (1, 1, 1, 1), "reflect")
        with torch.no_grad():
            intermediate.add_(3)
        del intermediate
        self.assertIsNone(reference())
        output.sum().backward()
        self.assertIsNotNone(source.grad)

    def test_no_grad_and_disabled_compression_preserve_native_validation(self):
        x = torch.randn(2, 3, 5, 7, requires_grad=True)
        with torch.no_grad():
            self.assertIsNone(do_pad(x, (1, 1, 1, 1), "reflect").grad_fn)
        with self.assertRaises(RuntimeError):
            do_pad(x, (7, 7, 1, 1), "reflect")
        with self.assertRaises(RuntimeError):
            do_pad(x, (1, 1, 1, 1), "reflect", value=2)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=False, config={}):
            self.assertNotEqual(type(F.pad(x, (1, 1, 1, 1), "reflect").grad_fn).__name__, "_PaddingFunctionBackward")


@unittest.skipUnless(torch.cuda.is_available(), "Strict zero-padding CUDA regression requires a GPU")
class MetadataPaddingStrictZeroCudaTests(unittest.TestCase):
    def test_actual_original224_zero_padding_backward_obeys_strict_policy(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 support required")
        with _strict_algorithms():
            for dtype in (torch.float32, torch.bfloat16):
                for channels_last in (False, True):
                    _zero_eager_vjp(self, device="cuda", dtype=dtype, channels_last=channels_last)
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
