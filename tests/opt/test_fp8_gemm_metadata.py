"""FP8 GEMM metadata probes must never enter a real CUDA executor."""

import sys
import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from deepopt.opt.backends import gemm_nt
from deepopt.opt.precision.gradient import GradientEdgePlan
from deepopt.opt.quantization.activation_tensor import FP8Activation
from deepopt.opt.quantization.gradient import GradientSlot
from deepopt.opt.quantization.tensor import QuantizedTensor


BLOCKED_EXECUTORS = {
    name: None
    for name in (
        "triton",
        "deepopt.opt.backends.triton.gemm.fp8",
        "deepopt.opt.backends.triton.quantization.fp8",
        "deepopt.opt.backends.triton.quantization.activation.fp8",
        "deepopt.opt.backends.triton.quantization.activation.affine",
    )
}


def operand(rows, cols, *, device="cuda:0", blocks=None, transposed=False):
    data = (
        torch.empty(cols, rows, device=device, dtype=torch.float8_e4m3fn).T
        if transposed
        else torch.empty(rows, cols, device=device, dtype=torch.float8_e4m3fn)
    )
    scale_shape = (
        (1, 1)
        if blocks is None
        else tuple((size + block - 1) // block for size, block in zip(data.shape, blocks))
    )
    return QuantizedTensor(data, torch.empty(scale_shape, device=device), blocks)


class GEMMMetadataTests(unittest.TestCase):
    def test_fake_cuda_gemm_needs_no_executor_device_query_or_data_pointer(self):
        with (
            FakeTensorMode(),
            patch.dict(sys.modules, BLOCKED_EXECUTORS),
            patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("GPU query")),
            patch.object(torch.Tensor, "data_ptr", side_effect=AssertionError("data pointer")),
        ):
            for backend in ("triton", "native_fast", "native_precise"):
                for rows, cols, reduction, blocks, transposed, strides in (
                    (34, 65, 128, None, False, (68, 72)),
                    (34, 64, 128, None, False, (64, 64)),
                    (7, 37, 65, (1, 32), False, (40, 40)),
                    (7, 64, 128, None, True, (64, 64)),
                ):
                    a = operand(rows, reduction, blocks=blocks, transposed=transposed)
                    b = operand(cols, reduction, blocks=blocks)
                    for dtype in (torch.float32, torch.bfloat16, torch.float16):
                        for bias in (None, torch.empty(cols, device="cuda:0")):
                            with self.subTest(backend=backend, shape=(rows, cols), dtype=dtype, bias=bias is not None):
                                actual = gemm_nt(a, b, backend=backend, out_dtype=dtype, bias=bias)
                                stride = strides[0 if dtype == torch.float32 else 1]
                                self.assertIsInstance(actual, FakeTensor)
                                self.assertEqual(actual.shape, (rows, cols))
                                self.assertEqual(actual.stride(), (stride, 1))
                                self.assertEqual(actual.dtype, dtype)
                                self.assertEqual(actual.device, a.data.device)

    def test_fake_cuda_gradient_output_keeps_its_logical_and_storage_contract(self):
        with FakeTensorMode(), patch.dict(sys.modules, BLOCKED_EXECUTORS):
            a, b = operand(34, 128), operand(65, 128)
            slot = GradientSlot(GradientEdgePlan(precision="fp8"), {}, device="cuda:0")
            with slot.output((2, 17, 65), torch.bfloat16) as output:
                actual = gemm_nt(a, b, backend="triton", gradient_output=output)
                self.assertIsInstance(actual, FP8Activation)
                self.assertIsInstance(actual._data, FakeTensor)
                self.assertEqual(actual.shape, (2, 17, 65))
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertEqual(actual._data.dtype, torch.float8_e5m2)
                self.assertEqual(actual.device, a.data.device)
                for backend in ("native_fast", "native_precise"):
                    with self.assertRaisesRegex(ValueError, "cannot emit scaled FP8 gradients"):
                        gemm_nt(a, b, backend=backend, gradient_output=output)

    def test_fake_cuda_gemm_preserves_structural_validation(self):
        with FakeTensorMode(), patch.dict(sys.modules, BLOCKED_EXECUTORS):
            a, b = operand(7, 32), operand(17, 32)
            for backend in ("triton", "native_fast", "native_precise"):
                for kwargs, message in (
                    ({"out_dtype": torch.float64}, "Output dtype"),
                    ({"bias": torch.empty(18, device="cuda:0")}, "Bias"),
                    ({"bias": torch.empty(17, device="cuda:0", dtype=torch.int32)}, "Bias"),
                    ({"bias": torch.empty(17)}, "Bias"),
                    ({"autotune": 1}, "autotune"),
                    ({"num_stages": 0}, "num_stages"),
                ):
                    with self.subTest(backend=backend, kwargs=kwargs), self.assertRaisesRegex((TypeError, ValueError), message):
                        gemm_nt(a, b, backend=backend, **kwargs)
                with self.assertRaisesRegex(ValueError, "reduction"):
                    gemm_nt(a, operand(17, 48), backend=backend)
                with self.assertRaisesRegex(ValueError, "same device"):
                    gemm_nt(a, operand(17, 32, device="cuda:1"), backend=backend)
                with self.assertRaisesRegex(ValueError, "CUDA"):
                    gemm_nt(operand(7, 32, device="cpu"), operand(17, 32, device="cpu"), backend=backend)
            with self.assertRaisesRegex(ValueError, "backend"):
                gemm_nt(a, b, backend="unknown")


if __name__ == "__main__":
    unittest.main()
