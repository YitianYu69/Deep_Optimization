import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import TrainingRuntime
from deepopt.observers import CheckObservers, FiniteCheck, all_finite


class ObserverDeviceChecks:
    def test_real_complex_strided_empty_and_integral_inputs(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                      torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype):
                values = torch.ones(128, device=self.device, dtype=dtype)[::2]
                self.assertTrue(all_finite([values, values[:0]]))
                for bad in (float("nan"), float("inf"), float("-inf")):
                    values[-1] = bad
                    self.assertFalse(all_finite([values]))
                if values.is_complex():
                    values[-1] = complex(1, float("inf"))
                    self.assertFalse(all_finite([values]))
        self.assertTrue(all_finite([torch.ones(8, dtype=torch.bool, device=self.device),
                                    torch.arange(8, device=self.device)]))

    def test_dispatch_preserves_device_storage_and_original_graph(self):
        source = torch.ones(4, device=self.device, requires_grad=True)
        value = source.square()
        seen = []

        def record(event):
            observed, = event.tensors
            self.assertEqual(observed.device, value.device)
            self.assertEqual(observed.data_ptr(), value.data_ptr())
            self.assertFalse(observed.requires_grad)
            self.assertIsNone(observed.grad_fn)
            self.assertFalse(torch.is_grad_enabled())
            seen.append(event.name)

        checks = CheckObservers([FiniteCheck(kinds={"activation"}), record])
        with patch("torch.Tensor.cpu", side_effect=AssertionError("Unexpected CPU copy")):
            checks.observe("activation", value, name="hidden")
        self.assertEqual(seen, ["hidden"])
        value.sum().backward()
        torch.testing.assert_close(source.grad, torch.full_like(source, 2))

    def test_sparse_duplicate_overflow_is_checked_after_coalescing(self):
        indices = torch.tensor([[0, 0]], device=self.device)
        values = torch.tensor([3e38, 3e38], device=self.device)
        sparse = torch.sparse_coo_tensor(indices, values, (1,))
        with self.assertRaises(FloatingPointError):
            CheckObservers().observe("gradients", sparse)

    def test_runtime_blocks_nonfinite_gradients_on_training_device(self):
        model = nn.Linear(1, 1, bias=False).to(self.device)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1), device=self.device)
        before = model.weight.detach().clone()
        runtime.backward(model(torch.ones(1, 1, device=self.device)).sum())
        model.weight.grad.fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            runtime.step()
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            runtime.step()


class CPUObserverDeviceTests(ObserverDeviceChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware and a CUDA-enabled PyTorch build are required")
class CUDAObserverDeviceTests(ObserverDeviceChecks, unittest.TestCase):
    device = "cuda:0"

    def test_mixed_cpu_cuda_event_checks_both_devices(self):
        cpu, cuda = torch.ones(8), torch.ones(8, device=self.device)
        self.assertTrue(all_finite([cpu, cuda]))
        cuda[-1] = float("nan")
        self.assertFalse(all_finite([cpu, cuda]))
        cuda[-1] = 1
        cpu[-1] = float("inf")
        self.assertFalse(all_finite([cpu, cuda]))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two CUDA devices are required")
    def test_multiple_cuda_devices_are_reduced_separately(self):
        tensors = [torch.ones(8, device=f"cuda:{index}") for index in (0, 1)]
        self.assertTrue(all_finite(tensors))
        tensors[1][-1] = float("nan")
        self.assertFalse(all_finite(tensors))
        tensors[1][-1] = 1
        tensors[0][-1] = float("inf")
        self.assertFalse(all_finite(tensors))


if __name__ == "__main__":
    unittest.main()
