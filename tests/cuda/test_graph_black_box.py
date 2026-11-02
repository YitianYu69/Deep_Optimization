"""Public CUDA Graph behavior against eager PyTorch; real GPU execution only."""

from copy import deepcopy
import io
import unittest

import torch
from torch import nn

from deepopt import CUDAGraphCore, Trainer, TrainingRuntime, cuda_graph_core


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is required")
class CUDAGraphBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.core = CUDAGraphCore(self.device)

    def test_raw_context_has_native_graph_and_replays_new_buffer_values(self):
        x = torch.ones(4, device=self.device)
        self.core.warmup(lambda: x.sin() * 2)
        with self.core.capture() as handle:
            output = x.sin() * 2
        self.assertIsInstance(handle.graph, torch.cuda.CUDAGraph)
        for value in (1., 2., -3.):
            x.fill_(value)
            handle.replay()
            torch.testing.assert_close(output, x.sin() * 2)
        handle.close()
        with self.assertRaises(RuntimeError):
            handle.replay()

    def test_decorator_warms_once_reuses_outputs_and_validates_new_inputs(self):
        calls = []
        @self.core.capture(warmup=2)
        def transform(x):
            calls.append(1)
            return x.square() + 3
        x = torch.randn(4, device=self.device)
        first = transform(x)
        saved = first.clone()
        y = torch.randn_like(x)
        second = transform(y)
        self.assertIs(first, second)
        torch.testing.assert_close(saved, x.square() + 3)
        torch.testing.assert_close(second, y.square() + 3)
        self.assertEqual(len(calls), 3)  # Two warmups and one capture; replay skips Python.
        for bad in (torch.ones(3, device=self.device), y.double(), y.cpu(), y.clone().requires_grad_()):
            with self.subTest(spec=(bad.shape, bad.dtype, bad.device)), self.assertRaises(ValueError):
                transform(bad)
        with self.assertRaises(TypeError):
            transform(x=y)
        transform.close()

    def test_bare_decorator_tuple_outputs_and_zero_argument_closure(self):
        x = torch.ones(3, device=self.device)
        @cuda_graph_core.capture
        def transform():
            return x + 1, x * 2
        actual = transform()
        torch.testing.assert_close(actual, (x + 1, x * 2))
        x.fill_(3)
        torch.testing.assert_close(transform(), (x + 1, x * 2))
        transform.close()

    def test_high_level_function_inference_and_autograd(self):
        def function(x, scale):
            return x.sin() * scale
        x = torch.randn(4, device=self.device)
        scale = torch.tensor(2., device=self.device)
        inference = self.core.capture_model(function, (x, scale), autograd=False)
        self.assertIsInstance(inference.graph, torch.cuda.CUDAGraph)
        torch.testing.assert_close(inference(x, scale), function(x, scale))
        inference.close()
        x.requires_grad_()
        scale.requires_grad_()
        captured = self.core.capture_callable(function, (x, scale))
        for _ in range(3):
            actual = captured(x, scale)
            expected = function(x, scale)
            torch.testing.assert_close(actual, expected)
            actual_grads = torch.autograd.grad(actual.sum(), (x, scale))
            expected_grads = torch.autograd.grad(expected.sum(), (x, scale))
            for a, b in zip(actual_grads, expected_grads):
                torch.testing.assert_close(a, b)
        captured.close()

    def test_model_runtime_training_accumulation_and_eager_validation(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)).to(self.device)
        reference = deepcopy(model)
        x = torch.randn(3, 4, device=self.device)
        capture = self.core.capture_model(model, (x,))
        optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.5)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.02, momentum=.5)
        runtime = TrainingRuntime(model, optimizer, device=self.device, execution_mode="max_speed")
        for update in range(3):
            for _ in range(2):
                value = torch.randn_like(x)
                actual = runtime.compute(value, model)
                expected = reference(value)
                torch.testing.assert_close(actual, expected)
                runtime.backward(actual.square().mean())
                expected.square().mean().backward()
            runtime.step(gradient_divisor=2)
            for parameter in reference.parameters():
                parameter.grad.div_(2)
            ref_optimizer.step()
            ref_optimizer.zero_grad(set_to_none=True)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b)
            runtime.set_execution_mode("analyze" if update == 0 else "max_speed")
        model.eval()
        reference.eval()
        with torch.no_grad():
            value = torch.randn(1, 4, device=self.device)
            torch.testing.assert_close(model(value), reference(value))
        capture.close()

    def test_trainer_uses_captured_model_and_checkpoint_state_remains_portable(self):
        model = nn.Linear(4, 2).to(self.device)
        reference = deepcopy(model)
        x, y = torch.randn(3, 4, device=self.device), torch.randn(3, 2, device=self.device)
        capture = self.core.capture_model(model, (x,))
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), device=self.device,
                          execution_mode="max_speed", loss_fn=nn.functional.mse_loss, accumulation_steps=2)
        eager = Trainer(reference, torch.optim.SGD(reference.parameters(), lr=.01), device=self.device,
                        execution_mode="max_speed", loss_fn=nn.functional.mse_loss, accumulation_steps=2)
        trainer.train_epoch([(x, y)] * 3)
        eager.train_epoch([(x, y)] * 3)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = nn.Linear(4, 2).to(self.device)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        capture.close()
        recaptured = self.core.capture_model(restored, (x,))
        torch.testing.assert_close(restored(x), reference(x))
        recaptured.close()

    def test_capture_preserves_batchnorm_buffers_rng_and_example_input(self):
        model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 2)).to(self.device)
        x = torch.randn(3, 4, device=self.device)
        saved_x = x.clone()
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        capture = self.core.capture_model(model, (x,))
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name])
        torch.testing.assert_close(x, saved_x)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(self.device)))
        model(x).sum().backward()
        self.assertEqual(model[0].num_batches_tracked.item(), 1)
        capture.close()

    def test_frozen_model_inference_returns_native_graph(self):
        model = nn.Linear(4, 2).to(self.device).eval().requires_grad_(False)
        reference = deepcopy(model)
        x = torch.randn(3, 4, device=self.device)
        capture = self.core.capture_model(model, (x,), autograd=False)
        self.assertIsInstance(capture.graph, torch.cuda.CUDAGraph)
        torch.testing.assert_close(capture.replay(x), reference(x))
        self.assertFalse(capture(x).requires_grad)
        capture.close()

    def test_raw_context_can_capture_caller_owned_backward_and_sgd(self):
        model = nn.Linear(4, 2).to(self.device)
        reference = deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, foreach=False)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01, foreach=False)
        x = torch.randn(3, 4, device=self.device)
        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            optimizer.step()
        self.core.warmup(update, model, optimizer, iterations=3)
        for _ in range(3):
            update(reference, reference_optimizer)
        optimizer.zero_grad(set_to_none=True)
        with self.core.capture() as capture:
            model(x).square().mean().backward()
            optimizer.step()
        for _ in range(3):
            x.normal_()
            capture.replay()
            update(reference, reference_optimizer)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b)
        capture.close()

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 is required")
    def test_bf16_model_capture_preserves_autograd(self):
        model = nn.Linear(4, 2).to(self.device)
        reference = deepcopy(model)
        x = torch.randn(3, 4, device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            capture = self.core.capture_model(model, (x,))
            actual = model(x)
            expected = reference(x)
        torch.testing.assert_close(actual, expected)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        capture.close()


if __name__ == "__main__":
    unittest.main()
