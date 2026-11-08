"""Exercise autograd callbacks without the caller's Python context variables."""

from contextlib import contextmanager, nullcontext
from contextvars import Context
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import FP8Config, DelayedScalingConfig, DiagnosticPolicy, TrainingRuntime, fp8_checkpoint, fp8_scaling_scope
from deepopt.opt.do_nn.AOTAutograd_Func.linear_autograd_func import _DOLinearFunction
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.scaling_strategy.delayed import DelayedScalingState, current_scaling_batch, delayed_states


def settings():
    return FP8Config(enabled=True, scaling_recipe="delayed", intercept_linear=False,
                     delayed_scaling=DelayedScalingConfig(history_length=3))


@contextmanager
def isolated_autograd_context():
    original = _DOLinearFunction.backward

    def backward(ctx, gradient):
        # Keep the real autograd graph task and tensor operations, but remove
        # Python ContextVars as happens on a CUDA autograd worker thread.
        return Context().run(original, ctx, gradient)

    with patch.object(_DOLinearFunction, "backward", new=backward):
        yield


class DelayedAutogradContextTests(unittest.TestCase):
    def test_backward_on_another_python_thread_joins_the_explicit_boundary(self):
        model, _ = prepare_fp8_model(nn.Linear(5, 3), settings())
        with fp8_scaling_scope(), ThreadPoolExecutor(max_workers=1) as worker:
            value = model(torch.ones(7, 5, requires_grad=True))
            worker.submit(value.sum().backward).result(timeout=15)
            self.assertEqual(model.fp8_scaling.updates.tolist(), [0, 0, 0])
        self.assertEqual(model.fp8_scaling.updates.tolist(), [1, 1, 1])

    def test_runtime_worker_context_accumulates_into_the_forward_window(self):
        for mode in ("max_speed", "balanced", "analyze"):
            with self.subTest(mode=mode):
                source = nn.Sequential(nn.Linear(5, 7), nn.Sigmoid(), nn.Linear(7, 3))
                oracle_source = deepcopy(source)
                actual = TrainingRuntime(source, torch.optim.SGD(source.parameters(), lr=.001),
                                         fp8=settings(), execution_mode=mode)
                oracle = TrainingRuntime(oracle_source, torch.optim.SGD(oracle_source.parameters(), lr=.001),
                                         fp8=settings(), execution_mode=mode)
                for runtime in (actual, oracle):
                    for magnitude in (1., 3.):
                        loss = runtime.compute(torch.full((7, 5), magnitude),
                                               lambda x: runtime.model(x).square().mean())
                        with isolated_autograd_context() if runtime is actual else nullcontext():
                            runtime.backward(loss)
                        for state in delayed_states(runtime.model):
                            self.assertEqual(state.updates.tolist(), [0, 0, 0])
                            torch.testing.assert_close(state.scales, torch.ones(3), rtol=0, atol=0)
                    runtime.step()
                for value, expected in zip(actual.model.parameters(), oracle.model.parameters()):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    torch.testing.assert_close(value.grad, expected.grad, rtol=0, atol=0)
                for state, expected in zip(delayed_states(actual.model), delayed_states(oracle.model)):
                    self.assertEqual(state.updates.tolist(), [1, 1, 1])
                    torch.testing.assert_close(state.scales, expected.scales, rtol=0, atol=0)
                    torch.testing.assert_close(state.history, expected.history, rtol=0, atol=0)

    def test_explicit_boundary_and_checkpoint_recomputation_survive_context_loss(self):
        for checkpointed in (False, True):
            with self.subTest(checkpointed=checkpointed):
                model, _ = prepare_fp8_model(nn.Sequential(nn.Linear(5, 7), nn.Sigmoid(), nn.Linear(7, 3)), settings())
                x = torch.ones(7, 5, requires_grad=True)
                with fp8_scaling_scope():
                    value = fp8_checkpoint(model, x) if checkpointed else model(x)
                    with isolated_autograd_context():
                        value.sum().backward()
                    for state in delayed_states(model):
                        self.assertEqual(state.updates.tolist(), [0, 0, 0])
                for state in delayed_states(model):
                    self.assertEqual(state.updates.tolist(), [1, 1, 1])
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_standalone_backward_does_not_reopen_a_finished_forward_boundary(self):
        model, _ = prepare_fp8_model(nn.Linear(5, 3), settings())
        value = model(torch.ones(7, 5, requires_grad=True))
        state = model.fp8_scaling
        self.assertEqual(state.updates.tolist(), [1, 1, 0])
        with isolated_autograd_context():
            value.sum().backward()
        self.assertEqual(state.updates.tolist(), [1, 1, 1])
        model.state_dict()

    def test_worker_context_keeps_gradient_diagnostic_cadence(self):
        source = nn.Linear(5, 3)
        runtime = TrainingRuntime(source, torch.optim.SGD(source.parameters(), lr=.001), fp8=settings(),
                                  diagnostics=DiagnosticPolicy(loss_finite_every=3, gradient_finite_every=2))
        original, records = DelayedScalingState.quantize, []

        def quantize(state, x, role, **kwargs):
            if role == "grad":
                records.append((runtime.mode_controller.completed_updates + 1, kwargs["check_finite"]))
            return original(state, x, role, **kwargs)

        with patch.object(DelayedScalingState, "quantize", new=quantize), isolated_autograd_context():
            for _ in range(6):
                loss = runtime.compute(torch.ones(7, 5), lambda x: runtime.model(x).square().mean())
                runtime.backward(loss)
                runtime.step()
        self.assertEqual(records, [(update, update % 2 == 0) for update in range(1, 7)])

    def test_backward_failure_releases_owner_and_restores_worker_context(self):
        source = nn.Linear(5, 3)
        runtime = TrainingRuntime(source, torch.optim.SGD(source.parameters(), lr=.001), fp8=settings(), checks=False)
        loss = runtime.compute(torch.ones(7, 5), lambda x: runtime.model(x).sum())
        original, worker = _DOLinearFunction.backward, Context()
        with patch.object(_DOLinearFunction, "backward", new=lambda ctx, grad: worker.run(original, ctx, grad)), \
             patch("deepopt.opt.do_nn.AOTAutograd_Func.linear_autograd_func.gemm_nt", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                runtime.backward(loss)
        self.assertIsNone(worker.run(current_scaling_batch))
        self.assertIsNone(runtime._delayed_window)
        self.assertIsNone(runtime.model.fp8_scaling._pending_owner)
        self.assertEqual(runtime.model.fp8_scaling.updates.tolist(), [0, 0, 0])

    def test_different_live_backward_boundary_is_still_rejected(self):
        model, _ = prepare_fp8_model(nn.Linear(5, 3), settings())

        def conflicting_backward(value):
            with fp8_scaling_scope():
                value.sum().backward()

        with fp8_scaling_scope():
            value = model(torch.ones(7, 5))
            with self.assertRaisesRegex(RuntimeError, "Overlapping delayed-scaling"):
                Context().run(conflicting_backward, value)
            self.assertEqual(model.fp8_scaling.updates.tolist(), [0, 0, 0])
        self.assertEqual(model.fp8_scaling.updates.tolist(), [1, 1, 0])

    def test_independent_forward_boundaries_still_reject_shared_projection(self):
        model, _ = prepare_fp8_model(nn.Linear(5, 3), settings())
        with fp8_scaling_scope():
            model(torch.ones(7, 5))
            with self.assertRaisesRegex(RuntimeError, "Overlapping delayed-scaling"):
                Context().run(model, torch.ones(7, 5))
            self.assertEqual(model.fp8_scaling.updates.tolist(), [0, 0, 0])
        self.assertEqual(model.fp8_scaling.updates.tolist(), [1, 1, 0])
