"""Inspect core boundaries without pretending mocked collectives test DDP math.

Real multi-rank numerical behavior and artifact recovery live in the black-box
suite. These tests isolate delegation, device ownership and forbidden side effects.
"""

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from deepopt import DDPCore, Trainer, TrainingCore, TrainingRuntime
from deepopt.observers import CheckObservers


class TrainingCoreWhiteBoxTests(unittest.TestCase):
    def test_core_ownership_and_public_namespace(self):
        import deepopt.distributed as package
        from deepopt.engine.core import TrainingCore as EngineCore
        runtime = TrainingRuntime(nn.Linear(1, 1))
        self.assertIs(type(runtime.core), EngineCore)
        self.assertIs(TrainingCore, EngineCore)
        self.assertIs(DDPCore, package.DDPCore)
        self.assertEqual(package.__all__, ["DDPCore"])
        with self.assertRaisesRegex(ValueError, "multi-rank"):
            DDPCore(runtime.model, "cpu")

    def test_regular_core_preserves_higher_order_graph_without_collectives(self):
        model = nn.Linear(1, 1, bias=False)
        nn.init.ones_(model.weight)
        core = TrainingCore(model, torch.device("cpu"))
        with ExitStack() as stack:
            for operation in ("all_reduce", "broadcast", "all_gather_object", "init_process_group"):
                stack.enter_context(patch(f"torch.distributed.{operation}", side_effect=AssertionError("Local core used a collective")))
            loss = core.compute(torch.tensor([[3.]]), lambda batch: model(batch).square().sum())
            core.observe(CheckObservers(), "loss", loss)
            gradient, = torch.autograd.grad(loss, model.weight, create_graph=True)
            core.backward(gradient.sum())
            core.sync_buffers()
            self.assertEqual(core.gradient_divisor(7), 7)
        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 18))

    def test_runtime_delegates_execution_and_owns_optimizer_cleanup(self):
        model = nn.Linear(1, 1, bias=False)
        nn.init.ones_(model.weight)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.1))
        core = runtime.core
        with patch.object(core, "compute", wraps=core.compute) as forward, \
                patch.object(core, "backward", wraps=core.backward) as backward, \
                patch.object(core, "gradient_divisor", wraps=core.gradient_divisor) as divisor:
            loss = runtime.compute(torch.ones(2, 1), lambda batch: model(batch).sum())
            runtime.backward(loss)
            runtime.step(gradient_divisor=2)
            forward.assert_called_once()
            backward.assert_called_once()
            divisor.assert_called_once_with(2., normalize=True, reduction="legacy")
        self.assertIsNone(model.weight.grad)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, .9))

    def test_memory_estimation_does_not_clone_or_execute_a_distributed_wrapper(self):
        from deepopt.memory import estimate_training_memory

        class UncopyableReducer:
            def __deepcopy__(self, memo):
                raise AssertionError("A DDP reducer must not enter fake execution")

        model = nn.Linear(1, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1), loss_fn=nn.functional.mse_loss)
        sentinel = SimpleNamespace(distributed=True, wrapper=UncopyableReducer())
        trainer.runtime.core = sentinel
        with patch("torch.distributed.all_reduce", side_effect=AssertionError("Tracing invoked communication")):
            report = estimate_training_memory(trainer, (torch.ones(2, 1), torch.zeros(2, 1)))
        self.assertGreater(report.peak_bytes_by_device["cpu"], 0)
        self.assertIs(trainer.runtime.core, sentinel)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


class DDPCoreWhiteBoxTests(unittest.TestCase):
    def make_core(self):
        model = nn.Linear(1, 1, bias=False)
        model.register_buffer("counter", torch.tensor(0))
        wrapper = Mock(side_effect=lambda batch, step: step(batch))
        with patch("deepopt.distributed.ddp.distributed.world_size", return_value=2), \
                patch("deepopt.distributed.ddp.distributed.collective_errors", side_effect=lambda _: nullcontext()), \
                patch("deepopt.distributed.ddp.distributed.require_equal") as agreement, \
                patch("deepopt.distributed.ddp.DistributedDataParallel.__new__", return_value=wrapper) as constructor:
            core = DDPCore(model, "cpu")
            self.assertEqual([call.args[1] for call in agreement.call_args_list],
                             ["FP8 configuration", "DDP options"])
        return core, wrapper, constructor.call_args

    def test_training_enters_ddp_but_uneven_evaluation_bypasses_forward_collectives(self):
        core, wrapper, constructor = self.make_core()
        self.assertIs(constructor.args[1].model, core.model)
        batch = torch.ones(1, 1)
        with patch("torch.distributed.broadcast") as broadcast:
            core.compute(batch, lambda value: core.model(value).sum())
            wrapper.assert_called_once()
            broadcast.assert_not_called()  # Native DDP owns training broadcasts.
            self.assertTrue(constructor.kwargs["broadcast_buffers"])
            wrapper.reset_mock()
            broadcast.reset_mock()
            core.model.eval()
            with torch.no_grad():
                core.compute(batch, lambda value: core.model(value).sum())
            wrapper.assert_not_called()
            broadcast.assert_not_called()

    def test_gradient_divisor_corrects_for_native_ddp_averaging(self):
        core, _, _ = self.make_core()
        with patch("deepopt.distributed.ddp.distributed.reduce_number", return_value=4) as reduce, \
                patch("deepopt.distributed.ddp.distributed.world_size", return_value=3):
            self.assertEqual(core.gradient_divisor(2), 4)
            reduce.assert_called_once_with(2 / 3)

    def test_backward_rejects_bypassed_forward_and_higher_order_sync(self):
        core, _, _ = self.make_core()
        loss = core.model(torch.ones(1, 1)).sum()
        with patch("deepopt.distributed.ddp.distributed.collective_errors", side_effect=lambda _: nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "runtime.compute"):
                core.backward(loss)
            core.forward_pending = True
            with self.assertRaisesRegex(ValueError, "higher-order"):
                core.backward(loss, create_graph=True)
        self.assertIsNone(core.model.weight.grad)

    def test_invalid_ddp_options_do_not_construct_native_wrapper(self):
        with patch("deepopt.distributed.ddp.distributed.world_size", return_value=2), \
                patch("deepopt.distributed.ddp.distributed.collective_errors", side_effect=lambda _: nullcontext()), \
                patch("deepopt.distributed.ddp.DistributedDataParallel.__new__") as constructor:
            for options in ({"process_group": object()}, {"bucket_cap_mb": 0}, {"static_graph": 1}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    DDPCore(nn.Linear(1, 1), "cpu", options)
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
