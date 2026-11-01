"""Static batch-size choices and explicit loader resize contracts."""

import random
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils.data import BatchSampler, DataLoader, TensorDataset

from deepopt import Trainer
from deepopt.data.resumable import ResumableDataLoader
from deepopt.memory.planning import apply_batch_size_plan, loader_batch_size, plan_batch_size, set_loader_batch_size


def estimate(peak, *, initial=100, device="cpu"):
    return SimpleNamespace(as_dict=lambda: {
        "peak_bytes_by_device": {device: peak}, "initial_bytes_by_device": {device: initial},
        "persistent_bytes_by_device": {device: initial}, "assumptions": ["shape-only test estimate"],
    })


class BatchSizePlanningTests(unittest.TestCase):
    def test_manual_and_non_cuda_without_budget_leave_requested_size_untouched(self):
        trainer = SimpleNamespace(device="cpu")
        with patch("deepopt.memory.estimate_training_memory", side_effect=AssertionError("must not estimate")):
            manual = plan_batch_size(trainer, requested_batch_size=13, mode="manual")
            automatic = plan_batch_size(trainer, requested_batch_size=13)
        self.assertEqual(manual["status"], "manual")
        self.assertEqual(automatic["status"], "skipped")
        self.assertEqual(manual["new_batch_size"], 13)
        self.assertEqual(automatic["new_batch_size"], 13)
        self.assertEqual(automatic["candidates"], [])

    def test_candidate_factory_uses_fake_tensors_and_first_halved_fit_is_selected(self):
        requested = []

        def sample(size):
            requested.append(size)
            return {"tokens": torch.empty(7, size, 3), "targets": torch.zeros(size, dtype=torch.long)}

        def static_estimate(trainer, batch):
            self.assertIsInstance(batch["tokens"], FakeTensor)
            self.assertFalse(batch["tokens"].fake_mode.allow_fallback_kernels)
            size = batch["targets"].shape[0]
            return estimate(100 + size * 100)

        with patch("deepopt.memory.estimate_training_memory", side_effect=static_estimate):
            plan = plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=9,
                                          sample_batch_fn=sample, memory_budget_bytes=350)
        self.assertEqual(requested, [9, 4, 2])
        self.assertEqual(plan["new_batch_size"], 2)
        self.assertEqual(plan["status"], "planned")
        self.assertEqual([candidate["fits"] for candidate in plan["candidates"]], [False, False, True])
        self.assertEqual(plan["report"]["peak_bytes_by_device"]["cpu"], 300)

    def test_cuda_default_budget_credits_only_reported_live_initial_bytes(self):
        with patch("torch.cuda.mem_get_info", return_value=(900, 5000)) as available, \
                patch("torch.cuda.memory_reserved", side_effect=AssertionError("reserved memory is not free")), \
                patch("deepopt.memory.estimate_training_memory", return_value=estimate(790, device="cuda:0")):
            plan = plan_batch_size(SimpleNamespace(device="cuda:0"), requested_batch_size=7,
                                          sample_batch_fn=lambda size: torch.empty(size, 3))
        self.assertEqual(plan["memory_budget_bytes"], 800)
        self.assertEqual(plan["new_batch_size"], 7)
        available.assert_called_once_with(torch.device("cuda:0"))

    def test_explicit_budget_does_not_query_cuda_memory(self):
        with patch("torch.cuda.mem_get_info", side_effect=AssertionError("explicit budget")), \
                patch("deepopt.memory.estimate_training_memory", return_value=estimate(100, device="cuda:0")):
            plan = plan_batch_size(SimpleNamespace(device="cuda:0"), requested_batch_size=3,
                                          sample_batch_fn=lambda size: torch.empty(size), memory_budget_bytes=100)
        self.assertEqual(plan["memory_budget_bytes"], 100)

    def test_factory_and_estimator_failures_propagate_without_trying_smaller_sizes(self):
        for origin in ("factory", "estimator"):
            calls = []

            def sample(size):
                calls.append(size)
                if origin == "factory":
                    raise RuntimeError("Unsupported representative shape")
                return torch.empty(size)

            with self.subTest(origin=origin), \
                    patch("deepopt.memory.estimate_training_memory", side_effect=RuntimeError("Unsupported fake operation")), \
                    self.assertRaisesRegex(RuntimeError, "Unsupported"):
                plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=8,
                                       sample_batch_fn=sample, memory_budget_bytes=1000)
            self.assertEqual(calls, [8])

    def test_factory_random_state_is_restored_on_success_and_error(self):
        for fails in (False, True):
            python_before, torch_before = random.getstate(), torch.get_rng_state()
            numpy = sys.modules.get("numpy")
            numpy_before = numpy.random.get_state() if numpy is not None else None

            def sample(size):
                random.random()
                torch.manual_seed(919)
                if numpy is not None:
                    numpy.random.random()
                if fails:
                    raise RuntimeError("factory failed")
                return torch.empty(size)

            with self.subTest(fails=fails), patch("deepopt.memory.estimate_training_memory", return_value=estimate(100)):
                if fails:
                    with self.assertRaisesRegex(RuntimeError, "factory failed"):
                        plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=2,
                                               sample_batch_fn=sample, memory_budget_bytes=100)
                else:
                    plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=2,
                                           sample_batch_fn=sample, memory_budget_bytes=100)
            self.assertEqual(random.getstate(), python_before)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
            if numpy is not None:
                after = numpy.random.get_state()
                self.assertEqual(after[0], numpy_before[0])
                self.assertEqual(after[1].tolist(), numpy_before[1].tolist())
                self.assertEqual(after[2:], numpy_before[2:])

    def test_planner_inside_an_existing_fake_mode_restores_real_rng_state(self):
        before = torch.get_rng_state()
        with FakeTensorMode(allow_fallback_kernels=False), \
                patch("deepopt.memory.estimate_training_memory", return_value=estimate(100)):
            plan = plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=2,
                                          sample_batch_fn=lambda size: torch.empty(size), memory_budget_bytes=100)
        self.assertEqual(plan["new_batch_size"], 2)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_real_tensor_propagation_is_rejected_before_calling_batch_factory(self):
        from torch._functorch import config

        factory = Mock(side_effect=AssertionError("must not create a batch"))
        with config.patch(fake_tensor_propagate_real_tensors=True), \
                patch("deepopt.memory.estimate_training_memory") as estimator, \
                self.assertRaisesRegex(ValueError, "real tensor propagation disabled"):
            plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=4,
                            sample_batch_fn=factory, memory_budget_bytes=1000)
        factory.assert_not_called()
        estimator.assert_not_called()

    def test_no_candidate_fits_rejects_before_training(self):
        sizes = []
        with patch("deepopt.memory.estimate_training_memory", return_value=estimate(501)), \
                self.assertRaisesRegex(ValueError, "batch size 1 needs 501 bytes, budget is 500"):
            plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=7,
                                   sample_batch_fn=lambda size: sizes.append(size) or torch.empty(size),
                                   memory_budget_bytes=500)
        self.assertEqual(sizes, [7, 3, 1])

    def test_cuda_rng_state_is_restored_and_oom_never_triggers_smaller_candidates(self):
        cuda_states = [torch.tensor([3, 7], dtype=torch.uint8)]
        sizes = []
        with patch("torch.cuda.is_initialized", return_value=True), \
                patch("torch.cuda.get_rng_state_all", return_value=cuda_states), \
                patch("torch.cuda.set_rng_state_all") as restore, \
                patch("deepopt.memory.estimate_training_memory", side_effect=torch.OutOfMemoryError("user error")), \
                self.assertRaises(torch.OutOfMemoryError):
            plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=8,
                                   sample_batch_fn=lambda size: sizes.append(size) or torch.empty(size),
                                   memory_budget_bytes=1000)
        self.assertEqual(sizes, [8])
        restore.assert_called_once()
        self.assertIs(restore.call_args.args[0], cuda_states)

    def test_missing_factory_and_multi_rank_are_actionable_errors(self):
        with self.assertRaisesRegex(ValueError, r"sample_batch_fn\(size\)"):
            plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=2, memory_budget_bytes=100)
        with patch("torch.distributed.is_available", return_value=True), \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=2), \
                self.assertRaisesRegex(ValueError, "multi-rank"):
            plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=2,
                                   memory_budget_bytes=100, sample_batch_fn=lambda size: torch.empty(size))

    def test_invalid_options_and_incomplete_reports_are_rejected(self):
        for options in ({"requested_batch_size": True}, {"requested_batch_size": 0},
                        {"requested_batch_size": 1.5}, {"mode": "other"},
                        {"memory_budget_bytes": True}, {"memory_budget_bytes": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan_batch_size(SimpleNamespace(device="cpu"), **({"requested_batch_size": 2} | options))
        for field in ("peak_bytes_by_device", "initial_bytes_by_device"):
            report = estimate(100).as_dict()
            report[field] = {}
            with self.subTest(field=field), \
                    patch("deepopt.memory.estimate_training_memory", return_value=SimpleNamespace(as_dict=lambda: report)), \
                    self.assertRaisesRegex(ValueError, "did not account"):
                plan_batch_size(SimpleNamespace(device="cpu"), requested_batch_size=2,
                                       sample_batch_fn=lambda size: torch.empty(size), memory_budget_bytes=100)

    def test_real_cpu_shape_estimate_does_not_update_model_or_optimizer(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer, loss_fn=nn.functional.mse_loss)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        plan = plan_batch_size(trainer, requested_batch_size=4, memory_budget_bytes=10_000_000,
                                      sample_batch_fn=lambda size: (torch.empty(size, 3), torch.empty(size, 2)))
        self.assertEqual(plan["new_batch_size"], 4)
        self.assertGreater(plan["report"]["peak_bytes_by_device"]["cpu"], 0)
        self.assertEqual(optimizer.state, {})
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))


class LoaderBatchSizeHelpersTests(unittest.TestCase):
    def test_optional_or_unknown_loaders_have_no_assumed_batch_size(self):
        self.assertIsNone(loader_batch_size(None))
        self.assertIsNone(loader_batch_size(SimpleNamespace()))
        self.assertIsNone(loader_batch_size(DataLoader(TensorDataset(torch.arange(4)), batch_size=None)))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            loader_batch_size(SimpleNamespace(batch_size=False))

    def test_standard_loader_resizes_exact_batch_sampler_before_iteration(self):
        loader = DataLoader(TensorDataset(torch.arange(11)), batch_size=4)
        self.assertIs(type(loader.batch_sampler), BatchSampler)
        set_loader_batch_size(loader, 3)
        self.assertEqual(loader_batch_size(loader), 3)
        self.assertEqual(loader.batch_size, 3)
        self.assertEqual([len(batch[0]) for batch in loader], [3, 3, 3, 2])

    def test_actual_sampler_size_takes_precedence_and_step_changes_need_capability(self):
        loader = DataLoader(TensorDataset(torch.arange(11)), batch_size=4)
        loader.batch_sampler.batch_size = 3
        self.assertEqual(loader_batch_size(loader), 3)
        with self.assertRaisesRegex(ValueError, "supports_batch_size_change"):
            set_loader_batch_size(loader, 2, require_resume_support=True)
        self.assertEqual(loader_batch_size(loader), 3)

    def test_custom_loader_requires_explicit_setter_and_checks_its_result(self):
        class CustomLoader(DataLoader):
            pass

        custom = CustomLoader(TensorDataset(torch.arange(8)), batch_size=4)
        with self.assertRaisesRegex(ValueError, "implement set_batch_size"):
            set_loader_batch_size(custom, 2)
        custom.set_batch_size = lambda size: None
        with self.assertRaisesRegex(ValueError, "did not apply"):
            set_loader_batch_size(custom, 2)

    def test_resumable_loader_uses_its_cursor_preserving_setter(self):
        loader = ResumableDataLoader(TensorDataset(torch.arange(11)), batch_size=4,
                                     generator=torch.Generator().manual_seed(7))
        iterator = iter(loader)
        next(iterator)
        order = loader.batch_sampler.order.clone()
        set_loader_batch_size(loader, 3, require_resume_support=True)
        self.assertEqual(loader.batch_sampler.position, 4)
        self.assertEqual(next(iterator)[0].tolist(), order[4:7].tolist())
        self.assertIsNone(loader.batch_size)


class ApplyBatchSizePlanTests(unittest.TestCase):
    @staticmethod
    def loader(size=4):
        return DataLoader(TensorDataset(torch.arange(11)), batch_size=size)

    @staticmethod
    def plan(requested=4, selected=2):
        return {"requested_batch_size": requested, "new_batch_size": selected, "mode": "auto",
                "status": "planned", "reason": "Selected an estimated fit", "memory_budget_bytes": 200,
                "report": estimate(150).as_dict(), "candidates": []}

    def test_fresh_plan_resizes_matching_loaders_and_records_and_logs_estimate(self):
        trainer, train, validation = SimpleNamespace(scheduler=None), self.loader(), self.loader()
        plan = self.plan()
        with self.assertLogs("deepopt.engine.batch_size", level="INFO") as log:
            apply_batch_size_plan(trainer, train, validation, plan)
        self.assertEqual((loader_batch_size(train), loader_batch_size(validation)), (2, 2))
        self.assertEqual(plan["validation_batch_size"], 2)
        self.assertIs(trainer.batch_size_adjustment, plan)
        self.assertTrue(any("estimated peak tensor bytes={'cpu': 150}" in line for line in log.output))
        self.assertTrue(any("requested 4 to 2" in line for line in log.output))

    def test_independent_validation_batching_and_absent_validation_are_preserved(self):
        for validation in (self.loader(3), None):
            with self.subTest(validation=validation):
                trainer, train, plan = SimpleNamespace(scheduler=None), self.loader(), self.plan()
                with self.assertLogs("deepopt.engine.batch_size"):
                    apply_batch_size_plan(trainer, train, validation, plan)
                self.assertEqual(loader_batch_size(train), 2)
                self.assertEqual(plan["validation_batch_size"], 3 if validation is not None else None)

    def test_one_cycle_rejects_changed_size_before_mutating_loaders_or_metadata(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=4)
        trainer, train, validation, plan = SimpleNamespace(scheduler=scheduler), self.loader(), self.loader(), self.plan()
        with self.assertRaisesRegex(ValueError, "OneCycleLR.*manual"):
            apply_batch_size_plan(trainer, train, validation, plan)
        self.assertEqual((loader_batch_size(train), loader_batch_size(validation)), (4, 4))
        self.assertNotIn("validation_batch_size", plan)
        self.assertFalse(hasattr(trainer, "batch_size_adjustment"))
        unchanged = plan_batch_size(trainer, requested_batch_size=4, mode="manual")
        apply_batch_size_plan(trainer, train, validation, unchanged)
        self.assertIs(trainer.batch_size_adjustment, unchanged)

    def test_step_resize_keeps_consumed_cursor_and_rejects_ordinary_loaders(self):
        trainer = SimpleNamespace(scheduler=None)
        with self.assertRaisesRegex(ValueError, "supports_batch_size_change"):
            apply_batch_size_plan(trainer, self.loader(), None, self.plan(), step_resume=True)
        loader = ResumableDataLoader(TensorDataset(torch.arange(11)), batch_size=4,
                                     generator=torch.Generator().manual_seed(7))
        iterator = iter(loader)
        next(iterator)
        order = loader.batch_sampler.order.clone()
        plan = self.plan() | {"old_batch_size": 4}
        apply_batch_size_plan(trainer, loader, None, plan, step_resume=True)
        self.assertEqual(loader.batch_sampler.position, 4)
        self.assertEqual(next(iterator)[0].tolist(), order[4:6].tolist())

    def test_resume_plan_leaves_the_saved_to_selected_change_log_to_checkpoint(self):
        plan = self.plan() | {"old_batch_size": 8}
        with self.assertLogs("deepopt.engine.batch_size", level="INFO") as log:
            apply_batch_size_plan(SimpleNamespace(scheduler=None), self.loader(), None, plan)
        self.assertEqual(len(log.records), 1)
        self.assertEqual(log.records[0].levelname, "INFO")

    def test_cpu_skip_records_actual_validation_size_and_reason(self):
        trainer = SimpleNamespace(scheduler=None, device="cpu")
        plan = plan_batch_size(trainer, requested_batch_size=4)
        with self.assertLogs("deepopt.engine.batch_size", level="WARNING") as log:
            apply_batch_size_plan(trainer, self.loader(), self.loader(3), plan)
        self.assertEqual(plan["validation_batch_size"], 3)
        self.assertIn("cpu", log.output[0])
        self.assertIs(trainer.batch_size_adjustment, plan)

    def test_plan_for_a_different_loader_is_rejected_before_mutation(self):
        trainer, loader, plan = SimpleNamespace(scheduler=None), self.loader(3), self.plan()
        with self.assertRaisesRegex(ValueError, "differs from the training loader"):
            apply_batch_size_plan(trainer, loader, None, plan)
        self.assertEqual(loader_batch_size(loader), 3)
        self.assertFalse(hasattr(trainer, "batch_size_adjustment"))


if __name__ == "__main__":
    unittest.main()
