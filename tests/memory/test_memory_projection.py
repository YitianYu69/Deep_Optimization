"""Persistent-memory forecasts stay separate from allocated tensor storage."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from deepopt import MultiOptimizer
from deepopt.memory import summarize_model_memory


class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters):
        super().__init__(parameters, defaults={})

    def step(self, closure=None):
        return None


class CustomAdamW(torch.optim.AdamW):
    pass


class MemoryProjectionTests(unittest.TestCase):
    def setUp(self):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.addCleanup(torch.set_default_dtype, previous_dtype)

    def test_builtin_forecasts_match_state_allocated_by_a_real_update(self):
        cases = (
            (torch.optim.Adam, {}, 72),
            (torch.optim.Adam, {"amsgrad": True}, 104),
            (torch.optim.AdamW, {}, 72),
            (torch.optim.AdamW, {"amsgrad": True}, 104),
            (torch.optim.SGD, {"momentum": 0.0}, 0),
            (torch.optim.SGD, {"momentum": 0.9}, 32),
        )
        for optimizer_type, options, expected in cases:
            with self.subTest(optimizer=optimizer_type.__name__, options=options):
                model = nn.Linear(3, 2)
                optimizer = optimizer_type(model.parameters(), lr=0.01, **options)
                before = summarize_model_memory(model, optimizer)
                self.assertEqual(before.optimizer_bytes, 0)
                self.assertEqual(before.estimated_gradient_bytes, 32)
                self.assertEqual(before.estimated_optimizer_bytes, expected)
                self.assertEqual(before.estimated_persistent_bytes, 64 + expected)

                model(torch.ones(2, 3)).sum().backward()
                optimizer.step()
                after = summarize_model_memory(model, optimizer)
                self.assertEqual(after.optimizer_bytes, expected)
                self.assertEqual(after.estimated_optimizer_bytes, after.optimizer_bytes)
                self.assertEqual(after.estimated_persistent_bytes, before.estimated_persistent_bytes)

    def test_projection_units_and_dictionary_fields_without_an_optimizer(self):
        report = summarize_model_memory(nn.Linear(3, 2))
        self.assertEqual(report.estimated_gradient_bytes, 32)
        self.assertEqual(report.estimated_optimizer_bytes, 0)
        self.assertEqual(report.estimated_persistent_bytes, 64)
        for field in ("estimated_gradient", "estimated_optimizer", "estimated_persistent"):
            self.assertEqual(getattr(report, field + "_gb"), getattr(report, field + "_bytes") / 1_000_000_000)
            self.assertEqual(report.as_dict()[field + "_bytes"], getattr(report, field + "_bytes"))
            self.assertEqual(report.as_dict()[field + "_gb"], getattr(report, field + "_gb"))

    def test_partial_initialization_adds_only_the_remaining_lazy_state(self):
        model = nn.Module()
        model.first = nn.Parameter(torch.ones(3))
        model.second = nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW(model.parameters())
        model.first.sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report = summarize_model_memory(model, optimizer)
        self.assertEqual(report.optimizer_bytes, 28)
        self.assertEqual(report.estimated_optimizer_bytes, 48)
        self.assertEqual(len(optimizer.state), 1)
        (model.first.sum() + model.second.sum()).backward()
        optimizer.step()
        self.assertEqual(summarize_model_memory(model, optimizer).optimizer_bytes, 48)

    def test_frozen_parameters_keep_existing_state_without_forecasting_new_state(self):
        model = nn.Linear(3, 2)
        model.bias.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters())
        report = summarize_model_memory(model, optimizer)
        self.assertEqual(report.estimated_gradient_bytes, 24)
        self.assertEqual(report.estimated_optimizer_bytes, 52)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.weight.requires_grad_(False)
        frozen = summarize_model_memory(model, optimizer)
        self.assertEqual(frozen.estimated_gradient_bytes, 0)
        self.assertEqual(frozen.optimizer_bytes, 52)
        self.assertEqual(frozen.estimated_optimizer_bytes, 52)
        self.assertEqual(frozen.estimated_persistent_bytes, 84)

    def test_existing_gradients_make_frozen_optimizer_parameters_eligible(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones(3), requires_grad=False)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer = torch.optim.AdamW(model.parameters())
        before = summarize_model_memory(model, optimizer)
        self.assertEqual(before.estimated_gradient_bytes, 12)
        self.assertEqual(before.estimated_optimizer_bytes, 28)
        optimizer.step()
        self.assertEqual(summarize_model_memory(model, optimizer).optimizer_bytes, 28)

    def test_group_settings_and_actual_parameter_dtypes_determine_state_size(self):
        model = nn.Module()
        model.first = nn.Parameter(torch.ones(3, dtype=torch.float32))
        model.second = nn.Parameter(torch.ones(2, dtype=torch.float64))
        optimizer = torch.optim.AdamW([
            {"params": [model.first]},
            {"params": [model.second], "amsgrad": True},
        ])
        before = summarize_model_memory(model, optimizer)
        self.assertEqual(before.estimated_gradient_bytes, 28)
        self.assertEqual(before.estimated_optimizer_bytes, 80)
        (model.first.sum() + model.second.sum()).backward()
        optimizer.step()
        self.assertEqual(summarize_model_memory(model, optimizer).optimizer_bytes, 80)

    def test_adam_scalar_state_uses_the_default_dtype(self):
        torch.set_default_dtype(torch.float64)
        model = nn.Linear(3, 2, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters())
        report = summarize_model_memory(model, optimizer)
        self.assertEqual(report.estimated_optimizer_bytes, 80)
        model(torch.ones(2, 3, dtype=torch.float32)).sum().backward()
        optimizer.step()
        self.assertEqual(summarize_model_memory(model, optimizer).optimizer_bytes, 80)
        fused = torch.optim.AdamW(model.parameters(), fused=True)
        self.assertEqual(summarize_model_memory(model, fused).estimated_optimizer_bytes, 72)

    def test_tied_parameters_and_distinct_views_have_independent_state_rules(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones(8))
        model.tied = model.weight
        model.view = nn.Parameter(model.weight[:4])
        optimizer = torch.optim.AdamW(model.parameters())
        report = summarize_model_memory(model, optimizer)
        self.assertEqual(report.model_bytes, 32)
        self.assertEqual(report.estimated_gradient_bytes, 48)
        self.assertEqual(report.estimated_optimizer_bytes, 104)
        (model.weight.sum() + model.view.sum()).backward()
        optimizer.step()
        self.assertEqual(summarize_model_memory(model, optimizer).optimizer_bytes, 104)

    def test_initialized_aliased_state_is_not_counted_twice(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        for state in optimizer.state.values():
            state["exp_avg_sq"] = state["exp_avg"]
        report = summarize_model_memory(model, optimizer)
        self.assertEqual(report.optimizer_bytes, 40)
        self.assertEqual(report.estimated_optimizer_bytes, 40)

    def test_multioptimizer_forecasts_each_child_once(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
        optimizer = MultiOptimizer({
            "first": torch.optim.AdamW(model[0].parameters()),
            "second": torch.optim.SGD(model[1].parameters(), lr=0.01, momentum=0.9),
        })
        report = summarize_model_memory(model, optimizer)
        self.assertEqual(report.estimated_gradient_bytes, 44)
        self.assertEqual(report.estimated_optimizer_bytes, 84)
        self.assertEqual(report.estimated_persistent_bytes, 172)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        self.assertEqual(summarize_model_memory(model, optimizer).optimizer_bytes, 84)

    def test_custom_optimizers_and_subclasses_report_unknown_forecasts(self):
        for optimizer_type in (CustomOptimizer, CustomAdamW):
            with self.subTest(optimizer=optimizer_type.__name__):
                model = nn.Linear(3, 2)
                report = summarize_model_memory(model, optimizer_type(model.parameters()))
                self.assertEqual(report.optimizer_bytes, 0)
                self.assertEqual(report.estimated_gradient_bytes, 32)
                self.assertIsNone(report.estimated_optimizer_bytes)
                self.assertIsNone(report.estimated_optimizer_gb)
                self.assertIsNone(report.estimated_persistent_bytes)
                self.assertIsNone(report.as_dict()["estimated_persistent_gb"])

    def test_unmaterialized_models_have_unknown_projections(self):
        for model in (nn.LazyLinear(2), nn.Linear(3, 2, device="meta")):
            with self.subTest(model=type(model).__name__, device=str(next(model.parameters()).device)):
                report = summarize_model_memory(model, torch.optim.AdamW(model.parameters()))
                self.assertIsNone(report.estimated_gradient_bytes)
                self.assertIsNone(report.estimated_optimizer_bytes)
                self.assertIsNone(report.estimated_persistent_bytes)
        with FakeTensorMode():
            model = nn.Linear(3, 2)
            report = summarize_model_memory(model, torch.optim.AdamW(model.parameters()))
        self.assertEqual(report.model_bytes, 0)
        self.assertIsNone(report.estimated_gradient_bytes)
        self.assertIsNone(report.estimated_optimizer_bytes)
        self.assertIsNone(report.estimated_persistent_bytes)

    def test_partial_or_malformed_adam_state_is_not_treated_as_uninitialized(self):
        for malformed in (False, True):
            with self.subTest(malformed=malformed):
                model = nn.Linear(3, 2)
                optimizer = torch.optim.AdamW(model.parameters())
                optimizer.state[model.weight]["step"] = torch.tensor(0.0)
                if malformed:
                    optimizer.state[model.weight]["exp_avg"] = torch.zeros_like(model.weight)
                    optimizer.state[model.weight]["exp_avg_sq"] = "not a tensor"
                report = summarize_model_memory(model, optimizer)
                self.assertIsNone(report.estimated_optimizer_bytes)
                self.assertIsNone(report.estimated_persistent_bytes)

    def test_adam_moments_require_real_matching_storage_and_dtype(self):
        for invalid_state in ("meta", "fake", "wrong_dtype"):
            with self.subTest(state=invalid_state):
                model = nn.Linear(3, 2)
                optimizer = torch.optim.AdamW(model.parameters())
                model(torch.ones(2, 3)).sum().backward()
                optimizer.step()
                if invalid_state == "meta":
                    moment = torch.zeros(model.weight.shape, device="meta")
                elif invalid_state == "fake":
                    with FakeTensorMode():
                        moment = torch.zeros(model.weight.shape)
                else:
                    moment = torch.zeros(model.weight.shape, dtype=torch.float64)
                optimizer.state[model.weight]["exp_avg"] = moment
                report = summarize_model_memory(model, optimizer)
                self.assertEqual(report.model_bytes, 32)
                self.assertEqual(report.estimated_gradient_bytes, 32)
                self.assertIsNone(report.estimated_optimizer_bytes)
                self.assertIsNone(report.estimated_persistent_bytes)

    def test_tensor_capturable_setting_is_not_read_or_coerced(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.param_groups[0]["capturable"] = torch.tensor(True)
        with patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("no tensor truth read")), \
                patch.object(torch.Tensor, "item", side_effect=AssertionError("no scalar read")):
            report = summarize_model_memory(model, optimizer)
        self.assertIsNone(report.estimated_optimizer_bytes)
        self.assertIsNone(report.estimated_persistent_bytes)

    def test_sgd_none_momentum_buffers_are_forecast_as_missing_state(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for parameter in model.parameters():
            optimizer.state[parameter]["momentum_buffer"] = None
        before = summarize_model_memory(model, optimizer)
        self.assertEqual(before.optimizer_bytes, 0)
        self.assertEqual(before.estimated_optimizer_bytes, 32)
        self.assertTrue(all(state["momentum_buffer"] is None for state in optimizer.state.values()))
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        after = summarize_model_memory(model, optimizer)
        self.assertEqual(after.optimizer_bytes, before.estimated_optimizer_bytes)
        self.assertEqual(after.estimated_optimizer_bytes, after.optimizer_bytes)

    def test_shared_uninitialized_state_mappings_have_unknown_forecasts(self):
        for optimizer_type, options in ((torch.optim.AdamW, {}), (torch.optim.SGD, {"momentum": 0.9})):
            with self.subTest(optimizer=optimizer_type.__name__):
                model = nn.Linear(3, 2)
                optimizer = optimizer_type(model.parameters(), lr=0.01, **options)
                shared_state = {}
                optimizer.state[model.weight] = shared_state
                optimizer.state[model.bias] = shared_state
                report = summarize_model_memory(model, optimizer)
                self.assertEqual(report.optimizer_bytes, 0)
                self.assertIsNone(report.estimated_optimizer_bytes)
                self.assertIsNone(report.estimated_persistent_bytes)
                self.assertEqual(shared_state, {})
                self.assertIs(optimizer.state[model.weight], optimizer.state[model.bias])

    def test_optimizer_parameters_outside_the_model_make_the_total_unknown(self):
        model = nn.Linear(3, 2)
        external = nn.Parameter(torch.ones(3))
        optimizer = torch.optim.AdamW([*model.parameters(), external])
        report = summarize_model_memory(model, optimizer)
        self.assertEqual(report.model_bytes, 32)
        self.assertEqual(report.optimizer_bytes, 0)
        self.assertEqual(report.estimated_optimizer_bytes, 100)
        self.assertIsNone(report.estimated_gradient_bytes)
        self.assertIsNone(report.estimated_persistent_bytes)

    def test_forecasting_does_not_execute_or_materialize_training(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        parameters = [parameter.detach().clone() for parameter in model.parameters()]
        rng = torch.get_rng_state().clone()
        with patch.object(model, "forward", side_effect=AssertionError("no forward")), \
                patch.object(optimizer, "step", side_effect=AssertionError("no update")), \
                patch.object(optimizer, "state_dict", side_effect=AssertionError("no serialization")), \
                patch("torch.empty", side_effect=AssertionError("no tensor allocation")), \
                patch("torch.zeros", side_effect=AssertionError("no tensor allocation")), \
                patch("torch.zeros_like", side_effect=AssertionError("no tensor allocation")):
            report = summarize_model_memory(model, optimizer)
        self.assertEqual(report.estimated_optimizer_bytes, 72)
        self.assertFalse(optimizer.state)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        torch.testing.assert_close(rng, torch.get_rng_state())
        for before, parameter in zip(parameters, model.parameters()):
            torch.testing.assert_close(before, parameter)


if __name__ == "__main__":
    unittest.main()
