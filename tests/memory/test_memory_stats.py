import io
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from deepopt import MultiOptimizer, Trainer
from deepopt.config import get_config, load_config, reset_config
from deepopt.logger import logging_session
from deepopt.memory import summarize_model_memory


class MemoryStatsTests(unittest.TestCase):
    def test_counts_dtype_buffers_and_decimal_gb(self):
        model = nn.Linear(3, 2, dtype=torch.float64)
        model.bias.requires_grad_(False)
        model.register_buffer("counter", torch.zeros(3, dtype=torch.int64), persistent=False)
        report = summarize_model_memory(model)
        self.assertEqual((report.total_parameters, report.trainable_parameters, report.frozen_parameters), (8, 6, 2))
        self.assertEqual((report.total_percent, report.trainable_percent, report.frozen_percent), (100.0, 75.0, 25.0))
        self.assertEqual(report.as_dict()["trainable_percent"], 75.0)
        self.assertEqual(report.parameter_bytes, 64)
        self.assertEqual(report.buffer_bytes, 24)
        self.assertEqual(report.model_bytes, 88)
        self.assertEqual(report.model_gb, 88 / 1_000_000_000)
        self.assertEqual(report.optimizer_bytes, 0)
        self.assertEqual(report.as_dict()["model_gb"], report.model_gb)

    def test_tied_parameters_and_views_deduplicate_storage(self):
        model = nn.Module()
        model.a = nn.Parameter(torch.ones(8))
        model.b = model.a
        model.c = nn.Parameter(model.a[:4])
        model.register_buffer("alias", model.a.detach())
        report = summarize_model_memory(model)
        self.assertEqual(report.total_parameters, 12)
        self.assertEqual(report.parameter_bytes, 32)
        self.assertEqual(report.buffer_bytes, 32)
        self.assertEqual(report.model_bytes, 32)

    def test_adam_is_read_only_and_reports_lazy_then_materialized_state(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        rng = torch.get_rng_state().clone()
        with patch.object(optimizer, "step", side_effect=AssertionError("no updates")), \
             patch.object(optimizer, "state_dict", side_effect=AssertionError("no serialization")), \
             patch.object(model, "forward", side_effect=AssertionError("no forward")):
            before = summarize_model_memory(model, optimizer)
        self.assertEqual(before.optimizer_bytes, 0)
        self.assertFalse(optimizer.state)
        torch.testing.assert_close(rng, torch.get_rng_state())
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        after = summarize_model_memory(model, optimizer)
        # Two FP32 moment tensors for each parameter, plus two FP32 step counters.
        self.assertEqual(after.optimizer_bytes, 2 * 8 * 4 + 2 * 4)
        self.assertEqual(after.model_bytes, 8 * 4)  # Existing gradients are excluded.
        self.assertEqual(after.optimizer_gb, after.optimizer_bytes / 1_000_000_000)
        restored = torch.optim.AdamW(model.parameters())
        restored.load_state_dict(optimizer.state_dict())
        self.assertEqual(summarize_model_memory(model, restored).optimizer_bytes, after.optimizer_bytes)

    def test_multioptimizer_nested_state_aliases_and_tensor_group_options(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        a = torch.optim.SGD(model[0].parameters(), lr=0.1)
        b = torch.optim.SGD(model[1].parameters(), lr=0.1)
        optimizer = MultiOptimizer({"a": a, "b": b})
        shared = torch.zeros(5, dtype=torch.float64)
        a.state[model[0].weight]["nested"] = [shared, {"view": shared[:2]}]
        b.state[model[1].weight]["same"] = shared
        a.state[model[0].weight]["parameter_alias"] = model[0].weight.detach()
        a.param_groups[0]["tensor_option"] = torch.ones(1)
        b.param_groups[0]["tensor_option"] = torch.ones(2)
        self.assertEqual(summarize_model_memory(model, optimizer).optimizer_bytes, 40 + 4 + 8)

    def test_sparse_buffers_include_indices_and_values(self):
        model = nn.Module()
        model.register_buffer("sparse", torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]]), torch.ones(2), size=(3, 3)))
        report = summarize_model_memory(model)
        self.assertEqual(report.total_parameters, 0)
        self.assertIsNone(report.total_percent)
        self.assertIsNone(report.trainable_percent)
        self.assertIsNone(report.frozen_percent)
        self.assertEqual(report.buffer_bytes, 4 * 8 + 2 * 4)

    def test_lazy_and_meta_are_not_fabricated_allocations(self):
        lazy = nn.LazyLinear(2)
        report = summarize_model_memory(lazy)
        self.assertIsNone(report.total_parameters)
        self.assertIsNone(report.trainable_percent)
        self.assertIsNone(report.model_bytes)
        lazy(torch.ones(1, 3))
        self.assertEqual(summarize_model_memory(lazy).total_parameters, 8)
        meta = summarize_model_memory(nn.Linear(3, 2, device="meta"))
        self.assertEqual(meta.total_parameters, 8)
        self.assertEqual(meta.model_bytes, 0)
        with FakeTensorMode():
            fake = summarize_model_memory(nn.Linear(3, 2))
        self.assertEqual(fake.model_bytes, 0)
        self.assertTrue(fake.notes)

    def test_unavailable_storage_reports_unknown(self):
        model = nn.Linear(3, 2)
        with patch.object(torch.Tensor, "untyped_storage", side_effect=NotImplementedError):
            report = summarize_model_memory(model)
        self.assertEqual(report.total_parameters, 8)
        self.assertIsNone(report.model_bytes)
        self.assertIsNone(report.model_gb)

    def test_trainer_initialization_logs_to_active_session(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            with logging_session({"name": "memory-init"}, directory) as session:
                model = nn.Linear(3, 2)
                Trainer(model, torch.optim.AdamW(model.parameters()), loss_fn=F.mse_loss)
            text = Path(session.path).read_text()
            self.assertIn("Model parameters: total=8 (100.00%) trainable=8 (100.00%) frozen=0 (0.00%)", text)
            self.assertIn("Model tensor storage (parameters + buffers): 0.000000032 GB", text)
            self.assertIn("Optimizer tensor storage (currently allocated): 0.000000000 GB", text)
            self.assertIn("Estimated dense gradient storage: 0.000000032 GB", text)
            self.assertIn("Estimated optimizer storage after state initialization: 0.000000072 GB", text)
            self.assertIn("Estimated persistent training storage (model + gradients + optimizer): 0.000000136 GB", text)
            self.assertIn("lazy state may grow", text)
            self.assertNotIn("Training started", text)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_accounting_does_not_allocate_payload(self):
        model = nn.Linear(3, 2, device="cuda")
        before = torch.cuda.memory_allocated()
        report = summarize_model_memory(model)
        self.assertEqual(report.model_bytes, 32)
        self.assertEqual(before, torch.cuda.memory_allocated())


class EpochMemoryLoggingTests(unittest.TestCase):
    def setUp(self):
        previous = get_config()
        reset_config()
        self.addCleanup(lambda: load_config(previous) if previous is not None else reset_config())
        self.loader = [(torch.ones(2, 3), torch.zeros(2, 2))]

    def trainer(self, config=None):
        model = nn.Linear(3, 2)
        return Trainer(model, torch.optim.AdamW(model.parameters()), loss_fn=F.mse_loss, config=config)

    def test_enabled_reports_fresh_storage_after_each_epoch_to_file(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(stderr):
            with logging_session(directory=directory, level="WARNING", file_level="INFO") as session:
                trainer = self.trainer({"logging": {"memory_each_epoch": True}})
                trainer.fit(self.loader, self.loader, epochs=2, progress_bar=False)
            text = session.path.read_text()
        self.assertIn("Optimizer tensor storage (currently allocated): 0.000000000 GB", text)
        for epoch in (1, 2):
            self.assertEqual(text.count(f"Epoch {epoch} memory (cpu):"), 1)
            self.assertIn(f"Epoch {epoch} memory (cpu): model=0.000000032 GB optimizer=0.000000072 GB", text)
            self.assertIn(f"Epoch {epoch} estimated persistent training storage (model + gradients + optimizer): 0.000000136 GB", text)
            self.assertLess(text.index(f"Epoch {epoch}/2:"), text.index(f"Epoch {epoch} memory (cpu):"))
        self.assertNotIn("memory (cpu):", stderr.getvalue())
        self.assertNotIn("CUDA memory", text)

    def test_disabled_or_omitted_skips_epoch_accounting(self):
        for config in (None, {"logging": {"memory_each_epoch": False}}):
            with self.subTest(config=config), redirect_stderr(io.StringIO()):
                trainer = self.trainer(config)
                with patch("deepopt.engine.training.trainer.summarize_model_memory",
                           side_effect=AssertionError("disabled epoch accounting")), \
                        patch("torch.cuda.memory_allocated", side_effect=AssertionError("no CUDA query")), \
                        patch("torch.cuda.memory_reserved", side_effect=AssertionError("no CUDA query")):
                    trainer.fit(self.loader, epochs=2, progress_bar=False)

    def test_trainer_owns_memory_policy_after_source_and_global_changes(self):
        source = {"name": "memory", "seed": 1,
                  "training": {"epochs": 1, "batch_size": 2, "learning_rate": .01},
                  "logging": {"memory_each_epoch": True}}
        load_config(source)
        with redirect_stderr(io.StringIO()):
            trainer = self.trainer()
            source["logging"]["memory_each_epoch"] = False
            load_config(source)
            with patch("deepopt.engine.training.trainer.summarize_model_memory",
                       wraps=summarize_model_memory) as summarize:
                trainer.fit(self.loader, epochs=1, progress_bar=False)
        summarize.assert_called_once_with(trainer.model, trainer.optimizer)

    def test_failed_validation_and_non_main_rank_do_not_report_epoch_memory(self):
        with redirect_stderr(io.StringIO()):
            trainer = self.trainer({"logging": {"memory_each_epoch": True}})
            with patch.object(trainer, "evaluate", side_effect=RuntimeError("validation failed")), \
                    patch("deepopt.engine.training.trainer.summarize_model_memory") as summarize:
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    trainer.fit(self.loader, self.loader, epochs=1, progress_bar=False)
            summarize.assert_not_called()
            trainer = self.trainer({"logging": {"memory_each_epoch": True}})
            with patch("deepopt.engine.training.trainer.distributed.is_main_process", return_value=False), \
                    patch("deepopt.engine.training.trainer.summarize_model_memory") as summarize:
                trainer.fit(self.loader, epochs=1, progress_bar=False)
            summarize.assert_not_called()

    def test_cuda_reporting_queries_selected_device_without_resetting_peaks(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            trainer = self.trainer({"logging": {"memory_each_epoch": True}})
            # Exercise reporting only; this is not a CUDA training test.
            trainer.runtime.device = torch.device("cuda:2")
            trainer.state.completed_epochs = 1
            with patch("torch.cuda.memory_allocated", return_value=125_000_000) as allocated, \
                    patch("torch.cuda.memory_reserved", return_value=250_000_000) as reserved, \
                    patch("torch.cuda.max_memory_allocated", return_value=500_000_000) as peak_allocated, \
                    patch("torch.cuda.max_memory_reserved", return_value=750_000_000) as peak_reserved, \
                    patch("torch.cuda.reset_peak_memory_stats", side_effect=AssertionError("must not reset peaks")), \
                    patch("torch.cuda.synchronize", side_effect=AssertionError("must not synchronize")):
                trainer._log_epoch_memory()
        allocated.assert_called_once_with(torch.device("cuda:2"))
        reserved.assert_called_once_with(torch.device("cuda:2"))
        peak_allocated.assert_called_once_with(torch.device("cuda:2"))
        peak_reserved.assert_called_once_with(torch.device("cuda:2"))
        self.assertIn("Epoch 1 CUDA memory (cuda:2): allocated=0.125000000 GB reserved=0.250000000 GB",
                      stderr.getvalue())
        self.assertIn("Epoch 1 CUDA peak memory since last reset (cuda:2): allocated=0.500000000 GB reserved=0.750000000 GB",
                      stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
