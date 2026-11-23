"""Observable allocation policy, restoration and concurrent scope ownership."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import unittest

import torch
from torch import nn

from deepopt import TrainingRuntime
from deepopt._memory_policy import allocation_fill_scope, configured_fill, fill_spec
from deepopt._config_resolution import resolve_config


class MemoryFillTests(unittest.TestCase):
    def setUp(self):
        self.fill = torch.utils.deterministic.fill_uninitialized_memory
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(setattr, torch.utils.deterministic, "fill_uninitialized_memory", self.fill)
        self.addCleanup(torch.use_deterministic_algorithms, self.deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = True

    def test_nested_exception_and_explicit_values_restore(self):
        with allocation_fill_scope(False):
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with allocation_fill_scope(True):
                    self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
                    raise RuntimeError("test failure")
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
        self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)

    def test_other_thread_conflict_and_matching_policy(self):
        def enter(value):
            with allocation_fill_scope(value):
                return torch.utils.deterministic.fill_uninitialized_memory
        with ThreadPoolExecutor(1) as pool:
            with allocation_fill_scope(False):
                self.assertFalse(pool.submit(enter, False).result())
                with self.assertRaisesRegex(RuntimeError, "overlapping"):
                    pool.submit(enter, True).result()
            with allocation_fill_scope(True):
                with allocation_fill_scope(False):
                    # A matching inner flag must not outlive a different outer
                    # flag in another thread, which would prevent restoration.
                    with self.assertRaisesRegex(RuntimeError, "overlapping"):
                        pool.submit(enter, False).result()
        self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)

    def test_last_compatible_owner_restores_original_process_flag(self):
        entered, release = Event(), Event()
        def worker():
            with allocation_fill_scope(False):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("scope release timeout")
        with ThreadPoolExecutor(1) as pool:
            with allocation_fill_scope(False):
                future = pool.submit(worker)
                self.assertTrue(entered.wait(5))
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            release.set()
            future.result()
        self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)

    def test_runtime_scopes_forward_backward_optimizer_and_override(self):
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic)
            for override in (None, True, False):
                model = nn.Linear(2, 1)
                optimizer = torch.optim.AdamW(model.parameters())
                runtime = TrainingRuntime(model, optimizer, execution_mode="max_speed",
                                          fill_uninitialized_memory=override)
                expected = override if override is not None else False
                seen = []
                model.register_forward_pre_hook(lambda *args: seen.append(fill_spec()))
                optimizer.register_step_pre_hook(lambda *args: seen.append(fill_spec()))
                with runtime.autocast():
                    loss = runtime.model(torch.ones(2, 2)).square().mean()
                    loss.register_hook(lambda value: seen.append(fill_spec()) or value)
                runtime.backward(loss)
                runtime.step()
                self.assertEqual(seen, [(deterministic, deterministic and expected)] * 3)
                self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
                self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)

    def test_config_validation_and_mode_defaults(self):
        for mode in (None, "balanced", "analyze", "max_speed"):
            config = resolve_config({"runtime": {"execution_mode": mode}})
            self.assertEqual(configured_fill(config), False if mode == "max_speed" else None)
        with self.assertRaisesRegex(ValueError, "fill_uninitialized_memory"):
            resolve_config({"runtime": {"fill_uninitialized_memory": 1}})
