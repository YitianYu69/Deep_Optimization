"""Inspect initialization metadata sources, placement, and group ownership."""

from dataclasses import FrozenInstanceError
from datetime import timedelta
import os
import unittest
from unittest.mock import patch

import torch

from deepopt import DistributedContext, initialize_distributed
from deepopt.distributed import _process


class ProcessMetadataWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch.object(_process, "_owns_group", False))

    def test_cpu_without_distributed_build_uses_local_defaults(self):
        with patch.object(_process.dist, "is_available", return_value=False), \
                patch.object(_process.dist, "is_initialized") as initialized, \
                patch("torch.cuda.is_available", return_value=False):
            context = initialize_distributed("auto")
        initialized.assert_not_called()
        self.assertIsInstance(context, DistributedContext)
        self.assertEqual(context.as_dict(), {
            "device": "cpu", "global_rank": 0, "local_rank": 0, "world_size": 1,
            "local_world_size": 1, "backend": None, "is_initialized": False,
            "owns_process_group": False, "is_distributed": False, "is_main_process": True,
        })
        with self.assertRaises(FrozenInstanceError):
            context.global_rank = 9

    def test_reused_group_is_authoritative_for_global_metadata(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "999", "RANK": "999", "LOCAL_RANK": "2", "LOCAL_WORLD_SIZE": "4"}), \
                patch.object(_process.dist, "is_available", return_value=True), \
                patch.object(_process.dist, "is_initialized", return_value=True), \
                patch.object(_process.dist, "get_rank", return_value=6), \
                patch.object(_process.dist, "get_world_size", return_value=8), \
                patch.object(_process.dist, "get_backend", return_value="gloo"), \
                patch.object(_process.dist, "init_process_group") as initialize, \
                patch.object(_process.dist, "destroy_process_group") as destroy:
            context = initialize_distributed("cpu", backend="nccl")
            _process.shutdown_distributed()
        self.assertEqual((context.global_rank, context.local_rank, context.world_size, context.local_world_size), (6, 2, 8, 4))
        self.assertEqual(context.backend, "gloo")
        self.assertFalse(context.owns_process_group)
        self.assertFalse(context.is_main_process)
        initialize.assert_not_called()
        destroy.assert_not_called()

    def test_external_group_without_launcher_has_unknown_local_topology(self):
        with patch.object(_process.dist, "is_available", return_value=True), \
                patch.object(_process.dist, "is_initialized", return_value=True), \
                patch.object(_process.dist, "get_rank", return_value=3), \
                patch.object(_process.dist, "get_world_size", return_value=4), \
                patch.object(_process.dist, "get_backend", return_value="gloo"):
            context = initialize_distributed("cpu")
        self.assertIsNone(context.local_rank)
        self.assertIsNone(context.local_world_size)
        self.assertEqual(context.global_rank, 3)
        self.assertTrue(context.is_initialized)

    def test_cuda_placement_uses_local_rank_not_global_rank(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "2", "LOCAL_WORLD_SIZE": "4"}), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.set_device") as set_device, \
                patch("torch.cuda.current_device", side_effect=AssertionError("LOCAL_RANK determines placement")), \
                patch.object(_process.dist, "is_available", return_value=True), \
                patch.object(_process.dist, "is_initialized", return_value=True), \
                patch.object(_process.dist, "get_rank", return_value=6), \
                patch.object(_process.dist, "get_world_size", return_value=8), \
                patch.object(_process.dist, "get_backend", return_value="nccl"):
            context = initialize_distributed("auto")
        set_device.assert_called_once_with(torch.device("cuda:2"))
        self.assertEqual(context.device, torch.device("cuda:2"))
        self.assertEqual((context.local_rank, context.global_rank), (2, 6))

    def test_initialization_reuse_and_shutdown_preserve_metadata_snapshots(self):
        active = [False]
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "2"}), \
                patch.object(_process.dist, "is_available", return_value=True), \
                patch.object(_process.dist, "is_initialized", side_effect=lambda: active[0]), \
                patch.object(_process.dist, "get_rank", return_value=1), \
                patch.object(_process.dist, "get_world_size", return_value=2), \
                patch.object(_process.dist, "get_backend", return_value="gloo"), \
                patch.object(_process.dist, "init_process_group", side_effect=lambda **kw: active.__setitem__(0, True)) as initialize, \
                patch.object(_process.dist, "destroy_process_group", side_effect=lambda: active.__setitem__(0, False)) as destroy:
            context = initialize_distributed("cpu", timeout=45)
            repeated = initialize_distributed("cpu")
            self.assertEqual(context, repeated)
            self.assertIsNot(context, repeated)
            self.assertTrue(context.owns_process_group)
            initialize.assert_called_once_with(backend="gloo", timeout=timedelta(seconds=45))
            _process.shutdown_distributed()
            self.assertFalse(active[0])
            destroy.assert_called_once()
            self.assertTrue(context.is_initialized)

    def test_invalid_local_topology_fails_before_group_creation(self):
        for environment in ({"WORLD_SIZE": "0"}, {"LOCAL_RANK": "-1"}, {"LOCAL_RANK": "bad"},
                            {"LOCAL_WORLD_SIZE": "0"}, {"LOCAL_RANK": "2", "LOCAL_WORLD_SIZE": "2"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
                    patch.object(_process.dist, "init_process_group") as initialize:
                with self.assertRaises(ValueError):
                    initialize_distributed("cpu")
                initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
