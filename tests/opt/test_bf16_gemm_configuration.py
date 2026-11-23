"""BF16 launch contracts, bounded metadata, and opt-in execution evidence."""

from dataclasses import FrozenInstanceError, replace
from itertools import product as cartesian_product
import json
import unittest
from unittest.mock import patch

import torch

from deepopt.kernel_policy import kernel_policy_scope
from deepopt.opt.backends._shared.bf16_gemm import (
    BF16GemmConfig, BF16GemmKey, MAX_BF16_CANDIDATES, MAX_BF16_METADATA_ENTRIES,
    bf16_gemm_candidates, bf16_gemm_selection, bf16_source_version,
    bf16_tile_coordinates, legacy_bf16_config,
)
from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms
from deepopt.opt.backends.bf16 import product


def key(**updates):
    values = dict(role="forward", m=32, n=4096, k=1024, device="cuda:0", capability=(9, 0),
        a_dtype="torch.bfloat16", b_dtype="torch.bfloat16", a_stride=(1024, 1), b_stride=(1024, 1),
        compression=("dense",), epilogue=(None, None, False), deterministic=False,
        precision_policy=("bf16_multiplicands_fp32_accumulation", "fp_fusion_disabled"),
        kernel_policy="test-policy", runtime_version=("torch-test", "cuda-test", "triton-test"),
        source_version="source-test")
    return BF16GemmKey(**{**values, **updates})


class BF16ConfigurationTests(unittest.TestCase):
    def test_legacy_config_preserves_existing_launch_and_never_selects_candidate(self):
        for tile, stages in cartesian_product((32, 64, 128), (1, 2, 3, 4, 5)):
            selection = bf16_gemm_selection(key(role="dw", k=100864), tile=tile, num_stages=stages)
            self.assertEqual(selection.config, BF16GemmConfig(tile, tile, 32, 4, stages))
            self.assertEqual(selection.config.ordering, "column_major")
            self.assertEqual(selection.reason, "declared_legacy_baseline_no_qualified_measurement")
            self.assertEqual(selection.qualification, "unmeasured")
        self.assertIs(legacy_bf16_config(), legacy_bf16_config())
        # A warmed integer specialization must not bypass bool validation.
        with self.assertRaisesRegex(ValueError, "positive integer"):
            legacy_bf16_config(64, True)
        bf16_gemm_selection(key(), num_stages=1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            bf16_gemm_selection(key(), num_stages=True)

    def test_explicit_independent_config_is_selected_with_experimental_packed_evidence(self):
        config = BF16GemmConfig(32, 128, 64, 8, 4, "grouped_m", 8)
        selection = bf16_gemm_selection(key(role="dw", b_dtype="torch.int32",
            compression=("affine_flat", "source_KN_right_KN", 2, 1024)), config)
        self.assertIs(selection.config, config)
        self.assertEqual(selection.reason, "explicit_configuration")
        self.assertEqual(selection.packed_read_status, "experimental_explicit_control")
        self.assertEqual(selection.qualification, "unmeasured")
        self.assertEqual(config.kernel_arguments, dict(BM=32, BN=128, BK=64, num_warps=8,
            num_stages=4, ORDER="grouped_m", GROUP_M=8, enable_fp_fusion=False))
        self.assertIn("source_version", json.loads(json.dumps(selection.as_dict()))["key"])

    def test_candidate_tables_are_role_shape_aware_bounded_and_explain_rejections(self):
        tables = []
        for role in ("forward", "dx", "dw"):
            candidates = bf16_gemm_candidates(key(role=role))
            self.assertLessEqual(len(candidates), MAX_BF16_CANDIDATES)
            self.assertEqual(len({item.config for item in candidates}), len(candidates))
            self.assertEqual(candidates[0].config, legacy_bf16_config())
            self.assertTrue(any(item.config.block_m != item.config.block_n for item in candidates))
            self.assertTrue(any(item.config.block_k > 32 for item in candidates))
            self.assertEqual({item.config.num_warps for item in candidates}, {4, 8})
            self.assertEqual({item.config.ordering for item in candidates}, {"column_major", "row_major", "grouped_m"})
            tables.append(candidates)
        self.assertNotEqual(tables[0], tables[2])
        self.assertNotEqual(bf16_gemm_candidates(key()), bf16_gemm_candidates(key(m=4096)))
        short = bf16_gemm_candidates(key(m=1, n=1, k=17))
        self.assertTrue(any(item.rejection_reason and "redundant" in item.rejection_reason for item in short))
        self.assertTrue(all(item.rejection_reason for item in bf16_gemm_candidates(key(capability=(7, 5)))))
        with self.assertRaisesRegex(ValueError, "capability"):
            bf16_gemm_selection(key(capability=(7, 5)))
        for changes, reason in (({"device": "cpu"}, "CUDA device"),
                ({"b_dtype": "torch.int32"}, "storage format"),
                ({"compression": ("other_codec",)}, "compression format"),
                ({"b_dtype": "torch.int32", "compression": ("affine_flat", "wrong_orientation", 2, 1024)}, "orientation")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, reason):
                bf16_gemm_selection(key(**changes), BF16GemmConfig())

    def test_key_separates_role_layout_codec_recipe_device_and_source(self):
        initial = key()
        changes = ({"role": "dx"}, {"role": "dw"}, {"m": 33}, {"n": 7}, {"k": 4096},
            {"a_dtype": "torch.float32"}, {"a_stride": (2048, 2)}, {"b_stride": (1, 4096)},
            {"a_storage_offset": 1}, {"b_storage_offset": 3},
            {"compression": ("affine_flat", "source_KN_right_KN", 2, 1024)},
            {"compression": ("affine_flat", "other_orientation", 2, 1024)},
            {"epilogue": (("gelu", "tanh"),)}, {"deterministic": True},
            {"precision_policy": ("different_rounding",)}, {"kernel_policy": "new-policy"},
            {"device": "cuda:1"}, {"capability": (8, 0)}, {"source_version": "new-source"},
            {"runtime_version": ("new-torch", "cuda-test", "triton-test")}, {"tuning_version": "new-tuning"})
        self.assertEqual(len({initial, *(replace(initial, **value) for value in changes)}), len(changes) + 1)
        with self.assertRaises(FrozenInstanceError):
            initial.role = "dw"
        with self.assertRaises(FrozenInstanceError):
            legacy_bf16_config().block_m = 128
        for value in (torch.ones(1), {"tensor": torch.ones(1)}, ["mutable"]):
            with self.assertRaises(TypeError):
                key(epilogue=(value,))
        self.assertEqual(len(bf16_source_version()), 64)

    def test_metadata_caches_are_bounded_and_never_store_invocation_tensors(self):
        bf16_gemm_selection.cache_clear()
        bf16_gemm_candidates.cache_clear()
        for m in range(1, MAX_BF16_METADATA_ENTRIES + 40):
            metadata = key(m=m)
            bf16_gemm_selection(metadata)
            bf16_gemm_candidates(metadata)
        self.assertEqual(bf16_gemm_selection.cache_info().currsize, MAX_BF16_METADATA_ENTRIES)
        self.assertEqual(bf16_gemm_candidates.cache_info().currsize, MAX_BF16_METADATA_ENTRIES)
        # JSON serialization rejects tensor/storage payloads without custom hooks.
        json.dumps(bf16_gemm_selection(key()).as_dict())
        self.assertIs(bf16_gemm_candidates(key()), bf16_gemm_candidates(key()))

    def test_program_ordering_covers_every_tile_including_partial_groups(self):
        configurations = (BF16GemmConfig(), BF16GemmConfig(ordering="row_major"),
                          *(BF16GemmConfig(ordering="grouped_m", group_m=g) for g in (1, 4, 8, 16)))
        for rows, columns, config in cartesian_product((1, 3, 8, 9, 17), (1, 3, 9), configurations):
            actual = [bf16_tile_coordinates(pid, rows, columns, config) for pid in range(rows * columns)]
            self.assertEqual(len(set(actual)), len(actual))
            self.assertEqual(set(actual), set(cartesian_product(range(rows), range(columns))))
        self.assertEqual([bf16_tile_coordinates(pid, 3, 2, configurations[0]) for pid in range(6)],
                         [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)])

    def test_invalid_config_and_route_are_rejected_before_importing_generated_backend(self):
        for changes in ({"block_m": True}, {"block_n": 63}, {"block_k": 16}, {"num_stages": 6},
                        {"num_warps": 16}, {"ordering": "persistent"}, {"group_m": 8}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                BF16GemmConfig(**changes)
        a, b = torch.randn(3, 5), torch.randn(7, 5)
        with patch("deepopt.opt.backends.bf16.import_module", side_effect=AssertionError("unexpected GPU import")):
            for backend in ("native", "tilelang"):
                with self.assertRaisesRegex(ValueError, "Triton"):
                    product(a, b, backend=backend, kernel_config=BF16GemmConfig())
            with self.assertRaisesRegex(TypeError, "BF16GemmConfig"):
                product(a, b, backend="triton", kernel_config={"block_m": 64})
            with self.assertRaisesRegex(ValueError, "role"):
                product(a, b, backend="triton", role="db")

    def test_cpu_generated_request_is_labeled_reference_and_records_ambient_role(self):
        a, b = torch.randn(3, 5), torch.randn(7, 5)
        expected = torch.nn.functional.linear(a.bfloat16(), b.bfloat16())
        with patch("deepopt.opt.backends.bf16.import_module", side_effect=AssertionError("unexpected GPU import")):
            with record_gemms() as records, kernel_policy_scope(role="linear.dw"):
                actual = product(a, b, backend="triton", kernel_config=BF16GemmConfig(32, 64, 64))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        record, = records
        self.assertEqual(record["requested_backend"], "triton")
        self.assertEqual(record["effective_backend"], "reference")
        self.assertEqual(record["execution"], "semantic_reference")
        self.assertEqual(record["role"], "dw")
        self.assertNotIn("kernel_launch", record)
        json.dumps(record)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BF16ConfigurationCUDA(unittest.TestCase):
    def test_independent_tiles_orders_and_launch_evidence(self):
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("NVIDIA compute capability >= 8 required")
        a = torch.randn(129, 83, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(137, 83, device="cuda", dtype=torch.bfloat16)
        configurations = (BF16GemmConfig(32, 64, 64, 4, 1, "row_major"),
                          BF16GemmConfig(64, 32, 32, 8, 3, "grouped_m", 4))
        reference = torch.nn.functional.linear(a, b)
        with record_gemms() as records:
            for config in configurations:
                actual = product(a, b, backend="triton", role="dx", kernel_config=config)
                torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
        self.assertEqual(len(records), len(configurations))
        for record, config in zip(records, configurations):
            self.assertEqual(record["execution"], "generated")
            self.assertEqual(record["kernel_launch"]["selected_config"], config.as_dict())
            self.assertEqual(record["kernel_launch"]["selection"]["key"]["role"], "dx")
            self.assertEqual(record["kernel_launch"]["selection"]["qualification"], "unmeasured")
            self.assertIn("spills", record["kernel_launch"]["resources"])
            json.dumps(record)


if __name__ == "__main__":
    unittest.main()
