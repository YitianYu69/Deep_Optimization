"""Physical-storage evidence and fail-closed capacity policy regressions."""

import gc
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from deepopt.memory import admit_cuda_capacity, cuda_capacity_admission
from deepopt.memory.ledger import StorageLedger, StorageObserver
from deepopt.opt.quantization.activation_tensor import FP8Activation


class StorageLedgerTests(unittest.TestCase):
    def test_physical_ranges_preserve_alias_offsets_and_generations_without_payload_reads(self):
        ledger = StorageLedger()
        value = torch.empty(10, 12)
        alias = value[:, 3:5]
        ledger.observe((value, alias))
        report = ledger.snapshot()
        storage = report["storages"][0]
        self.assertEqual(storage["physical_address"], value.untyped_storage().data_ptr())
        self.assertEqual(storage["physical_byte_range"], [storage["physical_address"], storage["physical_address"] + 480])
        self.assertEqual(report["logical_values"][1]["storage_byte_span"], [12, 452])
        self.assertEqual(len(ledger.live_storage_records()), 1)
        with FakeTensorMode(allow_fallback_kernels=False):
            fake = torch.empty(3)
            other = StorageLedger()
            other.observe(fake)
            self.assertIsNone(other.snapshot()["storages"][0]["physical_address"])

    def test_tied_parameter_aliases_and_derived_copies_have_distinct_accounting(self):
        parameter = torch.nn.Parameter(torch.randn(2048))
        alias = parameter[8:]
        clone = parameter.detach().clone()
        cast = parameter.detach().bfloat16()
        ledger = StorageLedger()
        ledger.observe((parameter, parameter), owner="model_parameter")
        keys = [ledger.save(value) for value in (parameter, alias, clone, cast)]
        self.assertEqual(len(ledger.snapshot()["storages"]), 3)
        self.assertEqual(ledger.saved_snapshot()["live_unique_bytes"], 20480)
        self.assertEqual(ledger.saved_snapshot()["live_nonpersistent_bytes"], 12288)
        for key in keys:
            ledger.release_saved(key)
        self.assertEqual(ledger.saved_snapshot()["live_unique_bytes"], 0)

    def test_optimizer_parameter_keys_are_never_formatted(self):
        parameter = torch.nn.Parameter(torch.empty(4))
        state = torch.empty(4)
        ledger = StorageLedger()
        with patch.object(torch.Tensor, "__repr__", side_effect=AssertionError("payload formatting")):
            ledger.observe({parameter: {"momentum": state}}, owner="optimizer_state")
        self.assertEqual(ledger.snapshot()["logical_values"][0]["component"], "value.entry_0.momentum")

    def test_views_and_saved_references_share_backing_with_separate_owners(self):
        ledger = StorageLedger()
        value = torch.empty(10, 12)
        alias = value[:, 3:5]
        ledger.observe(value, owner="caller_input")
        first, second = ledger.save(value), ledger.save(alias)
        ledger.advance("use")
        ledger.observe(alias, consumer="backward:dW")
        ledger.unpack(first)
        report = ledger.snapshot()
        self.assertEqual(len(report["storages"]), 1)
        record = report["storages"][0]
        self.assertEqual(record["owners"], ["caller_input", "saved_for_backward"])
        self.assertEqual(record["saved_reference_count"], 2)
        self.assertEqual(record["last_use"], 1)
        self.assertEqual(report["logical_values"][1]["storage_offset"], 3)
        self.assertEqual(report["logical_values"][1]["strides"], [12, 1])
        self.assertEqual(report["peak_visible_bytes_by_device"]["cpu"], 480)
        self.assertEqual(report["saved_storage_lifetimes"]["cumulative_unique_storage_bytes"], 480)
        ledger.release_saved(first)
        self.assertEqual(ledger.saved_snapshot()["live_unique_bytes"], 480)
        ledger.release_saved(second)
        self.assertEqual(ledger.saved_snapshot()["live_unique_bytes"], 0)
        self.assertEqual(ledger.snapshot()["live_visible_bytes_by_device"], {"cpu": 480})

    def test_records_do_not_retain_payload_and_release_is_observed(self):
        ledger = StorageLedger()
        value = torch.empty(100)
        reference = weakref.ref(value)
        ledger.observe(value, producer="forward:temporary")
        report = ledger.snapshot()
        del value
        gc.collect()
        self.assertIsNone(reference())
        ledger.advance()
        after = ledger.snapshot()
        self.assertEqual(after["live_visible_bytes_by_device"], {})
        self.assertEqual(after["storages"][0]["release_observed_at"], 1)
        json.dumps(report)
        json.dumps(after)

    def test_expired_storage_identity_starts_a_new_generation(self):
        ledger = StorageLedger()
        value = torch.empty(7)
        first = ledger.observe(value)
        # Simulate address reuse deterministically, without depending on the
        # host allocator returning a particular storage implementation pointer.
        entry = next(iter(ledger._live.values()))
        entry["reference"] = SimpleNamespace(expired=lambda: True)
        ledger.advance()
        second = ledger.observe(value)
        self.assertNotEqual(first, second)
        self.assertEqual([row["generation"] for row in ledger.snapshot()["storages"]], [1, 2])

    def test_fp8_wrapper_and_explicit_children_deduplicate_physical_bytes(self):
        data = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
        scale = torch.empty(1, 1)
        value = FP8Activation(data, scale, (2, 4), layout="last")
        ledger = StorageLedger()
        ledger.observe((value, data, scale), owner="saved_for_backward")
        report = ledger.snapshot()
        self.assertEqual(len(report["storages"]), 2)
        self.assertEqual(report["peak_visible_bytes_by_device"]["cpu"], 12)
        self.assertEqual(sum(row["storage_bytes"] for row in report["storages"]), 12)

    def test_mutation_and_rebinding_end_logical_versions_without_releasing_backing(self):
        ledger = StorageLedger()
        value = torch.zeros(4)
        alias = value[:]
        ledger.observe(value)
        value.add_(1)
        ledger.advance()
        ledger.observe(value)
        replacement = torch.empty(8)
        value.set_(replacement)
        ledger.advance()
        ledger.observe(value)
        report = ledger.snapshot()
        versions = report["logical_values"]
        self.assertEqual(versions[0]["binding_end_reason"], "version_changed")
        self.assertEqual(versions[0]["binding_end_observed_at"], 1)
        self.assertEqual(versions[1]["binding_end_reason"], "storage_rebound")
        self.assertEqual(versions[1]["binding_end_observed_at"], 2)
        self.assertIsNone(report["storages"][0]["release_observed_at"])
        self.assertEqual(report["live_visible_bytes_by_device"], {"cpu": 48})
        self.assertEqual(alias.numel(), 4)

    def test_dispatch_captures_coexisting_temporaries_and_fake_metadata(self):
        for fake in (False, True):
            with self.subTest(fake=fake):
                ledger = StorageLedger()
                mode = FakeTensorMode(allow_fallback_kernels=False) if fake else None
                from contextlib import nullcontext
                with mode if fake else nullcontext(), StorageObserver(ledger):
                    source = torch.empty(100)
                    copy = source.clone()
                    del source
                    output = copy[:2]
                ledger.observe(output, owner="escaped_output")
                report = ledger.snapshot()
                self.assertEqual(report["peak_visible_bytes_by_device"]["cpu"], 800)
                self.assertEqual(report["live_visible_bytes_by_device"]["cpu"], 400)
                self.assertIsNone(report["opaque_workspace_bytes"])
                self.assertTrue(any(row["consumers"] for row in report["storages"]))

    def test_metadata_limit_does_not_limit_peak_accounting(self):
        ledger = StorageLedger(max_records=2)
        values = [torch.empty(10), torch.empty(20), torch.empty(30)]
        ledger.observe(values)
        report = ledger.snapshot()
        self.assertTrue(report["metadata_truncated"])
        self.assertEqual(len(report["storages"]), 2)
        self.assertEqual(len(report["logical_values"]), 2)
        self.assertEqual(report["peak_visible_bytes_by_device"], {"cpu": 240})
        del values
        gc.collect()
        self.assertEqual(ledger.snapshot()["live_visible_bytes_by_device"], {})


class CapacityAdmissionTests(unittest.TestCase):
    def admission(self, **changes):
        values = dict(requested_workload={"physical_local_batch": 128, "input": [128, 3, 224, 224]},
                      device_available=True, total_bytes=10000, free_bytes=7000,
                      already_live_bytes=3000, additional_tensor_bytes=5000, workspace_budget_bytes=500,
                      margin_fraction=.1, margin_bytes=1)
        values.update(changes)
        return admit_cuda_capacity(**values)

    def test_known_fit_and_live_state_is_not_subtracted_twice(self):
        result = self.admission()
        self.assertTrue(result["admitted"])
        self.assertEqual(result["required_additional_bytes_including_margin"], 6500)
        self.assertEqual(result["requested_workload"]["physical_local_batch"], 128)

    def test_unknown_workspace_device_or_capacity_rejects_unchanged_workload(self):
        for changes in ({"workspace_budget_bytes": None}, {"free_bytes": None},
                        {"already_live_bytes": None}, {"device_available": None},
                        {"additional_tensor_bytes": None}, {"device_available": False}):
            with self.subTest(changes=changes):
                result = self.admission(**changes)
                self.assertFalse(result["admitted"])
                self.assertTrue(result["reasons"])
                self.assertEqual(result["requested_workload"]["physical_local_batch"], 128)

    def test_both_capacity_counters_and_default_margin_are_checked(self):
        self.assertFalse(self.admission(additional_tensor_bytes=6500)["admitted"])
        self.assertFalse(self.admission(free_bytes=11000)["admitted"])
        self.assertFalse(self.admission(already_live_bytes=5000)["admitted"])
        self.assertFalse(self.admission(margin_bytes=2**30)["admitted"])
        for bad in (-1, 1.2, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.admission(workspace_budget_bytes=bad)

    def test_cuda_query_failure_and_cpu_fail_closed(self):
        cuda = SimpleNamespace(is_available=lambda: True,
                               mem_get_info=lambda device: (_ for _ in ()).throw(RuntimeError("no counters")))
        fake_torch = SimpleNamespace(cuda=cuda)
        result = cuda_capacity_admission(fake_torch, torch.device("cuda"), requested_workload={"batch": 128},
                                         additional_tensor_bytes=100, workspace_budget_bytes=10)
        self.assertFalse(result["admitted"])
        self.assertIn("no counters", result["query_error"])
        result = cuda_capacity_admission(fake_torch, torch.device("cpu"), requested_workload={"batch": 128},
                                         additional_tensor_bytes=100, workspace_budget_bytes=10)
        self.assertFalse(result["admitted"])


if __name__ == "__main__":
    unittest.main()
