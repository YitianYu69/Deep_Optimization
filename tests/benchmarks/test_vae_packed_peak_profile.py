"""Metadata converter guards; scalar fixtures provide no CUDA evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from examples.vae import packed_peak_profile
from tests.benchmarks.test_vae_packed_offload_benchmark import arguments, offload, successful


def save(path, value):
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def fixture(directory):
    root = Path(directory)
    plan = offload.benchmark_plan(arguments(root))
    resident, target = successful(plan, "resident"), successful(plan, "demand_offload")
    regions = [{"forward_signature": "a" * 64, "backward_signature": "b" * 64, "recipe_identity_sha256": "c" * 64},
        {"forward_signature": "d" * 64, "backward_signature": "e" * 64, "recipe_identity_sha256": "f" * 64}]
    boundaries = []
    for invocation, (region, phase) in enumerate(((0, "forward"), (1, "forward"), (1, "backward"), (0, "backward")), 1):
        boundaries.append({"invocation": invocation, "native_region_index": region, "phase": phase, "index": 7,
            **{"native_" + name: value for name, value in regions[region].items()}})
    storage = {"finished": True, "metadata_only": True, "cuda_execution_evidence": True,
        "attribution_complete": False, "errors": [], "omitted_boundaries": 0, "omitted_allocator_samples": 0,
        "boundaries": boundaries, "allocator_samples": [{**row, "allocator_allocated_bytes": 400} for row in boundaries]}
    joined = {"events": [{"total_allocated_bytes": 450, "native_boundary": boundaries[2]}], "omitted_events": 0}
    compiler = {"native_autodiff_graphs": [{"packed_activation_offload": {"model_scope": {
        "identity": "1" * 64, "regions": regions, "inventory": {}, "peak_windows": []}}}]}
    resident["profile"] = {"complete": True, "cuda_execution_evidence": True,
        "storage_ledger": save(root / "storage-ledger.json", storage),
        "allocator_storage_join": save(root / "allocator-storage-join.json", joined)}
    target["compiler_report_artifact"] = save(root / "compiler-report.json", compiler)
    resident_path, target_path = root / "resident.json", root / "target.json"
    save(resident_path, resident)
    save(target_path, target)
    return resident_path, target_path, resident, target, storage, joined


class PackedPeakProfileTests(unittest.TestCase):
    def test_full_report_size_cannot_hide_declared_scope_or_later_conflicts(self):
        scope = {"identity": "1" * 64, "regions": [{"forward_signature": "a" * 64}],
            "inventory": {}, "peak_windows": []}
        region = {"packed_activation_offload": {"transport_envelope_version": 2, "model_scope": scope}}
        graph = {"native_autodiff_regions": 2, "native_autodiff_graphs": [region, deepcopy(region)],
            "final_schedule": [{} for _ in range(500001)]}
        report = {"captured_graphs": 1, "native_autodiff_regions": 2, "graphs": [graph]}
        self.assertEqual(packed_peak_profile._model_scope(report), scope)
        graph["native_autodiff_graphs"][1]["packed_activation_offload"]["model_scope"]["peak_windows"] = [
            {"region": 0, "phase": "forward", "boundary": 1, "baseline_bytes": 1}]
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            packed_peak_profile._model_scope(report)
        graph["native_autodiff_graphs"][1] = {}
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            packed_peak_profile._model_scope(report)

    def test_exact_resident_target_signatures_produce_explicit_profile_and_hashed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            resident_path, target_path, _, _, _, _ = fixture(directory)
            profile, evidence = packed_peak_profile.derive_profile(resident_path, target_path)
            from deepopt.compiler.model_offload_plan import resolve_peak_profile

            resolved = resolve_peak_profile(profile)
            self.assertEqual(resolved.scope_identity, "1" * 64)
            self.assertEqual(len(resolved.windows), 4)
            self.assertEqual(max(window.baseline_bytes for window in resolved.windows), 450)
            self.assertEqual(profile["source_receipt_sha256"], hashlib.sha256(resident_path.read_bytes()).hexdigest())
            self.assertFalse(evidence["automatic_policy_promotion"])
            self.assertFalse(evidence["physical_storage_attribution_complete"])

    def test_CPU_enabled_baseline_identity_drift_and_incomplete_diagnostics_are_rejected(self):
        for change in ("CPU", "enabled", "input", "profile"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                source, target_path, resident, _, _, _ = fixture(directory)
                if change == "CPU":
                    resident["device"] = "cpu"
                elif change == "enabled":
                    resident["settings"]["runtime"]["packed_activation_offload"]["enabled"] = True
                elif change == "input":
                    resident["input"]["sha256"] = "9" * 64
                else:
                    resident["profile"]["complete"] = False
                save(source, resident)
                with self.assertRaises(ValueError):
                    packed_peak_profile.derive_profile(source, target_path)

    def test_changed_artifact_branch_order_or_unmapped_peak_never_produces_a_profile(self):
        for change in ("hash", "branch", "outside", "mapping"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                source, target_path, resident, _, storage, joined = fixture(directory)
                if change == "hash":
                    Path(resident["profile"]["storage_ledger"]["path"]).write_text("{}")
                elif change == "branch":
                    storage["boundaries"][0]["native_region_index"] = 1
                    resident["profile"]["storage_ledger"] = save(Path(directory) / "storage-ledger.json", storage)
                elif change == "mapping":
                    storage["boundaries"][0].pop("native_recipe_identity_sha256")
                    resident["profile"]["storage_ledger"] = save(Path(directory) / "storage-ledger.json", storage)
                else:
                    joined["events"].append({"total_allocated_bytes": 500, "native_boundary": None})
                    resident["profile"]["allocator_storage_join"] = save(Path(directory) / "allocator-storage-join.json", joined)
                save(source, resident)
                with self.assertRaises(ValueError):
                    packed_peak_profile.derive_profile(source, target_path)

    def test_ambiguous_or_changed_native_target_signatures_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target_path, _, target, _, _ = fixture(directory)
            compiler_path = Path(target["compiler_report_artifact"]["path"])
            value = json.loads(compiler_path.read_text())
            extra = deepcopy(value["native_autodiff_graphs"][0])
            extra["packed_activation_offload"]["model_scope"]["identity"] = "2" * 64
            value["native_autodiff_graphs"].append(extra)
            target["compiler_report_artifact"] = save(compiler_path, value)
            save(target_path, target)
            with self.assertRaisesRegex(ValueError, "unambiguous"):
                packed_peak_profile.derive_profile(source, target_path)


if __name__ == "__main__":
    unittest.main()
