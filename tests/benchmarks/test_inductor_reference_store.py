"""Permanent compiler-reference storage without serialized tensor payloads."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from benchmarks.compiler.inductor_reference_store import (
    create_run, finalize_run, find_references, refresh_index, save_artifacts, verify_run,
)


class InductorReferenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.store = self.root / "references"
        self.source = self.root / "working"
        self.source.mkdir()

    def run_dir(self, **request):
        return create_run(self.store, {"model": "resnet18", "recipe": "torch-bf16", "device_type": "cuda", **request})

    def code(self):
        (self.source / "output_code.py").write_text("# generated code\n")

    def test_unique_permanent_runs_no_overwrite(self):
        first, second = self.run_dir(), self.run_dir()
        self.assertNotEqual(first, second)
        self.code()
        save_artifacts(first, self.source)
        finalized = finalize_run(first, {"torch_version": "2.15"}, "complete")
        self.assertEqual(finalized["status"], "complete")
        self.assertTrue(verify_run(first)["valid"])
        with self.assertRaises(FileExistsError):
            finalize_run(first, {}, "failed")
        with self.assertRaises(FileExistsError):
            save_artifacts(first, self.source, namespace="again")
        self.assertTrue(second.is_dir())
        self.assertEqual(len(refresh_index(self.store)["runs"]), 1)

    def test_graph_ir_and_nested_manifests_are_inventory_hashed(self):
        run = self.run_dir()
        self.code()
        for name in ("fx_graph_readable.py", "ir_pre_fusion.txt", "ir_post_fusion.txt", "manifest.json", "kernel.ttgir", "kernel.ptx"):
            (self.source / name).write_text("{}\n")
        report = save_artifacts(run, self.source)
        self.assertEqual(len(report["copied"]), 7)
        manifest = finalize_run(run, {}, "complete", required_artifacts=("fx_graph", "ir_pre_fusion", "ir_post_fusion"))
        self.assertEqual(manifest["status"], "complete")
        self.assertIn("artifacts/inductor/manifest.json", manifest["files"])
        (run / "artifacts/inductor/manifest.json").write_text("tampered")
        self.assertFalse(verify_run(run)["valid"])
        self.assertEqual(find_references(self.store), [])

    def test_tensor_pickle_and_unknown_files_are_never_copied(self):
        run = self.run_dir()
        self.code()
        for name in ("weights.pt", "snapshot.pt.tmp", "state.pth", "x.npy", "x.npz", "x.safetensors", "fx.pkl", "x.pickle", "x.tensor", "x.bin", "cache.lock"):
            (self.source / name).write_bytes(b"forbidden contents")
        report = save_artifacts(run, self.source)
        self.assertEqual(len(report["copied"]), 1)
        self.assertEqual(len(report["omitted"]), 11)
        self.assertFalse(report["all_files_saved"])
        self.assertEqual(report["tensor_files_saved"], 0)
        self.assertEqual([x.name for x in (run / "artifacts/inductor").iterdir()], ["output_code.py"])
        finalize_run(run, {}, "complete")
        self.assertTrue(verify_run(run)["valid"])

    def test_missing_dumps_and_explicit_blocked_attempt_are_not_references(self):
        empty = self.run_dir()
        report = finalize_run(empty, {"reason": "CUDA unavailable"}, "blocked")
        self.assertEqual(report["status"], "blocked")
        another = self.run_dir()
        self.assertEqual(finalize_run(another, {}, "complete")["status"], "partial")
        self.assertEqual(find_references(self.store), [])
        self.assertEqual(len(find_references(self.store, include_incomplete=True)), 2)

    def test_copy_limits_across_namespaces_downgrade_complete(self):
        run = self.run_dir()
        self.code()
        size = (self.source / "output_code.py").stat().st_size
        save_artifacts(run, self.source, namespace="debug", max_total_bytes=size + 1)
        report = save_artifacts(run, self.source, namespace="cache", max_total_bytes=size + 1)
        self.assertEqual(report["copied"], [])
        self.assertEqual(report["errors"][0]["reason"], "artifact_byte_limit")
        self.assertEqual(finalize_run(run, {}, "complete")["status"], "partial")

    def test_symlink_file_directory_root_and_namespace_traversal_rejected(self):
        run = self.run_dir()
        outside = self.root / "outside.py"
        outside.write_text("do not copy")
        (self.source / "link.py").symlink_to(outside)
        (self.source / "directory").symlink_to(self.root, target_is_directory=True)
        report = save_artifacts(run, self.source)
        self.assertGreaterEqual(len(report["errors"]), 2)
        self.assertEqual(report["copied"], [])
        with self.assertRaises(ValueError):
            save_artifacts(run, self.source, namespace="../escape")
        linked = self.root / "linked"
        linked.symlink_to(self.store, target_is_directory=True)
        with self.assertRaises(ValueError):
            create_run(linked, {})

    def test_late_unindexed_or_forbidden_files_fail_integrity(self):
        run = self.run_dir()
        self.code()
        save_artifacts(run, self.source)
        finalize_run(run, {}, "complete")
        (run / "unexpected.log").write_text("late file")
        self.assertFalse(verify_run(run)["valid"])
        (run / "weights.pt").write_text("never unpickle")
        with self.assertRaises(ValueError):
            verify_run(run)

    def test_reference_filters_separate_cpu_gpu_recipe_and_versions(self):
        self.code()
        for device, recipe, version in (("cpu", "torch-bf16", "2.15"), ("cuda", "torch-bf16", "2.15"), ("cuda", "deepopt-bf16-compressed", "2.16")):
            run = self.run_dir(device_type=device, recipe=recipe)
            save_artifacts(run, self.source)
            finalize_run(run, {"torch_version": version}, "complete")
        refs = find_references(self.store, model="resnet18", recipe="torch-bf16", device_type="cuda", torch_version="2.15")
        self.assertEqual(len(refs), 1)
        index = json.loads((self.store / "index.json").read_text())
        self.assertEqual(len(index["runs"]), 3)
        self.assertTrue(all(not Path(entry["manifest"]).is_absolute() for entry in index["runs"]))

    def test_byte_limits_and_non_json_request_fail_before_copy(self):
        with self.assertRaises(TypeError):
            create_run(self.store, {"tensor": object()})
        run = self.run_dir()
        self.code()
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                save_artifacts(run, self.source, max_file_bytes=limit)

    def test_fifo_never_blocks_artifact_copy(self):
        run = self.run_dir()
        os.mkfifo(self.source / "output_code.py")
        report = save_artifacts(run, self.source)
        self.assertEqual(report["copied"], [])
        self.assertIn("regular file", report["errors"][0]["reason"])

    def test_damaged_attempt_does_not_hide_healthy_references(self):
        self.code()
        healthy, damaged = self.run_dir(), self.run_dir()
        for run in (healthy, damaged):
            save_artifacts(run, self.source)
            finalize_run(run, {}, "complete")
        (damaged / "manifest.json").write_text("not JSON")
        refs = find_references(self.store)
        self.assertEqual([entry["run_id"] for entry in refs], [healthy.name])
        self.assertEqual(refresh_index(self.store)["runs"][1]["status"], "integrity_error")

    def test_unknown_direct_artifact_cannot_bypass_copy_policy(self):
        run = self.run_dir()
        self.code()
        save_artifacts(run, self.source)
        (run / "unrecognized.archive").write_bytes(b"not a compiler artifact")
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            finalize_run(run, {}, "complete")


if __name__ == "__main__":
    unittest.main()
