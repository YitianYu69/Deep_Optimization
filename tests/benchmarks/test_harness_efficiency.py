"""Bounded diagnostics, complete compact exports and unchanged numerical gates."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

import torch

from analysis.measurement import errors, measure
from analysis.telemetry import UtilizationTelemetry
from compiler.export_evidence import export_evidence, read_evidence


class HarnessEfficiencyTests(unittest.TestCase):
    def test_chunked_comparison_matches_full_tensor_metrics_and_bounds_temporaries(self):
        torch.manual_seed(91)
        expected = torch.randn(137, 73, dtype=torch.float64).T
        actual = expected.clone()
        actual[-1, -1] += .125
        sizes = []
        original = torch.Tensor.double

        def convert(value, *args, **kwargs):
            sizes.append(value.numel())
            return original(value, *args, **kwargs)

        with patch.object(torch.Tensor, "double", convert):
            report = errors({"output": actual}, {"output": expected}, .01, .001, chunk_elements=37)
        result = report["tensors"]["output"]
        self.assertLessEqual(max(sizes), 37)
        self.assertEqual(result["allclose_passed"], bool(torch.allclose(actual, expected, rtol=.01, atol=.001)))
        self.assertAlmostEqual(result["relative_l2"], float((actual - expected).norm() / expected.norm()), places=14)
        self.assertEqual(result["max_abs"], .125)
        self.assertEqual(report["comparison_chunk_elements"], 37)
        actual[-1, -1] = float("nan")
        self.assertFalse(errors({"x": actual}, {"x": expected}, .01, .001, chunk_elements=37)["finite"])

    def test_integer_exactness_does_not_disappear_in_double_conversion(self):
        actual = torch.tensor([2**60 + 1], dtype=torch.int64)
        expected = torch.tensor([2**60 + 2], dtype=torch.int64)
        self.assertFalse(errors({"x": actual}, {"x": expected}, 1., 1., chunk_elements=1)["passed"])
        self.assertTrue(errors({"x": torch.empty(0)}, {"x": torch.empty(0)}, 0., 0.)["passed"])

    def test_telemetry_is_bounded_and_cannot_query_inside_throughput_window(self):
        queries = []
        telemetry = UtilizationTelemetry("cuda:0", interval=1, max_samples=2,
                                         query=lambda: queries.append(1) or {"utilization_percent": 0})

        def call():
            thread = threading.Thread(target=telemetry.sample)
            thread.start()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

        measure(call, torch, torch.device("cpu"), 0, 1, 1)
        self.assertFalse(queries)
        self.assertEqual(telemetry.skipped, 1)
        for _ in range(3):
            telemetry.sample()
        report = telemetry.report()
        self.assertEqual(len(report["samples"]), 2)
        self.assertEqual(report["dropped_oldest_samples"], 1)
        self.assertFalse(report["hot_window_sampling"])

    def test_telemetry_unavailable_cpu_and_invalid_intervals_are_explicit(self):
        def missing():
            raise FileNotFoundError("nvidia-smi")

        telemetry = UtilizationTelemetry("cuda:0", 1, query=missing)
        telemetry.sample()
        self.assertEqual(telemetry.report()["status"], "unavailable")
        with UtilizationTelemetry("cpu", 1, query=missing) as cpu:
            self.assertEqual(cpu.report()["status"], "disabled")
        for value in (-1, .1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                UtilizationTelemetry("cuda:0", value)

    def test_named_group_slots_preserve_independent_cases_under_one_recipe(self):
        from analysis.matrix import DEFAULT_SETTINGS, variants
        from analysis.scheduler import effective_task

        grid = {"compression.group_size": [1024],
                "runtime.qualification_case": [f"group1024-slot-{n}" for n in range(5)],
                "compression.bits": [1, 2, 4, 8]}
        rows = list(variants(deepcopy(DEFAULT_SETTINGS), grid, "cartesian"))
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({key for key, _ in rows}), 20)
        self.assertEqual({value["compression"]["group_size"] for _, value in rows}, {1024})
        self.assertEqual({value["compression"]["bits"] for _, value in rows}, {1, 2, 4, 8})
        # Effective-setting deduplication must retain the independent repetitions.
        controls = [effective_task({"case": "torch_bf16", "settings": value}) for _, value in rows]
        self.assertEqual(len({json.dumps(value, sort_keys=True) for value in controls}), 5)

    def test_export_deduplicates_complete_reports_and_roundtrips_proof_and_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "reports"
            source.mkdir()
            proof = {"compiler_regions": {f"region-{n}": {"kernel_names": ["_bf16_gemm"],
                "linked_cuda_launches_per_step": n + 1} for n in range(400)},
                "timing": {"wall_ms": {"samples": [float(n) for n in range(50)]}}}
            originals = {}
            for n in range(3):
                value = {"case": n, "profile": proof, "strict_launch_proof": True}
                name = f"case-{n}.json"
                originals[name] = value
                (source / name).write_text(json.dumps(value, indent=2))
            (source / "worker.log").write_text("complete log\n")
            exported = export_evidence(source, root / "export.zip")
            self.assertEqual(exported["logical_files"], 4)
            with zipfile.ZipFile(root / "export.zip") as archive:
                manifest = json.loads(archive.read("evidence-manifest.json"))
                self.assertLess(sum(value["bytes"] for value in manifest["blobs"].values()), exported["source_bytes"])
                self.assertIsNone(archive.testzip())
            for name, value in originals.items():
                self.assertEqual(read_evidence(root / "export.zip", name), value)
            self.assertEqual(read_evidence(root / "export.zip", "worker.log"), b"complete log\n")
            with self.assertRaises(FileExistsError):
                export_evidence(source, root / "export.zip")

    def test_export_refuses_partial_tensor_files_and_external_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "reports"
            source.mkdir()
            bad = source / "snapshot.pt.tmp"
            bad.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "tensor"):
                export_evidence(source, root / "export.zip")
            bad.unlink()
            (root / "outside.json").write_text("{}")
            (source / "bad.json").symlink_to(root / "outside.json")
            with self.assertRaisesRegex(ValueError, "symlink"):
                export_evidence(source, root / "export.zip")

    def test_export_reader_rejects_corruption_and_bounded_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "reports"
            source.mkdir()
            (source / "report.json").write_text(json.dumps({"samples": list(range(3000))}))
            path = root / "export.zip"
            export_evidence(source, path)
            with self.assertRaisesRegex(ValueError, "limit|budget"):
                read_evidence(path, "report.json", max_bytes=1000)
            with zipfile.ZipFile(path) as archive:
                contents = {name: archive.read(name) for name in archive.namelist()}
            key = next(name for name in contents if name.startswith("blobs/json/"))
            contents[key] = b" " * len(contents[key])
            corrupted = root / "corrupt.zip"
            with zipfile.ZipFile(corrupted, "w") as archive:
                for name, data in contents.items():
                    archive.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "identity"):
                read_evidence(corrupted, "report.json")

    def test_export_source_mutation_fails_closed_and_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "reports"
            source.mkdir()
            report = source / "report.json"
            report.write_text('{"case": 1}')
            original = Path.read_bytes
            reads = []

            def read(path):
                data = original(path)
                if path.resolve() == report.resolve():
                    reads.append(1)
                    if len(reads) == 1:
                        path.write_text('{"case": 2}')
                return data

            with patch.object(Path, "read_bytes", read), self.assertRaisesRegex(RuntimeError, "changed"):
                export_evidence(source, root / "export.zip")
            self.assertFalse((root / "export.zip").exists())

            def add_tensor(path):
                data = original(path)
                if path.resolve() == report.resolve():
                    (source / "late-snapshot.pt").write_bytes(b"")
                return data

            with patch.object(Path, "read_bytes", add_tensor), self.assertRaisesRegex(RuntimeError, "membership"):
                export_evidence(source, root / "export.zip")
            self.assertFalse((root / "export.zip").exists())


if __name__ == "__main__":
    unittest.main()
