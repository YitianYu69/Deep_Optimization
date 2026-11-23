"""Runtime report exports must preserve the generated code used in analysis."""

from pathlib import Path
import tempfile
import unittest

from benchmarks.compiler.export_evidence import export_evidence, read_evidence


class InductorEvidenceExportTests(unittest.TestCase):
    def test_generated_code_ir_and_assembly_round_trip_with_runtime_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "benchmark-output"
            code = source / "cases/native/inductor/runs/reference/artifacts/compiler"
            code.mkdir(parents=True)
            (source / "comparison.json").write_text('{"native_inductor_wall_ms": 1.25}')
            files = {"output_code.py": b"# generated native TorchInductor wrapper\n",
                     "kernel.ttir": b"module { }\n", "kernel.ptx": b".version 8.0\n",
                     "kernel.cubin": b"\x7fELF\x00compiled-kernel", "ir_post_fusion.txt": b"scheduler nodes\n"}
            for name, data in files.items():
                (code / name).write_bytes(data)
            archive = root / "evidence.zip"
            report = export_evidence(source, archive)
            self.assertEqual(report["logical_files"], len(files) + 1)
            self.assertEqual(read_evidence(archive, "comparison.json"), {"native_inductor_wall_ms": 1.25})
            for name, data in files.items():
                self.assertEqual(read_evidence(archive, (code / name).relative_to(source).as_posix()), data)

    def test_tensor_or_pickle_dumps_cannot_enter_analysis_export(self):
        for name in ("state.pt.tmp", "graph.pkl", "weights.safetensors", "data.bin"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                source = root / "output"
                source.mkdir()
                (source / name).write_bytes(b"forbidden")
                with self.assertRaisesRegex(ValueError, "tensor artifacts"):
                    export_evidence(source, root / "evidence.zip")


if __name__ == "__main__":
    unittest.main()
