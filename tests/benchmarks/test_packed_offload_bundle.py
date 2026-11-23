"""The offload release profile does not replace existing optimization controls."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tests.benchmarks.test_qualification_bundle_export import EXPORTER, _Fixture, _sha, _wheel


class PackedOffloadBundleTests(unittest.TestCase):
    def fixture(self, directory):
        value = _Fixture(directory)
        value.fusion_sources()
        for name in EXPORTER["OFFLOAD_MODULES"]:
            content = (b"class PackedActivationOffloadPolicy:\n    enabled: bool = False\n"
                if name.endswith("/_packed_offload.py") else b"# source-only offload fixture\n")
            value.write(name, content)
            value.candidate_package[name.removeprefix("src/")] = content
        for name in EXPORTER["OFFLOAD_GUIDES"]:
            value.write(name, b"{}\n" if name.endswith(".json") else b"# transport guide\n")
        value.write("examples/vae/benchmark_packed_offload.py", b"# stdlib planner fixture\n")
        value.write("scripts/run_h200_vae_packed_offload.sh", b"#!/usr/bin/env bash\n")
        value.wheel_sha = _wheel(value.wheel, "2.0+candidate", value.candidate_package)
        frozen = json.loads(value.production.read_text())
        frozen["source_sha256"].update({"src/" + name: _sha(data) for name, data in value.candidate_package.items()})
        value.production.write_text(json.dumps(frozen))
        value.freeze_benchmark()
        return value

    def test_independent_profile_exports_same_candidate_guide_without_changing_eight_options(self):
        from examples.vae.benchmark_optimization import OPTION_DEFAULTS

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            description = fixture.plan(Path(directory) / "bundle")
            self.assertEqual(description["experiment_profile"], "native_training_fusion_v3")
            self.assertEqual(description["packed_offload_profile"], "packed_offload_v1")
            self.assertNotIn("packed_activation_offload", OPTION_DEFAULTS)
            self.assertIn("scripts/run_h200_vae_packed_offload.sh", description["source_sha256"])
            EXPORTER["export"](fixture.root, description, fixture.production, fixture.benchmark)
            output = Path(description["output"])
            self.assertTrue((output / "RUN_PACKED_ACTIVATION_OFFLOAD.md").is_file())
            self.assertIn("positive integer budgets", (output / "RUN_PACKED_ACTIVATION_OFFLOAD.md").read_text())
            self.assertFalse(json.loads((output / "manifest.json").read_text())["cuda_qualified"])
            with zipfile.ZipFile(description["archive"]) as archive:
                self.assertTrue(any(name.endswith("/RUN_PACKED_ACTIVATION_OFFLOAD.md") for name in archive.namelist()))
                self.assertFalse(any("/src/deepopt/" in name for name in archive.namelist()))

    def test_partial_runtime_missing_guide_or_enabled_production_default_is_rejected(self):
        for part in ("runtime", "guide", "default"):
            with self.subTest(part=part), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                if part == "runtime":
                    (fixture.root / EXPORTER["OFFLOAD_MODULES"][-1]).unlink()
                elif part == "guide":
                    (fixture.root / EXPORTER["OFFLOAD_GUIDES"][-1]).unlink()
                else:
                    fixture.write("src/deepopt/_packed_offload.py", b"class PackedActivationOffloadPolicy:\n    enabled: bool = True\n")
                with self.assertRaises(ValueError):
                    EXPORTER["sources"](fixture.root)

    def memory_v2_fixture(self, directory):
        fixture = self.fixture(directory)
        for name in EXPORTER["MEMORY_V2_MODULES"]:
            fixture.write(name, b"# metadata-only release fixture\n")
            fixture.candidate_package[name.removeprefix("src/")] = b"# metadata-only release fixture\n"
        for name in EXPORTER["MEMORY_V2_GUIDES"]:
            fixture.write(name, b"{}\n" if name.endswith(".json") else b"# v2 guide or launcher\n")
        fixture.wheel_sha = _wheel(fixture.wheel, "2.0+candidate", fixture.candidate_package)
        frozen = json.loads(fixture.production.read_text())
        frozen["source_sha256"].update({"src/" + name: _sha(data) for name, data in fixture.candidate_package.items()})
        fixture.production.write_text(json.dumps(frozen))
        fixture.freeze_benchmark()
        return fixture

    def test_v2_bundle_pins_full_candidate_and_rejects_changed_helper_before_install(self):
        from benchmarks.compiler.run_packed_memory_experiments import candidate_spec

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.memory_v2_fixture(directory)
            description = fixture.plan(Path(directory) / "bundle")
            self.assertEqual(description["packed_memory_profile"], "packed_memory_v2")
            EXPORTER["export"](fixture.root, description, fixture.production, fixture.benchmark)
            output = Path(description["output"])
            candidate = candidate_spec(output, output / "work/candidate")
            self.assertEqual(candidate["expected_installed"]["version"], "2.0+candidate")
            self.assertEqual(candidate["expected_installed"]["files_hashed"], len(fixture.candidate_package))
            self.assertTrue((output / "RUN_PACKED_MEMORY_V2.md").is_file())
            helper = output / "examples/vae/benchmark_packed_offload.py"
            helper.write_text("# changed after freeze\n")
            with self.assertRaisesRegex(ValueError, "Bundle helper/source changed"):
                candidate_spec(output, output / "work/other-candidate")

    def test_partial_v2_modules_and_missing_v2_guide_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            fixture.write(EXPORTER["MEMORY_V2_MODULES"][0], b"# partial v2\n")
            with self.assertRaisesRegex(ValueError, "requires all"):
                EXPORTER["sources"](fixture.root)
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.memory_v2_fixture(directory)
            (fixture.root / EXPORTER["MEMORY_V2_GUIDES"][0]).unlink()
            with self.assertRaisesRegex(ValueError, "requires matching guide"):
                EXPORTER["sources"](fixture.root)


if __name__ == "__main__":
    unittest.main()
