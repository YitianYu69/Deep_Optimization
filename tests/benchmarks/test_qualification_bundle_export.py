"""Candidate and historical-control exports keep separate immutable identities."""

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/compiler/export_qualification_bundle.py"
EXPORTER = runpy.run_path(str(SCRIPT), run_name="qualification_export_test")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _wheel(path, version, package, *, bad_record=False, omit_metadata=False):
    directory = f"deep_optimization-{version}.dist-info/"
    members = {**package,
        directory + "METADATA": f"Metadata-Version: 2.1\nName: deep-optimization\nVersion: {version}\n".encode(),
        directory + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    for name, payload in members.items():
        if omit_metadata and name.endswith("/METADATA"):
            continue
        hashed = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        writer.writerow((name, "sha256=" + ("wrong" if bad_record and name.startswith("deepopt/") else hashed), len(payload)))
    writer.writerow((directory + "RECORD", "", ""))
    members[directory + "RECORD"] = stream.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return _sha(path.read_bytes())


class _Fixture:
    def __init__(self, directory):
        self.root = Path(directory) / "project"
        self.root.mkdir()
        required = ("LICENSE", "AGENTS.md", "compiler_references/inductor/README.md")
        for name in required:
            self.write(name, b"fixture source\n")
        self.write("benchmarks/compiler/export_qualification_bundle.py", SCRIPT.read_bytes())
        self.write("README.md", b"current candidate readme\n")
        self.candidate_package = {"deepopt/__init__.py": b"release = 'candidate'\n",
                                  "deepopt/config.toml": b"enabled = false\n"}
        self.control_package = {"deepopt/__init__.py": b"release = 'historical'\n",
                                "deepopt/config.toml": b"enabled = true\n",
                                "deepopt/a.py": b"# module sharing a directory prefix\n",
                                "deepopt/a/value.txt": b"package data\n"}
        for name, payload in self.candidate_package.items():
            self.write("src/" + name, payload)
        self.production = self.root / "release-inputs/candidate-production.json"
        self.control_production = self.root / "release-inputs/control-production.json"
        self.benchmark = self.root / "release-inputs/benchmark.json"
        self.wheel = self.root / "wheels/deep_optimization-2.0+candidate-py3-none-any.whl"
        self.control_wheel = self.root / "wheels/deep_optimization-1.0+historical-py3-none-any.whl"
        self.wheel_sha = _wheel(self.wheel, "2.0+candidate", self.candidate_package)
        self.control_sha = _wheel(self.control_wheel, "1.0+historical", self.control_package)
        for path, version, package, readme in (
                (self.production, "2.0+candidate", self.candidate_package, b"current candidate readme\n"),
                (self.control_production, "1.0+historical", self.control_package, b"immutable old readme\n")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"version": version, "source_sha256": {
                **{"src/" + name: _sha(payload) for name, payload in package.items()}, "README.md": _sha(readme)}}))
        self.freeze_benchmark()

    def write(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def freeze_benchmark(self):
        self.benchmark.write_text(json.dumps({"source_sha256": {
            name: _sha(path.read_bytes()) for name, path in EXPORTER["sources"](self.root).items()}}))

    def kwargs(self):
        return {"control_wheel": self.control_wheel, "control_wheel_sha256": self.control_sha,
                "control_production_freeze": self.control_production}

    def plan(self, output, *, control=True, **overrides):
        return EXPORTER["plan"](self.root, output, self.wheel, self.wheel_sha,
            self.production, self.benchmark, **({**self.kwargs(), **overrides} if control else overrides))

    def optimization_sources(self):
        self.write("examples/vae/benchmark_optimization.py", b"# metadata-only fixture\n")
        self.write("scripts/run_h200_vae_optimization.sh", b"#!/usr/bin/env bash\n")
        self.write("docs/execution/backend_qualification.md", b"# Qualification\n")
        self.freeze_benchmark()

    def convolution_sources(self, *, modules=2):
        self.optimization_sources()
        frozen = json.loads(self.production.read_text())
        for name in EXPORTER["CONVOLUTION_MODULES"][:modules]:
            payload = b"# code-only convolution fixture; never import\n"
            self.write(name, payload)
            self.candidate_package[name.removeprefix("src/")] = payload
            frozen["source_sha256"][name] = _sha(payload)
        self.production.write_text(json.dumps(frozen))
        self.wheel_sha = _wheel(self.wheel, "2.0+candidate", self.candidate_package)
        if modules == 2:
            for name in EXPORTER["CONVOLUTION_GUIDES"]:
                self.write(name, b"{}\n" if name.endswith(".json") else b"# Convolution wave guide\n")
        self.freeze_benchmark()

    def fusion_sources(self, *, complete_config=True):
        self.convolution_sources()
        frozen = json.loads(self.production.read_text())
        files = {name: b"# code-only fusion fixture; never import\n" for name in EXPORTER["FUSION_MODULES"]}
        defaults = dict(EXPORTER["FUSION_CONFIG_DEFAULTS"])
        if not complete_config:
            defaults.pop("swiglu_fusion_stage")
        files["src/deepopt/compiler/backend.py"] = ("raise RuntimeError('candidate must never be imported by exporter')\n"
            "class BackendConfig:\n" + "".join(f"    {name}: {type(value).__name__} = {value!r}\n"
                for name, value in defaults.items())).encode()
        for name, payload in files.items():
            self.write(name, payload)
            self.candidate_package[name.removeprefix("src/")] = payload
            frozen["source_sha256"][name] = _sha(payload)
        self.production.write_text(json.dumps(frozen))
        self.wheel_sha = _wheel(self.wheel, "2.0+candidate", self.candidate_package)
        for name in EXPORTER["FUSION_GUIDES"]:
            self.write(name, b"{}\n" if name.endswith(".json") else b"# Fusion/parity guide\n")
        self.write("examples/vae/parity_execution.py", b"# code-only parity worker fixture\n")
        self.write("examples/vae/benchmark_parity.py", b"# code-only parity supervisor fixture\n")
        self.freeze_benchmark()


class QualificationBundleExportTests(unittest.TestCase):
    def test_legacy_positional_api_still_exports_one_wheel(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            local_evidence = ("compiler_references/inductor/index.json",
                "compiler_references/inductor/imports/index.json", "work/benchmark-result.json",
                "examples/vae/inductor_vae_v2_b16_224_local/output_code.py")
            for name in local_evidence:
                fixture.write(name, b"local capture\n")
            output = fixture.root / "legacy-delivery"
            planned = EXPORTER["plan"](fixture.root, output, fixture.wheel, fixture.wheel_sha,
                                       fixture.production, fixture.benchmark)
            self.assertNotIn("control", planned)
            self.assertEqual(EXPORTER["verify_wheel"](fixture.wheel, fixture.wheel_sha,
                fixture.production, fixture.root)["version"], "2.0+candidate")
            result = EXPORTER["export"](fixture.root, planned, fixture.production, fixture.benchmark)
            self.assertTrue(Path(result["archive"]).is_file())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotIn("control", manifest)
            for name in local_evidence:
                self.assertNotIn(name, manifest["files"])
            self.assertEqual(len(list((output / "packages").glob("*.whl"))), 1)

    def test_historical_control_uses_own_freeze_and_both_releases_are_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            fixture.optimization_sources()
            output = fixture.root / "paired-delivery"
            planned = fixture.plan(output)
            self.assertNotIn("experiment_profile", planned)
            control = planned["control"]
            self.assertEqual(control["version"], "1.0+historical")
            self.assertEqual(control["files_hashed"], 4)
            identity = hashlib.sha256()
            installed = fixture.root / "preserved-control/deepopt"
            for name, payload in fixture.control_package.items():
                fixture.write("preserved-control/" + name, payload)
            for path in sorted(installed.rglob("*")):
                if path.is_file():
                    identity.update(path.relative_to(installed).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
            self.assertEqual(control["package_source_sha256"], identity.hexdigest())
            self.assertEqual(control["verification"], "own_production_freeze_not_current_sources")
            self.assertEqual(control["production_freeze_sha256"], _sha(fixture.control_production.read_bytes()))
            # Export recovers the pinned control from the description; callers
            # can keep using the legacy export positional arguments.
            result = EXPORTER["export"](fixture.root, planned, fixture.production, fixture.benchmark)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["control"], control)
            self.assertEqual((output / control["bundle_wheel"]).read_bytes(), fixture.control_wheel.read_bytes())
            self.assertEqual((output / control["bundle_production_freeze"]).read_bytes(), fixture.control_production.read_bytes())
            self.assertEqual((output / "release/production-freeze.json").read_bytes(), fixture.production.read_bytes())
            for name, expected in manifest["files"].items():
                self.assertEqual(_sha((output / name).read_bytes()), expected)
            guide = (output / "RUN_VAE_OPTIMIZATION.md").read_text()
            self.assertIn("one A/B/C triage block: three fresh", guide)
            self.assertIn("--trials 3", guide)
            self.assertIn("nine fresh workers", guide)
            for name in ("examples/vae/benchmark_optimization.py", "scripts/run_h200_vae_optimization.sh",
                         "docs/execution/backend_qualification.md"):
                self.assertIn(name, manifest["files"])
            with zipfile.ZipFile(result["archive"]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertFalse(any(name.startswith(output.name + "/src/") for name in archive.namelist()))

    def test_convolution_profile_preserves_both_wheels_and_exports_new_guide_and_typed_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            fixture.convolution_sources()
            output = fixture.root / "convolution-delivery"
            planned = fixture.plan(output)
            self.assertEqual(planned["experiment_profile"], "native_convolution_wave2")
            result = EXPORTER["export"](fixture.root, planned, fixture.production, fixture.benchmark)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["experiment_profile"], "native_convolution_wave2")
            self.assertEqual(len(list((output / "packages").glob("*.whl"))), 2)
            self.assertEqual((output / "packages" / fixture.control_wheel.name).read_bytes(), fixture.control_wheel.read_bytes())
            self.assertEqual((output / "release/control-production-freeze.json").read_bytes(), fixture.control_production.read_bytes())
            for name in EXPORTER["CONVOLUTION_GUIDES"]:
                self.assertIn(name, manifest["files"])
                self.assertEqual((output / name).read_bytes(), (fixture.root / name).read_bytes())
            guide = (output / "RUN_VAE_OPTIMIZATION.md").read_text()
            for fragment in ("--passes conv-both", "--passes conv-order", "--passes conv-geometry", "--trials 3",
                    "previous norm-cat and donation controls remain disabled", *EXPORTER["CONVOLUTION_GUIDES"],
                    "docs/execution/backend_qualification.md", "fail on skipped tests"):
                self.assertIn(fragment, guide)
            for name, expected in manifest["files"].items():
                self.assertEqual(_sha((output / name).read_bytes()), expected)
            with zipfile.ZipFile(result["archive"]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertFalse(any(name.startswith(output.name + "/src/") for name in archive.namelist()))
            self.assertFalse(manifest["cuda_qualified"])
            self.assertEqual(manifest["tensor_files_included"], 0)

    def test_convolution_profile_requires_both_core_modules_and_all_required_guides(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            fixture.convolution_sources(modules=1)
            self.assertNotIn("experiment_profile", fixture.plan(fixture.root / "legacy-partial-core"))
        for missing in EXPORTER["CONVOLUTION_GUIDES"]:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(temporary)
                fixture.convolution_sources()
                (fixture.root / missing).unlink()
                with self.assertRaisesRegex(ValueError, "Convolution experiment.*guide"):
                    fixture.plan(fixture.root / "not-written")
                self.assertFalse((fixture.root / "not-written").exists())

    def test_fusion_profile_requires_modules_and_typed_config_and_exports_parity_runbooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            fixture.fusion_sources()
            output = fixture.root / "fusion-delivery"
            planned = fixture.plan(output)
            self.assertEqual(planned["experiment_profile"], "native_training_fusion_v3")
            result = EXPORTER["export"](fixture.root, planned, fixture.production, fixture.benchmark)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["experiment_profile"], "native_training_fusion_v3")
            for name in (*EXPORTER["CONVOLUTION_GUIDES"], *EXPORTER["FUSION_GUIDES"],
                         "examples/vae/parity_execution.py", "examples/vae/benchmark_parity.py"):
                self.assertEqual((output / name).read_bytes(), (fixture.root / name).read_bytes())
                self.assertIn(name, manifest["files"])
            guide = (output / "RUN_VAE_OPTIMIZATION.md").read_text()
            for fragment in ("--passes fusion-both --fusion-stage forward_backward", "--passes norm-activation",
                    "--passes swiglu", "--fusion-stage forward", "Every legacy memory/convolution option",
                    *EXPORTER["FUSION_GUIDES"], "compiler_references/inductor/README.md"):
                self.assertIn(fragment, guide)
            self.assertIn("compiler_references/inductor/README.md", (output / "RUN_VAE_PARITY.md").read_text())
            self.assertFalse(manifest["cuda_qualified"])
            for name, expected in manifest["files"].items():
                self.assertEqual(_sha((output / name).read_bytes()), expected)
            with zipfile.ZipFile(result["archive"]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertFalse(any(name.startswith(output.name + "/src/") for name in archive.namelist()))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            fixture.fusion_sources(complete_config=False)
            self.assertEqual(fixture.plan(fixture.root / "prior-profile")["experiment_profile"], "native_convolution_wave2")

    def test_fusion_profile_requires_all_new_guides_and_preserves_legacy_membership(self):
        for missing in EXPORTER["FUSION_GUIDES"]:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(temporary)
                fixture.fusion_sources()
                (fixture.root / missing).unlink()
                with self.assertRaisesRegex(ValueError, "Training fusion experiment.*guide"):
                    fixture.plan(fixture.root / "not-written")
                self.assertFalse((fixture.root / "not-written").exists())

    def test_control_arguments_are_all_or_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            for missing in fixture.kwargs():
                with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "supplied together"):
                    fixture.plan(fixture.root / "not-written", **{missing: None})
            self.assertFalse((fixture.root / "not-written").exists())

    def test_control_sha_record_coverage_and_freeze_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            output = fixture.root / "not-written"
            with self.assertRaisesRegex(ValueError, "SHA256"):
                fixture.plan(output, control_wheel_sha256="0" * 64)
            for record_error in ("hash", "coverage"):
                with self.subTest(record_error=record_error):
                    bad_sha = _wheel(fixture.control_wheel, "1.0+historical", fixture.control_package,
                                     bad_record=record_error == "hash", omit_metadata=record_error == "coverage")
                    with self.assertRaisesRegex(ValueError, "RECORD"):
                        fixture.plan(output, control_wheel_sha256=bad_sha)
            fixture.control_sha = _wheel(fixture.control_wheel, "1.0+historical", fixture.control_package)
            frozen = json.loads(fixture.control_production.read_text())
            frozen["source_sha256"]["src/deepopt/__init__.py"] = "0" * 64
            fixture.control_production.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, "production freeze"):
                fixture.plan(output)
            self.assertFalse(output.exists())

    def test_candidate_still_checks_current_source_bytes_and_membership(self):
        for change in ("bytes", "membership"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(temporary)
                fixture.write("src/deepopt/__init__.py" if change == "bytes" else "src/deepopt/new.py", b"changed\n")
                with self.assertRaisesRegex(ValueError, "Production source"):
                    fixture.plan(fixture.root / "not-written")
                self.assertFalse((fixture.root / "not-written").exists())

    def test_changed_source_or_control_freeze_after_planning_refuses_before_writing(self):
        for change in ("production", "benchmark", "control_freeze"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(temporary)
                output = fixture.root / "not-written"
                planned = fixture.plan(output)
                if change == "production":
                    fixture.write("src/deepopt/__init__.py", b"changed\n")
                elif change == "benchmark":
                    fixture.write("benchmarks/new_worker.py", b"changed\n")
                else:
                    frozen = json.loads(fixture.control_production.read_text())
                    frozen["scope"] = "changed metadata after planning"
                    fixture.control_production.write_text(json.dumps(frozen))
                with self.assertRaises(ValueError):
                    EXPORTER["export"](fixture.root, planned, fixture.production, fixture.benchmark)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_name(output.name + ".zip").exists())

    def test_optimization_worker_requires_its_launcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            fixture.write("examples/vae/benchmark_optimization.py", b"# pending launcher\n")
            with self.assertRaisesRegex(ValueError, "optimization.*launcher"):
                EXPORTER["sources"](fixture.root)

    def test_stdlib_cli_dry_run_accepts_control_without_gpu_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(temporary)
            output = fixture.root / "not-written"
            wrapper = """import runpy, sys
class NoGpuImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','triton','tilelang','deepopt'}:
            raise AssertionError('Exporter imported runtime: ' + fullname)
sys.meta_path.insert(0, NoGpuImports())
script, *arguments = sys.argv[1:]
namespace = runpy.run_path(script, run_name='stdlib_export')
raise SystemExit(namespace['main'](arguments))
"""
            command = [sys.executable, "-S", "-c", wrapper,
                str(fixture.root / "benchmarks/compiler/export_qualification_bundle.py"),
                "--output", str(output), "--wheel", str(fixture.wheel),
                "--expected-wheel-sha256", fixture.wheel_sha, "--production-freeze", str(fixture.production),
                "--benchmark-freeze", str(fixture.benchmark), "--control-wheel", str(fixture.control_wheel),
                "--control-wheel-sha256", fixture.control_sha,
                "--control-production-freeze", str(fixture.control_production), "--dry-run"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["control"]["wheel_sha256"], fixture.control_sha)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".zip").exists())


if __name__ == "__main__":
    unittest.main()
