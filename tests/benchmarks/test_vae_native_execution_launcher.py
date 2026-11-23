"""Execution/paired benchmark launcher wiring; fake processes are not CUDA evidence."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SHIM = '''#!@PYTHON@
from pathlib import Path
import json, os, subprocess, sys
args = sys.argv[1:]
if args[:2] == ["-m", "pip"]:
    raise SystemExit(0)
if any(name in args for name in ("examples/vae/native_execution.py", "examples/vae/benchmark_execution.py")):
    Path("request.json").write_text(json.dumps({"arguments": args,
        "repro_after_present": "TORCHDYNAMO_REPRO_AFTER" in os.environ,
        "workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}))
    passed = os.environ["SIMULATED_EXECUTION"] == "pass"
    output = Path(args[args.index("--output") + 1])
    output.write_text(json.dumps({"execution_completed": passed,
        "status": "passed" if passed else "failed", "correctness_assessed": False}))
    artifacts = output.parent / (output.stem + "-artifacts")
    artifacts.mkdir()
    (artifacts / "compiled_forward.py").write_text("# code-only fixture\\n")
    print("simulated native execution output", flush=True)
    raise SystemExit(0 if passed else 1)
if args and args[0] == "-":
    source = sys.stdin.read()
    if "torch.cuda.is_available" in source:
        raise SystemExit(0)
    raise SystemExit(subprocess.run([@PYTHON_LITERAL@, *args], input=source, text=True).returncode)
os.execv(@PYTHON_LITERAL@, [@PYTHON_LITERAL@, *args])
'''


class VAENativeExecutionLauncherTests(unittest.TestCase):
    def check_launcher(self, passed, *, paired=False, candidate=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            name = "run_h200_vae_benchmark.sh" if paired else "run_h200_vae_native128.sh"
            shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
            (root / "packages").mkdir()
            wheel = root / "packages/candidate.whl"
            if candidate:
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("deepopt/__init__.py", "# fake package; no CUDA execution\n")
                    archive.writestr("deep_optimization-0.1.1.dist-info/METADATA",
                        "Name: deep-optimization\nVersion: 0.1.1+launcher.fixture\n")
            else:
                wheel.write_bytes(b"never installed: launcher fixture")
            manifest = {"wheel": wheel.name, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
            if candidate:
                manifest["version"] = "0.1.1+launcher.fixture"
            (root / "manifest.json").write_text(json.dumps(manifest))
            previous = root / "original.json"
            previous.write_text("{}")
            binary = root / "bin"
            binary.mkdir()
            shim = binary / "python"
            shim.write_text(SHIM.replace("@PYTHON@", sys.executable).replace("@PYTHON_LITERAL@", repr(sys.executable)))
            shim.chmod(0o755)
            environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                SIMULATED_EXECUTION="pass" if passed else "fail", TORCHDYNAMO_REPRO_AFTER="invalid")
            environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
            completed = subprocess.run(["bash", "scripts/" + name, str(previous)],
                cwd=root, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0 if passed else 1, completed.stdout + completed.stderr)
            request = json.loads((root / "request.json").read_text())
            arguments = request["arguments"]
            self.assertEqual(arguments[arguments.index("--from-result") + 1], str(previous.resolve()))
            self.assertEqual(arguments[arguments.index("--batch-size") + 1], "128")
            self.assertEqual(arguments[arguments.index("--device") + 1], "cuda:0")
            self.assertEqual(arguments[arguments.index("--warmup") + 1], "5" if paired else "1")
            self.assertEqual(arguments[arguments.index("--updates") + 1], "10" if paired else "2")
            if paired:
                self.assertEqual(arguments[arguments.index("--trials") + 1], "1")
                self.assertIn("examples/vae/benchmark_execution.py", arguments)
            if candidate:
                self.assertIn("--arm-manifest", arguments)
                pin_path = Path(arguments[arguments.index("--arm-manifest") + 1])
                pin = json.loads((pin_path if pin_path.is_absolute() else root / pin_path).read_text())
                self.assertEqual(pin["expected_installed"]["version"], manifest["version"])
                self.assertEqual(pin["expected_installed"]["files_hashed"], 1)
                digest = hashlib.sha256()
                digest.update(b"__init__.py\0# fake package; no CUDA execution\n\0")
                self.assertEqual(pin["expected_installed"]["source_sha256"], digest.hexdigest())
                self.assertEqual(pin["wheel_sha256"], manifest["wheel_sha256"])
                self.assertIsNone(pin["native_options_sha256"])
            else:
                self.assertNotIn("--arm-manifest", arguments)
            self.assertNotIn("--diagnose", arguments)
            self.assertNotIn("--qualify", arguments)
            self.assertNotIn("--synthetic", arguments)
            self.assertFalse(request["repro_after_present"])
            self.assertIsNone(request["workspace"])
            archives = list((root / "work").glob("*.zip"))
            self.assertEqual(len(archives), 1)
            with zipfile.ZipFile(archives[0]) as archive:
                names = archive.namelist()
                report = next(name for name in names if name.endswith(".json"))
                self.assertEqual(json.loads(archive.read(report))["execution_completed"], passed)
                self.assertIn("compiled_forward.py", " ".join(names))
                if candidate:
                    self.assertTrue(any(name.endswith("-arm.json") for name in names))
                log = next(name for name in names if name.endswith(".log"))
                self.assertIn("simulated native execution output", archive.read(log).decode())

    def test_successful_execution_is_archived(self):
        self.check_launcher(True)

    def test_execution_failure_survives_log_pipeline_and_is_archived(self):
        self.check_launcher(False)

    def test_paired_benchmark_uses_original128_warmup5_updates10_and_is_archived(self):
        self.check_launcher(True, paired=True)

    def test_paired_benchmark_failure_is_archived_with_original_exit_code(self):
        self.check_launcher(False, paired=True)

    def test_candidate_paired_benchmark_pins_the_wheel_version_and_package_hash(self):
        self.check_launcher(True, paired=True, candidate=True)

    def test_candidate_pin_does_not_mask_paired_worker_failure(self):
        self.check_launcher(False, paired=True, candidate=True)

    def test_candidate_native128_launcher_pins_its_unique_wheel(self):
        self.check_launcher(True, candidate=True)

    def test_paired_launcher_rejects_untracked_output_override(self):
        for option in (("--output", "/tmp/override.json"), ("--output=/tmp/override.json",)):
            completed = subprocess.run(["bash", str(ROOT / "scripts/run_h200_vae_benchmark.sh"),
                "/nonexistent/original.json", *option], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("direct Python runner", completed.stderr)


if __name__ == "__main__":
    unittest.main()
