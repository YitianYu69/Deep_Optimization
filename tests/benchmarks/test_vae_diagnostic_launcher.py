"""Launcher control flow only; fake processes cannot establish CUDA correctness."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SHIM = '''#!@PYTHON@
from pathlib import Path
import json, os, subprocess, sys
args = sys.argv[1:]
if args[:2] == ["-m", "pip"]:
    raise SystemExit(0)
if "benchmarks/compiler/qualification_tests.py" in args:
    Path("preflight-started").touch()
    print("simulated CUDA contract output", flush=True)
    Path("preflight-request.json").write_text(json.dumps({"arguments": args,
        "workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "repro_after_present": "TORCHDYNAMO_REPRO_AFTER" in os.environ}))
    passed = os.environ["SIMULATED_PREFLIGHT"] == "pass"
    Path(args[args.index("--output") + 1]).write_text(json.dumps({"passed": passed}))
    raise SystemExit(0 if passed else 1)
if "examples/vae/benchmark_vae.py" in args:
    Path("matrix-started").touch()
    Path(args[args.index("--output") + 1]).write_text(json.dumps({"status": "passed"}))
    raise SystemExit(0)
if args and args[0] == "-":
    source = sys.stdin.read()
    if "torch.cuda.is_available" in source:
        raise SystemExit(0)
    raise SystemExit(subprocess.run([@PYTHON_LITERAL@, *args], input=source, text=True).returncode)
os.execv(@PYTHON_LITERAL@, [@PYTHON_LITERAL@, *args])
'''


class VAEDiagnosticLauncherTests(unittest.TestCase):
    def run_launcher(self, temporary, passed):
        root = Path(temporary)
        (root / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts/run_h200_vae_diagnostics.sh", root / "scripts/run_h200_vae_diagnostics.sh")
        (root / "packages").mkdir()
        wheel = root / "packages/fake.whl"
        wheel.write_bytes(b"launcher fixture; never installed")
        (root / "manifest.json").write_text(json.dumps({"wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}))
        previous = root / "previous.json"
        previous.write_text("{}")
        binary = root / "bin"
        binary.mkdir()
        shim = binary / "python"
        shim.write_text(SHIM.replace("@PYTHON@", sys.executable).replace("@PYTHON_LITERAL@", repr(sys.executable)))
        shim.chmod(0o755)
        environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                           SIMULATED_PREFLIGHT="pass" if passed else "fail")
        result = subprocess.run(["bash", "scripts/run_h200_vae_diagnostics.sh", str(previous)],
                                cwd=root, env=environment, capture_output=True, text=True, timeout=30)
        self.assertTrue((root / "preflight-started").exists(), result.stderr)
        preflight_request = json.loads((root / "preflight-request.json").read_text())
        for module in ("tests.compiler.test_native_activation_saved_abi.NativeActivationSavedABICudaTests",
                       "tests.opt.test_padding.MetadataPaddingStrictZeroCudaTests",
                       "tests.compiler.test_native_padding.NativePaddingStrictZeroCudaTests",
                       "tests.compiler.test_native_deterministic_padding_capture.NativeDeterministicPaddingCaptureCudaTests"):
            self.assertIn(module, preflight_request["arguments"])
        self.assertEqual(preflight_request["workspace"], ":4096:8")
        self.assertFalse(preflight_request["repro_after_present"])
        self.assertEqual((root / "matrix-started").exists(), passed, result.stdout + result.stderr)
        archives = list((root / "work").glob("*.zip"))
        self.assertEqual(len(archives), 1, result.stdout + result.stderr)
        import zipfile

        with zipfile.ZipFile(archives[0]) as bundle:
            names = bundle.namelist()
            preflight = next(name for name in names if name.endswith("-saved-abi.json"))
            main = next(name for name in names if name != preflight)
            self.assertEqual(json.loads(bundle.read(preflight))["passed"], passed)
            self.assertEqual(json.loads(bundle.read(main))["status"], "passed" if passed else "blocked")
            preflight_log = next(name for name in names if name.endswith("-preflight.log"))
            self.assertIn("simulated CUDA contract output", bundle.read(preflight_log).decode())
        return result

    def test_failed_cuda_gate_preserves_archive_and_does_not_start_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_launcher(temporary, False)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_successful_cuda_gate_runs_matrix_and_archives_both_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_launcher(temporary, True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
