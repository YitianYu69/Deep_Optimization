"""Contract qualification must retain failures, skips and source identity."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/compiler/qualification_tests.py"


class BackendContractRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.runner = self.root / "benchmarks/compiler/qualification_tests.py"
        self.runner.parent.mkdir(parents=True)
        shutil.copyfile(RUNNER, self.runner)
        # The runner's result accounting is tested without site-packages.
        # A narrow preflight double lets these fixtures stay independent of
        # the machine's installed Torch/DeepOpt versions. Runtime provenance
        # rejection is covered separately by test_installed_runtime.
        self.runtime = self.runner.with_name("installed_runtime.py")
        self.runtime.write_text(
            "import sys\n"
            "def prepare_imports(root):\n"
            "    sys.path.insert(0, str(root))\n"
            "def installed_identity(root):\n"
            "    return {'distribution': 'fixture-only', 'package_paths': []}\n"
        )

    def run_fixture(self, body, *flags, module="contract_fixture"):
        (self.root / "contract_fixture.py").write_text("import unittest\n" + body)
        output = self.root / "contracts.json"
        result = subprocess.run([sys.executable, "-S", str(self.runner), "--modules", module,
                                 "--output", str(output), *flags], cwd=self.root,
                                env={**os.environ, "PYTHONPATH": str(self.root)},
                                text=True, capture_output=True, timeout=20)
        self.assertTrue(output.exists(), result.stdout + result.stderr)
        return result, json.loads(output.read_text())

    def test_help_and_module_import_need_only_standard_library(self):
        result = subprocess.run([sys.executable, "-S", str(self.runner), "--help"],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-S", "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1],run_name='contract_runner'); "
            "assert not {'torch','triton','deepopt'} & sys.modules.keys()", str(self.runner)],
            text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_success_records_per_test_timing_and_source_hashes(self):
        result, report = self.run_fixture("class Contracts(unittest.TestCase):\n"
            "    def test_ok(self): self.assertEqual(2 + 2, 4)\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["passed"] and report["qualified"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["counts"]["executed"], 1)
        row, = report["tests"]
        self.assertEqual(row["status"], "passed")
        self.assertGreaterEqual(row["duration_seconds"], 0)
        self.assertTrue(report["source_identity"]["unchanged"])
        self.assertEqual(len(report["source_identity"]["before_digest"]), 64)
        self.assertIn("contract_fixture", report["imported_sources"])
        self.assertEqual(len(report["imported_sources"]["contract_fixture"]["sha256"]), 64)

    def test_missing_module_is_a_recorded_error(self):
        result, report = self.run_fixture("", module="missing_qualification_module")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report["passed"])
        self.assertEqual(report["counts"]["errors"], 1)
        self.assertIn("ModuleNotFoundError", report["tests"][0]["traceback"])

    def test_skips_are_explicit_and_strict_mode_fails(self):
        body = ("class Contracts(unittest.TestCase):\n"
                "    def test_ok(self): pass\n"
                "    @unittest.skip('backend unavailable')\n"
                "    def test_skip(self): pass\n")
        for flags, accepted in (((), True), (("--fail-on-skip",), False)):
            with self.subTest(flags=flags):
                result, report = self.run_fixture(body, *flags)
                self.assertEqual(result.returncode == 0, accepted)
                self.assertEqual(report["passed"], accepted)
                self.assertFalse(report["qualified"])
                self.assertEqual(report["counts"]["skipped"], 1)
                skipped, = [row for row in report["tests"] if row["status"] == "skipped"]
                self.assertEqual(skipped["reason"], "backend unavailable")

    def test_empty_and_entirely_skipped_suites_cannot_pass(self):
        bodies = ("", "class Contracts(unittest.TestCase):\n"
                  "    @unittest.skip('no backend')\n"
                  "    def test_skip(self): pass\n")
        for body in bodies:
            with self.subTest(body=body):
                result, report = self.run_fixture(body)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(report["status"], "no_tests")
                self.assertEqual(report["counts"]["executed"], 0)
                self.assertFalse(report["passed"])

    def test_failures_errors_and_subtest_details_survive_json(self):
        body = ("class Contracts(unittest.TestCase):\n"
                "    def test_fail(self): self.fail('incorrect derivative')\n"
                "    def test_error(self): raise RuntimeError('missing saved state')\n"
                "    def test_subtest(self):\n"
                "        with self.subTest(shape=17): self.assertEqual(1, 2)\n"
                "    def test_subtest_skip(self):\n"
                "        with self.subTest(backend='optional'): self.skipTest('backend unavailable')\n")
        result, report = self.run_fixture(body)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["counts"]["failures"], 2)
        self.assertEqual(report["counts"]["errors"], 1)
        self.assertEqual(report["counts"]["skipped"], 1)
        rows = {row["id"].rsplit(".", 1)[-1]: row for row in report["tests"]}
        self.assertIn("incorrect derivative", rows["test_fail"]["traceback"])
        self.assertIn("missing saved state", rows["test_error"]["traceback"])
        self.assertIn("shape=17", rows["test_subtest"]["subtests"][0]["id"])
        self.assertEqual(rows["test_subtest_skip"]["subtests"][0]["reason"], "backend unavailable")

    def test_fixture_errors_and_skips_do_not_break_recording(self):
        for error, expected in (("raise RuntimeError('fixture failed')", "error"),
                                ("raise unittest.SkipTest('fixture unavailable')", "skipped")):
            with self.subTest(expected=expected):
                result, report = self.run_fixture("class Contracts(unittest.TestCase):\n"
                    "    @classmethod\n"
                    f"    def setUpClass(cls): {error}\n"
                    "    def test_not_reached(self): pass\n")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(report["tests"][0]["status"], expected)
                self.assertEqual(report["tests"][0]["kind"], "fixture")
                self.assertEqual(report["counts"]["executed"], 0)

    def test_preflight_failure_has_json_without_loading_the_test_module(self):
        result, report = self.run_fixture("raise RuntimeError('must not load tests')\n",
                                         "--require-cuda", "--device", "cuda:0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["status"], "preflight_failed")
        self.assertIn("torch", report["error"])
        self.assertNotIn("must not load tests", report["error"])
        self.assertEqual(report["tests"], [])

    def test_cpu_contracts_require_installed_distribution_before_loading_tests(self):
        self.runtime.write_text(
            "def prepare_imports(root): pass\n"
            "def installed_identity(root):\n"
            "    raise RuntimeError('DeepOpt must be installed in this interpreter')\n"
        )
        result, report = self.run_fixture("raise RuntimeError('must not load tests')\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["status"], "preflight_failed")
        self.assertIn("must be installed", report["error"])
        self.assertNotIn("must not load tests", report["error"])
        self.assertEqual(report["tests"], [])

    def test_source_mutation_invalidates_an_otherwise_passing_suite(self):
        result, report = self.run_fixture("from pathlib import Path\n"
            "class Contracts(unittest.TestCase):\n"
            "    def test_edits_source(self):\n"
            "        p = Path(__file__); p.write_text(p.read_text() + '# changed\\n')\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["tests"][0]["status"], "passed")
        self.assertFalse(report["passed"])
        self.assertFalse(report["source_identity"]["unchanged"])
        self.assertIn(str(self.root / "contract_fixture.py"), report["source_identity"]["changed_paths"])

    def test_method_selector_hashes_its_module_without_importing_before_preflight(self):
        result, report = self.run_fixture("class Contracts(unittest.TestCase):\n"
            "    def test_ok(self): pass\n", module="contract_fixture.Contracts.test_ok")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.root / "contract_fixture.py"), report["source_identity"]["before"])

    def test_cuda_device_selection_and_fp8_capability_are_explicit(self):
        spec = importlib.util.spec_from_file_location("backend_contract_runner", RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        selected = []
        cuda = SimpleNamespace(is_available=lambda: True, set_device=selected.append,
            get_device_capability=lambda: (8, 0), get_device_name=lambda: "test GPU",
            is_bf16_supported=lambda: True)
        torch = SimpleNamespace(__version__="test", __file__="torch.py", version=SimpleNamespace(cuda="test"),
                                cuda=cuda, device=lambda text: SimpleNamespace(type=text.split(":")[0], index=2))
        args = module.parser().parse_args(["--modules", "fixture", "--output", "out.json",
                                          "--require-cuda", "--device", "cuda:2"])
        helper = SimpleNamespace(installed_identity=lambda root: {"distribution": "test-installed"})
        with patch.object(module, "_installed_runtime", return_value=helper), \
                patch.object(module.importlib, "import_module", return_value=torch):
            metadata = module._preflight(args)
            self.assertEqual(metadata["capability"], [8, 0])
            self.assertEqual(selected[0].index, 2)
            args.require_fp8 = True
            with self.assertRaisesRegex(RuntimeError, "capability >= 9"):
                module._preflight(args)
            args.require_fp8 = False
            args.device = "cpu"
            with self.assertRaisesRegex(ValueError, "--device cuda:N"):
                module._preflight(args)


if __name__ == "__main__":
    unittest.main()
