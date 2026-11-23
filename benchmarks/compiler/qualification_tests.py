"""Test the installed DeepOpt distribution and publish source-identified results.

This entry point imports only the standard library until actual execution, so
``python qualification_tests.py --help`` also works before installing PyTorch.
Use fully qualified unittest module, class or method names with ``--modules``.
"""

import argparse
import hashlib
import importlib
from importlib.machinery import PathFinder
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
import unittest


ROOT = Path(__file__).resolve().parents[2]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--modules", nargs="+", required=True,
                        help="Fully qualified unittest modules, classes or methods")
    result.add_argument("--output", type=Path, required=True, help="Machine-readable JSON result")
    result.add_argument("--device", default="cpu", help="Selected device; CUDA requirements need cuda:N")
    result.add_argument("--fail-on-skip", action="store_true", help="Any skipped test fails qualification")
    result.add_argument("--require-cuda", action="store_true", help="Require CUDA with BF16 capability >= 8")
    result.add_argument("--require-triton", action="store_true", help="Require CUDA/BF16 and an importable Triton")
    result.add_argument("--require-fp8", action="store_true", help="Require CUDA compute capability >= 9")
    return result


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _module_source(name):
    # Method/class names are legal unittest selectors. Find their containing
    # module without assuming every dotted component is an importable package.
    pieces = name.split(".")
    search = None
    source = None
    for index in range(len(pieces)):
        try:
            spec = PathFinder.find_spec(".".join(pieces[:index + 1]), search)
        except (ImportError, AttributeError, ValueError):
            spec = None
        if spec is None:
            break
        if spec is not None and spec.origin and spec.origin.endswith(".py"):
            source = Path(spec.origin).resolve()
        search = spec.submodule_search_locations
        if search is None:
            break
    return source


def _source_identity(modules):
    paths = {Path(__file__).resolve()}
    folders = [ROOT / "tests", ROOT / "benchmarks", ROOT / "examples"]
    try:
        specification = importlib.util.find_spec("deepopt")
    except (ImportError, AttributeError, ValueError):
        specification = None
    if specification is not None:
        folders.extend(Path(folder) for folder in specification.submodule_search_locations or ())
    for folder in folders:
        if folder.is_dir():
            paths.update(path.resolve() for path in folder.rglob("*")
                         if path.is_file() and path.suffix in (".py", ".toml")
                         and "__pycache__" not in path.parts)
    for module in modules:
        source = _module_source(module)
        if source is not None:
            paths.add(source)
    result = {}
    for path in sorted(paths):
        try:
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            result[str(path)] = {"error": f"{type(error).__name__}: {error}"}
    return result


def _digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _installed_runtime():
    # Keep module import and --help independent of DeepOpt, Torch and Triton.
    # Loading by sibling path also supports a standalone extracted bundle.
    path = Path(__file__).with_name("installed_runtime.py")
    specification = importlib.util.spec_from_file_location("qualification_installed_runtime", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _preflight(arguments, environment=None):
    environment = {} if environment is None else environment
    environment.update(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       working_directory=str(Path.cwd()), root=str(ROOT), device=arguments.device)
    environment["deepopt"] = _installed_runtime().installed_identity(ROOT)
    if not (arguments.require_cuda or arguments.require_triton or arguments.require_fp8):
        return environment
    torch = importlib.import_module("torch")
    environment.update(torch=torch.__version__, torch_path=torch.__file__, cuda=torch.version.cuda)
    device = torch.device(arguments.device)
    if device.type != "cuda":
        raise ValueError(f"CUDA qualification requires --device cuda:N; got {arguments.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    torch.cuda.set_device(device)
    capability = tuple(torch.cuda.get_device_capability())
    environment.update(gpu=torch.cuda.get_device_name(), capability=list(capability))
    if capability < (8, 0) or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 qualification requires CUDA compute capability >= 8; got {capability}")
    if arguments.require_fp8 and capability < (9, 0):
        raise RuntimeError(f"FP8 qualification requires CUDA compute capability >= 9; got {capability}")
    if arguments.require_triton:
        triton = importlib.import_module("triton")
        environment.update(triton=getattr(triton, "__version__", "unknown"), triton_path=triton.__file__)
    return environment


def _runtime_environment(environment):
    # CPU contracts may themselves import Torch. Record that actual runtime
    # without introducing a dependency for standard-library-only test suites.
    for name in ("torch", "triton"):
        module = sys.modules.get(name)
        if module is not None:
            environment[name] = str(getattr(module, "__version__", "unknown"))
            environment[name + "_path"] = str(getattr(module, "__file__", "unknown"))
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "version"):
        environment["cuda"] = getattr(torch.version, "cuda", None)


class RecordingResult(unittest.TextTestResult):
    """Keep actual unittest outcomes, including failures inside subtests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.current = None
        self.started = None

    def startTest(self, test):
        super().startTest(test)
        self.started = time.perf_counter()
        self.current = {"id": test.id(), "status": "running", "duration_seconds": 0., "subtests": []}
        self.records.append(self.current)

    def stopTest(self, test):
        self.current["duration_seconds"] = time.perf_counter() - self.started
        if self.current["status"] == "running":
            self.current["status"] = "error"
            self.current["reason"] = "Test completed without a recognized unittest outcome"
        super().stopTest(test)
        self.current = None

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.current["status"] == "running":
            self.current["status"] = "passed"

    def _problem(self, status, test, error):
        record = self.current
        if record is None:
            record = {"id": test.id(), "duration_seconds": 0., "subtests": [], "kind": "fixture"}
            self.records.append(record)
        record.update(status=status, traceback=self._exc_info_to_string(error, test))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._problem("failed", test, err)

    def addError(self, test, err):
        super().addError(test, err)
        self._problem("error", test, err)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if isinstance(test, unittest.case._SubTest):
            self.current["subtests"].append({"id": test.id(), "status": "skipped", "reason": reason})
        elif self.current is None:
            self.records.append({"id": test.id(), "status": "skipped", "reason": reason,
                                 "duration_seconds": 0., "subtests": [], "kind": "fixture"})
        else:
            self.current.update(status="skipped", reason=reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._problem("expected_failure", test, err)

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.current.update(status="unexpected_success", reason="An expected-failure test unexpectedly passed")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            status = "failed" if issubclass(err[0], test.failureException) else "error"
            detail = {"id": subtest.id(), "status": status, "traceback": self._exc_info_to_string(err, subtest)}
            self.current["subtests"].append(detail)
            self.current["status"] = status


def _imported_sources(identity):
    result = {}
    for name, module in tuple(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if not location or not str(location).endswith(".py"):
            continue
        location = str(Path(location).resolve())
        if location in identity or name == "deepopt" or name.startswith("deepopt."):
            result[name] = {"path": location, "sha256": identity.get(location)}
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    started = time.perf_counter()
    report = {"schema_version": 1, "modules": arguments.modules, "passed": False, "qualified": False,
              "status": "failed", "device": arguments.device, "environment": {},
              "requirements": {name: getattr(arguments, name) for name in
                  ("require_cuda", "require_triton", "require_fp8", "fail_on_skip")}, "tests": [],
              "counts": {"loaded": 0, "total": 0, "executed": 0, "passed": 0, "skipped": 0,
                         "failures": 0, "errors": 0, "expected_failures": 0, "unexpected_successes": 0}}
    before = {}
    exit_code = 1
    try:
        try:
            _installed_runtime().prepare_imports(ROOT)
            _preflight(arguments, report["environment"])
        except Exception as error:
            report.update(status="preflight_failed", error=f"{type(error).__name__}: {error}",
                          traceback=traceback.format_exc())
        else:
            before = _source_identity(arguments.modules)
            suite = unittest.defaultTestLoader.loadTestsFromNames(arguments.modules)
            report["counts"]["loaded"] = suite.countTestCases()
            result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
            report["tests"] = result.records
            report["counts"].update(total=result.testsRun,
                executed=sum(row["status"] != "skipped" and row.get("kind") != "fixture" for row in result.records),
                passed=sum(row["status"] == "passed" for row in result.records), skipped=len(result.skipped),
                failures=len(result.failures), errors=len(result.errors), expected_failures=len(result.expectedFailures),
                unexpected_successes=len(result.unexpectedSuccesses))
            ran = report["counts"]["executed"] > 0
            accepted = ran and result.wasSuccessful() and not (arguments.fail_on_skip and result.skipped)
            report["passed"] = bool(accepted)
            report["qualified"] = bool(accepted and not result.skipped and not result.expectedFailures)
            report["status"] = ("failed" if not result.wasSuccessful() else "no_tests" if not ran else "failed" if not accepted else
                                "passed_with_skips" if result.skipped else "passed")
            if not ran:
                report["error"] = "No test body executed (the suite was empty or every test was skipped)"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        _runtime_environment(report["environment"])
        after = _source_identity(arguments.modules)
        changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        readable = all(isinstance(value, str) for value in (*before.values(), *after.values()))
        unchanged = bool(before) and not changed and readable
        report["source_identity"] = {"before": before, "after": after, "unchanged": unchanged,
                                     "changed_paths": changed, "before_digest": _digest(before),
                                     "after_digest": _digest(after)}
        report["imported_sources"] = _imported_sources(after)
        if before:
            try:
                installed_before = report["environment"]["deepopt"]
                installed_after = _installed_runtime().installed_identity(ROOT)
                installed_unchanged = installed_before == installed_after
                report["installed_identity"] = {"before": installed_before, "after": installed_after,
                                                "unchanged": installed_unchanged}
                if not installed_unchanged:
                    report.update(passed=False, qualified=False, status="failed",
                                  error="Installed DeepOpt changed during qualification")
            except Exception as error:
                report.update(passed=False, qualified=False, status="failed",
                              error=f"Installed DeepOpt validation failed: {type(error).__name__}: {error}")
        report["duration_seconds"] = time.perf_counter() - started
        if not unchanged:
            report.update(passed=False, qualified=False)
            if report["status"] not in ("preflight_failed", "no_tests"):
                report.update(status="failed", error="Source changed or could not be read during qualification")
        _write_json(arguments.output, report)
        exit_code = 0 if report["passed"] else 1
    print(f"Contract qualification: {report['status']}; JSON: {arguments.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
