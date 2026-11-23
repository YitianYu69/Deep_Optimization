"""Save a benchmark's actual Inductor code beside its runtime measurements.

This module does not time or execute generated files.  The normal analysis
measurement and profiler passes remain the source of runtime evidence.  Compiler
working caches are temporary; only verified code/IR/diagnostics are retained in
the case output, without tensor checkpoints or pickle caches.
"""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .inductor_reference_store import create_run, finalize_run, save_artifacts, verify_run


ENVIRONMENT_FLAG = "DEEPOPT_INDUCTOR_BENCHMARK_DUMPS"
_NATIVE_NAMESPACES = frozenset({"aten", "prims", "prim", "higher_order", "inductor", "rngprims"})
_SOURCE_SCAN_BYTES = 32 * 2**20
_OP_NAMESPACE = r"(?<![\w.])(?:torch\.)?ops\.([A-Za-z_][A-Za-z0-9_]*)\."


def reference_controls_valid(controls, preset=None):
    if not isinstance(controls, dict):
        return False
    name = preset or controls.get("preset")
    if name not in ("default", "cast-preserving") or controls.get("preset") != name:
        return False
    expected = {"backward_pass_autocast": "off", "fallback_random": True,
                "emulate_precision_casts": name == "cast-preserving"}
    return all(isinstance(controls.get(key), dict) and controls[key] == expected
               and all(type(controls[key][name]) is type(value) for name, value in expected.items())
               for key in ("requested", "applied", "effective"))


@contextmanager
def inductor_reference_context(preset="default"):
    """Scope the compiled-backward contract across every lazy execution.

    This is a native-Torch arithmetic experiment, independent of DeepOpt's
    storage policy. Unsupported settings fail closed rather than being ignored.
    """
    if preset not in ("default", "cast-preserving"):
        raise ValueError(f"Unknown Inductor arithmetic preset: {preset}")
    functorch = importlib.import_module("torch._functorch.config")
    inductor = importlib.import_module("torch._inductor.config")
    torch = importlib.import_module("torch")
    requested = {"backward_pass_autocast": "off", "fallback_random": True,
                 "emulate_precision_casts": preset == "cast-preserving"}
    for key in requested:
        owner = functorch if key == "backward_pass_autocast" else inductor
        if not hasattr(owner, key):
            raise RuntimeError(f"Required Inductor numerical control unavailable: {key}")
    with ExitStack() as stack:
        stack.enter_context(functorch.patch({"backward_pass_autocast": "off"}))
        stack.enter_context(inductor.patch({key: value for key, value in requested.items()
                                           if key != "backward_pass_autocast"}))
        effective = {key: getattr(functorch if key == "backward_pass_autocast" else inductor, key)
                     for key in requested}
        if effective != requested:
            raise RuntimeError("Effective Inductor numerical settings differ from request")
        yield {"preset": preset, "requested": requested, "applied": effective.copy(), "effective": effective,
               "backward_execution": "outside autocast", "native_torch_only": True,
               "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
               "tf32_cudnn": torch.backends.cudnn.allow_tf32,
               "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
               "random_policy": "fallback_random=True (explicit eager RNG diagnostic)"}


def _read_code(path):
    # Source inspection is bounded and never imports generated Python.
    if path.stat().st_size > _SOURCE_SCAN_BYTES:
        raise ValueError(f"Generated source exceeds inspection bound: {path.name}")
    return path.read_text(errors="replace")


def _has_graph_breaks(counters):
    if not isinstance(counters, dict):
        return True
    for name, value in counters.items():
        if "graph_break" in str(name):
            values = value.values() if isinstance(value, dict) else (value,)
            if any(type(count) is not int or count != 0 for count in values):
                return True
    return False


def inspect_artifacts(run_dir, copied):
    """Classify saved compiler outputs and audit captured FX operator domains."""
    run = Path(run_dir)
    sources, phases, fx, triton, external, namespaces, opaque = [], {"forward": [], "backward": []}, [], [], [], set(), []
    for entry in copied:
        relative = entry["path"]
        path = run / relative
        if path.suffix.lower() not in {".py", ".cpp", ".cu"} and not path.name.startswith("fx_graph"):
            continue
        code = _read_code(path)
        is_fx = path.name.startswith("fx_graph")
        if is_fx:
            fx.append(relative)
            # Match the actual torch.ops root. Runnable FX repro headers contain
            # torch._higher_order_ops.triton_kernel_wrap.kernel_side_table
            # .reset_table(); the 'ops' suffix in that Python module is not an
            # operator namespace or a captured custom-kernel call.
            found = set(re.findall(_OP_NAMESPACE, code))
            namespaces.update(found)
            opaque.extend({"path": relative, "namespace": name} for name in sorted(found - _NATIVE_NAMESPACES))
            for line in code.splitlines():
                if "deepopt" in line.lower() and re.search(_OP_NAMESPACE + r"|call_function|target=", line):
                    opaque.append({"path": relative, "target": line.strip()[:1000]})
                if re.search(r"autograd_function_apply|PythonOp|python_call|torch\.autograd\.Function", line):
                    opaque.append({"path": relative, "target": line.strip()[:1000]})
                if re.search(r"\btriton_kernel_wrapper_(?:functional|mutation)\s*\(", line):
                    opaque.append({"path": relative, "target": line.strip()[:1000]})
        if path.name == "output_code.py" or (path.suffix == ".py" and "AOT ID:" in code):
            sources.append(relative)
            for phase in phases:
                if re.search(r"AOT ID:.*?[^A-Za-z]" + phase + r"[^A-Za-z]", code) or re.search(r"(?:_|/)" + phase + r"(?:_|/)", relative):
                    phases[phase].append(relative)
            if re.search(r"@triton\.jit|async_compile\.triton\(", code):
                triton.append(relative)
            external.extend({"path": relative, "call": name} for name in sorted(set(
                re.findall(r"(?:extern_kernels|torch\.ops\.aten)\.([A-Za-z_][A-Za-z0-9_.]*)", code))))
        elif path.suffix == ".py" and re.search(r"@triton\.jit|async_compile\.triton\(", code):
            triton.append(relative)
    return {"output_code_paths": sources, "compiled_phase_artifacts": phases,
            "triton_source_paths": sorted(set(triton)), "fx_graph_paths": fx,
            "external_library_calls": external,
            "native_operator_audit": {"status": "failed" if opaque else "passed" if fx else "unverified",
                "captured_fx_files": len(fx), "operator_namespaces": sorted(namespaces),
                "opaque_or_custom_targets": opaque,
                "scope": "Captured FX targets only; graph-break execution remains separately reported."}}


def validate_inductor_artifacts(report, case_directory):
    """Verify saved code before accepting an Inductor runtime reference.

    Paths are resolved relative to this case output.  A forged success flag,
    changed source, missing compiler phase, or custom operator cannot qualify.
    CPU and library-only generated wrappers are allowed but explicitly labeled.
    """
    try:
        if not isinstance(report, dict) or report.get("reference_valid") is not True:
            raise ValueError("Inductor artifact report is absent or not valid")
        directory = Path(case_directory).resolve()
        manifest_relative = Path(report["manifest"])
        if manifest_relative.is_absolute() or ".." in manifest_relative.parts:
            raise ValueError("Manifest must be inside the benchmark case output")
        manifest_path = directory / manifest_relative
        if not manifest_path.resolve().is_relative_to(directory):
            raise ValueError("Manifest escapes the benchmark output")
        run = manifest_path.parent
        verified = verify_run(run)
        if not verified["valid"]:
            raise ValueError("Saved Inductor artifacts fail membership/hash verification")
        manifest = json.loads(manifest_path.read_text())
        if manifest["request"] != json.loads((run / "request.json").read_text()):
            raise ValueError("Manifest request does not match its saved task identity")
        task = manifest["request"]["task"]
        runtime = task["settings"]["runtime"]
        if task["case"] != "torch_bf16" or runtime.get("execution") != "compile" or runtime["compile"]["backend"] != "inductor":
            raise ValueError("Reference task is not the raw Torch Inductor control")
        metadata = manifest["metadata"]
        if not reference_controls_valid(metadata.get("configuration", {}).get("numerical_controls")):
            raise ValueError("Required compiled-backward arithmetic contract is missing or unsupported")
        if (manifest.get("status") != "complete" or manifest.get("reference_eligible") is not True
                or metadata.get("benchmark_execution_observed") is not True):
            raise ValueError("Saved capture is incomplete or lacks benchmark execution")
        # Reinspect the verified contents instead of trusting stored audit booleans.
        entries = [{"path": name, **entry} for name, entry in manifest["files"].items()
                   if name.startswith("artifacts/")]
        inspected = inspect_artifacts(run, entries)
        if inspected["native_operator_audit"]["status"] != "passed":
            raise ValueError("Captured FX graph includes custom targets or is missing")
        if _has_graph_breaks(metadata.get("graph_breaks_and_fallbacks", {})):
            raise ValueError("Graph breaks leave untraced execution outside the native operator audit")
        required = ["forward"] + ([] if task.get("workload", {}).get("gradient_mode", "all") == "none" else ["backward"])
        if not inspected["output_code_paths"] or any(not inspected["compiled_phase_artifacts"][phase] for phase in required):
            raise ValueError("Generated forward/backward code is missing")
        prefix = run.relative_to(directory).as_posix()
        return {"passed": True, "native_ops_only": True,
                "numerical_controls": metadata["configuration"]["numerical_controls"],
                "generated_code_files": [f"{prefix}/{name}" for name in inspected["output_code_paths"]],
                "triton_code_files": [f"{prefix}/{name}" for name in inspected["triton_source_paths"]],
                "triton_generated": bool(inspected["triton_source_paths"]),
                "cuda_execution_observed": metadata.get("cuda_execution_observed") is True,
                "device_type": metadata.get("device_type"),
                "external_library_calls": inspected["external_library_calls"],
                "graph_breaks_and_fallbacks": metadata.get("graph_breaks_and_fallbacks", {}),
                "manifest": manifest_relative.as_posix(), "files_checked": verified["files_checked"]}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"passed": False, "native_ops_only": False, "generated_code_files": [],
                "triton_code_files": [], "reason": f"{type(error).__name__}: {error}"}


class InductorBenchmarkArtifacts:
    """A fresh worker's dump lifetime, configured before Session compilation.

    ``finish(row=...)`` returns JSON metadata with paths relative to the benchmark
    case directory.  It must run after timing/profile passes.  On failure the
    same report is saved locally and attached to ``error.inductor_artifacts``.
    """

    def __init__(self, directory, task, *, enabled=None):
        # Canonicalize platform aliases such as macOS /var -> /private/var.
        self.directory = Path(directory).resolve()
        self.task = deepcopy(task)
        self.enabled = os.environ.get(ENVIRONMENT_FLAG) == "1" if enabled is None else bool(enabled)
        self.stack, self.run, self.scratch, self.report = ExitStack(), None, None, None
        self.configuration = {}
        self.preserve_scratch = False

    def __enter__(self):
        if not self.enabled:
            return self
        runtime = self.task["settings"]["runtime"]
        if (self.task.get("case") != "torch_bf16" or runtime.get("execution") != "compile"
                or runtime.get("compile", {}).get("backend") != "inductor"):
            raise ValueError("Inductor runtime references require raw torch_bf16 with backend=inductor; DeepOpt/custom/compressed tracks are excluded")
        request = {"model": self.task.get("workload", {}).get("name", self.task.get("workload", {}).get("op", "benchmark")),
                   "recipe": "torch-bf16", "task": self.task,
                   "purpose": "runtime baseline with code saved in benchmark output",
                   "compiler": "TorchInductor", "deepopt_native_autodiff": False}
        self.run = create_run(self.directory / "inductor", request)
        self.scratch = Path(tempfile.mkdtemp(prefix="deepopt-inductor-benchmark-")).resolve()
        self.stack.callback(lambda: None if self.preserve_scratch else shutil.rmtree(self.scratch, ignore_errors=True))
        environment = {
            "TORCH_COMPILE_DEBUG": "1", "TORCH_COMPILE_DEBUG_SAVE_REAL": "0",
            "TORCH_COMPILE_DEBUG_DIR": str(self.scratch / "debug"),
            "TORCHINDUCTOR_CACHE_DIR": str(self.scratch / "inductor-cache"),
            "TRITON_CACHE_DIR": str(self.scratch / "triton-cache"),
            "TRITON_DUMP_DIR": str(self.scratch / "triton-dumps"), "TRITON_KERNEL_DUMP": "1",
            "TORCHINDUCTOR_SAVE_COMPILE_ARGS": "0", "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0", "TORCHDYNAMO_REPRO_LEVEL": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        # Dynamo treats an empty string (and the string "None") as an invalid
        # repro target. Absence selects Python None when config is first loaded.
        removed = {"TORCH_TRACE", "TRITON_OVERRIDE_DIR", "TORCHDYNAMO_REPRO_AFTER"}
        previous = {name: os.environ.get(name) for name in set(environment) | removed}

        def restore_environment():
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self.stack.callback(restore_environment)
        for name in removed:
            os.environ.pop(name, None)
        os.environ.update(environment)
        try:
            # The analysis engine may already have imported Torch/Dynamo, so
            # changing the environment alone cannot reset its cached config.
            dynamo = importlib.import_module("torch._dynamo.config")
            dynamo_requested = {"repro_after": None, "repro_level": 0}
            dynamo_unavailable = [key for key in dynamo_requested if not hasattr(dynamo, key)]
            self.configuration = {"environment": {"applied": environment, "removed": sorted(removed)},
                "dynamo": {"requested": dynamo_requested, "applied": {}, "unavailable": dynamo_unavailable}}
            if dynamo_unavailable:
                raise RuntimeError("Dynamo repro dumping cannot be disabled on this Torch version: "
                                   + ", ".join(dynamo_unavailable))
            self.stack.enter_context(dynamo.patch(dynamo_requested))
            self.configuration["dynamo"]["applied"] = dynamo_requested.copy()
            self.configuration["numerical_controls"] = self.stack.enter_context(
                inductor_reference_context("cast-preserving" if runtime.get("compile", {}).get(
                    "options", {}).get("emulate_precision_casts") else "default"))
            config = importlib.import_module("torch._inductor.config")
            requested = {"trace.enabled": True, "trace.save_real_tensors": False,
                "trace.debug_dir": str(self.scratch / "debug"), "trace.fx_graph": True,
                "trace.fx_graph_transformed": True, "trace.ir_pre_fusion": True,
                "trace.ir_post_fusion": True, "trace.output_code": True,
                "trace.debug_log": True, "trace.info_log": True, "trace.compile_profile": False,
                "fx_graph_cache": False, "fx_graph_remote_cache": False, "autotune_remote_cache": False}
            applied, unavailable = {}, []
            for key, value in requested.items():
                owner = config
                try:
                    for part in key.split("."):
                        owner = getattr(owner, part)
                    applied[key] = value
                except AttributeError:
                    unavailable.append(key)
            if applied.get("trace.save_real_tensors") is not False:
                raise RuntimeError("Inductor real-tensor dumping cannot be disabled on this Torch version")
            self.configuration.update({"requested": requested, "applied": applied, "unavailable": unavailable,
                                       "real_tensor_dumping": False})
            self.stack.enter_context(config.patch(applied))
        except BaseException as error:
            self.finish(error=error)
            self.stack.close()
            raise
        return self

    def finish(self, *, row=None, error=None):
        if not self.enabled or self.report is not None:
            return self.report
        if self.run is None:
            return None
        try:
            copied = save_artifacts(self.run, self.scratch, namespace="compiler")
            # Byte-limit or unstable-source failures may leave useful generated
            # code uncopied. Keep that working directory for explicit recovery;
            # the incomplete capture still cannot qualify as a reference.
            self.preserve_scratch = bool(copied["errors"])
            evidence = inspect_artifacts(self.run, copied["copied"])
            required_phases = ["forward"]
            if self.task.get("workload", {}).get("gradient_mode", "all") != "none":
                required_phases.append("backward")
            missing_phases = [phase for phase in required_phases if not evidence["compiled_phase_artifacts"][phase]]
            observed = bool(row and row.get("status") == "ok" and row.get("timing") and row.get("finite_after_timing"))
            counters = (row or {}).get("execution", {}).get("compile_counters", {})
            complete = bool(error is None and observed and not missing_phases and evidence["output_code_paths"]
                            and not _has_graph_breaks(counters)
                            and evidence["native_operator_audit"]["status"] == "passed")
            metadata = {**evidence, "configuration": self.configuration, "artifact_copy_report": copied,
                "benchmark_execution_observed": observed, "cuda_execution_observed": observed and
                    str((row or {}).get("environment", {}).get("device", "")).startswith("cuda"),
                "device_type": str((row or {}).get("environment", {}).get("device", "unknown")).split(":")[0],
                "environment": (row or {}).get("environment", {}), "required_phases": required_phases,
                "missing_compiled_phases": missing_phases, "graph_breaks_and_fallbacks":
                    counters,
                "compile_scope": {"forward": "TorchInductor", "backward": "PyTorch AOTAutograd",
                                  "loss": "eager", "optimizer": "eager", "deepopt_native_autodiff": False},
                "triton_generation": "source_saved" if evidence["triton_source_paths"] else
                    "no_generated_triton_source; inspect external library calls and CPU/device identity",
                "timing_scope": deepcopy((row or {}).get("timing_scope", self.task.get("timing_scope",
                    self.task.get("timing_implementation", "Existing unprofiled analysis.measure samples; compilation/warmup, artifact copy/hash and separate profiles are excluded")))),
                "uncopied_working_directory": str(self.scratch) if self.preserve_scratch else None,
                "failure": f"{type(error).__name__}: {error}" if error is not None else None}
            manifest = finalize_run(self.run, metadata, "complete" if complete else "partial",
                                    required_artifacts=("generated_code", "fx_graph"))
            prefix = self.run.relative_to(self.directory).as_posix()
            self.report = {"status": manifest["status"], "reference_valid": manifest["reference_eligible"],
                "numerical_controls": self.configuration.get("numerical_controls"),
                "manifest": f"{prefix}/manifest.json", "run_id": self.run.name,
                "cuda_execution_observed": metadata["cuda_execution_observed"],
                **{key: metadata[key] for key in ("native_operator_audit", "compile_scope", "triton_generation", "timing_scope",
                                                 "missing_compiled_phases", "graph_breaks_and_fallbacks")},
                **{key: [f"{prefix}/{name}" for name in evidence[key]] for key in
                   ("output_code_paths", "triton_source_paths", "fx_graph_paths")},
                "compiled_phase_artifacts": {phase: [f"{prefix}/{name}" for name in names]
                                             for phase, names in evidence["compiled_phase_artifacts"].items()},
                "files": {f"{prefix}/{name}": entry for name, entry in manifest["files"].items()},
                "uncopied_working_directory": metadata["uncopied_working_directory"],
                "tensor_files_saved": 0}
        except Exception as capture_error:
            # Never hide a successfully measured row or erase the original
            # exception; unavailable dumps remain an explicit failed reference.
            self.preserve_scratch = bool(self.scratch and self.scratch.exists())
            self.report = {"status": "failed", "reference_valid": False,
                "cuda_execution_observed": False, "tensor_files_saved": 0,
                "reason": f"{type(capture_error).__name__}: {capture_error}",
                "run_directory": str(self.run.relative_to(self.directory)),
                "uncopied_working_directory": str(self.scratch) if self.preserve_scratch else None,
                "uncopied_scope": "Recovery working files only; not a validated benchmark code reference"}
        report_path = self.directory / "inductor-artifacts.json"
        report_path.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        if error is not None:
            error.inductor_artifacts = self.report
        return self.report

    def __exit__(self, kind, error, traceback):
        try:
            if self.report is None:
                self.finish(error=error)
        finally:
            self.stack.close()
        return False
