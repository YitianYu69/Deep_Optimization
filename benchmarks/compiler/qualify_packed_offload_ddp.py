"""Short, pinned, installed-only packed-offload DDP qualification.

The CUDA suite uses real local NCCL ranks. ``--device cpu`` is an explicit
Gloo/testing-transport diagnostic, never CUDA qualification. Help and dry-run
planning use only the standard library. No capacity sweep or model benchmark
is run by this entry point.
"""

import argparse
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import importlib
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "packed_offload_ddp_v1"
WORKER = "tests/distributed/_packed_offload_worker.py"
REQUIRED_SOURCES = ("benchmarks/compiler/qualify_packed_offload_ddp.py",
                    "benchmarks/compiler/installed_runtime.py", WORKER,
                    "scripts/run_h200_packed_offload_ddp.sh",
                    "tests/compiler/test_compiled_packed_offload.py",
                    "tests/compiler/__init__.py", "tests/distributed/__init__.py")
REQUIRED_PACKAGE = ("__init__.py", "distributed/offload.py", "compiler/runtime/packed_offload.py")
COMMON_CHECKS = ("native_fullgraph", "native_autodiff", "fp32_parameter_gradients",
    "exact_manual_all_reduce_oracle", "no_sync_accumulation", "retained_backward_in_no_sync",
    "gradient_bucket_views", "parameter_identity", "rank_parameters_equal", "multiple_ddp_buckets", "checkpoint_restore",
    "checkpoint_next_update", "exact_optimizer_state", "rank_local_checkpoint_state",
    "rank_local_rng_restored",
    "policy_mismatch_cooperative", "collective_usable_after_rejection", "cooperative_forward_error",
    "no_backward_after_forward_error", "source_identity_stable")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    result.add_argument("--installed-target", type=Path, help="Pinned non-editable target; fresh when --install is used")
    result.add_argument("--install", action="store_true", help="Install the pinned wheel into a fresh target after device preflight")
    result.add_argument("--processes", type=int, default=2, help="Local ranks/visible GPUs; at least two, e.g.4")
    result.add_argument("--timeout-seconds", type=int, default=600, help="Whole torchrun group timeout, at most1800 seconds")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--require-h200", action="store_true", help="Require every selected CUDA device to be an H200")
    result.add_argument("--output", type=Path, help="Fresh JSON result; default work/packed_offload_ddp_TIMESTAMP.json")
    result.add_argument("--dry-run", action="store_true", help="Validate manifest/wheel/source pins without importing Torch or allocating")
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(2**20), b""):
            digest.update(data)
    return digest.hexdigest()


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _hash_pin(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _resolve(value, parent):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _source_pins(sources):
    if not isinstance(sources, dict) or any(name not in sources for name in REQUIRED_SOURCES):
        raise ValueError("Manifest must pin every qualification helper, launcher and worker source")
    references = {}
    for name, pin in sources.items():
        relative = Path(name)
        path = ROOT / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or relative.suffix not in (".py", ".sh", ".md", ".toml", ".json", ".txt")
                or not path.resolve().is_relative_to(ROOT.resolve()) or not isinstance(pin, dict)
                or type(pin.get("bytes")) is not int or pin["bytes"] < 0 or not _hash_pin(pin.get("sha256"))):
            raise ValueError("Invalid qualification source pin: " + str(name))
        actual = _reference(path)
        if any(actual[key] != pin[key] for key in ("bytes", "sha256")):
            raise ValueError("Qualification source differs from its release pin: " + name)
        references[name] = actual
    return references


def _wheel_pin(candidate, manifest_path):
    if not isinstance(candidate, dict) or not isinstance(candidate.get("wheel"), dict):
        raise ValueError("Manifest requires candidate wheel and installed identity pins")
    expected, wheel = candidate.get("expected_installed"), candidate["wheel"]
    if (not isinstance(expected, dict) or not isinstance(expected.get("version"), str) or not expected["version"]
            or not _hash_pin(expected.get("source_sha256")) or type(expected.get("files_hashed")) is not int
            or expected["files_hashed"] < 1 or not _hash_pin(wheel.get("sha256"))):
        raise ValueError("Invalid candidate version/source/wheel pin")
    path = _resolve(wheel["path"], manifest_path.parent)
    if path.is_symlink() or _sha(path) != wheel["sha256"]:
        raise ValueError("Candidate wheel checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or archive.testzip() is not None
                or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names)):
            raise ValueError("Candidate wheel membership/CRC/path mismatch")
        metadata_name, = [name for name in names if name.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        if metadata["Name"] != "deep-optimization" or metadata["Version"] != expected["version"]:
            raise ValueError("Candidate wheel version/distribution differs from release pin")
        files = {name.removeprefix("deepopt/"): archive.read(name)
                 for name in names if name.startswith("deepopt/") and not name.endswith("/")}
    if any(name not in files for name in REQUIRED_PACKAGE):
        raise ValueError("Candidate wheel lacks compiled packed-offload DDP support")
    digest = hashlib.sha256()
    for name in sorted(files, key=lambda item: Path(item).parts):
        digest.update(name.encode() + b"\0" + files[name] + b"\0")
    if digest.hexdigest() != expected["source_sha256"] or len(files) != expected["files_hashed"]:
        raise ValueError("Candidate package bytes differ from release source identity")
    return _reference(path), {key: expected[key] for key in ("version", "source_sha256", "files_hashed")}


def qualification_plan(args):
    if type(args.processes) is not int or args.processes < 2:
        raise ValueError("DDP qualification requires at least two local processes")
    if type(args.timeout_seconds) is not int or not 1 <= args.timeout_seconds <= 1800:
        raise ValueError("timeout-seconds must be1..1800")
    if args.device == "cpu" and args.require_h200:
        raise ValueError("H200 qualification cannot substitute CPU/Gloo")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("profile") != PROFILE):
        raise ValueError("Require a schema1 packed_offload_ddp_v1 release manifest")
    sources = _source_pins(manifest.get("sources"))
    wheel, expected = _wheel_pin(manifest.get("candidate"), manifest_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "work" / ("packed_offload_ddp_" + stamp + ".json")).resolve()
    target = (args.installed_target or output.parent / (output.stem + "-install")).resolve()
    if not args.install and args.installed_target is None:
        raise ValueError("Supply --installed-target or use --install for a fresh isolated target")
    return {"schema_version": 1, "profile": PROFILE, "scope": "compiled_packed_offload_DDP_contract_suite",
        "manifest": _reference(manifest_path), "sources": sources, "wheel": wheel, "expected_installed": expected,
        "installed_target": str(target), "output": str(output), "processes": args.processes,
        "device": args.device, "backend": "nccl" if args.device == "cuda" else "gloo",
        "require_h200": args.require_h200, "timeout_seconds": args.timeout_seconds,
        "case": "suite", "source_substitution_allowed": False, "CUDA_execution_proven": False,
        "performance_qualified": False, "tensor_files_saved": 0}


def _device_preflight(plan):
    os.environ.pop("TORCHDYNAMO_REPRO_AFTER", None)
    os.environ["TORCHDYNAMO_REPRO_LEVEL"] = "0"
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run the qualification supervisor with Python, not inside torchrun")
    torch = importlib.import_module("torch")
    dynamo_config = sys.modules.get("torch._dynamo.config")
    if dynamo_config is not None:
        dynamo_config.repro_after, dynamo_config.repro_level = None, 0
    result = {"torch": str(torch.__version__), "torch_git": getattr(torch.version, "git_version", None),
        "cuda": torch.version.cuda, "device": plan["device"], "devices": [], "model_allocated": False}
    if plan["device"] == "cpu":
        if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
            raise RuntimeError("Explicit CPU diagnostic requires Gloo")
        return result
    if (not torch.cuda.is_available() or torch.cuda.device_count() < plan["processes"]
            or not torch.distributed.is_available() or not torch.distributed.is_nccl_available()):
        raise RuntimeError(f"Require at least{plan['processes']} visible CUDA GPUs and NCCL; no CPU substitution")
    for index in range(plan["processes"]):
        properties = torch.cuda.get_device_properties(index)
        capability = (properties.major, properties.minor)
        if capability < (8, 0) or plan["require_h200"] and "H200" not in properties.name:
            raise RuntimeError("Every selected GPU must support BF16" + (" and be H200" if plan["require_h200"] else ""))
        result["devices"].append({"local_rank": index, "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "unavailable")), "capability": list(capability),
            "total_memory_bytes": properties.total_memory})
    triton = importlib.import_module("triton")
    result["triton"] = str(triton.__version__)
    return result


def _stop_group(process, *, force=False):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        pass


def _stream_process(command, log, environment, timeout_seconds, *, stage):
    """Stream progress while bounding the complete child process group."""
    if os.name != "posix":
        raise RuntimeError("Bounded whole-group qualification requires a POSIX host")
    started, last_progress = time.monotonic(), time.monotonic()
    print(f"[packed-offload DDP] START {stage} timeout={timeout_seconds}s log={log}", flush=True)
    timed_out = False
    with Path(log).open("xb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map() or process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed >= timeout_seconds:
                        timed_out = True
                        _stop_group(process)
                        # Do not reap the leader before terminating its group:
                        # surviving ranks can outlive an exited torchrun
                        # leader. Keep the owned group through escalation.
                        _stop_group(process, force=True)
                        break
                    for key, _ in selector.select(timeout=min(.25, max(.001, timeout_seconds - elapsed))):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        stream.write(data)
                        stream.flush()
                        print(data.decode(errors="replace"), end="", flush=True)
                    if time.monotonic() - last_progress >= 15:
                        print(f"[packed-offload DDP] {stage} elapsed={time.monotonic()-started:.1f}s log={log}", flush=True)
                        last_progress = time.monotonic()
            remaining, _ = process.communicate(timeout=5)
            if remaining:
                stream.write(remaining)
                print(remaining.decode(errors="replace"), end="", flush=True)
        except BaseException as error:
            try:
                _stop_group(process, force=True)
                process.wait(timeout=5)
            except Exception as cleanup_error:
                error.add_note(f"Owned process-group cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
            raise
        finally:
            process.stdout.close()
    result = {"command": command, "exit_code": process.returncode, "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started, "log": _reference(log)}
    print(f"[packed-offload DDP] COMPLETE {stage} exit={process.returncode} timeout={timed_out} elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    return result


def _installed(plan):
    from benchmarks.compiler.installed_runtime import installed_identity

    sys.path.insert(0, plan["installed_target"])
    identity = installed_identity(ROOT, full=True)
    if (identity["install_root"] != plan["installed_target"]
            or any(identity[key] != value for key, value in plan["expected_installed"].items())):
        raise ValueError("Imported DeepOpt differs from the pinned installed candidate")
    return identity


def _rank_results(directory, plan, identity):
    rows, errors = [], []
    for rank in range(plan["processes"]):
        path = directory / f"rank-{rank}.json"
        row = {"rank": rank}
        try:
            row["artifact"] = _reference(path)
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("Rank receipt must be an object")
            row["receipt"] = raw
            checks = raw.get("checks")
            required_checks = (*COMMON_CHECKS, "actual_cuda_startup_preflight", "whole_model_offload_plan_preserved",
                "one_training_runtime_transport_owner_per_rank") if plan["device"] == "cuda" else (
                *COMMON_CHECKS, "cpu_offload_rejected", "generated_joint_saved_abi", "graph_leases_released", "ddp_optimizer_graph_split")
            expected_cases = {"preflight", "forward_error", "native_demand", "checkpoint_demand",
                "native_memory_first_overlap", "checkpoint_memory_first_overlap",
                "native_demand_torch_amp", "checkpoint_demand_torch_amp"} if plan["device"] == "cuda" else {
                "preflight", "forward_error", "native_resident", "checkpoint_resident", "generated_testing_transport"}
            cases = raw.get("cases")
            expected_scope = "CUDA_NCCL_real_packed_offload" if plan["device"] == "cuda" else "CPU_Gloo_native_and_explicit_test_transport"
            if (type(raw.get("schema_version")) is not int or raw["schema_version"] != 1
                    or raw.get("status") != "completed" or type(raw.get("rank")) is not int or raw["rank"] != rank
                    or type(raw.get("local_rank")) is not int or raw["local_rank"] != rank
                    or type(raw.get("world_size")) is not int or raw["world_size"] != plan["processes"]
                    or raw.get("device") != (f"cuda:{rank}" if plan["device"] == "cuda" else "cpu")
                    or raw.get("scope") != expected_scope or raw.get("backend") != plan["backend"]
                    or raw.get("installed") != identity or raw.get("installed_after") != identity
                    or raw.get("group_initialized_after_shutdown") is not False or raw.get("cleanup_exception")
                    or raw.get("shutdown_exception") or raw.get("exception")
                    or type(raw.get("tensor_files_saved")) is not int or raw["tensor_files_saved"] != 0
                    or not isinstance(checks, dict) or any(checks.get(name) is not True for name in required_checks)
                    or any(value is not True for value in checks.values())):
                raise ValueError("Rank completion/topology/installed identity/checks did not qualify")
            if (not isinstance(cases, list) or len(cases) != len(expected_cases)
                    or any(not isinstance(case, dict) or not isinstance(case.get("name"), str) for case in cases)
                    or {case["name"] for case in cases} != expected_cases
                    or any(not isinstance(case.get("checks"), dict) or not case["checks"]
                           or any(value is not True for value in case["checks"].values())
                           or case.get("status", "completed") not in ("completed", "passed") for case in cases)):
                raise ValueError("The complete required suite must execute; missing/skipped/failed cases do not qualify")
            reported_checks = {name for case in cases for name in case["checks"]}
            if any(name not in reported_checks for name in required_checks if name != "source_identity_stable"):
                raise ValueError("Required checks are absent from executed cases")
            for case in cases:
                if case["name"].startswith(("native_", "checkpoint_")):
                    compiler = case.get("compiler")
                    if (not isinstance(compiler, dict) or any(type(compiler.get(name)) is not int or compiler[name] <= 0
                            for name in ("captured_graphs", "native_autodiff_operations", "native_autodiff_regions"))
                            or type(compiler.get("runtime_fallbacks")) is not int or compiler["runtime_fallbacks"] != 0
                            or compiler.get("legacy_backward_operations") != []
                            or plan["device"] == "cuda" and compiler["captured_graphs"] != 1):
                        raise ValueError("Executed native cases must retain full native AutoDiff ownership with zero fallback")
                if plan["device"] == "cuda" and case["name"].startswith("native_"):
                    scope = case.get("compiler_scope")
                    if (not isinstance(scope, dict) or scope.get("ddp_optimizer_partitioning") != "disabled"
                            or scope.get("communication_overlap_qualified") is not False):
                        raise ValueError("Offload must preserve its recorded whole-model DDP compiler scope")
            if plan["device"] == "cuda":
                if raw.get("cuda_execution_verified") is not True or raw.get("test_transport") is not False:
                    raise ValueError("Real CUDA execution and production transport are required")
            elif raw.get("cuda_execution_verified") is not False or raw.get("test_transport") is not True:
                raise ValueError("CPU diagnostic must explicitly identify testing transport and disclaim CUDA evidence")
            row["valid"] = True
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
            row.update(valid=False, error=f"{type(error).__name__}: {error}")
            errors.append(f"rank{rank}: {row['error']}")
        rows.append(row)
    return rows, errors


def _archive(output, directory, report):
    archive = output.with_suffix(".zip")
    records, omissions = {}, []
    sources = {Path(pin["path"]) for pin in report["sources"].values()}
    paths = [output, *sorted(directory.rglob("*")), *sorted(sources)]
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as target:
        total = 0
        for path in paths:
            if not path.is_file() or path.is_symlink() or (path not in sources and path.suffix not in (".json", ".log")):
                continue
            size = path.stat().st_size
            name = ("sources/" + path.relative_to(ROOT.resolve()).as_posix() if path in sources
                    else "results/" + path.relative_to(output.parent).as_posix())
            if len(records) >= 256 or total + size > 64 * 2**20:
                omissions.append({"path": str(path), "bytes": size, "reason": "metadata_archive_budget"})
                continue
            target.write(path, name)
            records[name] = {"bytes": size, "sha256": _sha(path)}
            total += size
        target.writestr("archive-manifest.json", json.dumps({"schema_version": 1, "files": records,
            "omissions": omissions, "tensor_files_saved": 0, "install_trees_saved": False}, indent=2))
    report["archive"] = _reference(archive)
    report["archive_omissions"] = omissions


def run(args):
    plan = qualification_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    output, target = Path(plan["output"]), Path(plan["installed_target"])
    directory = output.parent / (output.stem + "-artifacts")
    if output.exists() or directory.exists() or output.with_suffix(".zip").exists() or args.install and target.exists():
        raise FileExistsError("Choose fresh qualification output and installation paths")
    directory.mkdir(parents=True)
    report = {**plan, "status": "running", "stage": "device_preflight", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands": [], "ranks": [], "error": None, "CUDA_execution_proven": False}
    _write(output, report)
    environment = dict(os.environ)
    environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
    environment.pop("DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY", None)
    environment.update(TORCHDYNAMO_REPRO_LEVEL="0", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    try:
        report["environment"] = _device_preflight(plan)
        if args.install:
            report["stage"] = "install_candidate"
            row = _stream_process([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
                "--target", str(target), plan["wheel"]["path"]], directory / "install.log", environment, min(args.timeout_seconds, 300), stage=report["stage"])
            report["commands"].append(row)
            if row["exit_code"] != 0 or row["timed_out"]:
                raise RuntimeError("Pinned candidate installation failed or timed out")
        report["stage"] = "installed_identity"
        identity = report["installed"] = _installed(plan)
        from benchmarks.compiler.installed_runtime import child_environment

        environment = child_environment(ROOT, identity=identity, base=environment)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        ranks = directory / "ranks"
        ranks.mkdir()
        command = [sys.executable, "-B", "-m", "torch.distributed.run", "--nnodes=1",
            f"--nproc-per-node={args.processes}", "--max-restarts=0", "--rdzv-backend=static",
            "--master-addr=127.0.0.1", f"--master-port={port}", str(ROOT / WORKER),
            "--device", args.device, "--case", "suite", "--output-dir", str(ranks)]
        if args.device == "cuda":
            command.append("--require-cuda")
        report.update(stage="multi_rank_suite", worker_command=command)
        _write(output, report)
        row = _stream_process(command, directory / "torchrun.log", environment, args.timeout_seconds, stage=report["stage"])
        report["commands"].append(row)
        report["ranks"], errors = _rank_results(ranks, plan, identity)
        report["validation_errors"] = errors
        if row["exit_code"] != 0 or row["timed_out"] or errors:
            raise RuntimeError("DDP qualification failed: " + (errors[0] if errors else "torchrun failed or timed out"))
        if _installed(plan) != identity or _source_pins(json.loads(args.manifest.read_text())["sources"]) != plan["sources"]:
            raise RuntimeError("Qualification package or helper sources changed")
        if _reference(args.manifest) != plan["manifest"] or _reference(plan["wheel"]["path"]) != plan["wheel"]:
            raise RuntimeError("Release manifest or wheel changed during qualification")
        report.update(status="completed", stage="completed", CUDA_execution_proven=args.device == "cuda")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write(output, report)
        _archive(output, directory, report)
        _write(output, report)
    print(json.dumps({"status": report["status"], "stage": report["stage"], "output": str(output),
        "archive": report["archive"]["path"], "CUDA_execution_proven": report["CUDA_execution_proven"],
        "performance_qualified": False, "error": report["error"]}, allow_nan=False), flush=True)
    return 0 if report["status"] == "completed" else 1


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
