"""Permanent, verified compiler reference files; never tensor checkpoints.

Compiler working directories are transient. This store copies their permitted
code and diagnostics into unique runs and keeps completed and failed attempts
in an index. It never executes, imports, or unpickles a saved artifact.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid


ALLOWED_SUFFIXES = frozenset("""
.py .txt .log .json .csv .md .yaml .yml .dot .svg .html .css .js
.cpp .cc .c .h .hpp .cu .cuh .hip .ptx .ttir .ttgir .llir .mlir
.bc .cubin .hsaco .spv .so .dylib .dll .s .asm .sass .o
""".split())
FORBIDDEN_SUFFIXES = frozenset(".pt .pth .pkl .pickle .safetensors .npy .npz .tensor .bin".split())
STATUSES = frozenset({"complete", "partial", "blocked", "failed"})
SCHEMA_VERSION = 1
CHUNK_BYTES = 1024 * 1024


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_new(path, value):
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_new(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-")[:64] or "unspecified"


def _no_symlinks(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlink paths are not permitted: {part}")
    return path


def _run_path(run_dir, *, mutable=False):
    path = _no_symlinks(run_dir)
    if path.parent.name != "runs" or not (path / "request.json").is_file():
        raise ValueError("Not a reference-store run directory")
    if mutable and (path / "manifest.json").exists():
        raise FileExistsError("Finalized reference runs are immutable; create a new run")
    return path


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _digest(path):
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Cannot hash a non-regular artifact")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for data in iter(lambda: stream.read(CHUNK_BYTES), b""):
                digest.update(data)
                size += len(data)
    finally:
        os.close(descriptor)
    return size, digest.hexdigest()


def artifact_category(path):
    """Classify saved files without treating a profile as generated code."""
    path = Path(path)
    name = path.name.lower()
    if name.startswith("ir_pre_fusion"):
        return "ir_pre_fusion"
    if name.startswith("ir_post_fusion"):
        return "ir_post_fusion"
    if name.startswith("fx_graph"):
        return "fx_graph"
    if path.suffix.lower() in {".ttir", ".ttgir", ".llir", ".mlir", ".bc"}:
        return "kernel_ir"
    if path.suffix.lower() in {".dot", ".svg"}:
        return "graph_diagram"
    if path.suffix.lower() in {".ptx", ".cubin", ".hsaco", ".spv", ".sass"}:
        return "kernel_code"
    if path.suffix.lower() in {".py", ".cpp", ".cc", ".c", ".cu", ".hip", ".so", ".dylib", ".dll", ".s", ".asm", ".o"}:
        return "generated_code"
    if path.suffix.lower() in {".log", ".txt"}:
        return "logs"
    return "metadata"


def create_run(output_root, request):
    """Create one permanent, unique model/recipe attempt and return its path."""
    if not isinstance(request, dict):
        raise TypeError("Reference request must be a JSON object")
    _json_bytes(request)
    root = _no_symlinks(output_root)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    _no_symlinks(root / "runs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{stamp}_{_slug(request.get('model', 'model'))}_{_slug(request.get('recipe', 'recipe'))}_{uuid.uuid4().hex[:12]}"
    run = root / "runs" / name
    run.mkdir()
    _write_new(run / "request.json", {**request, "created_at": _timestamp()})
    return run


def _copy_file(source, destination, *, limit):
    _no_symlinks(source)
    if not stat.S_ISREG(source.lstat().st_mode):
        raise ValueError("Artifact is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(source, flags)
    owned = False
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("Artifact is not a bounded regular file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _no_symlinks(destination)
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "rb", closefd=False) as source_stream, destination.open("xb") as target:
            owned = True
            for data in iter(lambda: source_stream.read(CHUNK_BYTES), b""):
                size += len(data)
                if size > limit:
                    raise ValueError("Artifact grew beyond the copy budget")
                digest.update(data)
                target.write(data)
        if _stamp(before) != _stamp(os.fstat(descriptor)) or _stamp(before) != _stamp(source.stat()):
            raise RuntimeError("Artifact changed during copying")
        if _digest(source) != (size, digest.hexdigest()) or _digest(destination) != (size, digest.hexdigest()):
            raise RuntimeError("Artifact source/copy hash mismatch")
        return {"bytes": size, "sha256": digest.hexdigest(), "category": artifact_category(source)}
    except BaseException:
        if owned:
            destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)


def save_artifacts(run_dir, source_dir, *, namespace="inductor", max_file_bytes=256 * 2**20, max_total_bytes=2 * 2**30):
    """Copy permitted artifacts and persist explicit omission/error evidence.

    The total bound covers all artifact namespaces already saved for this run.
    Tensor and pickle files are identified by *any* suffix, including .pt.tmp.
    Symlink files/directories are recorded and never followed.
    """
    run = _run_path(run_dir, mutable=True)
    if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", namespace):
        raise ValueError("Artifact namespace must be one safe path component")
    if any(type(value) is not int or value <= 0 for value in (max_file_bytes, max_total_bytes)):
        raise ValueError("Artifact byte limits must be positive integers")
    source = _no_symlinks(source_dir)
    if not source.is_dir() or run.is_relative_to(source) or source.is_relative_to(run):
        raise ValueError("Artifact source must be a separate existing working directory")
    destination = run / "artifacts" / namespace
    if destination.exists():
        raise FileExistsError("Use a distinct namespace; saved artifacts are never overwritten")
    destination.mkdir(parents=True)
    report = {"schema_version": SCHEMA_VERSION, "namespace": namespace, "copied": [], "omitted": [], "errors": [],
              "max_file_bytes": max_file_bytes, "max_total_bytes": max_total_bytes, "tensor_files_saved": 0}
    used = sum(path.stat().st_size for path in (run / "artifacts").rglob("*") if path.is_file() and not path.is_symlink())
    membership = []
    for current, directories, files in os.walk(source, followlinks=False):
        current = Path(current)
        directories.sort()
        for name in sorted(directories[:]):
            if (current / name).is_symlink():
                report["errors"].append({"path": (current / name).relative_to(source).as_posix(), "reason": "symlink_directory"})
                directories.remove(name)
        for name in sorted(files):
            path = current / name
            relative = path.relative_to(source).as_posix()
            membership.append(relative)
            if path.is_symlink():
                report["errors"].append({"path": relative, "reason": "symlink_file"})
                continue
            suffixes = {suffix.lower() for suffix in path.suffixes}
            if suffixes & FORBIDDEN_SUFFIXES:
                report["omitted"].append({"path": relative, "reason": "tensor_or_pickle_forbidden"})
                continue
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                report["omitted"].append({"path": relative, "reason": "unrecognized_artifact_type"})
                continue
            try:
                size = path.stat().st_size
                if size > max_file_bytes or used + size > max_total_bytes:
                    report["errors"].append({"path": relative, "reason": "artifact_byte_limit", "bytes": size})
                    continue
                entry = _copy_file(path, destination / relative, limit=min(max_file_bytes, max_total_bytes - used))
                used += entry["bytes"]
                report["copied"].append({"path": f"artifacts/{namespace}/{relative}", **entry})
            except (OSError, ValueError, RuntimeError) as error:
                report["errors"].append({"path": relative, "reason": str(error), "error_type": type(error).__name__})
    after = sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file() or path.is_symlink())
    # Do not falsely report changing compiler directories as frozen captures.
    before = sorted(membership + [item["path"] for item in report["errors"] if item["reason"] == "symlink_directory"])
    if after != before:
        report["errors"].append({"path": ".", "reason": "source_membership_changed_or_symlink_tree"})
    for entry in report["copied"]:
        original = source / Path(entry["path"]).relative_to(f"artifacts/{namespace}")
        try:
            _no_symlinks(original)
            if _digest(original) != (entry["bytes"], entry["sha256"]):
                raise RuntimeError("Artifact changed after its copy snapshot")
        except (OSError, ValueError, RuntimeError) as error:
            report["errors"].append({"path": original.relative_to(source).as_posix(), "reason": str(error)})
    report.update(bytes_copied=sum(item["bytes"] for item in report["copied"]),
                  status="partial" if report["errors"] else "saved", all_files_saved=not report["errors"] and not report["omitted"])
    reports = run / "copy_reports"
    reports.mkdir(exist_ok=True)
    _write_new(reports / f"{namespace}.json", report)
    return report


def _inventory(run):
    files = {}
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError("Reference run contains a symlink")
        if path.is_file() and path != run / "manifest.json":
            name = path.relative_to(run).as_posix()
            if {suffix.lower() for suffix in path.suffixes} & FORBIDDEN_SUFFIXES:
                raise ValueError("Reference run contains a forbidden tensor/pickle file")
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                raise ValueError("Reference run contains an unrecognized artifact type")
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("Reference run contains a non-regular file")
            size, digest = _digest(path)
            files[name] = {"bytes": size, "sha256": digest,
                           "category": artifact_category(path) if name.startswith("artifacts/") else "store_metadata"}
    return files


def finalize_run(run_dir, metadata, status, *, required_artifacts=()):
    """Freeze an attempt, retain partial failure evidence, and refresh the index."""
    run = _run_path(run_dir, mutable=True)
    if status not in STATUSES or not isinstance(metadata, dict):
        raise ValueError("A valid reference status and JSON metadata object are required")
    _json_bytes(metadata)
    files = _inventory(run)
    categories = sorted({entry["category"] for entry in files.values() if entry["category"] != "store_metadata"})
    required = sorted(set(required_artifacts) | {"generated_code"})
    missing = sorted(set(required) - set(categories))
    copy_errors = []
    for path in sorted((run / "copy_reports").glob("*.json")):
        report = json.loads(path.read_text())
        copy_errors.extend({"namespace": report["namespace"], **error} for error in report["errors"])
    effective = "partial" if status == "complete" and (missing or copy_errors) else status
    manifest = {"schema_version": SCHEMA_VERSION, "run_id": run.name, "finalized_at": _timestamp(),
                "status": effective, "requested_status": status, "request": json.loads((run / "request.json").read_text()),
                "metadata": metadata, "files": files, "categories": categories, "required_artifacts": required,
                "missing_artifacts": missing, "copy_errors": copy_errors, "tensor_files_saved": 0,
                "total_bytes": sum(item["bytes"] for item in files.values()),
                "reference_eligible": effective == "complete", "storage_policy": "permanent_no_automatic_eviction"}
    _write_new(run / "manifest.json", manifest)
    refresh_index(run.parent.parent)
    return manifest


def verify_run(run_dir):
    """Verify file membership, byte sizes and SHA256s before using a reference."""
    run = _run_path(run_dir)
    manifest_path = _no_symlinks(run / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("run_id") != run.name:
        raise ValueError("Invalid reference manifest identity/schema")
    actual = _inventory(run)
    expected = manifest["files"]
    errors = [{"path": name, "reason": "unexpected_or_missing_or_changed"}
              for name in sorted(set(actual) | set(expected)) if actual.get(name) != expected.get(name)]
    return {"run_id": run.name, "valid": not errors, "files_checked": len(actual), "errors": errors,
            "manifest_sha256": _digest(manifest_path)[1]}


def refresh_index(output_root):
    """Rebuild a portable index; existing runs and artifacts are never deleted."""
    root = _no_symlinks(output_root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for run in sorted((root / "runs").glob("*")):
        try:
            _no_symlinks(run)
            if not run.is_dir() or not (run / "manifest.json").exists():
                continue
            manifest = json.loads(_no_symlinks(run / "manifest.json").read_text())
            verified = verify_run(run)
            records.append({"run_id": run.name, "manifest": f"runs/{run.name}/manifest.json",
                            "status": manifest["status"], "reference_eligible": manifest["reference_eligible"] and verified["valid"],
                            "integrity_valid": verified["valid"], "request": manifest["request"],
                            "metadata": manifest["metadata"], "categories": manifest["categories"],
                            "total_bytes": manifest["total_bytes"], "manifest_sha256": verified["manifest_sha256"]})
        except (OSError, ValueError, KeyError, TypeError) as error:
            records.append({"run_id": run.name, "manifest": f"runs/{run.name}/manifest.json",
                            "status": "integrity_error", "reference_eligible": False, "integrity_valid": False,
                            "error": str(error), "request": {}, "metadata": {}, "categories": [], "total_bytes": None})
    index = {"schema_version": SCHEMA_VERSION, "updated_at": _timestamp(), "storage_policy": "permanent_no_automatic_eviction",
             "tensor_files_saved": 0, "runs": records}
    _atomic_json(root / "index.json", index)
    return index


def find_references(output_root, *, model=None, recipe=None, device_type=None, torch_version=None, include_incomplete=False):
    """Find verified references with explicit hardware/software/recipe filters."""
    index = refresh_index(output_root)
    results = []
    for entry in index["runs"]:
        if not entry["integrity_valid"] or (not include_incomplete and not entry["reference_eligible"]):
            continue
        details = {**entry["request"], **entry["metadata"]}
        if all(value is None or details.get(key) == value for key, value in
               (("model", model), ("recipe", recipe), ("device_type", device_type), ("torch_version", torch_version))):
            results.append(entry)
    return results
