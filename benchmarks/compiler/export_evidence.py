"""Export complete scalar reports with content-addressed JSON subtree sharing.

This optional archival format leaves live qualification reports untouched.
Every logical file remains indexed and every timing sample/launch proof is
recoverable with read_evidence. Tensor artifacts are rejected, never exported.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


REFERENCE = "$deepopt_evidence_blob"
MAX_FILE_BYTES = 256 * 2**20
REPORT_SUFFIXES = {".json", ".csv", ".log", ".md", ".txt", ".gz"}
COMPILER_SUFFIXES = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".cuh", ".hip", ".ptx", ".ttir",
                     ".ttgir", ".llir", ".mlir", ".bc", ".cubin", ".hsaco", ".spv", ".so", ".dylib", ".dll",
                     ".s", ".asm", ".sass", ".o", ".dot", ".svg", ".html", ".css", ".js", ".yaml", ".yml"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts) or "\\" in name:
        raise ValueError("Evidence contains an unsafe relative path")
    return path.as_posix()


def export_evidence(source, destination, *, subtree_bytes=4096, max_file_bytes=MAX_FILE_BYTES):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if type(subtree_bytes) is not int or subtree_bytes < 256 or type(max_file_bytes) is not int or max_file_bytes < 1:
        raise ValueError("Evidence export bounds must be positive")
    if not source.is_dir() or destination.is_relative_to(source):
        raise ValueError("Evidence destination must be outside the report source directory")
    paths = sorted(path for path in source.rglob("*") if path.is_file())
    membership = [path.relative_to(source).as_posix() for path in paths]
    if any(path.is_symlink() or not path.resolve().is_relative_to(source) for path in paths):
        raise ValueError("Evidence source contains external/symlink files")
    tensors = [path for path in paths if set(suffix.lower() for suffix in path.suffixes)
               & {".pt", ".pth", ".safetensors", ".npy", ".npz", ".pkl", ".pickle", ".tensor", ".bin"}]
    if tensors:
        raise ValueError("Checkpoint-free evidence export refuses tensor artifacts")
    paths = [path for path in paths if path.suffix.lower() in REPORT_SUFFIXES | COMPILER_SUFFIXES]
    if destination.exists():
        raise FileExistsError("Use a fresh evidence export path; historical artifacts are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "files": {}, "blobs": {}, "tensor_files": 0,
                "lossless_json_values": True, "subtree_bytes": subtree_bytes,
                "qualification_reports_modified": False}
    stamps = {}
    owned = False
    try:
        stream = destination.open("xb")
        owned = True
        with stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            def blob(data, kind):
                digest = hashlib.sha256(data).hexdigest()
                name = f"blobs/{kind}/{digest}"
                if name not in manifest["blobs"]:
                    archive.writestr(name, data)
                    manifest["blobs"][name] = {"bytes": len(data), "sha256": digest, "kind": kind}
                return name

            def encode(value, *, root=False):
                if isinstance(value, dict):
                    if REFERENCE in value:
                        raise ValueError("Source JSON uses the evidence format's reserved reference key")
                    value = {key: encode(child) for key, child in value.items()}
                elif isinstance(value, list):
                    value = [encode(child) for child in value]
                else:
                    return value
                data = _json(value)
                return {REFERENCE: blob(data, "json")} if root or len(data) >= subtree_bytes else value

            for path in paths:
                before = path.stat()
                if path.stat().st_size > max_file_bytes:
                    raise ValueError(f"Evidence file exceeds the explicit per-file limit: {path.name}")
                data = path.read_bytes()
                name = _safe_name(path.relative_to(source).as_posix())
                entry = {"source_bytes": len(data), "source_sha256": hashlib.sha256(data).hexdigest()}
                if path.suffix.lower() == ".json":
                    value = json.loads(data)
                    entry.update(kind="json", canonical_sha256=hashlib.sha256(_json(value)).hexdigest(),
                                 value=encode(value, root=True))
                else:
                    entry.update(kind="raw", blob=blob(data, "raw"))
                if path.read_bytes() != data:
                    raise RuntimeError(f"Evidence changed during export: {name}")
                after = path.stat()
                stamps[path] = (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                if stamps[path] != (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
                    raise RuntimeError(f"Evidence changed during export: {name}")
                manifest["files"][name] = entry
            archive.writestr("evidence-manifest.json", _json(manifest))
        # No source file may disappear or appear while an export is frozen.
        after = sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file())
        if after != membership:
            raise RuntimeError("Evidence file membership changed during export")
        for path, stamp in stamps.items():
            current = path.stat()
            if stamp != (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns):
                raise RuntimeError("Evidence changed after its export snapshot")
    except BaseException:
        if owned:
            destination.unlink(missing_ok=True)
        raise
    return {"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "bytes": destination.stat().st_size, "logical_files": len(manifest["files"]),
            "unique_blobs": len(manifest["blobs"]), "source_bytes": sum(x["source_bytes"] for x in manifest["files"].values())}


def read_evidence(path, name, *, max_bytes=MAX_FILE_BYTES):
    """Reconstruct one complete report, verifying every referenced blob hash."""
    name = _safe_name(name)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(_safe_name(item) != item for item in names):
            raise ValueError("Evidence archive has duplicate or unsafe member names")
        if archive.getinfo("evidence-manifest.json").file_size > max_bytes:
            raise ValueError("Evidence manifest exceeds its byte limit")
        manifest = json.loads(archive.read("evidence-manifest.json"))
        if manifest.get("schema_version") != 1 or set(names) != set(manifest["blobs"]) | {"evidence-manifest.json"}:
            raise ValueError("Evidence manifest/membership is invalid")
        entry = manifest["files"][name]
        remaining = max_bytes

        def load_blob(key):
            nonlocal remaining
            record = manifest["blobs"][key]
            size = archive.getinfo(key).file_size
            remaining -= size
            if remaining < 0 or size != record["bytes"]:
                raise ValueError("Evidence reconstruction exceeds its bounded byte budget")
            data = archive.read(key)
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Evidence blob identity does not match its content")
            return data

        def decode(value, depth=0):
            if depth > 100:
                raise ValueError("Evidence references exceed the nesting limit")
            if isinstance(value, dict):
                if REFERENCE in value:
                    if set(value) != {REFERENCE}:
                        raise ValueError("Malformed evidence reference")
                    return decode(json.loads(load_blob(value[REFERENCE])), depth + 1)
                return {key: decode(child, depth + 1) for key, child in value.items()}
            if isinstance(value, list):
                return [decode(child, depth + 1) for child in value]
            return value

        if entry["kind"] == "raw":
            data = load_blob(entry["blob"])
            if hashlib.sha256(data).hexdigest() != entry["source_sha256"]:
                raise ValueError("Reconstructed raw report differs from the source identity")
            return data
        value = decode(entry["value"])
        if hashlib.sha256(_json(value)).hexdigest() != entry["canonical_sha256"]:
            raise ValueError("Reconstructed JSON report differs from the source identity")
        return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export_evidence(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
