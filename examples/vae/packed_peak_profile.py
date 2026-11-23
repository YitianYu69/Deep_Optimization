"""Convert hashed resident CUDA diagnostics into an explicit target peak profile.

This offline metadata tool never executes a model or changes a policy. Enabled
offload allocations cannot be used as a resident baseline. Missing native
scope mappings, branching execution or a peak outside the mapped native scope
are rejected rather than projected onto an arbitrary schedule boundary.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

MAX_JSON_BYTES = 256 * 2**20


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    def constant(value):
        raise ValueError("Nonfinite JSON metadata: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate metadata key: " + key)
            result[key] = value
        return result
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("Metadata exceeds the explicit byte bound")
    value = json.loads(data, parse_constant=constant, object_pairs_hook=pairs)
    if type(value) is not dict:
        raise ValueError("Metadata must be an object")
    return value, {"path": str(Path(path).resolve()), "bytes": len(data), "sha256": _sha(data)}


def _artifact(receipt, reference):
    if (type(reference) is not dict or type(reference.get("path")) is not str
            or type(reference.get("bytes")) is not int or not 0 <= reference["bytes"] <= MAX_JSON_BYTES
            or type(reference.get("sha256")) is not str or len(reference["sha256"]) != 64):
        raise ValueError("Artifact requires bounded byte count and SHA256")
    supplied = Path(reference["path"])
    if supplied.suffix != ".json":
        raise ValueError("Only scalar JSON artifacts are accepted")
    candidates = [supplied if supplied.is_absolute() else receipt.parent / supplied]
    # Uploaded bundles relocate absolute worker paths. Search only this
    # receipt's own artifact tree, then still require the exact declared hash.
    own = receipt.parent / (receipt.stem + "-artifacts")
    if own.is_dir():
        candidates += list(own.rglob(supplied.name))[:4096]
    for path in dict.fromkeys(candidates):
        if path.is_file() and path.stat().st_size == reference["bytes"]:
            value, metadata = _read(path)
            if metadata["sha256"] == reference["sha256"]:
                return value, metadata
    raise ValueError("Missing or hash-mismatched declared artifact: " + supplied.name)


def _model_scope(report):
    from examples.vae.execution_inventory import _declared_model_scopes

    records, errors = _declared_model_scopes(report)
    matches = {}
    for _, value in records:
        if any(name not in value for name in ("identity", "regions", "inventory", "peak_windows")):
            raise ValueError("Target needs one unambiguous compiled whole-model scope")
        matches[_sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())] = value
    if errors or len(matches) != 1:
        raise ValueError("Target needs one unambiguous compiled whole-model scope")
    return next(iter(matches.values()))


def derive_profile(resident_path, target_path):
    from examples.vae.benchmark_packed_offload import _identity

    resident_path, target_path = Path(resident_path).resolve(), Path(target_path).resolve()
    resident, resident_ref = _read(resident_path)
    target, target_ref = _read(target_path)
    for name, row in (("resident", resident), ("target", target)):
        if (row.get("status") != "completed" or row.get("execution_completed") is not True
                or row.get("native_ownership_verified") is not True or not str(row.get("device", "")).startswith("cuda")
                or any(row.get(key) is not True for key in ("installed_stable", "source_stable", "helper_source_stable", "input_stable"))):
            raise ValueError(name + " must be a completed source-pinned native CUDA execution")
        if not row.get("software", {}).get("gpu", {}).get("uuid"):
            raise ValueError("Missing actual GPU identity")
    policy = resident.get("settings", {}).get("runtime", {}).get("packed_activation_offload", {})
    target_policy = target.get("settings", {}).get("runtime", {}).get("packed_activation_offload", {})
    if policy.get("enabled") is not False or target_policy.get("enabled") is not True:
        raise ValueError("A disabled resident diagnostic and separate enabled target are required")
    if _identity(resident) != _identity(target):
        raise ValueError("Installed/workload/input/state/RNG/software/native math identities differ")
    identity = _identity(resident)
    for name in ("origin_sha256", "initial_model_sha256", "initial_optimizer_sha256", "installed", "common_settings", "helpers"):
        if not identity.get(name):
            raise ValueError("Missing comparison identity: " + name)
    profile = resident.get("profile", {})
    if profile.get("complete") is not True or profile.get("cuda_execution_evidence") is not True:
        raise ValueError("Resident requires complete actual CUDA profiler activity")
    storage, storage_ref = _artifact(resident_path, profile.get("storage_ledger"))
    joined, joined_ref = _artifact(resident_path, profile.get("allocator_storage_join"))
    compiled, compiled_ref = _artifact(target_path, target.get("compiler_report_artifact"))
    if (storage.get("finished") is not True or storage.get("metadata_only") is not True
            or storage.get("cuda_execution_evidence") is not True or storage.get("errors")
            or storage.get("omitted_boundaries", 0) or storage.get("omitted_allocator_samples", 0)
            or joined.get("omitted_events", 0)):
        raise ValueError("Resident native/allocator boundary coverage is incomplete")
    scope = _model_scope(compiled)
    regions = scope["regions"]
    if not isinstance(regions, list) or not regions:
        raise ValueError("Target lacks ordered native region signatures")
    invocations, windows = {}, {}

    def boundary_key(row):
        index, phase, boundary = row.get("native_region_index"), row.get("phase"), row.get("index")
        if type(index) is not int or not 0 <= index < len(regions) or phase not in ("forward", "backward") or type(boundary) is not int or boundary < 0:
            raise ValueError("Missing native region/phase/boundary mapping")
        expected = regions[index]
        for actual, key in (("native_forward_signature", "forward_signature"),
                ("native_backward_signature", "backward_signature"), ("native_recipe_identity_sha256", "recipe_identity_sha256")):
            value = row.get(actual)
            if type(value) is not str or len(value) != 64 or value != expected.get(key):
                raise ValueError("Resident and target native schedule/recipe signatures differ")
        return index, phase, boundary

    for row in storage.get("boundaries", ()):
        index, phase, _ = boundary_key(row)
        invocation = row.get("invocation")
        if type(invocation) is not int or invocations.setdefault(invocation, (index, phase)) != (index, phase):
            raise ValueError("Ambiguous native invocation mapping")
    expected = [(index, "forward") for index in range(len(regions))] + [(index, "backward") for index in reversed(range(len(regions)))]
    actual = [invocations[key] for key in sorted(invocations)]
    if not actual or len(actual) % len(expected) or actual != expected * (len(actual) // len(expected)):
        raise ValueError("Only a proven ordered forward/reverse-backward chain is supported")
    for row in storage.get("allocator_samples", ()):
        value = row.get("allocator_allocated_bytes")
        if row.get("native_region_index") is not None and type(value) is int and value >= 0:
            key = boundary_key(row)
            windows[key] = max(windows.get(key, 0), value)
    observed_peak = 0
    for event in joined.get("events", ()):
        value = event.get("total_allocated_bytes")
        if type(value) is not int or value < 0:
            continue
        observed_peak = max(observed_peak, value)
        row = event.get("native_boundary")
        if row is not None:
            key = boundary_key(row)
            windows[key] = max(windows.get(key, 0), value)
    if not windows or max(windows.values()) < observed_peak:
        raise ValueError("The allocator peak is outside the mapped native scope; no arbitrary boundary projection")
    result = {"schema_version": 1, "scope_identity": scope["identity"], "cuda_execution_verified": True,
        "source_receipt_sha256": resident_ref["sha256"], "windows": [
            {"region": region, "phase": phase, "boundary": boundary, "baseline_bytes": value,
                "source": "target_storage_ledger"} for (region, phase, boundary), value in sorted(windows.items())]}
    evidence = {"schema_version": 1, "resident_receipt": resident_ref, "target_receipt": target_ref,
        "converter_source_sha256": _sha(Path(__file__).read_bytes()),
        "storage_ledger": storage_ref, "allocator_join": joined_ref, "target_compiler_report": compiled_ref,
        "comparison_identity_sha256": _sha(json.dumps(identity, sort_keys=True).encode()),
        "baseline": "resident_Torch_allocator_bytes_not_enabled_offload_allocations",
        "physical_storage_attribution_complete": storage.get("attribution_complete") is True,
        "unknown_vendor_workspace": True, "automatic_policy_promotion": False,
        "scope": "explicit_import_candidate_only_not_a_performance_or_release_qualification"}
    return result, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resident-result", type=Path, required=True)
    parser.add_argument("--target-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile, evidence = derive_profile(args.resident_result, args.target_result)
    if args.output.exists() or args.output.with_suffix(".evidence.json").exists():
        raise FileExistsError("Choose a fresh peak-profile output and companion receipt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, value in ((args.output, profile), (args.output.with_suffix(".evidence.json"), evidence)):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "windows": len(profile["windows"]), "automatic_policy_promotion": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
