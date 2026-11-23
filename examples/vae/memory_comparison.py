"""Validate memory-isolation receipts without importing Torch in supervisors."""


DEFAULT_START_FREE_TOLERANCE_BYTES = 256 * 2**20
CUDA_FIELDS = ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
               "inactive_split_bytes", "allocation_retries", "ooms", "device_free_bytes", "device_total_bytes")


def _integer(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def worker_memory_errors(row):
    audit = row.get("memory_audit")
    if not isinstance(audit, dict) or audit.get("enabled") is not True:
        return ["memory_audit_missing_or_disabled"]
    errors = []
    if type(audit.get("schema_version")) is not int or audit["schema_version"] != 1:
        errors.append("memory_audit_schema")
    if audit.get("complete") is not True or audit.get("status") != "complete":
        errors.append("memory_audit_incomplete")
    process = audit.get("process") or {}
    if not isinstance(process, dict) or not _integer(process.get("pid"), positive=True):
        errors.append("memory_audit_process_identity")
    if not isinstance(audit.get("errors"), list) or audit["errors"]:
        errors.append("memory_audit_collection_errors")
    cleanup = audit.get("cleanup") or {}
    if (not isinstance(cleanup, dict) or cleanup.get("attempted") is not True
            or cleanup.get("completed") is not True):
        errors.append("memory_cleanup_incomplete")
    snapshots = audit.get("snapshots") or {}
    if not isinstance(snapshots, dict):
        return errors + ["memory_audit_snapshot_structure"]
    required = ("before_model", "after_setup", "after_first_update", "after_warmup",
                "measurement_start", "measurement_end", "after_postchecks", "cleanup")
    if any(not isinstance(snapshots.get(name), dict) for name in required):
        errors.append("memory_audit_missing_stage")
    if any(not isinstance(snapshots.get(name), dict) or snapshots[name].get("complete") is not True
           for name in required):
        errors.append("memory_audit_incomplete_stage")
    if str(row.get("device", "")).startswith("cuda"):
        for name in required:
            stage = snapshots.get(name)
            if not isinstance(stage, dict):
                continue
            if (any(not _integer(stage.get(field)) for field in CUDA_FIELDS)
                    or not _integer(stage.get("device_total_bytes"), positive=True)
                    or stage.get("device_free_bytes", 0) > stage.get("device_total_bytes", 0)
                    or stage.get("synchronized") is not True):
                errors.append("memory_audit_invalid_CUDA_stage:" + name)
        before = snapshots.get("before_model") or {}
        if not isinstance(before, dict):
            return errors + ["memory_audit_start_structure"]
        if (not _integer(before.get("allocated_bytes")) or not _integer(before.get("reserved_bytes"))
                or before["allocated_bytes"] != 0 or before["reserved_bytes"] != 0):
            errors.append("memory_audit_start_allocator_not_empty")
        if (not _integer(before.get("device_free_bytes"), positive=True)
                or not _integer(before.get("device_total_bytes"), positive=True)
                or before.get("device_free_bytes", 0) > before.get("device_total_bytes", 0)):
            errors.append("memory_audit_start_device_memory_missing")
        if before.get("synchronized") is not True:
            errors.append("memory_audit_start_not_synchronized")
    return errors


def compare_memory_baselines(rows, *, tolerance_bytes=DEFAULT_START_FREE_TOLERANCE_BYTES):
    if not _integer(tolerance_bytes):
        raise ValueError("Starting free-memory tolerance must be a nonnegative integer byte count")
    result = {"valid": False, "invalid_reasons": [], "tolerance_bytes": tolerance_bytes,
              "starting_free_bytes": [], "evidence_scope": "CPU_diagnostic_no_CUDA_allocator_evidence",
              "worker_cleanup_is_context_exit": False,
              "release_boundary": "supervisor waits for fresh worker process exit"}
    for row in rows:
        label = f"{row.get('block', row.get('trial', 1))}:{row.get('arm', row.get('execution_mode', 'worker'))}"
        errors = worker_memory_errors(row)
        result["invalid_reasons"].extend(label + ":" + error for error in errors)
        if str(row.get("device", "")).startswith("cuda") and not errors:
            before = row["memory_audit"]["snapshots"]["before_model"]
            result["starting_free_bytes"].append({"worker": label,
                "pid": row["memory_audit"]["process"]["pid"],
                "bytes": before["device_free_bytes"], "total_bytes": before["device_total_bytes"]})
    starts = result["starting_free_bytes"]
    if starts:
        values = [entry["bytes"] for entry in starts]
        result.update(evidence_scope="CUDA_worker_start_allocator_and_free_memory_audit",
                      starting_free_range_bytes=max(values) - min(values))
        if max(values) - min(values) > tolerance_bytes:
            result["invalid_reasons"].append("starting_free_memory_differs_beyond_declared_tolerance")
        if len({entry["total_bytes"] for entry in starts}) != 1:
            result["invalid_reasons"].append("starting_device_capacity_mismatch")
    if not rows:
        result["invalid_reasons"].append("no_worker_memory_receipts")
    result["valid"] = not result["invalid_reasons"]
    return result
