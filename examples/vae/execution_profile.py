"""Separate cold or post-measurement diagnostics; never a timing benchmark.

Steady profiling runs extra updates after the clean measurement is frozen.
Cold diagnostics run the fresh worker's first update, including lazy capture.
Only JSON metadata and a Chrome trace leave this helper. Runtime tensors remain
invocation-local and are never serialized.
"""

import hashlib
import json
import math
from pathlib import Path
from contextlib import nullcontext
import traceback


MAX_TRACE_BYTES = 256 * 2**20
MAX_SUMMARY_ROWS = 1024
MAX_ALLOCATOR_RECORDS = 4096


def _save_json(directory, name, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    path = directory / name
    with path.open("xb") as stream:
        stream.write(data)
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


class _PhaseRanges:
    """Close each host phase even when a profiled update raises."""

    def __init__(self, torch, context):
        self.torch, self.context, self.active = torch, context, None

    def __call__(self, event):
        self.close()
        phase = event["phase"]
        self.context["phase"] = phase
        self.active = self.torch.profiler.record_function("vae.profile." + phase)
        self.active.__enter__()

    def close(self):
        if self.active is not None:
            active, self.active = self.active, None
            active.__exit__(None, None, None)


def _allocator_snapshot(torch, device):
    if device.type != "cuda":
        return {"scope": "CPU_no_CUDA_allocator_evidence", "segments": [], "omitted_records": 0}
    index = device.index if device.index is not None else torch.cuda.current_device()
    segments, records, omitted = [], 0, 0
    for segment in torch.cuda.memory_snapshot():
        if segment.get("device") != index:
            continue
        blocks = segment.get("blocks", ())
        if records >= MAX_ALLOCATOR_RECORDS:
            omitted += 1 + len(blocks)
            continue
        row = {key: segment[key] for key in ("device", "address", "total_size", "allocated_size", "active_size", "stream")
               if type(segment.get(key)) in (int, str, bool)}
        row["blocks"] = []
        records += 1
        for block in blocks:
            if records >= MAX_ALLOCATOR_RECORDS:
                omitted += 1
                continue
            row["blocks"].append({key: block[key] for key in ("address", "size", "requested_size", "state")
                                  if type(block.get(key)) in (int, str, bool)})
            records += 1
        segments.append(row)
    return {"scope": "bounded_post_profile_Torch_allocator_blocks_not_tensor_payloads",
            "segments": segments, "records": records, "record_limit": MAX_ALLOCATOR_RECORDS,
            "omitted_records": omitted, "vendor_workspace_complete": False}


def _operator_summary(profile):
    events = profile.key_averages()
    rows = []
    for event in events:
        row = {"name": str(event.key), "count": int(event.count)}
        for name in ("cpu_time_total", "self_cpu_time_total", "device_time_total", "self_device_time_total",
                     "cpu_memory_usage", "self_cpu_memory_usage", "device_memory_usage", "self_device_memory_usage"):
            row[name] = _number(getattr(event, name, None))
        rows.append(row)
    rows.sort(key=lambda item: item.get("self_device_time_total") or item.get("self_cpu_time_total") or 0, reverse=True)
    device_events = sum("CUDA" in str(event.device_type).upper() for event in profile.events())
    return {"rows": rows[:MAX_SUMMARY_ROWS], "row_limit": MAX_SUMMARY_ROWS,
            "omitted_rows": max(0, len(rows) - MAX_SUMMARY_ROWS), "time_units": "microseconds",
            "cuda_activity_events": device_events,
            "cuda_activity_count_is_kernel_launch_count": False,
            "latencies_are_profiled_not_benchmark_samples": True}


def run_execution_profile(*, torch, device, directory, update, first_update, updates,
                          record_memory=False, record_shapes=False, ownership=None,
                          expected_ownership=None, allocator_diagnostics=None, postchecks=None, context=None,
                          storage_trace=None, diagnostic_phase="steady", allow_initial_compilation=False):
    """Collect extra-update diagnostics without mutating the clean receipt.

    ``update(number, progress)`` submits the ordinary training step. It must not
    hash tensors or perform tensor checkpoint export. Metadata observers may
    be installed explicitly by this helper in separate diagnostic workers. Exceptions in
    this optional path are returned as incomplete diagnostics, never re-raised
    into the completed benchmark.
    """
    result = {"requested": True, "status": "incomplete", "complete": False,
        "evidence_scope": "target_CUDA_profile" if device.type == "cuda" else "CPU_diagnostic_profile_only",
        "cuda_execution_evidence": False, "measurement_eligible": False,
        "first_update": first_update, "requested_updates": updates, "updates_completed": 0,
        "diagnostic_phase": diagnostic_phase,
        "record_memory": record_memory, "record_shapes": record_shapes, "with_stack": False,
        "tensor_files_written": 0, "files": [], "omissions": [],
        "phase_note": ("fresh first update including lazy compilation; no warmup or measured samples"
            if diagnostic_phase == "cold" else
            "extra updates after frozen measurement; forward_and_loss is one host range; native forward ranges nest inside it")}
    directory = Path(directory)
    context = {} if context is None else context
    context.update(stage="cold_first_update_diagnostic" if diagnostic_phase == "cold" else "profile_after_measurement",
                   phase="profile_setup", update=None)
    profiler = None
    try:
        if type(updates) is not int or not 1 <= updates <= 3:
            raise ValueError("Profile update count must be between 1 and 3")
        directory.mkdir(parents=True, exist_ok=False)
        before = ownership() if ownership is not None else None
        result["ownership_before"] = before
        result["expected_ownership"] = expected_ownership
        if expected_ownership is not None and before != expected_ownership:
            raise RuntimeError("Profile no longer owns the frozen measured program")
        if allocator_diagnostics is not None:
            result["allocator_before"] = allocator_diagnostics()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                raise RuntimeError("CUDA profiler activity is unavailable; no CPU profile substitutes for CUDA evidence")
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities, record_shapes=record_shapes,
            profile_memory=record_memory, with_stack=False)
        with profiler, storage_trace.capture() if storage_trace is not None else nullcontext():
            for number in range(first_update, first_update + updates):
                context.update(phase="schedule", update=number)
                phases = _PhaseRanges(torch, context)
                try:
                    with torch.profiler.record_function("vae.profile.update"):
                        update(number, phases)
                finally:
                    phases.close()
                result["updates_completed"] += 1
                profiler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        context.update(phase="profile_export", update=None)
        summary = _operator_summary(profiler)
        result["cuda_execution_evidence"] = device.type == "cuda" and summary["cuda_activity_events"] > 0
        result["files"].append(_save_json(directory, "operators.json", summary))
        trace = directory / "trace.json"
        profiler.export_chrome_trace(str(trace))
        if trace.stat().st_size > MAX_TRACE_BYTES:
            size = trace.stat().st_size
            trace.unlink()
            result["omissions"].append({"artifact": "trace.json", "reason": "trace_byte_limit",
                                        "bytes": size, "limit_bytes": MAX_TRACE_BYTES})
        else:
            data = trace.read_bytes()
            result["files"].append({"path": str(trace), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if storage_trace is not None:
            from examples.vae.execution_storage import join_allocator_trace

            storage_trace.finish()
            result["storage_ledger"] = storage_trace.save(directory)
            result["files"].append(result["storage_ledger"])
            result["storage_metadata_complete"] = result["storage_ledger"]["status"] == "complete_observed_boundaries"
            result["storage_attribution_complete"] = False
            if result["storage_ledger"].get("omission"):
                result["omissions"].append({"artifact": "storage-ledger.json", "reason": result["storage_ledger"]["omission"]})
            if record_memory and trace.is_file():
                joined = join_allocator_trace(trace, storage_trace.as_dict())
                result["allocator_storage_join"] = _save_json(directory, "allocator-storage-join.json", joined)
                result["files"].append(result["allocator_storage_join"])
        if record_memory:
            snapshot = _allocator_snapshot(torch, device)
            result["files"].append(_save_json(directory, "allocator.json", snapshot))
            if snapshot["omitted_records"]:
                result["omissions"].append({"artifact": "allocator.json", "reason": "record_limit",
                                            "records": snapshot["omitted_records"]})
        after = ownership() if ownership is not None else None
        result["ownership_after"] = after
        result["same_measured_program"] = None if allow_initial_compilation else before is not None and before == after
        result["initial_compilation_allowed"] = allow_initial_compilation
        if before != after and not allow_initial_compilation:
            raise RuntimeError("Ownership or graph count changed during optional profiling")
        if postchecks is not None:
            checks = postchecks()
            result["postchecks"] = checks
            if not checks or any(value is not True for value in checks.values()):
                raise RuntimeError("Optional profile postchecks found nonfinite state")
        if device.type == "cuda" and not result["cuda_execution_evidence"]:
            raise RuntimeError("CUDA profile contains no observed CUDA activity")
        result["complete"] = not result["omissions"]
        result["status"] = "completed" if result["complete"] else "incomplete"
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      failure_context=dict(context))
    finally:
        if storage_trace is not None and "storage_ledger" not in result and directory.is_dir():
            try:
                storage_trace.finish()
                result["storage_ledger"] = storage_trace.save(directory)
                result["files"].append(result["storage_ledger"])
            except Exception as error:
                result.setdefault("collection_errors", []).append({"field": "storage_ledger",
                    "error": f"{type(error).__name__}: {error}"})
        # Profile allocator collection can fail independently of execution.
        if allocator_diagnostics is not None:
            try:
                result["allocator_after"] = allocator_diagnostics()
            except Exception as error:
                result.setdefault("collection_errors", []).append({"field": "allocator_after",
                    "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
                result.update(complete=False, status="incomplete")
        context.update(phase="profile_complete", update=None)
        if directory.is_dir():
            try:
                metadata = {key: value for key, value in result.items() if key != "files"}
                result["files"].append(_save_json(directory, "profile.json", metadata))
            except Exception as error:
                result.setdefault("collection_errors", []).append({"field": "profile_json",
                    "error": f"{type(error).__name__}: {error}"})
                result.update(complete=False, status="incomplete")
    return result
