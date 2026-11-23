"""Staged allocator metadata outside the VAE worker's measured updates.

This helper owns no model, optimizer, runtime or tensor references. Fresh
processes remain the isolation boundary; allocator snapshots cannot establish
the lifetime of opaque vendor workspaces or another process's allocations.
"""

from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import traceback


MAX_PROCESS_ROWS = 64
MAX_PROCESS_OUTPUT_BYTES = 64 * 1024
_MEMORY_FIELDS = (
    "allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
    "inactive_split_bytes", "allocation_retries", "ooms", "device_free_bytes", "device_total_bytes",
)


def host_capacity_snapshot(*, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup"), environment=None):
    """Read physical/job RAM limits; unavailable metadata never means free RAM."""
    environment = os.environ if environment is None else environment
    result = {"schema_version": 1, "status": "unknown", "available_bytes": None,
        "MemAvailable_bytes": None, "cgroup_limits": [], "slurm": {}, "errors": [],
        "scope": "read_only_host_and_job_capacity_snapshot_not_a_reservation"}
    capacities, unresolved_limits = [], False
    try:
        for line in (proc_root / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) == 3 and parts[2] == "kB":
                    available = int(parts[1]) * 1024
                    if available < 0:
                        raise ValueError("Negative MemAvailable")
                    result["MemAvailable_bytes"] = available
                    capacities.append(available)
    except (OSError, ValueError) as error:
        result["errors"].append({"source": "MemAvailable", "error_type": type(error).__name__})
    relative_paths = []
    try:
        for line in (proc_root / "self/cgroup").read_text().splitlines():
            _, controllers, name = line.split(":", 2)
            path = Path(name.lstrip("/"))
            if ".." not in path.parts and (not controllers or "memory" in controllers.split(",")):
                prefix = cgroup_root if not controllers else cgroup_root / "memory"
                relative_paths.extend((prefix / ancestor, str(ancestor)) for ancestor in (path, *path.parents))
    except (OSError, ValueError) as error:
        result["errors"].append({"source": "process_cgroup", "error_type": type(error).__name__})
    relative_paths.append((cgroup_root, "."))
    seen = set()
    for folder, relative in relative_paths[:128]:
        for limit_name, current_name in (("memory.max", "memory.current"), ("memory.limit_in_bytes", "memory.usage_in_bytes")):
            key = str(folder / limit_name)
            if key in seen or not (folder / limit_name).is_file():
                continue
            seen.add(key)
            try:
                raw = (folder / limit_name).read_text().strip()
                if raw == "max":
                    continue
                limit = int(raw)
                if limit >= 1 << 60:
                    continue
                if limit < 0:
                    raise ValueError("Negative memory limit")
                current = int((folder / current_name).read_text().strip())
                if current < 0:
                    raise ValueError("Negative cgroup memory usage")
                available = max(0, limit - current)
                result["cgroup_limits"].append({"source": key, "relative_scope": relative,
                    "limit_bytes": limit, "current_bytes": current, "available_bytes": available})
                capacities.append(available)
            except (OSError, ValueError) as error:
                unresolved_limits = True
                result["errors"].append({"source": key, "error_type": type(error).__name__})
    slurm_keys = ("SLURM_JOB_ID", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_ON_NODE")
    result["slurm"]["environment"] = {key: environment.get(key) for key in slurm_keys}
    job_limit = None
    try:
        if environment.get("SLURM_MEM_PER_NODE"):
            value = int(environment["SLURM_MEM_PER_NODE"])
            if value < 0:
                raise ValueError("Negative Slurm memory limit")
            job_limit = value * 2**20 if value > 0 else None
        elif environment.get("SLURM_MEM_PER_CPU"):
            memory, cpus = int(environment["SLURM_MEM_PER_CPU"]), int(environment["SLURM_CPUS_ON_NODE"])
            if memory < 0 or cpus < 1:
                raise ValueError("Invalid Slurm memory/CPU limit")
            job_limit = memory * cpus * 2**20 if memory > 0 else None
    except (ValueError, KeyError, TypeError):
        result["slurm"]["status"] = "unknown_invalid_limit_metadata"
    result["slurm"]["requested_limit_bytes"] = job_limit
    job_id = environment.get("SLURM_JOB_ID")
    if job_limit is not None:
        # Use the whole job cgroup, rather than a task's smaller live usage.
        # Matching job_1 must not accidentally match job_10.
        pattern = None if not job_id else re.compile(r"(?:^|/)job[_-]?" + re.escape(job_id) + r"(?:\.scope)?$")
        scoped = [row for row in result["cgroup_limits"] if pattern is not None and pattern.search(row["relative_scope"])]
        if scoped:
            remaining = min(max(0, job_limit - row["current_bytes"]) for row in scoped)
            result["slurm"].update(status="remaining_bounded_by_job_cgroup", available_bytes=remaining)
            capacities.append(remaining)
        else:
            result["slurm"].update(status="unknown_remaining_job_usage", available_bytes=None)
    result["unresolved_limits"] = unresolved_limits
    if capacities and not unresolved_limits and result["MemAvailable_bytes"] is not None and result["slurm"].get("status") not in (
            "unknown_remaining_job_usage", "unknown_invalid_limit_metadata"):
        result.update(status="available", available_bytes=min(capacities))
    return result


def host_capacity_admission(required_bytes, *, headroom_bytes, snapshot=None):
    if type(required_bytes) is not int or required_bytes < 0 or type(headroom_bytes) is not int or headroom_bytes < 0:
        raise ValueError("Host capacity and headroom must be nonnegative integer bytes")
    snapshot = host_capacity_snapshot() if snapshot is None else snapshot
    available = snapshot.get("available_bytes")
    known = snapshot.get("status") == "available" and type(available) is int and available >= 0
    allowed = known and required_bytes + headroom_bytes <= available
    return {"schema_version": 1, "admitted": bool(allowed), "status": "admitted" if allowed else "resource_limited" if known else "unknown",
        "required_physical_capacity_bytes": required_bytes, "reserved_headroom_bytes": headroom_bytes,
        "required_plus_headroom_bytes": required_bytes + headroom_bytes, "snapshot": snapshot,
        "requested_capacity_clamped": False, "reason": None if allowed else "host_or_job_capacity_unavailable_or_insufficient"}


def host_memory_snapshot():
    """OS process RSS/locked-page metadata; locked pages are not CUDA pin accounting."""
    result = {"current_rss_bytes": None, "peak_rss_bytes": None, "locked_bytes": None,
              "os_pinned_bytes": None, "scope": "OS_process_memory_not_authoritative_CUDA_pinned_pool"}
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_bytes"] = int(value) * (1 if platform.system() == "Darwin" else 1024)
    except (ImportError, OSError, ValueError):
        pass
    try:
        status = Path("/proc/self/status")
        if status.is_file():
            values = {line.split(":", 1)[0]: line.split(":", 1)[1].strip().split()
                      for line in status.read_text().splitlines() if ":" in line}
            for source, target in (("VmRSS", "current_rss_bytes"), ("VmLck", "locked_bytes"), ("VmPin", "os_pinned_bytes")):
                if source in values and len(values[source]) == 2 and values[source][1] == "kB":
                    result[target] = int(values[source][0]) * 1024
        elif platform.system() == "Darwin":
            query = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, timeout=2, check=False)
            if query.returncode == 0:
                result["current_rss_bytes"] = int(query.stdout.strip()) * 1024
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return result


def pinned_host_memory_snapshot(torch):
    """Torch pinned allocator cache/rounding metadata, separate from pool bytes."""
    function = getattr(getattr(torch.cuda, "memory", None), "host_memory_stats", None)
    result = {"status": "unavailable", "statistics": {},
        "scope": "process_Torch_pinned_allocator_active_and_cached_rounded_blocks",
        "peak_is_approximate": True, "peak_reset_performed": False,
        "managed_offload_budget_is_process_allocator_cap": False}
    if not callable(function):
        return {**result, "reason": "installed_Torch_has_no_host_memory_stats"}
    try:
        raw = function()
        keys = [name + "." + part for name in ("allocations", "allocated_bytes", "active_requests", "active_bytes")
                for part in ("current", "peak", "allocated", "freed")]
        keys += ["num_host_alloc", "num_host_free"]
        result["statistics"] = {name: raw[name] for name in keys if type(raw.get(name)) is int and raw[name] >= 0}
        result["status"] = "available" if "allocated_bytes.current" in result["statistics"] else "unavailable"
        if result["status"] != "available":
            result["reason"] = "allocator_returned_no_pinned_byte_counters"
    except Exception as error:
        result.update(reason=type(error).__name__, error=str(error)[:512])
    return result


def _integer(value):
    return value if type(value) is int and value >= 0 else None


def _process_occupancy(uuid):
    """Query bounded scalar process occupancy, without a Torch allocation."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "reason": "nvidia_smi_not_found", "processes": []}
    try:
        result = subprocess.run([executable, "--query-compute-apps=pid,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits"], capture_output=True, timeout=3, check=False)
        if result.returncode:
            return {"status": "unavailable", "reason": "nvidia_smi_failed", "exit_code": result.returncode,
                    "processes": []}
        if len(result.stdout) > MAX_PROCESS_OUTPUT_BYTES:
            return {"status": "incomplete", "reason": "process_output_byte_limit", "processes": [],
                    "limit_bytes": MAX_PROCESS_OUTPUT_BYTES}
        processes, omitted = [], 0
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 3:
                return {"status": "incomplete", "reason": "unrecognized_process_row", "processes": []}
            pid, used, gpu = fields
            if uuid is not None and gpu.removeprefix("GPU-") != uuid.removeprefix("GPU-"):
                continue
            try:
                row = {"pid": int(pid), "used_memory_bytes": int(used) * 2**20, "gpu_uuid": gpu}
            except ValueError:
                return {"status": "incomplete", "reason": "unavailable_process_memory", "processes": []}
            if row["pid"] < 0 or row["used_memory_bytes"] < 0:
                return {"status": "incomplete", "reason": "invalid_process_memory", "processes": []}
            if len(processes) < MAX_PROCESS_ROWS:
                processes.append(row)
            else:
                omitted += 1
        return {"status": "complete" if not omitted else "incomplete", "processes": processes,
                "omitted_processes": omitted, "process_limit": MAX_PROCESS_ROWS,
                "selected_device_filter_applied": uuid is not None}
    except Exception as error:
        return {"status": "unavailable", "reason": type(error).__name__, "processes": []}


class ExecutionMemoryAudit:
    """Collect JSON scalars and isolate diagnostic failures from execution."""

    def __init__(self, *, enabled=True, runtime_snapshot=None):
        self.torch = self.device = None
        self.runtime_snapshot = runtime_snapshot
        self.report = {"schema_version": 1, "enabled": enabled, "complete": False,
            "status": "not_started" if enabled else "disabled", "snapshots": {}, "errors": [],
            "scope": "staged_Torch_allocator_and_device_free_metadata",
            "cuda_execution_evidence": False, "process": {"pid": os.getpid(), "ppid": os.getppid()},
            "device": None, "allocator": {"backend": None, "configuration": {
                name: os.environ.get(name) for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}},
            "cleanup": {"attempted": False, "completed": False, "actions": []},
            "baseline_preparation": {"attempted": False, "completed": False, "actions": []},
            "first_update_profile": {"status": "not_requested", "implemented": True, "allocator_snapshot_available": False},
            "typed_storage_owner_ledger": {"status": "not_requested", "implemented": True},
            "vendor_workspace_complete": False, "fresh_process_is_primary_release_boundary": True,
            "tensor_files_written": 0}

    def record_error(self, stage, action, error):
        trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.report["errors"].append({"stage": stage, "action": action,
            "error_type": type(error).__name__, "error": str(error)[:4096], "traceback": trace[:16384],
            "traceback_truncated": len(trace) > 16384})

    def bind(self, torch, device):
        self.torch, self.device = torch, device
        cuda = device.type == "cuda"
        self.report["scope"] = ("CUDA_allocator_and_sampled_device_free" if cuda
                                else "CPU_no_CUDA_allocator_evidence")
        self.report["device"] = {"type": device.type, "index": device.index, "name": None, "uuid": None}
        if cuda and self.report["enabled"]:
            try:
                props = torch.cuda.get_device_properties(device)
                self.report["device"].update(name=props.name, uuid=str(props.uuid))
                if device.index is None:
                    self.report["device"]["index"] = torch.cuda.current_device()
            except Exception as error:
                self.record_error("device_setup", "device_identity", error)
            try:
                self.report["allocator"]["backend"] = torch.cuda.memory.get_allocator_backend()
            except Exception as error:
                self.record_error("device_setup", "allocator_backend", error)

    def _action(self, stage, action, function, actions):
        try:
            function()
            actions.append({"action": action, "completed": True})
            return True
        except Exception as error:
            self.record_error(stage, action, error)
            actions.append({"action": action, "completed": False})
            return False

    def prepare_baseline(self):
        if not self.report["enabled"] or self.device is None:
            return
        preparation = self.report["baseline_preparation"]
        preparation["attempted"] = True
        actions = preparation["actions"]
        self._action("before_model", "gc_collect", gc.collect, actions)
        if self.device.type == "cuda":
            self._action("before_model", "synchronize", lambda: self.torch.cuda.synchronize(self.device), actions)
            self._action("before_model", "empty_cache", self.torch.cuda.empty_cache, actions)
        preparation["completed"] = all(row["completed"] for row in actions)
        self.capture("before_model")

    def capture(self, stage, *, update=None, synchronize=True):
        if not self.report["enabled"] or self.device is None:
            return None
        if stage in self.report["snapshots"]:
            self.record_error(stage, "duplicate_stage", ValueError("Memory stage already recorded"))
            return None
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(),
            "update": update, "synchronized": False, "complete": True,
            **dict.fromkeys(_MEMORY_FIELDS)}
        row["host_process"] = host_memory_snapshot()
        errors_before = len(self.report["errors"])
        if self.runtime_snapshot is not None:
            try:
                row["packed_offload_runtime"] = self.runtime_snapshot()
            except Exception as error:
                self.record_error(stage, "packed_offload_metadata", error)
        if self.device.type == "cuda":
            if synchronize:
                try:
                    self.torch.cuda.synchronize(self.device)
                    row["synchronized"] = True
                except Exception as error:
                    self.record_error(stage, "synchronize", error)
            for name, function in (
                ("allocated_bytes", self.torch.cuda.memory_allocated),
                ("reserved_bytes", self.torch.cuda.memory_reserved),
                ("peak_allocated_bytes", self.torch.cuda.max_memory_allocated),
                ("peak_reserved_bytes", self.torch.cuda.max_memory_reserved),
            ):
                try:
                    row[name] = _integer(function(self.device))
                except Exception as error:
                    self.record_error(stage, name, error)
            try:
                stats = self.torch.cuda.memory_stats(self.device)
                row.update(inactive_split_bytes=_integer(stats.get("inactive_split_bytes.all.current")),
                    allocation_retries=_integer(stats.get("num_alloc_retries")),
                    ooms=_integer(stats.get("num_ooms")))
            except Exception as error:
                self.record_error(stage, "allocator_counters", error)
            try:
                free, total = self.torch.cuda.mem_get_info(self.device)
                row.update(device_free_bytes=_integer(free), device_total_bytes=_integer(total))
            except Exception as error:
                self.record_error(stage, "device_free_memory", error)
            row["complete"] = all(row[name] is not None for name in _MEMORY_FIELDS)
            row["process_occupancy"] = _process_occupancy(self.report["device"]["uuid"])
            row["pinned_host_allocator"] = pinned_host_memory_snapshot(self.torch)
        else:
            row["evidence_scope"] = "CPU_no_CUDA_allocator_evidence"
            row["pinned_host_allocator"] = {"status": "not_applicable_CPU", "statistics": {}}
            row["process_occupancy"] = {"status": "not_applicable_CPU", "processes": []}
        row["complete"] &= len(self.report["errors"]) == errors_before
        self.report["snapshots"][stage] = row
        if stage == "after_first_update":
            self.report["first_update_profile"]["allocator_snapshot_available"] = row["complete"]
            self.report["cuda_execution_evidence"] = self.device.type == "cuda" and row["complete"]
        return row

    def cleanup(self, *, release_actions=()):
        """Run supplied cache releases after the worker drops tensor owners."""
        cleanup = self.report["cleanup"]
        cleanup["attempted"] = True
        actions = cleanup["actions"]
        for name, function in release_actions:
            self._action("cleanup", name, function, actions)
        self._action("cleanup", "gc_collect", gc.collect, actions)
        if self.report["enabled"] and self.device is not None and self.device.type == "cuda":
            self._action("cleanup", "synchronize", lambda: self.torch.cuda.synchronize(self.device), actions)
            self._action("cleanup", "empty_cache", self.torch.cuda.empty_cache, actions)
        cleanup["completed"] = all(row["completed"] for row in actions)
        self.capture("cleanup")

    def finalize(self, *, expected_stages):
        if not self.report["enabled"]:
            return
        self.report["expected_stages"] = list(expected_stages)
        self.report["missing_stages"] = [name for name in expected_stages if name not in self.report["snapshots"]]
        self.report["complete"] = (not self.report["missing_stages"] and not self.report["errors"]
            and all(row["complete"] for row in self.report["snapshots"].values())
            and self.report["baseline_preparation"]["completed"] and self.report["cleanup"]["completed"])
        self.report["status"] = "complete" if self.report["complete"] else "incomplete"
