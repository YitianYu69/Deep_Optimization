"""Startup-only rank agreement and shared-host admission for packed saves.

All ranks enter this function, including ranks requesting disabled transport.
There are no update-time collectives, tensor transfers or runtime callbacks.
Capacity observations are not reservations and never silently reduce budgets.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import re
import socket

import torch

from deepopt._packed_offload import resolve_packed_offload
from . import _process


_MAX_RECIPE_BYTES = 64 * 1024
_MAX_RECIPE_ITEMS = 8192
_MAX_CONSTRAINTS = 128


def _recipe_json(recipe, *, enabled):
    if enabled and (not isinstance(recipe, Mapping) or not recipe):
        raise ValueError("Enabled packed offload requires a nonempty native numerical/compile recipe")
    remaining = _MAX_RECIPE_ITEMS

    def convert(value, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 16:
            raise ValueError("Packed offload recipe exceeds its bounded metadata schema")
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and isfinite(value):
            return value
        if isinstance(value, (torch.dtype, torch.device, torch.memory_format)):
            return str(value)
        if is_dataclass(value) and not isinstance(value, type):
            value = {field.name: getattr(value, field.name) for field in fields(value)}
        if isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                raise ValueError("Packed offload recipe keys must be strings")
            return {key: convert(child, depth + 1) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(child, depth + 1) for child in value]
        raise ValueError("Packed offload recipe must contain metadata only, never tensors or callbacks")

    result = json.dumps(convert(recipe), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(result.encode()) > _MAX_RECIPE_BYTES:
        raise ValueError("Packed offload recipe exceeds 64 KiB of metadata")
    return result


def _read_text(path, limit=64 * 1024):
    with path.open("r") as handle:
        result = handle.read(limit + 1)
    if len(result) > limit:
        raise ValueError("Host capacity metadata exceeds its bounded size")
    return result


def _host_capacity_snapshot(*, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup"), environment=None):
    """Observe host RAM and each finite memory-controller ancestor separately.

    Keeping scope identities allows shared job limits to charge every member
    while a private per-rank sub-cgroup charges only that rank. Slurm's declared
    limit is an upper bound; remaining job usage is never inferred from RSS.
    """
    environment = os.environ if environment is None else environment
    result = dict(schema_version=1, status="unknown", mem_available_bytes=None,
                  constraints=[], errors=[], warnings=[], snapshot_is_reservation=False)
    try:
        matches = [line.split() for line in _read_text(proc_root / "meminfo").splitlines()
                   if line.startswith("MemAvailable:")]
        if len(matches) != 1 or len(matches[0]) != 3 or matches[0][2] != "kB":
            raise ValueError("MemAvailable is absent or malformed")
        available = int(matches[0][1]) * 1024
        if available < 0:
            raise ValueError("MemAvailable is negative")
        result["mem_available_bytes"] = available
        locations = []
        for line in _read_text(proc_root / "self/cgroup").splitlines():
            _, controllers, relative = line.split(":", 2)
            if controllers and "memory" not in controllers.split(","):
                continue
            path = Path(relative.lstrip("/"))
            if ".." in path.parts:
                raise ValueError("Unsafe cgroup capacity scope")
            prefix = cgroup_root if not controllers else cgroup_root / "memory"
            names = ("memory.max", "memory.current") if not controllers else ("memory.limit_in_bytes", "memory.usage_in_bytes")
            for ancestor in (path, *path.parents):
                locations.append((prefix / ancestor, str(ancestor), names))
        if len(locations) > _MAX_CONSTRAINTS:
            raise ValueError("Too many memory controller ancestors")
        seen = set()
        for folder, relative, (limit_name, current_name) in locations:
            source = folder / limit_name
            if str(source) in seen or not source.is_file():
                continue
            seen.add(str(source))
            raw = _read_text(source, 128).strip()
            if raw == "max":
                continue
            limit = int(raw)
            if limit >= 1 << 60:
                continue  # Linux cgroup-v1's unlimited sentinel.
            current = int(_read_text(folder / current_name, 128).strip())
            if min(limit, current) < 0:
                raise ValueError("Negative memory controller capacity")
            result["constraints"].append(dict(scope_id=f"{limit_name}:{relative}", kind="cgroup",
                relative_scope=relative, limit_bytes=limit, current_bytes=current,
                available_bytes=max(0, limit-current), remaining_observed=True))
        job_limit = None
        if environment.get("SLURM_MEM_PER_NODE"):
            memory = int(environment["SLURM_MEM_PER_NODE"])
            if memory < 0:
                raise ValueError("Negative Slurm node memory")
            job_limit = memory * 2**20 or None
        elif environment.get("SLURM_MEM_PER_CPU"):
            memory, cpus = int(environment["SLURM_MEM_PER_CPU"]), int(environment["SLURM_CPUS_ON_NODE"])
            if memory < 0 or cpus < 1:
                raise ValueError("Invalid Slurm memory/CPU budget")
            job_limit = memory * cpus * 2**20 or None
        if job_limit is not None:
            job_id = environment.get("SLURM_JOB_ID", "unknown")
            if type(job_id) is not str or len(job_id) > 128:
                raise ValueError("Invalid Slurm job identity")
            pattern = re.compile(r"(?:^|/)job[_-]?" + re.escape(job_id) + r"(?:\.scope)?$")
            job_rows = [row for row in result["constraints"] if pattern.search(row["relative_scope"])]
            current = max((row["current_bytes"] for row in job_rows), default=None)
            result["constraints"].append(dict(scope_id="slurm_job:" + job_id, kind="slurm",
                limit_bytes=job_limit, current_bytes=current,
                available_bytes=max(0, job_limit-current) if current is not None else job_limit,
                remaining_observed=current is not None))
            if current is None:
                result["warnings"].append("Slurm declared limit checked; remaining job usage unavailable")
        result["status"] = "available"
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["errors"].append(dict(error_type=type(error).__name__, message=str(error)[:256]))
    return result


def _validate_context(context):
    actual_rank, actual_size = _process.rank(), _process.world_size()
    if (type(context.global_rank) is not int or context.global_rank != actual_rank
            or type(context.world_size) is not int or context.world_size != actual_size):
        raise ValueError("Packed offload context must match the active default process group")
    for name, minimum in (("local_rank", 0), ("local_world_size", 1)):
        value = getattr(context, name)
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError("Invalid packed offload local rank topology")
    if (context.local_rank is not None and context.local_world_size is not None
            and context.local_rank >= context.local_world_size):
        raise ValueError("Packed offload local rank exceeds its local world size")
    torch.device(context.device)


def _local_descriptor(policy, context):
    hostname = socket.gethostname()
    node = os.environ.get("SLURMD_NODENAME") or hostname
    if any(type(value) is not str or not value or len(value) > 255 for value in (hostname, node)):
        raise ValueError("Packed offload requires a bounded hostname/node identity")
    device = torch.device(context.device)
    result = dict(rank=context.global_rank, hostname=hostname, node_id=node,
        node_id_source="SLURMD_NODENAME" if os.environ.get("SLURMD_NODENAME") else "hostname",
        local_rank=context.local_rank, local_world_size=context.local_world_size,
        device=str(device), device_index=device.index, device_uuid=None, device_identity=None,
        backend=context.backend, requested_host_bytes=(policy.host_pinned_bytes + policy.host_pageable_bytes) if policy.enabled else 0,
        pinned_bytes=policy.host_pinned_bytes if policy.enabled else 0,
        pageable_bytes=policy.host_pageable_bytes if policy.enabled else 0, capacity=None)
    if not policy.enabled:
        return result
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise ValueError("Packed activation offload requires native compiled CUDA execution on each rank")
    if not 0 <= device.index < torch.cuda.device_count() or torch.cuda.current_device() != device.index:
        raise ValueError("Packed offload CUDA rank device is not the current visible device")
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    uuid = str(uuid) if uuid is not None else None
    if uuid and uuid.lower() not in ("none", "unavailable", "unknown"):
        result.update(device_uuid=uuid, device_identity="uuid:" + uuid)
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            entries = visible.split(",")
            if device.index >= len(entries) or not entries[device.index].strip():
                raise ValueError("Cannot establish packed offload CUDA device identity")
            identity = "visible:" + entries[device.index].strip()
        else:
            identity = "ordinal:" + str(device.index)
        result["device_identity"] = identity
    if any(len(value) > 256 for value in (result["device_identity"], result["device_uuid"] or "")):
        raise ValueError("CUDA device identity is too large")
    result["capacity"] = _host_capacity_snapshot()
    if result["capacity"]["status"] != "available":
        raise ValueError("Packed offload host capacity is unknown: " + json.dumps(result["capacity"]["errors"]))
    return result


def _validate_descriptors(descriptors, policy):
    """Pure shared-scope arithmetic over the gathered startup metadata."""
    for row in descriptors:
        if (not isinstance(row, dict) or type(row.get("rank")) is not int
                or any(type(row.get(key)) is not str or not row[key] or len(row[key]) > 255
                       for key in ("hostname", "node_id"))):
            raise ValueError("Invalid packed offload rank descriptor")
        expected = ((policy.host_pinned_bytes, policy.host_pageable_bytes) if policy.enabled else (0, 0))
        if any(type(row.get(key)) is not int or row[key] != value for key, value in
               zip(("pinned_bytes", "pageable_bytes", "requested_host_bytes"), (*expected, sum(expected)))):
            raise ValueError("Packed offload rank descriptor changed its declared host budget")
        if policy.enabled:
            capacity = row.get("capacity")
            if (not isinstance(capacity, dict) or capacity.get("status") != "available"
                    or type(capacity.get("mem_available_bytes")) is not int or capacity["mem_available_bytes"] < 0
                    or not isinstance(capacity.get("constraints"), (tuple, list))
                    or len(capacity["constraints"]) > _MAX_CONSTRAINTS):
                raise ValueError("Packed offload rank lacks observable host capacity")
            for limit in capacity["constraints"]:
                if (not isinstance(limit, dict) or type(limit.get("scope_id")) is not str
                        or not limit["scope_id"] or len(limit["scope_id"]) > 4096
                        or type(limit.get("limit_bytes")) is not int or limit["limit_bytes"] < 0
                        or type(limit.get("available_bytes")) is not int
                        or not 0 <= limit["available_bytes"] <= limit["limit_bytes"]
                        or type(limit.get("remaining_observed")) is not bool):
                    raise ValueError("Invalid packed offload shared capacity constraint")
                current = limit.get("current_bytes")
                if limit["remaining_observed"]:
                    if type(current) is not int or current < 0 or limit["available_bytes"] != max(0, limit["limit_bytes"]-current):
                        raise ValueError("Packed offload capacity does not match observed usage")
                elif limit.get("kind") != "slurm" or current is not None:
                    raise ValueError("Packed offload unobserved capacity is not a declared Slurm upper bound")
    if len(descriptors) != _process.world_size() or sorted(row["rank"] for row in descriptors) != list(range(len(descriptors))):
        raise ValueError("Packed offload rank descriptors do not cover the process group")
    groups = {}
    for row in descriptors:
        groups.setdefault(row["node_id"], []).append(row)
    nodes = []
    for node, members in sorted(groups.items()):
        ranks = tuple(sorted(row["rank"] for row in members))
        request = sum(row["requested_host_bytes"] for row in members)
        node_row = dict(node_id=node, ranks=ranks, requested_host_bytes=request,
                        pinned_bytes=sum(row["pinned_bytes"] for row in members),
                        pageable_bytes=sum(row["pageable_bytes"] for row in members), constraints=[])
        if policy.enabled:
            identities = [row["device_identity"] for row in members]
            if any(not identity for identity in identities) or len(set(identities)) != len(identities):
                raise ValueError(f"Packed offload requires distinct CUDA devices for ranks on node {node}")
            if not all(row["device_uuid"] for row in members) and len({identity.split(":", 1)[0] for identity in identities}) != 1:
                raise ValueError("Cannot compare mixed CUDA identity sources without UUIDs")
            local_sizes = {row["local_world_size"] for row in members if row["local_world_size"] is not None}
            local_ranks = [row["local_rank"] for row in members if row["local_rank"] is not None]
            launcher_matches = ((not local_sizes or local_sizes == {len(members)})
                                and len(set(local_ranks)) == len(local_ranks))
            if not launcher_matches and not all(row["device_uuid"] for row in members):
                raise ValueError("Packed offload local topology disagrees with gathered node membership")
            # Separate launcher agents can expose cuda:0/local_rank=0 on the
            # same host. Observed distinct UUIDs remain authoritative.
            node_row["launcher_local_topology_matches_node"] = launcher_matches
            capacity = min(row["capacity"]["mem_available_bytes"] for row in members)
            if request > capacity:
                raise ValueError(f"Packed offload node {node} declares {request} host bytes across ranks {ranks}; MemAvailable allows {capacity}")
            node_row.update(mem_available_bytes=capacity, mem_available_margin_bytes=capacity-request)
            scopes = {}
            for row in members:
                for limit in row["capacity"]["constraints"]:
                    scoped = scopes.setdefault(limit["scope_id"], dict(rows=[], ranks=set()))
                    scoped["rows"].append(limit)
                    scoped["ranks"].add(row["rank"])
            for scope, observation in sorted(scopes.items()):
                charged = sum(row["requested_host_bytes"] for row in members if row["rank"] in observation["ranks"])
                remaining = min(row["available_bytes"] for row in observation["rows"])
                if charged > remaining:
                    raise ValueError(f"Packed offload scope {node}/{scope} requires {charged} host bytes; observed bound is {remaining}")
                node_row["constraints"].append(dict(scope_id=scope, ranks=tuple(sorted(observation["ranks"])),
                    requested_host_bytes=charged, observed_bound_bytes=remaining, margin_bytes=remaining-charged,
                    remaining_observed=all(row["remaining_observed"] for row in observation["rows"])))
        nodes.append(node_row)
    return tuple(nodes)


def prepare_packed_offload(policy, distributed_context, *, recipe=None):
    """Agree rank policies/recipe and admit their combined host budgets once.

    Call on EVERY rank, even with disabled policy, before numerical model/DDP
    preparation. This function owns its collective phase boundaries; do not
    wrap the call in another ``collective_errors`` boundary.
    """
    with _process.collective_errors("packed offload configuration"):
        policy = resolve_packed_offload(policy)
        _validate_context(distributed_context)
        policy_wire = policy.as_dict()
        if len(json.dumps(policy_wire, allow_nan=False).encode()) > 16 * 1024:
            raise ValueError("Packed offload policy exceeds 16 KiB of metadata")
        encoded_recipe = _recipe_json(recipe, enabled=True) if policy.enabled else "null"
    _process.require_equal(policy_wire, "packed activation offload policies")
    if policy.enabled:  # Agreement above makes this branch uniform on all ranks.
        _process.require_equal(encoded_recipe, "packed offload native numerical/compile recipes")
    with _process.collective_errors("packed offload rank placement/capacity"):
        descriptor = _local_descriptor(policy, distributed_context)
        if len(json.dumps(descriptor, allow_nan=False).encode()) > 256 * 1024:
            raise ValueError("Packed offload rank descriptor exceeds 256 KiB of metadata")
    descriptors = [descriptor]
    if _process.world_size() > 1:
        descriptors = [None] * _process.world_size()
        _process.dist.all_gather_object(descriptors, descriptor)
    with _process.collective_errors("packed offload shared host admission"):
        nodes = _validate_descriptors(descriptors, policy)
    return dict(schema_version=1, status="admitted" if policy.enabled else "disabled", admitted=True,
        enabled=policy.enabled, policy=policy.as_dict(), recipe_sha256=hashlib.sha256(encoded_recipe.encode()).hexdigest() if policy.enabled else None,
        rank=distributed_context.global_rank, world_size=distributed_context.world_size,
        ranks=tuple(descriptors), nodes=nodes, requested_capacity_clamped=False,
        budget_semantics="per-rank declared pinned plus pageable, aggregated by node/shared scope",
        physical_host_budget=policy.physical_host_budget, preparation_only=True,
        host_capacity_is_reservation=False, cuda_execution_verified=False,
        qualification="startup metadata only; not CUDA transport, numerical or performance qualification")
