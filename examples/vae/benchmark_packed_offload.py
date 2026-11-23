"""Compare resident, demand and predictive packed saves on one pinned candidate.

GPU peak is the primary objective. Runtime is a reported tradeoff, never an
implicit veto on a lower-memory point. No numerical recipe or training loop
changes between these fresh sequential workers.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "packed_offload_v1"
ARMS = ("resident", "demand_offload", "predictive_offload")
BALANCED_ORDERS = (("resident", "demand_offload", "predictive_offload"),
    ("resident", "predictive_offload", "demand_offload"), ("demand_offload", "resident", "predictive_offload"),
    ("demand_offload", "predictive_offload", "resident"), ("predictive_offload", "resident", "demand_offload"),
    ("predictive_offload", "demand_offload", "resident"))
POLICY_DEFAULTS = {"enabled": False, "policy": "memory_first_overlap", "host_pinned_bytes": 0,
    "device_staging_bytes": 0, "d2h_inflight_bytes": 0, "copy_chunk_bytes": 16 * 2**20,
    "max_live_forwards": 1, "max_backward_tasks": 1, "peak_budget_bytes": None,
    "latency_target_percent": 1.0, "admission": "resident_fallback", "host_pageable_bytes": 0}


def normalize_policy(value):
    """Tensor-free planning validation; installed policy validation runs again."""
    if type(value) is not dict or set(value) - set(POLICY_DEFAULTS) - {"version"}:
        raise ValueError("Packed offload policy has unknown fields or is not an object")
    version = value.get("version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported packed offload policy version")
    policy = {**POLICY_DEFAULTS, **{key: part for key, part in value.items() if key != "version"}}
    if version == 2:
        policy["version"] = 2
    if type(policy["enabled"]) is not bool or policy["policy"] not in ("demand", "memory_first_overlap"):
        raise ValueError("Packed offload enabled/policy must use the strict supported types/values")
    for key in ("host_pinned_bytes", "device_staging_bytes", "d2h_inflight_bytes", "host_pageable_bytes"):
        if type(policy[key]) is not int or policy[key] < 0:
            raise ValueError(key + " must be nonnegative integer bytes")
    for key in ("copy_chunk_bytes", "max_live_forwards", "max_backward_tasks"):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if policy["peak_budget_bytes"] is not None and (type(policy["peak_budget_bytes"]) is not int or policy["peak_budget_bytes"] < 1):
        raise ValueError("peak_budget_bytes must be positive integer bytes or null")
    latency = policy["latency_target_percent"]
    try:
        finite_latency = type(latency) in (int, float) and math.isfinite(latency)
    except OverflowError:
        finite_latency = False
    if not finite_latency or latency < 0:
        raise ValueError("latency_target_percent must be finite and nonnegative")
    if policy["admission"] not in ("error", "resident_fallback"):
        raise ValueError("Packed offload admission must be error or resident_fallback")
    if policy["enabled"] and policy["host_pinned_bytes"] < 1:
        raise ValueError("Enabled offload requires a positive pinned host budget")
    return policy


def parser():
    from examples.vae.benchmark_execution import parser as existing

    result = existing()
    result.description = __doc__
    result.set_defaults(trials=6, output=Path("work/vae_packed_offload.json"))
    result.add_argument("--arms", choices=("all", "demand-only"), default="all",
                        help="Declare all three policies or resident/demand paired comparisons")
    result.add_argument("--cold-diagnostics", action="store_true", help="Run separate fresh first-update profile workers")
    result.add_argument("--native-options-json", type=Path, help="Immutable native pass controls, signed in every worker pin")
    result.add_argument("--inventory-output", type=Path, help="Export compact real compiled demand inventory for capacity-frontier reuse")
    result.add_argument("--host-headroom-mib", type=int, default=8192,
                        help="Reserve explicit RAM headroom before each version2 worker pins memory")
    result.add_argument("--experiment-manifest", type=Path, required=True)
    result.add_argument("--configuration-preflight-only", action="store_true",
                        help="Validate all three pinned configurations in fresh workers without model or CUDA execution")
    return result


def load_manifest(path):
    from examples.vae.native_execution import _json_input

    value, source = _json_input(path)
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("experiment_profile") != PROFILE or type(value.get("candidate")) is not dict
            or type(value.get("policies")) is not dict or set(value["policies"]) != set(ARMS)):
        raise ValueError("Require schema1 packed_offload_v1 candidate and all three policies")
    candidate = value["candidate"]
    if set(candidate) != {"wheel", "install_root", "expected_installed"}:
        raise ValueError("Offload candidate must pin wheel, installation and expected identity")
    expected, wheel = candidate["expected_installed"], candidate["wheel"]
    if (type(expected) is not dict or set(expected) != {"version", "source_sha256", "files_hashed"}
            or type(wheel) is not dict or set(wheel) != {"path", "sha256"}
            or type(expected["version"]) is not str or not expected["version"].strip()
            or type(expected["files_hashed"]) is not int or expected["files_hashed"] < 1):
        raise ValueError("Invalid packed offload installation identity schema")
    for name in (candidate["install_root"], wheel["path"]):
        if type(name) is not str or not name.strip():
            raise ValueError("Offload installation/wheel paths must be nonempty strings")
    for digest in (wheel["sha256"], expected["source_sha256"]):
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Offload identities require lowercase SHA256 values")
    policies = {name: normalize_policy(policy) for name, policy in value["policies"].items()}
    if policies["resident"]["enabled"] or policies["resident"]["policy"] != "memory_first_overlap":
        raise ValueError("Resident control must disable transport")
    if not policies["demand_offload"]["enabled"] or policies["demand_offload"]["policy"] != "demand":
        raise ValueError("Demand control must enable demand restoration")
    if not policies["predictive_offload"]["enabled"] or policies["predictive_offload"]["policy"] != "memory_first_overlap":
        raise ValueError("Predictive control must enable memory-first overlap")
    common = [{key: part for key, part in policies[name].items() if key not in ("enabled", "policy")} for name in ARMS]
    if not common[0] == common[1] == common[2]:
        raise ValueError("All arms must declare identical host/device/peak/admission budgets")
    value["policies"] = policies
    return value, source


def _apply_policy(settings, policy):
    result = deepcopy(settings)
    result["runtime"]["packed_activation_offload"] = deepcopy(policy)
    result["runtime"]["compile"]["options"]["packed_activation_offload"] = deepcopy(policy)
    return result


def benchmark_plan(args):
    from examples.vae import benchmark_execution as paired
    from examples.vae.benchmark_optimization import verify_arm

    if args.arm_manifest is not None or not args.memory_audit:
        raise ValueError("Offload arms require their experiment pins and the staged memory audit")
    if type(args.host_headroom_mib) is not int or args.host_headroom_mib < 0:
        raise ValueError("Host headroom must be a nonnegative integer MiB value")
    common = paired.benchmark_plan(args)
    manifest, source = load_manifest(args.experiment_manifest.resolve())
    identity = verify_arm(manifest["candidate"], args.experiment_manifest.resolve())
    native = common["workers"]["native"]
    if getattr(args, "native_options_json", None) is not None:
        from examples.vae import native_execution

        command = paired.worker_command(args, "native", args.output) + ["--native-options-json", str(args.native_options_json.resolve())]
        native = native_execution.execution_plan(native_execution.parser().parse_args(command[3:]))
    active_arms = ARMS[:2] if getattr(args, "arms", "all") == "demand-only" else ARMS
    native["host_capacity_policy"] = {"reserved_headroom_bytes": args.host_headroom_mib * 2**20,
        "headroom_mib": args.host_headroom_mib, "read_only_preallocation_check": True, "requested_capacity_clamped": False}
    order = ([list(active_arms if index % 2 == 0 else reversed(active_arms)) for index in range(args.trials)]
             if len(active_arms) == 2 else [list(BALANCED_ORDERS[index % 6]) for index in range(args.trials)])
    return {"schema_version": 1, "experiment_profile": PROFILE,
        "objective": "minimize_whole_run_GPU_allocated_peak_then_runtime_at_that_peak",
        "performance_policy_promoted": False, "CUDA_execution_proven": False,
        "manifest_source": source, "manifest": manifest, "installed": identity,
        "source_sha256": source_identity(), "origin": native["origin"],
        "workers": {name: {**deepcopy(native), "settings": _apply_policy(native["settings"], manifest["policies"][name]),
            "expected_installed_version": identity["version"]} for name in active_arms},
        "active_arms": list(active_arms), "arm_declaration": getattr(args, "arms", "all"), "order": order,
        "trials_requested": args.trials, "warmup": args.warmup, "updates": args.updates,
        "baseline_tolerance_bytes": args.start_free_tolerance_mib * 2**20,
        "host_capacity_policy": deepcopy(native["host_capacity_policy"]),
        "diagnostic_workers_requested": args.profile_after_measurement,
        "cold_diagnostic_workers_requested": getattr(args, "cold_diagnostics", False),
        "profiling_scope": "fresh separate workers after every clean block; diagnostic measurements excluded",
        "tensor_files_written": 0}


def source_identity():
    names = ("examples/vae/benchmark_packed_offload.py", "examples/vae/native_execution.py",
        "examples/vae/execution_memory.py", "examples/vae/execution_profile.py", "examples/vae/execution_storage.py",
        "examples/vae/execution_inventory.py",
        "examples/vae/memory_comparison.py", "examples/vae/benchmark_execution.py",
        "examples/vae/benchmark_optimization.py", "examples/vae/benchmark_vae.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def _identity(row):
    from examples.vae.benchmark_execution import comparable_identity

    result = comparable_identity(row)
    for name in ("common_settings", "effective_runtime_config"):
        value = result.get(name)
        if isinstance(value, dict) and isinstance(value.get("runtime"), dict):
            value["runtime"].pop("packed_activation_offload", None)
    result["native_math_options"] = deepcopy(row.get("settings", {}).get("runtime", {}).get("compile", {}).get("options", {}))
    result["native_math_options"].pop("packed_activation_offload", None)
    for name in ("rng_before_warmup_sha256", "rng_after_warmup_sha256", "rng_after_measurement_sha256"):
        result[name] = row.get(name)
    return result


def _runtime(row):
    receipt = row.get("packed_offload_runtime")
    if not isinstance(receipt, dict) or receipt.get("available") is not True or not isinstance(receipt.get("snapshot"), dict):
        raise ValueError("Packed offload runtime snapshot is unavailable")
    value = receipt["snapshot"]
    managers = value.get("managers")
    if managers is None and "policy" in value:
        managers = [value]
    if not isinstance(managers, (list, tuple)) or any(not isinstance(item, dict) for item in managers):
        raise ValueError("Packed offload runtime manager structure is invalid")
    return value, managers


def _proven_unused_manager(manager):
    """Recognize the real schema of a frozen owner with no initialized pool.

    Empty plans alone are insufficient: an initialized, failed or active owner
    still needs its allocator proof. Never mutate or discard the raw receipt.
    """
    if type(manager.get("schema_version")) is not int or manager["schema_version"] != 1:
        return False
    for key in ("plans", "events", "admission_counts", "admission_history", "cleanup_errors"):
        if not isinstance(manager.get(key), (list, tuple)) or manager[key]:
            return False
    if (manager.get("profiles") != {} or manager.get("admission_history_truncated") is not False
            or manager.get("cuda_execution_observed") is not False or manager.get("error") is not None
            or manager.get("testing_backend") is not False or manager.get("profile_frozen") is not True
            or manager.get("profile_learning") is not False):
        return False
    counters = manager.get("counters")
    if not isinstance(counters, dict) or any(type(value) is not int or value != 0 for value in counters.values()):
        return False
    expected = {
        "host": {kind + "_" + field for kind in ("pinned", "pageable")
                 for field in ("active_bytes", "cached_bytes", "peak_bytes")},
        "device": {"staging_bytes", "staging_peak_bytes", "pending_retire_bytes",
                   "d2h_inflight_bytes", "peak_d2h_inflight_bytes"},
    }
    for area, keys in expected.items():
        fields = manager.get(area)
        if (not isinstance(fields, dict) or set(fields) != keys
                or any(type(fields[key]) is not int or fields[key] != 0 for key in keys)):
            return False
    lifecycle = manager.get("lifecycle")
    counts = {"live_forwards", "regional_frames", "backward_tasks", "graph_leases", "task_leases", "dma_leases"}
    return (isinstance(lifecycle, dict) and set(lifecycle) == counts | {"owner_gone", "closed", "quiescent"}
        and all(type(lifecycle[key]) is int and lifecycle[key] == 0 for key in counts)
        and all(type(lifecycle[key]) is bool for key in ("owner_gone", "closed"))
        and lifecycle["quiescent"] is True)


def runtime_budget_errors(receipt, policy):
    """Validate a frozen runtime receipt outside measurement, without Torch.

    This is shared by the warmup preflight and final worker validation. It has
    no RNG, numerical-admission-window or measured-update requirements.
    """
    reasons = []
    try:
        policy = normalize_policy(policy)
        value, managers = _runtime({"packed_offload_runtime": receipt})
        if "managers" in value and value.get("frozen") is not True:
            reasons.append("runtime_profile_learning_during_measurement")
        domains = {}
        for manager in managers:
            if normalize_policy(manager.get("policy", policy)) != policy:
                reasons.append("runtime_policy_mismatch")
            if manager.get("testing_backend") or manager.get("error") is not None:
                reasons.append("testing_backend_or_runtime_error")
            cleanup_errors = manager.get("cleanup_errors", ())
            if not isinstance(cleanup_errors, (list, tuple)) or cleanup_errors:
                reasons.append("runtime_cleanup_errors")
            if manager.get("profile_frozen") is not True or manager.get("profile_learning") is not False:
                reasons.append("runtime_profile_learning_during_measurement")
            unused = _proven_unused_manager(manager)
            if policy.get("version", 1) == 2 and not unused:
                host = manager.get("host", {})
                domain = host.get("pinned_allocator_domain_id")
                if (host.get("pinned_accounting_verified") is not True
                        or type(domain) is not int or domain < 1
                        or any(type(host.get(key)) is not int or not 0 <= host[key] <= policy["host_pinned_bytes"]
                            for key in ("pinned_physical_capacity_bytes", "pinned_physical_peak_bytes"))):
                    reasons.append("physical_pinned_budget_unverified_or_exceeded")
                else:
                    observed = domains.setdefault(domain, [0, 0])
                    for index, key in enumerate(("pinned_physical_capacity_bytes", "pinned_physical_peak_bytes")):
                        observed[index] = max(observed[index], host[key])
            limits = dict(policy)
            for limit, requirement, concurrency in (
                    ("device_staging_bytes", "required_staging_bytes", policy["max_backward_tasks"]),
                    ("d2h_inflight_bytes", "required_d2h_inflight_bytes", 1)):
                if not policy["enabled"] or unused or policy[limit]:
                    continue
                floors = tuple(plan.get(requirement) for plan in manager.get("plans", ()))
                effective = manager.get("transport_limits", {}).get(limit)
                if (not floors or any(type(size) is not int or size < 0 for size in floors)
                        or type(effective) is not int or effective != max(floors) * concurrency):
                    reasons.append("automatic_managed_budget_unverified:" + limit)
                else:
                    limits[limit] = effective
            for area, key, limit in (("host", "pinned_peak_bytes", "host_pinned_bytes"),
                    ("host", "pageable_peak_bytes", "host_pageable_bytes"), ("device", "staging_peak_bytes", "device_staging_bytes"),
                    ("device", "peak_d2h_inflight_bytes", "d2h_inflight_bytes")):
                actual = manager.get(area, {}).get(key)
                if type(actual) is not int or actual < 0 or actual > limits[limit]:
                    reasons.append("missing_or_exceeded_managed_budget:" + key)
        if domains and any(sum(values[index] for values in domains.values()) > policy["host_pinned_bytes"]
                           for index in (0, 1)):
            reasons.append("physical_pinned_budget_unverified_or_exceeded")
    except (AttributeError, KeyError, TypeError, ValueError):
        reasons.append("runtime_metadata_unavailable_or_malformed")
    return list(dict.fromkeys(reasons))


def _selections(managers):
    selections = []
    for manager in managers:
        plans = manager.get("plans")
        if not isinstance(plans, (list, tuple)):
            raise ValueError("Missing regional offload plans")
        for plan in plans:
            if not isinstance(plan, dict) or type(plan.get("region_id")) is not str:
                raise ValueError("Missing stable offload region identity")
            ids = plan.get("selected_bundle_ids")
            if not isinstance(ids, (list, tuple)) or any(type(part) not in (str, int) for part in ids):
                raise ValueError("Malformed selected bundle identifiers")
            selections.append((plan["region_id"], tuple(sorted(str(part) for part in ids))))
    return sorted(selections)


def _actual_admissions(row):
    """Compare bounded executed residency, including regional fallback."""
    _, current = _runtime(row)
    _, frozen = _runtime({"packed_offload_runtime": row.get("packed_offload_profile")})

    def histories(managers):
        result = {}
        for manager in managers:
            planned = {region: ids for region, ids in _selections([manager])}
            if not planned:
                continue
            history = manager.get("admission_history")
            if manager.get("admission_history_truncated") is not False or not isinstance(history, (list, tuple)):
                raise ValueError("Missing or truncated actual admission history")
            records = []
            for item in history:
                if not isinstance(item, dict) or item.get("region_id") not in planned or type(item.get("resident_fallback")) is not bool:
                    raise ValueError("Malformed actual regional admission")
                ids = item.get("selected_bundle_ids")
                if not isinstance(ids, (list, tuple)) or any(type(value) not in (str, int) for value in ids):
                    raise ValueError("Malformed actual bundle selection")
                ids = tuple(sorted(str(value) for value in ids))
                if ids != (() if item["resident_fallback"] else planned[item["region_id"]]):
                    raise ValueError("Actual admission contradicts static selected inventory")
                records.append((item["region_id"], ids, item["resident_fallback"]))
            key = tuple(sorted(planned))
            if key in result:
                raise ValueError("Ambiguous repeated manager region identity")
            result[key] = tuple(records)
        return result

    after, before = histories(current), histories(frozen)
    if after.keys() != before.keys() or not after:
        raise ValueError("Warmup and measured runtime admission owners differ")
    result = []
    for key in sorted(after):
        prefix, complete = before[key], after[key]
        if complete[:len(prefix)] != prefix:
            raise ValueError("Measured admission history lost its frozen warmup prefix")
        measured = complete[len(prefix):]
        if not measured or any(sum(item[0] == region for item in measured) < row["updates"] for region in key):
            raise ValueError("Incomplete measured regional admission history")
        result.append((key, complete, measured))
    return tuple(result)


def _metrics(row):
    from examples.vae.benchmark_execution import _metrics as existing

    result = existing(row)
    stages = row.get("memory_audit", {}).get("snapshots", {})
    first = stages.get("after_first_update", {})
    host = stages.get("after_postchecks", {}).get("host_process", {})
    result.update(first_update_peak_allocated_bytes=first.get("peak_allocated_bytes"),
                  first_update_peak_reserved_bytes=first.get("peak_reserved_bytes"),
                  host_peak_rss_bytes=host.get("peak_rss_bytes"), host_rss_bytes=host.get("current_rss_bytes"))
    pinned = stages.get("after_postchecks", {}).get("pinned_host_allocator", {}).get("statistics", {})
    result["Torch_host_pinned_allocated_peak_bytes_approximate"] = pinned.get("allocated_bytes.peak")
    result["Torch_host_pinned_allocated_bytes"] = pinned.get("allocated_bytes.current")
    inventory = row.get("packed_offload_inventory", {})
    if isinstance(inventory, dict) and inventory.get("status") == "available":
        for name in ("eligible_history_bytes", "eligible_history_count", "selected_history_bytes", "selected_history_count",
                     "selected_physical_capacity_bytes"):
            value = inventory.get(name)
            result[name] = value if type(value) is int and value >= 0 else None
        eligible, selected = result.get("eligible_history_bytes"), result.get("selected_history_bytes")
        result["selected_history_coverage_percent"] = 100 * selected / eligible if eligible and selected is not None else None
    for key in ("physical_pinned_peak_bytes_upper_sum", "physical_pinned_peak_bytes",
                "actual_admitted_history_bytes", "actual_admitted_history_bytes_min",
                "actual_admitted_history_bytes_max", "actual_admitted_coverage_percent"):
        result[key] = None
    try:
        value, managers = _runtime(row)
        # Only owners proven to have no pool or execution are omitted from
        # exact singleton decisions. Raw snapshots and aggregate counters stay.
        managers = [manager for manager in managers if not _proven_unused_manager(manager)]
        for name in ("d2h_bytes", "h2d_bytes", "d2h_copies", "h2d_copies", "demand_misses", "prefetches", "cpu_copy_bytes"):
            amount = value.get("totals", {}).get("counters", {}).get(name)
            result["transport_warmup_and_measured_" + name] = amount if type(amount) in (int, float) and amount >= 0 else None
        for kind in ("pinned", "pageable"):
            result["host_" + kind + "_peak_bytes_upper_sum"] = sum(item.get("host", {}).get(kind + "_peak_bytes", 0) for item in managers)
            result["host_" + kind + "_peak_bytes"] = (managers[0].get("host", {}).get(kind + "_peak_bytes")
                if len(managers) == 1 else 0 if not managers else None)
            for field in ("active_bytes", "cached_bytes"):
                result["host_" + kind + "_" + field] = value.get("totals", {}).get("host", {}).get(kind + "_" + field)
        domains, physical_verified = {}, True
        for manager in managers:
            host = manager.get("host", {})
            domain, peak = host.get("pinned_allocator_domain_id"), host.get("pinned_physical_peak_bytes")
            if (type(domain) is not int or domain < 1 or host.get("pinned_accounting_verified") is not True
                    or type(peak) is not int or peak < 0):
                physical_verified = False
                break
            domains.setdefault(domain, []).append(peak)
        if physical_verified:
            result["physical_pinned_peak_bytes_upper_sum"] = sum(max(peaks) for peaks in domains.values())
            if not domains:
                result["physical_pinned_peak_bytes"] = 0
            elif len(domains) == 1:
                peaks = next(iter(domains.values()))
                if len(set(peaks)) == 1:
                    result["physical_pinned_peak_bytes"] = peaks[0]
        result["managed_staging_peak_bytes_upper_sum"] = sum(item.get("device", {}).get("staging_peak_bytes", 0) for item in managers)
        result["managed_staging_peak_bytes"] = (managers[0].get("device", {}).get("staging_peak_bytes")
            if len(managers) == 1 else 0 if not managers else None)
        admitted = []
        for manager in managers:
            plans = {plan["region_id"]: plan for plan in manager.get("plans", ())}
            per_region_ordinals, per_forward = {}, {}
            for event in manager.get("admission_history", ()):
                region = event.get("region_id")
                plan = plans.get(region)
                if plan is None:
                    continue
                number = per_region_ordinals.get(region, 0)
                per_region_ordinals[region] = number + 1
                ids = set(event.get("selected_bundle_ids", ()))
                sizes = plan.get("bundle_bytes", {})
                if not ids:
                    nbytes = 0
                elif isinstance(sizes, dict) and all(type(sizes.get(key)) is int for key in ids):
                    nbytes = sum(sizes[key] for key in ids)
                elif ids == set(plan.get("selected_bundle_ids", ())):
                    nbytes = plan.get("selected_bytes")
                else:
                    nbytes = None
                if nbytes is None or per_forward.get(number, 0) is None:
                    per_forward[number] = None
                else:
                    per_forward[number] = per_forward.get(number, 0) + nbytes
            admitted.extend(value for value in per_forward.values() if type(value) is int)
        if admitted and len(managers) == 1:
            import statistics

            result["actual_admitted_history_bytes"] = statistics.median(admitted)
            result["actual_admitted_history_bytes_min"] = min(admitted)
            result["actual_admitted_history_bytes_max"] = max(admitted)
            result["actual_admitted_coverage_percent"] = (100 * statistics.median(admitted) / result["eligible_history_bytes"]
                if result.get("eligible_history_bytes") else None)
    except ValueError:
        pass
    return result


def worker_errors(row, policy, expected_settings, identity, *, host_capacity_policy=None):
    from examples.vae.benchmark_execution import _worker_errors

    if type(row) is not dict:
        return ["malformed_worker_receipt"]
    try:
        reasons = _worker_errors(row, require_memory_audit=True)
    except (AttributeError, TypeError, ValueError, KeyError):
        return ["malformed_worker_receipt"]
    if row.get("installed") != identity or row.get("settings") != expected_settings:
        reasons.append("installation_or_declared_settings_mismatch")
    if host_capacity_policy is not None and row.get("host_capacity_policy") != host_capacity_policy:
        reasons.append("host_capacity_policy_mismatch")
    if policy.get("version", 1) == 2 and str(row.get("device", "")).startswith("cuda"):
        capacity_bytes = policy["host_pinned_bytes"] + policy["host_pageable_bytes"] if policy["enabled"] else 0
        capacity = row.get("host_capacity_preflight", {})
        capacity_policy = row.get("host_capacity_policy")
        headroom = capacity_policy.get("reserved_headroom_bytes") if isinstance(capacity_policy, dict) else None
        snapshot = capacity.get("snapshot") if isinstance(capacity, dict) else None
        available = snapshot.get("available_bytes") if isinstance(snapshot, dict) else None
        required = capacity.get("required_plus_headroom_bytes") if isinstance(capacity, dict) else None
        if (not isinstance(capacity, dict) or capacity.get("admitted") is not True
                or capacity.get("status") != "admitted" or type(headroom) is not int or headroom < 0
                or capacity.get("reserved_headroom_bytes") != headroom
                or not isinstance(snapshot, dict) or snapshot.get("status") != "available"
                or type(required) is not int or type(available) is not int or available < required
                or capacity.get("required_physical_capacity_bytes") != capacity_bytes
                or capacity.get("required_plus_headroom_bytes") != capacity_bytes + headroom
                or capacity.get("requested_capacity_clamped") is not False):
            reasons.append("physical_host_capacity_preflight_missing_or_failed")
    for location in (row.get("settings", {}), row.get("runtime_config", {})):
        if not isinstance(location, dict) or not isinstance(location.get("runtime"), dict):
            reasons.append("malformed_runtime_configuration")
            continue
        actual = location["runtime"].get("packed_activation_offload")
        try:
            if normalize_policy(actual) != policy:
                reasons.append("effective_packed_policy_mismatch")
        except (TypeError, ValueError):
            reasons.append("effective_packed_policy_missing")
    if any(not row.get(name) for name in ("rng_before_warmup_sha256", "rng_after_warmup_sha256", "rng_after_measurement_sha256")):
        reasons.append("missing_RNG_identity")
    try:
        value, managers = _runtime(row)
        reasons += runtime_budget_errors(row.get("packed_offload_runtime"), policy)
        frozen = row.get("packed_offload_profile")
        if not isinstance(frozen, dict) or frozen.get("snapshot", {}).get("frozen") is not True or value.get("frozen") is not True:
            reasons.append("offload_profile_not_frozen_before_measurement")
        selections = _selections(managers)
        sizes = [plan.get("selected_bytes") for manager in managers for plan in manager.get("plans", ())]
        if any(type(size) is not int or size < 0 for size in sizes):
            raise ValueError("Malformed selected bundle byte counts")
        selected = sum(sizes)
        if policy["enabled"] and (not selections or selected <= 0):
            reasons.append("no_selected_offload_bundles")
        if policy["enabled"]:
            _actual_admissions(row)
            mode = expected_settings.get("runtime", {}).get("compile", {}).get("options", {}).get("packed_offload_selection_mode", "peak")
            if mode == "all_eligible":
                inventory = row.get("packed_offload_inventory", {})
                if (not isinstance(inventory, dict) or inventory.get("status") != "available"
                        or inventory.get("all_eligible_selected") is not True
                        or type(inventory.get("eligible_history_bytes")) is not int
                        or inventory.get("eligible_history_bytes", 0) <= 0
                        or type(inventory.get("eligible_history_count")) is not int
                        or inventory.get("eligible_history_count", 0) <= 0
                        or inventory.get("selected_history_bytes") != inventory.get("eligible_history_bytes")
                        or inventory.get("selected_history_count") != inventory.get("eligible_history_count")
                        or selected != inventory.get("eligible_history_bytes")
                        or sum(len(ids) for _, ids in selections) != inventory.get("eligible_history_count")
                        or type(inventory.get("eligible_inventory_sha256")) is not str
                        or len(inventory.get("eligible_inventory_sha256", "")) != 64):
                    reasons.append("all_eligible_inventory_incomplete")
                for manager in managers:
                    by_region = {item["region_id"]: set(item.get("selected_bundle_ids", ())) for item in manager.get("plans", ())}
                    for event in manager.get("admission_history", ()):
                        if (event.get("resident_fallback") is not False
                                or set(event.get("selected_bundle_ids", ())) != by_region.get(event.get("region_id"))):
                            reasons.append("all_eligible_actual_admission_incomplete")
                            break
        if not policy["enabled"] and selected:
            reasons.append("resident_arm_selected_transport_bundles")
        if policy["enabled"] and str(row.get("device", "")).startswith("cuda"):
            counters = value.get("totals", {}).get("counters", {})
            if any(type(counters.get(key)) is not int or counters[key] <= 0 for key in ("d2h_bytes", "h2d_bytes", "d2h_copies", "h2d_copies")):
                reasons.append("CUDA_transport_not_observed")
        try:
            cleanup, cleanup_managers = _runtime({"packed_offload_runtime": row.get("packed_offload_cleanup")})
            lifecycle = cleanup.get("totals", {}).get("lifecycle", {})
            if lifecycle.get("quiescent") is not True or any(lifecycle.get(key, 0) != 0 for key in
                    ("live_forwards", "backward_tasks", "graph_leases", "task_leases", "dma_leases")):
                reasons.append("packed_cleanup_not_quiescent")
            if cleanup_managers and any(type(lifecycle.get(key)) is not int for key in
                    ("live_forwards", "backward_tasks", "graph_leases", "task_leases", "dma_leases")):
                reasons.append("packed_cleanup_lifecycle_fields_missing")
            for manager in cleanup_managers:
                errors = manager.get("cleanup_errors", ())
                if not isinstance(errors, (list, tuple)) or errors:
                    reasons.append("runtime_cleanup_errors")
        except (AttributeError, KeyError, TypeError, ValueError):
            reasons.append("packed_cleanup_unavailable")
    except (AttributeError, KeyError, TypeError, ValueError):
        reasons.append("runtime_metadata_unavailable_or_malformed")
    return reasons


def aggregate(results, plan):
    from examples.vae.benchmark_execution import _statistics
    from examples.vae.memory_comparison import compare_memory_baselines

    summary = {"valid": False, "invalid_reasons": [], "arms": {}, "blocks": [], "frontier": [],
        "minimum_memory_arms": [], "performance_policy_promoted": False,
        "objective": plan["objective"], "independent_blocks": len(plan["order"]), "runtime_is_a_tradeoff_not_a_memory_veto": True}
    # All arms execute native AD. The memory auditor's generic fallback label
    # is execution_mode; supply the experiment arm without changing that owner.
    audit_rows = [{**row, "arm": row.get("offload_arm", row.get("arm", "worker"))} for row in results]
    summary["memory_isolation"] = compare_memory_baselines(audit_rows, tolerance_bytes=plan["baseline_tolerance_bytes"])
    summary["invalid_reasons"] += summary["memory_isolation"]["invalid_reasons"]
    active_arms = tuple(plan.get("active_arms", ARMS))
    for arm in active_arms:
        selected = [row for row in results if row.get("offload_arm") == arm]
        good = [row for row in selected if not worker_errors(row, plan["manifest"]["policies"][arm], plan["workers"][arm]["settings"], plan["installed"],
            host_capacity_policy=plan.get("host_capacity_policy"))]
        metrics = [_metrics(row) for row in good]
        names = set().union(*(item.keys() for item in metrics)) if metrics else set()
        summary["arms"][arm] = {"successful_trials": len(good), "total_trials": len(selected),
            "metrics": {name: _statistics([item.get(name) for item in metrics]) for name in sorted(names)}}
    anchor = None
    for block in range(1, len(plan["order"]) + 1):
        rows = [row for row in results if row.get("block") == block]
        by_arm = {name: [row for row in rows if row.get("offload_arm") == name] for name in active_arms}
        reasons = []
        if len(rows) != len(active_arms) or any(len(value) != 1 for value in by_arm.values()):
            reasons.append("missing_or_duplicate_arms")
        else:
            values = {arm: by_arm[arm][0] for arm in active_arms}
            for arm, row in values.items():
                reasons += [arm + ":" + error for error in worker_errors(row, plan["manifest"]["policies"][arm], plan["workers"][arm]["settings"], plan["installed"],
                    host_capacity_policy=plan.get("host_capacity_policy"))]
            if not reasons:
                reference = _identity(values["resident"])
                if anchor is not None and reference != anchor:
                    reasons.append("identity_drift_across_blocks")
                anchor = reference if anchor is None else anchor
                for arm, row in values.items():
                    if _identity(row) != reference:
                        reasons.append(arm + ":math_state_input_RNG_source_or_GPU_drift")
            if not reasons and "predictive_offload" in active_arms:
                if _selections(_runtime(values["demand_offload"])[1]) != _selections(_runtime(values["predictive_offload"])[1]):
                    reasons.append("demand_predictive_selected_bundles_differ")
                if _actual_admissions(values["demand_offload"]) != _actual_admissions(values["predictive_offload"]):
                    reasons.append("demand_predictive_actual_admissions_differ")
        paired = {}
        if not reasons:
            reference_metrics = _metrics(values["resident"])
            for arm in active_arms:
                if arm == "resident":
                    continue
                actual_metrics = _metrics(values[arm])
                metrics = {}
                for key in ("wall_ms_per_update", "cuda_event_ms_per_update"):
                    before, after = reference_metrics.get(key), actual_metrics.get(key)
                    if type(before) in (int, float) and before > 0 and type(after) in (int, float) and after > 0:
                        metrics[key + "_overhead_percent"] = 100 * (after / before - 1)
                for scope in ("total", "startup", "steady", "first_update"):
                    for allocation in ("allocated", "reserved"):
                        key = scope + "_peak_" + allocation + "_bytes"
                        before, after = reference_metrics.get(key), actual_metrics.get(key)
                        if type(before) in (int, float) and before > 0 and type(after) in (int, float) and after >= 0:
                            metrics[key + "_saved"] = before - after
                            metrics[key + "_reduction_percent"] = 100 * (before - after) / before
                paired[arm] = metrics
        summary["blocks"].append({"block": block, "valid": not reasons, "invalid_reasons": reasons,
            "paired_vs_resident": paired})
        summary["invalid_reasons"] += [str(block) + ":" + reason for reason in reasons]
    summary["valid"] = bool(results) and not summary["invalid_reasons"]
    points = []
    for arm, value in summary["arms"].items():
        metrics = value["metrics"]
        peak, wall = metrics.get("total_peak_allocated_bytes"), metrics.get("wall_ms_per_update")
        if peak and wall:
            points.append({"arm": arm, "whole_run_peak_allocated_bytes": peak["median"], "wall_ms_per_update": wall["median"]})
    summary["available_points"] = points
    if points and summary["valid"]:
        minimum = min(item["whole_run_peak_allocated_bytes"] for item in points)
        summary["minimum_memory_arms"] = [item["arm"] for item in points if item["whole_run_peak_allocated_bytes"] == minimum]
        summary["frontier"] = [item for item in points if not any(
            other["whole_run_peak_allocated_bytes"] <= item["whole_run_peak_allocated_bytes"] and other["wall_ms_per_update"] <= item["wall_ms_per_update"]
            and (other["whole_run_peak_allocated_bytes"] < item["whole_run_peak_allocated_bytes"] or other["wall_ms_per_update"] < item["wall_ms_per_update"])
            for other in points)]
    summary["CUDA_execution_proven"] = summary["valid"] and all(str(row.get("device", "")).startswith("cuda") for row in results)
    summary["frontier_qualified"] = False
    summary["frontier_scope"] = "descriptive measured points; policy promotion requires separate target release gates"
    summary["managed_peak_accounting"] = "sum of regional high-water marks is an upper bound, not a simultaneous global host/device peak"
    summary["managed_exact_peak_scope"] = "reported only when one shared runtime manager owns the worker; multiple-manager peaks use upper sums"
    summary["declared_active_arms"] = list(active_arms)
    positions = {arm: [sum(order[index] == arm for order in plan["order"]) for index in range(len(active_arms))]
                 for arm in active_arms}
    summary["statistical_design"] = {"independent_blocks": len(plan["order"]), "required_blocks": 6,
        "position_counts": positions, "balanced_matrix_complete": len(plan["order"]) >= 6 and
            all(len(set(value)) == 1 for value in positions.values()),
        "consecutive_updates_are_independent_trials": False, "one_block_is_triage_only": True,
        "preferred_performance_target_qualified": False}
    summary["paired_comparisons"] = {}
    if summary["valid"]:
        for arm in active_arms:
            if arm == "resident":
                continue
            paired = [block["paired_vs_resident"][arm] for block in summary["blocks"]]
            names = set().union(*(value.keys() for value in paired))
            summary["paired_comparisons"][arm] = {"independent_pairs": len(paired),
                "metrics": {name: _statistics([value.get(name) for value in paired]) for name in sorted(names)},
                "scope": "descriptive matched independent process blocks; updates are not additional samples"}
    return summary


def _primary_failure(row, reasons=()):
    """One actual exception/phase for the terminal; raw receipts keep stacks."""
    context = row.get("failure_context")
    context = context if isinstance(context, dict) else {}
    error = row.get("error")
    if isinstance(error, str) and error:
        kind, message = "worker_exception", error
    elif row.get("worker_result_error"):
        kind, message = "worker_receipt", row["worker_result_error"]
    elif row.get("worker_exit_code", 0) != 0:
        kind, message = "worker_exit", f"Worker exited with status {row.get('worker_exit_code')}"
    elif reasons:
        kind, message = "evidence_validation", reasons[0]
    else:
        return None
    return {"kind": kind, "stage": context.get("stage"), "phase": context.get("phase"),
            "update": context.get("update"), "message": " ".join(str(message).split())[:2048]}


def _print_worker_failure(block, arm, primary, *, configuration=False, diagnostic=False):
    if primary is None:
        return
    label = "configuration" if configuration else "diagnostic" if diagnostic else f"block {block}"
    phase = "/".join(str(primary[key]) for key in ("stage", "phase") if primary.get(key)) or primary["kind"]
    print(f"Packed offload {label}, {arm}: {phase}: {primary['message']}", flush=True)


def _configuration_errors(row, plan, arm):
    errors = []
    if row.get("status") != "completed" or row.get("worker_exit_code") != 0:
        errors.append("configuration_worker_not_completed")
    preflight = row.get("configuration_preflight")
    if not isinstance(preflight, dict):
        return errors + ["configuration_preflight_receipt_missing"]
    checks = preflight.get("checks")
    if (preflight.get("requested") is not True or preflight.get("valid") is not True
            or not isinstance(checks, dict) or any(checks.get(name) is not True for name in
                ("installed_identity", "arm_policy_pins", "typed_config", "backend_config", "single_process_execution"))):
        errors.append("configuration_preflight_checks_incomplete")
    if (row.get("scope") != "configuration_preflight_only" or row.get("execution_completed") is not False
            or row.get("timing_eligible") is not False or preflight.get("no_model_allocated") is not True
            or preflight.get("no_input_allocated") is not True or preflight.get("cuda_execution") is not False):
        errors.append("configuration_preflight_execution_scope_invalid")
    if row.get("installed") != plan["installed"] or row.get("settings") != plan["workers"][arm]["settings"]:
        errors.append("configuration_preflight_settings_or_installation_mismatch")
    expected_capacity = plan["workers"][arm].get("host_capacity_policy")
    if expected_capacity is not None and row.get("host_capacity_policy") != expected_capacity:
        errors.append("configuration_preflight_host_capacity_policy_mismatch")
    if row.get("origin") != plan["origin"]:
        errors.append("configuration_preflight_origin_mismatch")
    if any(row.get(name) is not True for name in ("installed_stable", "helper_source_stable")):
        errors.append("configuration_preflight_source_unstable")
    settings = row.get("settings")
    runtime = settings.get("runtime") if isinstance(settings, dict) else None
    runtime = runtime if isinstance(runtime, dict) else {}
    compile_options = runtime.get("compile")
    compile_options = compile_options if isinstance(compile_options, dict) else {}
    native_options = compile_options.get("options")
    native_options = native_options if isinstance(native_options, dict) else {}
    if (compile_options.get("enabled") is not True or compile_options.get("backend") != "deepopt"
            or compile_options.get("fullgraph") is not True or compile_options.get("dynamic") is not False
            or compile_options.get("mode") is not None
            or any(native_options.get(name) is not True for name in ("use_native_autodiff", "require_full_aot"))):
        errors.append("configuration_preflight_native_owner_invalid")
    try:
        if normalize_policy(runtime.get("packed_activation_offload")) != plan["manifest"]["policies"][arm]:
            errors.append("configuration_preflight_policy_mismatch")
    except (TypeError, ValueError):
        errors.append("configuration_preflight_policy_missing")
    return errors


def _pinned_command(args, plan, directory, prefix, arm, *, diagnostic=False, configuration=False):
    from examples.vae import benchmark_execution as paired
    from examples.vae.native_execution import resolved_native_options_sha256

    path, log = directory / (prefix + ".json"), directory / (prefix + ".log")
    policy = directory / (prefix + ".policy.json")
    policy.write_text(json.dumps(plan["manifest"]["policies"][arm], sort_keys=True) + "\n")
    resolved = deepcopy(plan["workers"][arm]["settings"]["runtime"]["compile"]["options"])
    for name in ("bf16_policy", "activation_memory", "packed_activation_offload"):
        resolved.pop(name, None)
    native_options = directory / (prefix + ".native-options.json")
    native_payload = (json.dumps(resolved, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    native_options.write_bytes(native_payload)
    pin = directory / (prefix + ".arm.json")
    pin.write_text(json.dumps({"schema_version": 1, "arm": arm,
        "expected_installed": plan["manifest"]["candidate"]["expected_installed"],
        "wheel_sha256": plan["manifest"]["candidate"]["wheel"]["sha256"],
        "native_options_sha256": hashlib.sha256(native_payload).hexdigest(),
        "resolved_native_options_sha256": resolved_native_options_sha256(plan["workers"][arm]["settings"]),
        "packed_offload_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}, indent=2) + "\n")
    options = deepcopy(args)
    options.arm_manifest = pin
    options.native_options_json = native_options
    phase = "steady" if diagnostic is True else diagnostic
    options.profile_after_measurement = phase == "steady"
    command = paired.worker_command(options, "native", path) + ["--packed-offload-json", str(policy),
        "--native-options-json", str(native_options), "--host-headroom-mib", str(getattr(args, "host_headroom_mib", 8192))]
    if diagnostic:
        command += ["--diagnostic-phase", phase, "--storage-ledger"]
    if configuration:
        command.append("--configuration-preflight")
    return path, log, command


def _read_worker(path, *, exit_code, block, arm):
    from examples.vae.benchmark_execution import _reject_json_constant

    row = None
    try:
        row = json.loads(path.read_text(), parse_constant=_reject_json_constant)
        if not isinstance(row, dict):
            raise ValueError("Worker receipt must be an object")
    except Exception as error:
        row = {"status": "failed", "worker_result_error": f"{type(error).__name__}: {error}"}
    row.update(block=block, offload_arm=arm, arm=arm, worker_exit_code=exit_code)
    if row.get("status") not in ("completed", "failed"):
        row["status"] = "failed"
        row["worker_result_error"] = "Worker receipt omitted a recognized final status"
    return row


def _diagnostic_errors(row, plan, arm, phase):
    if phase == "steady":
        errors = worker_errors(row, plan["manifest"]["policies"][arm], plan["workers"][arm]["settings"], plan["installed"])
    else:
        errors = []
        if row.get("status") != "completed" or row.get("execution_completed") is not True or row.get("worker_exit_code") != 0:
            errors.append("cold_update_not_completed")
        if row.get("warmup_completed") != 0 or row.get("updates_completed") != 1 or "measurement" in row:
            errors.append("cold_update_window_invalid")
        if row.get("settings") != plan["workers"][arm]["settings"] or row.get("installed") != plan["installed"]:
            errors.append("diagnostic_settings_or_installation_mismatch")
        if row.get("native_ownership_verified") is not True:
            errors.append("diagnostic_native_owner_unverified")
        if any(row.get(key) is not True for key in ("source_stable", "installed_stable", "helper_source_stable")):
            errors.append("diagnostic_source_identity_unverified")
    profile = row.get("profile")
    if not isinstance(profile, dict) or profile.get("complete") is not True or profile.get("diagnostic_phase") != phase:
        errors.append("diagnostic_profile_incomplete")
    elif not isinstance(profile.get("storage_ledger"), dict) or profile["storage_ledger"].get("tensor_files_written") != 0:
        errors.append("diagnostic_weak_storage_ledger_missing")
    return errors


def _terminal_report(report, output):
    summary = report.get("summary", {})
    arms = {}
    for arm, row in summary.get("arms", {}).items():
        metrics = row.get("metrics", {})
        arms[arm] = {"successful_trials": row.get("successful_trials"), "total_trials": row.get("total_trials"),
            "metrics": {name: metrics[name] for name in ("wall_ms_per_update", "cuda_event_ms_per_update",
                "total_peak_allocated_bytes", "steady_peak_allocated_bytes", "total_peak_reserved_bytes") if name in metrics}}
    failures = [{"block": row.get("block"), "arm": row.get("arm"), "diagnostic": diagnostic,
                 **row["primary_failure"]}
                for diagnostic, rows in ((False, report.get("results", ())), (True, report.get("diagnostics", ())))
                for row in rows if row.get("primary_failure")]
    configuration = report.get("configuration_preflight", {})
    return {"status": report["status"], "output": str(output), "configuration_preflight": {
        "valid": configuration.get("valid"),
        "invalid_reason_count": len(configuration.get("invalid_reasons", ())),
        "first_invalid_reasons": configuration.get("invalid_reasons", [])[:8],
        "worker_failures": [{"arm": row["arm"], **row["primary_failure"]}
            for row in configuration.get("workers", ()) if row.get("primary_failure")]},
        "summary": {"valid": summary.get("valid"), "arms": arms, "worker_failures": failures,
            "invalid_reason_count": len(summary.get("invalid_reasons", ())),
            "first_invalid_reasons": summary.get("invalid_reasons", [])[:8],
            "full_validation_reasons": "Retained in the output JSON and linked raw worker receipts"},
        **({"supervisor_error": report["error"]} if report.get("error") else {})}


def run_benchmark(args):
    from examples.vae import benchmark_vae as benchmark
    from examples.vae import benchmark_execution as paired
    from examples.vae.benchmark_optimization import verify_arm

    plan = benchmark_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    output = args.output.resolve()
    directory = output.parent / (output.stem + "-artifacts")
    if output.exists() or directory.exists():
        raise FileExistsError("Choose a fresh offload experiment output")
    directory.mkdir(parents=True)
    report = {**plan, "status": "running", "results": [], "diagnostics": [], "inventories": [],
        "configuration_preflight_only": getattr(args, "configuration_preflight_only", False),
        "configuration_preflight": {"valid": False, "workers": [], "invalid_reasons": [],
                                    "scope": "typed_configuration_only_no_model_or_CUDA_execution"},
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    benchmark.write_json(output, report)
    rows = []
    try:
        from benchmarks.compiler.installed_runtime import child_environment

        report["helper_source_artifacts"] = []
        for name, expected in plan["source_sha256"].items():
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError("Helper source changed before saving: " + name)
            target = directory / "helper-sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            report["helper_source_artifacts"].append(paired._reference(target, output.parent))
        manifest_copy = directory / "experiment-manifest.json"
        manifest_copy.write_bytes(args.experiment_manifest.read_bytes())
        if hashlib.sha256(manifest_copy.read_bytes()).hexdigest() != plan["manifest_source"]["sha256"]:
            raise RuntimeError("Manifest changed before saving")
        report["manifest_artifact"] = paired._reference(manifest_copy, output.parent)
        environment = child_environment(ROOT, identity=plan["installed"])
        environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
        environment["TORCHDYNAMO_REPRO_LEVEL"] = "0"
        for arm in plan["active_arms"]:
            path, log, command = _pinned_command(args, plan, directory, "configuration-" + arm, arm, configuration=True)
            code = None
            with log.open("x") as stream:
                try:
                    code = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
                except OSError as error:
                    stream.write(str(error) + "\n")
            row = _read_worker(path, exit_code=code, block=0, arm=arm)
            reasons = _configuration_errors(row, plan, arm)
            primary = _primary_failure(row, reasons)
            _print_worker_failure(0, arm, primary, configuration=True)
            record = {"arm": arm, "status": row["status"], "worker_exit_code": code,
                "validation_errors": reasons, "primary_failure": primary, "log": paired._reference(log, output.parent)}
            if path.is_file():
                record["result"] = paired._reference(path, output.parent)
            report["configuration_preflight"]["workers"].append(record)
            report["configuration_preflight"]["invalid_reasons"] += [arm + ":" + reason for reason in reasons]
            benchmark.write_json(output, report)
        report["configuration_preflight"]["valid"] = not report["configuration_preflight"]["invalid_reasons"]
        if not report["configuration_preflight"]["valid"]:
            report["summary"] = {"valid": False, "invalid_reasons": ["configuration_preflight_failed"],
                                 "arms": {}, "CUDA_execution_proven": False, "frontier_qualified": False}
            report["status"] = "failed"
            report.update(execution_completed=False, timing_eligible=False)
            return_code = 1
        elif report["configuration_preflight_only"]:
            report["summary"] = {"valid": True, "invalid_reasons": [], "arms": {},
                                 "CUDA_execution_proven": False, "frontier_qualified": False}
            report["status"] = "completed"
            report.update(scope="configuration_preflight_only", execution_completed=False, timing_eligible=False)
            return_code = 0
        else:
            return_code = None
        jobs = [(block, arm, False) for block, order in enumerate(plan["order"], 1) for arm in order]
        if getattr(args, "cold_diagnostics", False):
            jobs += [(0, arm, "cold") for arm in plan["active_arms"]]
        if args.profile_after_measurement:
            jobs += [(0, arm, "steady") for arm in plan["active_arms"]]
        if return_code is not None:
            jobs = []
        for block, arm, diagnostic in jobs:
            prefix = (("diagnostic-cold-" if diagnostic == "cold" else "diagnostic-") if diagnostic else f"block-{block}-") + arm
            path, log, command = _pinned_command(args, plan, directory, prefix, arm, diagnostic=diagnostic)
            code = None
            with log.open("x") as stream:
                try:
                    code = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
                except OSError as error:
                    stream.write(str(error) + "\n")
            row = _read_worker(path, exit_code=code, block=block, arm=arm)
            if row.get("origin") != plan["origin"]:
                row.update(status="failed", worker_result_error="Worker original receipt differs from planning")
            validation = (_diagnostic_errors(row, plan, arm, diagnostic) if diagnostic else
                worker_errors(row, plan["manifest"]["policies"][arm], plan["workers"][arm]["settings"], plan["installed"]))
            row["supervisor_validation_errors"] = validation
            primary = _primary_failure(row, validation)
            _print_worker_failure(block, arm, primary, diagnostic=diagnostic)
            receipt = {"block": block, "arm": arm, "status": row["status"], "worker_exit_code": code,
                "diagnostic_phase": diagnostic or None,
                "error": row.get("error"), "measurement": row.get("measurement"), "log": paired._reference(log, output.parent),
                "primary_failure": primary, "validation_errors": validation}
            if path.is_file():
                receipt["result"] = paired._reference(path, output.parent)
            inventory = row.get("packed_offload_inventory")
            if (isinstance(inventory, dict) and inventory.get("status") == "available"
                    and row.get("installed") == plan["installed"] and row.get("origin") == plan["origin"]
                    and row.get("settings") == plan["workers"][arm]["settings"]):
                inventory_path = directory / (prefix + "-inventory.json")
                benchmark.write_json(inventory_path, inventory)
                report["inventories"].append({"block": block, "arm": arm,
                    "diagnostic_phase": diagnostic or None, "artifact": paired._reference(inventory_path, output.parent)})
                if arm == "demand_offload" and "inventory_artifact" not in report:
                    report["inventory_artifact"] = paired._reference(inventory_path, output.parent)
                    if args.inventory_output is not None:
                        destination = args.inventory_output.resolve()
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("xb") as stream:
                            stream.write(inventory_path.read_bytes())
                        report["inventory_export"] = {"path": str(destination), "bytes": destination.stat().st_size,
                            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
            report["diagnostics" if diagnostic else "results"].append(receipt)
            if not diagnostic:
                rows.append(row)
            benchmark.write_json(output, report)
        if verify_arm(plan["manifest"]["candidate"], args.experiment_manifest.resolve()) != plan["installed"]:
            raise RuntimeError("Candidate installation changed")
        if hashlib.sha256(args.experiment_manifest.read_bytes()).hexdigest() != plan["manifest_source"]["sha256"] or source_identity() != plan["source_sha256"]:
            raise RuntimeError("Offload experiment manifest or helper sources changed")
        if return_code is None:
            report["summary"] = aggregate(rows, plan)
            report["status"] = "completed" if report["summary"]["valid"] else "failed"
            report["CUDA_execution_proven"] = report["summary"]["CUDA_execution_proven"]
            if args.inventory_output is not None and "inventory_export" not in report:
                raise RuntimeError("Requested compact compiled demand inventory is unavailable; completed peers remain preserved")
    except Exception as error:
        report.update(benchmark.failure_error_details(error), status="failed")
        try:
            report["summary"] = aggregate(rows, plan)
        except Exception as nested:
            report["summary"] = {"invalid_reasons": [], "aggregation_error": benchmark.failure_error_details(nested)}
        report["summary"].update(valid=False, frontier_qualified=False)
        report["summary"]["invalid_reasons"].append("supervisor_failed")
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        benchmark.write_json(output, report)
    print(json.dumps(_terminal_report(report, output), indent=2))
    return 0 if report["status"] == "completed" else 1


def main(argv=None):
    return run_benchmark(parser().parse_args(argv))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
