"""Compact immutable compiled offload inventory, without numerical payloads."""

from copy import deepcopy
import hashlib
import json


INVENTORY_FIELDS = ("eligible_history_bytes", "eligible_history_count", "eligible_bundle_count", "all_eligible_host_physical_capacity_bytes",
    "required_staging_bytes", "required_d2h_inflight_bytes", "rejected_by_reason", "stage_floor_panel_proof",
    "selected_history_bytes", "selected_history_count", "selected_bundle_count", "selection_mode", "signature", "scope_identity",
    "selected_physical_capacity_bytes", "all_eligible_selected", "all_eligible_capacity_is_feasible_upper_bound_not_minimum",
    "eligible_inventory_sha256", "floor_proof")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def comparison_options(settings):
    options = deepcopy(settings["runtime"]["compile"].get("options", {}))
    options.pop("packed_activation_offload", None)
    options.pop("packed_offload_selection_mode", None)
    options.pop("packed_offload_peak_profile", None)
    return options


def _declared_native_regions(compiler_report):
    """Read every declared owner, without searching unrelated graph metadata."""
    records, errors = [], []
    if type(compiler_report) is not dict:
        return records, [{"path": "", "reason": "invalid_compiler_report"}]
    graphs = compiler_report.get("graphs", [compiler_report])
    if type(graphs) not in (list, tuple):
        return records, [{"path": "graphs", "reason": "invalid_graph_reports"}]
    if "graphs" in compiler_report and "captured_graphs" in compiler_report:
        count = compiler_report["captured_graphs"]
        if type(count) is not int or count != len(graphs):
            errors.append({"path": "graphs", "reason": "declared_graph_count_mismatch"})
    native_count = 0
    for index, graph in enumerate(graphs):
        path = f"graphs/{index}" if "graphs" in compiler_report else ""
        if type(graph) is not dict:
            errors.append({"path": path, "reason": "invalid_graph_report"})
            continue
        regions = graph.get("native_autodiff_graphs", ())
        if type(regions) not in (list, tuple):
            errors.append({"path": path + "/native_autodiff_graphs", "reason": "invalid_native_regions"})
            continue
        native_count += len(regions)
        if "native_autodiff_regions" in graph:
            count = graph["native_autodiff_regions"]
            if type(count) is not int or count != len(regions):
                errors.append({"path": path + "/native_autodiff_graphs", "reason": "declared_native_region_count_mismatch"})
        options = graph.get("effective_compiler_options", {})
        policy = options.get("packed_activation_offload", {}) if type(options) is dict else {}
        requires_v2 = (type(policy) is dict and policy.get("enabled") is True and policy.get("version") == 2)
        requires_v2 = requires_v2 or any(type(region) is dict
            and type(region.get("packed_activation_offload")) is dict
            and region["packed_activation_offload"].get("transport_envelope_version") == 2 for region in regions)
        for region_index, region in enumerate(regions):
            source = f"{path}/native_autodiff_graphs/{region_index}"
            if type(region) is not dict:
                errors.append({"path": source, "reason": "invalid_native_region"})
                continue
            records.append((source, region))
            if "packed_activation_offload" not in region:
                if requires_v2:
                    errors.append({"path": source + "/packed_activation_offload", "reason": "missing_v2_offload_record"})
                continue
            offload = region["packed_activation_offload"]
            if type(offload) is not dict:
                errors.append({"path": source + "/packed_activation_offload", "reason": "invalid_offload_record"})
                continue
            if requires_v2 and offload.get("transport_envelope_version") != 2:
                errors.append({"path": source + "/packed_activation_offload", "reason": "invalid_v2_offload_record"})
    if "graphs" in compiler_report and "native_autodiff_regions" in compiler_report:
        count = compiler_report["native_autodiff_regions"]
        if type(count) is not int or count != native_count:
            errors.append({"path": "graphs", "reason": "aggregate_native_region_count_mismatch"})
    return records, errors


def _declared_model_scopes(compiler_report):
    """Collect whole-model descriptors repeated by the declared native regions."""
    regions, errors = _declared_native_regions(compiler_report)
    records = []
    for path, region in regions:
        offload = region.get("packed_activation_offload")
        if type(offload) is dict:
            if "model_scope" not in offload and offload.get("transport_envelope_version") != 2:
                continue  # Legacy regional plans do not declare a v2 inventory.
            scope = offload.get("model_scope")
            source = path + "/packed_activation_offload/model_scope"
            if type(scope) is not dict or type(scope.get("inventory")) is not dict:
                errors.append({"path": source, "reason": "incomplete_model_scope_inventory"})
                continue
            records.append((source, scope))
    return records, errors


def extract_inventory(compiler_report, plan, installed):
    scopes, errors = _declared_model_scopes(compiler_report)
    records = []
    for path, scope in scopes:
        value = scope["inventory"]
        if any(key not in value for key in ("eligible_history_bytes", "all_eligible_host_physical_capacity_bytes")):
            errors.append({"path": path + "/inventory", "reason": "incomplete_model_scope_inventory"})
        else:
            records.append((path + "/inventory", value))
    candidates = {}
    for _, value in records:
        candidate = {key: deepcopy(value[key]) for key in INVENTORY_FIELDS if key in value}
        candidates[_digest(candidate)] = candidate
    result = {"schema_version": 1, "status": "unavailable", "candidate_metadata_only": True,
        "scope": "compiled_model_scope_eligible_inventory_not_actual_admission_or_GPU_peak",
        "installed": {key: installed[key] for key in ("version", "source_sha256", "files_hashed")},
        "origin": deepcopy(plan["origin"]), "requested_input_shape": list(plan["requested_input_shape"]),
        "model_options": deepcopy(plan["settings"]["application"]["model"]),
        "comparison_native_options_sha256": _digest(comparison_options(plan["settings"])),
        "comparison_native_options": comparison_options(plan["settings"]),
        "recipe": {key: deepcopy(plan["settings"]["runtime"].get(key)) for key in
            ("precision", "bf16_policy", "activation_compression", "activation_memory")},
        "tensor_files_written": 0, "search_truncated": False,
        "inventory_extraction": {"method": "declared_native_region_model_scope_records",
            "records_checked": len(records), "source_paths": [path for path, _ in records], "errors": errors},
        "model_scope_inventories_found": len(candidates)}
    if len(candidates) == 1 and not errors:
        result.update(next(iter(candidates.values())), status="available")
        for history, bundles in (("eligible_history_count", "eligible_bundle_count"), ("selected_history_count", "selected_bundle_count")):
            if history not in result and bundles in result:
                result[history] = result[bundles]
    else:
        result["reason"] = ("invalid_declared_model_scope_inventory" if errors
                            else "missing_or_ambiguous_model_scope_inventory")
    return result
