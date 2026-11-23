"""Immutable full-pipeline identities; no unqualified timing-based defaults."""

from dataclasses import asdict, dataclass

from deepopt._bf16_policy import BF16Policy, resolve_bf16_policy


@dataclass(frozen=True)
class BF16Route:
    role: str
    backend: str
    representation: str
    policy: BF16Policy = BF16Policy()
    forced: bool = True

    def __post_init__(self):
        object.__setattr__(self, "policy", resolve_bf16_policy(self.policy))
        if self.role not in ("forward", "dx", "dw", "unspecified"):
            raise ValueError("Unknown BF16 route role")
        if self.backend not in ("native", "triton", "tilelang"):
            raise ValueError("Unknown BF16 route backend")
        if self.representation not in ("dense", "materialized", "direct", "panel"):
            raise ValueError("Unknown BF16 route representation")
        if self.representation == "panel" and self.backend != "triton":
            raise ValueError("Panel route requires Triton")
        if self.backend == "native" and self.representation == "direct":
            raise ValueError("Native library GEMM requires a dense or materialized operand")
        if self.policy.storage_enabled and self.backend == "tilelang":
            raise ValueError("bf16_storage TileLang arithmetic is not qualified")

    def as_dict(self):
        native_control = None
        if self.backend == "native":
            import torch
            native_control = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        return {**asdict(self), "policy": self.policy.as_dict(),
                "arithmetic": ("bf16_multiplicands_fp32_accumulation" if self.backend == "triton"
                               else "internal_accumulation_unknown"),
                "requested_arithmetic": "bf16_multiplicands_fp32_accumulation",
                "observed_arithmetic": "unknown",
                "arithmetic_evidence": "generated_source_contract" if self.backend == "triton" else "library_or_backend_internal_unknown",
                "native_allow_bf16_reduced_precision_reduction": native_control,
                "rounding": "bf16_product_then_ordered_epilogue_boundaries",
                "parameter_gradient_storage_is_accumulation_precision": False,
                "differentiation_fallback": False,
                "automatic_promotion": False,
                "cost_scope": "decode_layout_allocation_gemm_epilogue_complete_pipeline"}


def route_alternatives(role, *, compressed=False, policy=None):
    policy = resolve_bf16_policy(policy)
    routes = [BF16Route(role, backend, "materialized" if compressed else "dense", policy)
              for backend in ("native", "triton")]
    if compressed:
        routes += [BF16Route(role, "triton", representation, policy) for representation in ("direct", "panel")]
    return tuple(route.as_dict() for route in routes)


def cost_recommendation(candidates, *, identity, objective="runtime", capacity_bytes=None):
    """Report-only ranking with exact identity/qualification gates.

    No saved Inductor latency is accepted implicitly; a caller must supply
    accepted same-policy complete-pipeline receipts from the target run.
    """
    if objective not in ("runtime", "capacity"):
        raise ValueError("Cost objective must be runtime or capacity")
    required_identity = {"device", "device_type", "software", "shape", "strides", "dtypes", "layouts", "bf16_policy", "codec",
                         "gradient_mask", "arithmetic", "schedule", "decode_route"}
    if (not isinstance(identity, dict) or not required_identity.issubset(identity)
            or any(identity[key] is None for key in required_identity)):
        return {"status": "unqualified", "selected_route": None,
                "reason": "incomplete_cost_identity", "changes_execution": False}
    if identity["device_type"] != "cuda" or str(identity["device"]).lower().startswith(("cpu", "mock", "structural")):
        return {"status": "unqualified", "selected_route": None,
                "reason": "requires_target_cuda_execution", "changes_execution": False}
    eligible, rejected = [], []
    for row in candidates:
        reason = None
        if row.get("execution_scope") != "target_cuda" or row.get("evidence_kind") != "runtime_measurement":
            reason = "requires_measured_target_execution"
        elif row.get("identity") != identity:
            reason = "identity_mismatch"
        elif row.get("decode_route") not in ("dense", "materialized", "direct", "panel"):
            reason = "missing_actual_decode_route"
        elif row["decode_route"] != (identity["decode_route"].get(row.get("route"))
                if isinstance(identity["decode_route"], dict) else identity["decode_route"]):
            reason = "decode_route_mismatch"
        elif not all(row.get(key) is True for key in ("numerical_accepted", "launch_observed", "resource_admitted")):
            reason = "missing_accepted_target_receipt"
        elif (row.get("cost_scope") != "complete_pipeline" or type(row.get("fresh_trials")) is not int
              or row["fresh_trials"] < 3):
            reason = "incomplete_pipeline_or_trials"
        elif type(row.get("latency_ms")) not in (int, float) or not 0 < row["latency_ms"] < float("inf"):
            reason = "invalid_latency"
        elif objective == "capacity" and (type(capacity_bytes) is not int or capacity_bytes < 0 or
                type(row.get("peak_bytes")) is not int or not 0 <= row["peak_bytes"] <= capacity_bytes):
            reason = "capacity_not_admitted"
        if reason:
            rejected.append({"route": row.get("route"), "reason": reason})
        else:
            eligible.append(row)
    winner = min(eligible, key=lambda row: row["latency_ms"], default=None)
    return {"status": "recommendation_only" if winner else "unqualified", "objective": objective,
            "selected_route": None if winner is None else winner["route"], "rejected": rejected,
            "changes_execution": False, "automatic_promotion": False}
