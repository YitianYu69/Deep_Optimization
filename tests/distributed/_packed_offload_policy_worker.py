"""Real Gloo collectives with explicit metadata-only CUDA/topology fixtures."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch.distributed as dist

from deepopt import initialize_distributed, shutdown_distributed
from deepopt._packed_offload import resolve_packed_offload
from deepopt.distributed import offload


def descriptor(policy, context, *, scope_limit=None, available=1 << 40):
    constraints = [] if scope_limit is None else [dict(scope_id="memory.max:job_1", kind="cgroup",
        limit_bytes=scope_limit, current_bytes=0, available_bytes=scope_limit, remaining_observed=True)]
    return dict(rank=context.global_rank, hostname="injected-one-node", node_id="injected-one-node",
        node_id_source="explicit_CPU_Gloo_metadata_fixture", local_rank=context.global_rank,
        local_world_size=context.world_size, device=f"cuda:{context.global_rank}",
        device_index=context.global_rank, device_uuid=f"metadata-GPU-{context.global_rank}",
        device_identity=f"uuid:metadata-GPU-{context.global_rank}", backend=context.backend,
        requested_host_bytes=policy.host_pinned_bytes+policy.host_pageable_bytes,
        pinned_bytes=policy.host_pinned_bytes, pageable_bytes=policy.host_pageable_bytes,
        capacity=dict(status="available", mem_available_bytes=available, constraints=constraints, errors=[], warnings=[]))


def run(directory):
    context = initialize_distributed("cpu", backend="gloo", timeout=30)
    records = []
    base = dict(version=2, enabled=True, policy="demand", host_pinned_bytes=8 << 20,
                device_staging_bytes=4 << 20, d2h_inflight_bytes=4 << 20)
    recipe = dict(precision="bf16", bf16_policy="bf16_storage", activation_compression=dict(enabled=True, bits=2, group_size=1024),
                  compile=dict(enabled=True, backend="deepopt", options=dict(use_native_autodiff=True, require_full_aot=True)))
    scenarios = ("disabled", "enabled_disabled_mismatch", "policy_mismatch", "recipe_mismatch", "bad_recipe",
                 "enabled_CPU_rejected", "metadata_admitted", "node_oversubscribed", "scope_oversubscribed", "local_descriptor_failure")
    try:
        for name in scenarios:
            selected = dict(base)
            selected_recipe = deepcopy(recipe)
            if name == "disabled":
                selected["enabled"] = False
            elif name == "enabled_disabled_mismatch" and context.global_rank == 1:
                selected["enabled"] = False
            elif name == "policy_mismatch" and context.global_rank == 1:
                selected["host_pinned_bytes"] //= 2
            elif name == "recipe_mismatch" and context.global_rank == 1:
                selected_recipe["bf16_policy"] = "torch_amp"
            elif name == "bad_recipe" and context.global_rank == 1:
                selected_recipe["callback"] = object()
            elif name == "node_oversubscribed":
                selected["host_pinned_bytes"] = 128 << 30
            record = dict(scenario=name, rank=context.global_rank, cpu_gloo_only=True, CUDA_execution_evidence=False)
            try:
                if name in ("metadata_admitted", "node_oversubscribed", "scope_oversubscribed", "local_descriptor_failure"):
                    def collect(policy, ctx):
                        if name == "local_descriptor_failure" and context.global_rank == 1:
                            raise ValueError("injected local descriptor failure")
                        return descriptor(policy, ctx,
                            scope_limit=12 << 20 if name == "scope_oversubscribed" else None,
                            available=200 << 30 if name == "node_oversubscribed" else 1 << 40)
                    with patch.object(offload, "_local_descriptor", side_effect=collect):
                        receipt = offload.prepare_packed_offload(resolve_packed_offload(selected), context, recipe=selected_recipe)
                else:
                    receipt = offload.prepare_packed_offload(resolve_packed_offload(selected), context, recipe=selected_recipe)
                record.update(status="passed", receipt=receipt)
            except Exception as error:
                record.update(status="rejected", error_type=type(error).__name__, error=str(error))
            records.append(record)
            dist.barrier()
    finally:
        shutdown_distributed()
    Path(directory, f"rank-{context.global_rank}.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    run(sys.argv[1])
