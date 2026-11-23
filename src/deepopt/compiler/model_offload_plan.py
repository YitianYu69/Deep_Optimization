"""Immutable whole-model packed-save selection before wrapper finalization.

Cold windows are logical estimates, never reported as observed GPU peaks.
An imported physical peak profile must match the full scope identity. All
resources and invocation ownership remain in the compiled runtime manager.
"""

from dataclasses import asdict, dataclass, replace
from math import prod
from collections.abc import Mapping

from .offload_plan import PackedOffloadPlan, PackedOffloadRejection, _digest


@dataclass(frozen=True)
class ScopePeakWindow:
    region: int
    phase: str
    boundary: int
    baseline_bytes: int
    source: str = "cold_logical_estimate"

    def __post_init__(self):
        if (type(self.region) is not int or self.region < 0 or self.phase not in ("forward", "backward")
                or type(self.boundary) is not int or self.boundary < 0
                or type(self.baseline_bytes) is not int or self.baseline_bytes < 0
                or self.source not in ("cold_logical_estimate", "target_storage_ledger")):
            raise ValueError("Invalid model offload peak window")


@dataclass(frozen=True)
class ScopePeakProfile:
    scope_identity: str
    source_receipt_sha256: str
    windows: tuple[ScopePeakWindow, ...]

    def __post_init__(self):
        for value in (self.scope_identity, self.source_receipt_sha256):
            if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Peak profile identities must be SHA256 metadata")
        if (type(self.windows) is not tuple or not self.windows or
                any(type(row) is not ScopePeakWindow or row.source != "target_storage_ledger" for row in self.windows)):
            raise ValueError("Peak profiles require immutable target storage windows")

    def as_dict(self):
        return dict(schema_version=1, scope_identity=self.scope_identity, cuda_execution_verified=True,
            source_receipt_sha256=self.source_receipt_sha256, windows=tuple(asdict(row) for row in self.windows))


def resolve_peak_profile(value):
    if value is None or isinstance(value, ScopePeakProfile):
        return value
    if (not isinstance(value, Mapping) or set(value) != {
            "schema_version", "scope_identity", "cuda_execution_verified", "source_receipt_sha256", "windows"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["cuda_execution_verified"] is not True or not isinstance(value["scope_identity"], str)
            or not isinstance(value["source_receipt_sha256"], str)
            or len(value["source_receipt_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in value["source_receipt_sha256"])):
        raise ValueError("Stale or unqualified whole-model peak profile")
    try:
        windows = tuple(ScopePeakWindow(**{**row, "source": "target_storage_ledger"}) for row in value["windows"])
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid target peak profile windows") from error
    if not windows:
        raise ValueError("Target peak profile must contain measured windows")
    return ScopePeakProfile(value["scope_identity"], value["source_receipt_sha256"], windows)


def scope_identity(plans):
    """The same scope/recipe can compare demand and predictive restoration."""
    plans = tuple(plans)
    return _digest(tuple((plan.forward_signature, plan.backward_signature, plan.profile_identity,
        {key: value for key, value in plan.policy.as_dict().items() if key != "policy"}) for plan in plans))


def _alive(bundle, region, window):
    if window.phase == "forward":
        return region < window.region or region == window.region and bundle.producer_boundary <= window.boundary
    # The scope describes an ordered chain. Unknown branch orders must use
    # cold conservative windows, not an imported reverse-order assumption.
    return region < window.region or region == window.region and window.boundary <= bundle.last_backward_use


def _absent(bundle, region, window):
    if not _alive(bundle, region, window):
        return False
    if window.phase == "forward":
        return region < window.region or window.boundary > bundle.last_forward_use
    return region < window.region or window.boundary < bundle.first_backward_use


def cold_peak_windows(plans, local_peaks=()):
    """Conservative ranking inputs, not a sum claimed to be a physical peak."""
    plans = tuple(plans)
    local_peaks = tuple(local_peaks)
    windows = []
    for region, plan in enumerate(plans):
        fw = max((bundle.last_forward_use for bundle in plan.bundles), default=0) + 1
        points = (("forward", fw), *(("backward", c.schedule_index) for c in plan.consumers))
        for phase, boundary in points:
            window = ScopePeakWindow(region, phase, boundary, 0)
            history = sum(bundle.bytes for index, other in enumerate(plans) for bundle in other.bundles
                          if _alive(bundle, index, window))
            dense = (local_peaks[region].get((phase, boundary), 0) if region < len(local_peaks) else 0)
            windows.append(replace(window, baseline_bytes=history + dense))
    return tuple(windows)


def local_logical_peak_hints(schedules):
    """Dense result liveness plus declared scratch; aliases stay unqualified."""
    from .runtime.memory import MemoryManager
    from .scheduler import scheduled_nodes

    result = {}
    for phase, schedule in zip(("forward", "backward"), schedules):
        manager = MemoryManager(schedule)
        manager.collect_schedule_level_remaining_users()
        rows = manager.lifetime_plan()
        for index, item in enumerate(schedule.items):
            live = sum(row["logical_bytes"] or 0 for row in rows
                       if row["owner"] == "invocation" and row["representation"] == "dense"
                       and row["birth"] <= index <= row["last_use"] and not row["storage_source"])
            scratch = 0
            for node in scheduled_nodes(item):
                # Decode scratch is hidden inside the trusted native dW call.
                spec = getattr(getattr(node, "target", None), "specification", None)
                if getattr(node, "op", None) == "convolution.native_dw" and getattr(spec, "compressed", False):
                    scratch += prod(spec.geometry.input_shape) * spec.compute_dtype.itemsize
            result[(phase, index)] = live + scratch
    return result


def _subset(plan, ids, signature, selection_mode):
    ids = frozenset(ids)
    bundles = tuple(bundle for bundle in plan.bundles if bundle.bundle_id in ids)
    sizes = {bundle.bundle_id: bundle.bytes for bundle in bundles}
    consumers = []
    for consumer in plan.consumers:
        kept = tuple(name for name in consumer.bundle_ids if name in ids)
        if kept:
            panel = consumer.panel if kept == consumer.bundle_ids else None
            consumers.append(replace(consumer, bundle_ids=kept, bytes=sum(sizes[name] for name in kept), panel=panel))
    progress = set()
    for consumer in consumers:
        before = [boundary for boundary in plan.progress_boundaries if boundary <= consumer.schedule_index]
        if before:
            progress.add(max(before))
    if consumers and plan.progress_boundaries:
        progress.add(min(plan.progress_boundaries))
    rejected = tuple(PackedOffloadRejection(tuple(c.name for c in bundle.components),
                     "whole_model_physical_capacity_selection") for bundle in plan.bundles if bundle.bundle_id not in ids)
    selected = replace(plan, bundles=bundles, consumers=tuple(consumers),
        rejections=plan.rejections + rejected, progress_boundaries=tuple(sorted(progress)),
        prefetches=(), signature="", model_scope_signature=signature, selection_mode=selection_mode)
    from .runtime.offload_cost import plan_prefetches

    return selected.with_prefetches(plan_prefetches(selected))


@dataclass(frozen=True)
class ModelOffloadPlan:
    regions: tuple[PackedOffloadPlan, ...]
    signature: str
    identity: str
    physical_host_bytes: int
    logical_host_bytes: int
    peak_windows: tuple[ScopePeakWindow, ...]
    profile_source: str
    estimated_peak_after_selection: int
    eligible_history_bytes: int = 0
    eligible_bundle_count: int = 0
    all_eligible_host_physical_capacity_bytes: int = 0
    required_staging_bytes: int = 0
    required_d2h_inflight_bytes: int = 0
    rejected_by_reason: tuple = ()
    selection_mode: str = "peak"
    eligible_inventory_sha256: str = ""

    def as_dict(self):
        return dict(schema_version=1, signature=self.signature, identity=self.identity,
            physical_host_bytes=self.physical_host_bytes, logical_host_bytes=self.logical_host_bytes,
            regions=tuple(dict(forward_signature=plan.forward_signature, backward_signature=plan.backward_signature,
                recipe_identity_sha256=_digest(plan.profile_identity),
                selected_bundle_ids=tuple(bundle.bundle_id for bundle in plan.bundles),
                selected_bytes=plan.selected_bytes) for plan in self.regions),
            peak_windows=tuple(asdict(window) for window in self.peak_windows), profile_source=self.profile_source,
            estimated_peak_after_selection=self.estimated_peak_after_selection,
            estimated_peak_is_observed=False, performance_qualified=False,
            inventory=self.inventory())

    def inventory(self):
        return dict(schema_version=1, scope_identity=self.identity,
            eligible_history_bytes=self.eligible_history_bytes, eligible_bundle_count=self.eligible_bundle_count,
            all_eligible_host_physical_capacity_bytes=self.all_eligible_host_physical_capacity_bytes,
            all_eligible_capacity_is_feasible_upper_bound_not_minimum=True,
            eligible_inventory_sha256=self.eligible_inventory_sha256,
            required_staging_bytes=self.required_staging_bytes,
            required_d2h_inflight_bytes=self.required_d2h_inflight_bytes,
            rejected_by_reason=dict(self.rejected_by_reason), selected_history_bytes=self.logical_host_bytes,
            selected_bundle_count=sum(len(p.bundles) for p in self.regions),
            selected_physical_capacity_bytes=self.physical_host_bytes, selection_mode=self.selection_mode,
            all_eligible_selected=self.logical_host_bytes == self.eligible_history_bytes,
            candidate_metadata_only=True)


def build_model_offload_plan(plans, *, peak_profile=None, local_peaks=(), selection_mode="peak"):
    """Select atomic groups by the remaining worst-window envelope and capacity.

    Selection is a deterministic greedy minimax heuristic, not a global optimum
    claim. Physical arena placement is checked for every prospective group.
    """
    from .runtime.pinned_arena import plan_pinned_placement
    from collections import Counter

    plans = tuple(plans)
    if not plans or any(type(plan) is not PackedOffloadPlan for plan in plans):
        raise ValueError("Whole-model selection requires immutable regional plans")
    policy = plans[0].policy
    if selection_mode not in ("peak", "all_eligible", "regional_order"):
        raise ValueError("Offload selection must be peak, all_eligible or regional_order")
    if any(plan.policy != policy for plan in plans) or not policy.enabled:
        raise ValueError("Whole-model selection requires one enabled scope policy")
    identity = scope_identity(plans)
    peak_profile = resolve_peak_profile(peak_profile)
    if peak_profile is not None:
        if peak_profile.scope_identity != identity:
            raise ValueError("Stale or unqualified whole-model peak profile")
        windows = peak_profile.windows
        if any(window.region >= len(plans) for window in windows):
            raise ValueError("Peak profile region is outside the compiled scope")
        source = "target_storage_ledger"
    else:
        windows, source = cold_peak_windows(plans, local_peaks), "cold_logical_estimate"
    if not windows:
        windows = (ScopePeakWindow(0, "forward", 0, 0),)
    candidates = {(region, bundle.bundle_id): bundle for region, plan in enumerate(plans) for bundle in plan.bundles}
    inventory_hash = _digest(tuple((plans[region].forward_signature, bid,
        tuple((c.name, c.shape, c.dtype, c.stride, c.device, c.nbytes) for c in candidates[(region, bid)].components))
        for region, bid in sorted(candidates)))
    all_components = tuple(component for key in sorted(candidates) for component in candidates[key].components)
    all_layout = plan_pinned_placement(all_components, budget_bytes=1 << 63)
    eligible_bytes = sum(c.nbytes for c in all_components)
    eligible_physical = all_layout.physical_bytes if all_layout is not None else 0
    rejected = Counter(row.reason for plan in plans for row in plan.rejections)
    staging_floor = max(plan.required_staging_bytes for plan in plans)
    d2h_floor = max(plan.required_d2h_inflight_bytes for plan in plans)
    # Named panel groups are indivisible; complete triplets always are.
    units = [{key} for key in candidates]
    for region, plan in enumerate(plans):
        for consumer in plan.consumers:
            if consumer.panel is not None:
                group = {(region, name) for name in consumer.bundle_ids}
                touching = [unit for unit in units if unit & group]
                for unit in touching:
                    units.remove(unit)
                    group.update(unit)
                units.append(group)
    selected = set()
    remaining = [window.baseline_bytes for window in windows]
    physical = logical = 0
    full = (plan_pinned_placement(all_components, budget_bytes=policy.host_pinned_bytes)
            if not policy.host_pageable_bytes else None)
    full_fits = (eligible_bytes <= policy.host_pageable_bytes
                 if policy.host_pageable_bytes else full is not None)
    if full_fits:
        selected = set(candidates)
        physical = full.physical_bytes if full is not None else 0
        remaining = [max(0, window.baseline_bytes - sum(bundle.bytes for (region, _), bundle in candidates.items()
                     if _absent(bundle, region, window))) for window in windows]
        units = []
    elif selection_mode == "all_eligible":
        raise ValueError("All-eligible packed offload exceeds the declared physical host capacity; partial selection forbidden")
    while units:
        best = None
        ordered = (sorted(units, key=lambda unit: tuple(sorted(
            (region, candidates[(region, name)].producer_boundary, name) for region, name in unit)))
            if selection_mode == "regional_order" else units)
        for unit in ordered:
            prospective = selected | unit
            components = tuple(c for key in sorted(prospective) for c in candidates[key].components)
            if policy.host_pageable_bytes:
                fits = sum(c.nbytes for c in components) <= policy.host_pageable_bytes
                capacity = 0
            else:
                placement = plan_pinned_placement(components, budget_bytes=policy.host_pinned_bytes)
                fits, capacity = placement is not None, placement.physical_bytes if placement is not None else 0
            if not fits:
                continue
            savings = tuple(sum(candidates[key].bytes for key in unit
                                if _absent(candidates[key], key[0], window)) for window in windows)
            after = tuple(max(0, value - saving) for value, saving in zip(remaining, savings))
            # Equal maxima are broken by the entire descending envelope, then
            # physical capacity. No region receives a forward-order privilege.
            score = tuple(sorted(after, reverse=True)), capacity, tuple(sorted(unit))
            if best is None or score < best[0]:
                best = score, unit, after, capacity
            if selection_mode == "regional_order":
                break  # Diagnostic first-fit ordering, with the same physical pool.
        if best is None:
            break
        _, unit, after, physical = best
        selected.update(unit)
        units.remove(unit)
        remaining = list(after)
    if policy.admission == "error" and units:
        if policy.host_pageable_bytes:
            capacity = f"full history uses {eligible_bytes} pageable bytes, rank budget {policy.host_pageable_bytes} bytes"
        elif all_layout is not None:
            capacity = f"full pinned layout uses {eligible_physical} bytes, rank budget {policy.host_pinned_bytes} bytes"
        else:
            capacity = f"full pinned layout unavailable, rank budget {policy.host_pinned_bytes} bytes"
        raise ValueError(f"Whole-model physical host admission rejected selected candidate groups ({capacity})")
    logical = sum(candidates[key].bytes for key in selected)
    signature = _digest(dict(identity=identity, selected=tuple(sorted(selected)), selection_mode=selection_mode,
        windows=tuple(asdict(window) for window in windows), source=source, physical=physical))
    regions = tuple(_subset(plan, (name for index, name in selected if index == region), signature, selection_mode)
                    for region, plan in enumerate(plans))
    return ModelOffloadPlan(regions, signature, identity, physical, logical, windows, source, max(remaining),
        eligible_bytes, len(candidates), eligible_physical,
        staging_floor, d2h_floor, tuple(sorted(rejected.items())), selection_mode, inventory_hash)


def finalize_model_offload_regions(regions):
    """Cold compiler finalization, before any generated outer forward runs."""
    regions = tuple(region for region in regions if getattr(region, "_offload_candidates", None) is not None)
    if not regions:
        return None
    if any((region.context.packed_offload_selection_mode != regions[0].context.packed_offload_selection_mode
            or region.context.packed_offload_peak_profile != regions[0].context.packed_offload_peak_profile)
           for region in regions):
        raise ValueError("All native regions must share the same selection mode and frozen peak profile")
    if any(region._offload_finalized for region in regions):
        identity = scope_identity(tuple(region._offload_candidates for region in regions))
        if all(region._offload_finalized and region.model_offload_plan["identity"] == identity for region in regions):
            return None  # Re-emitting the same outer wrapper cannot mutate live plans.
        raise RuntimeError("Cannot change the scope of an already finalized native offload region")
    plan = build_model_offload_plan(tuple(region._offload_candidates for region in regions),
        local_peaks=tuple(local_logical_peak_hints(region._schedules) for region in regions),
        selection_mode=regions[0].context.packed_offload_selection_mode,
        peak_profile=regions[0].context.packed_offload_peak_profile)
    from .runtime.manager import CompiledGraphRuntimeManager

    manager = CompiledGraphRuntimeManager(regions[0].context.packed_activation_offload) if plan.logical_host_bytes else None
    for index, (region, selected) in enumerate(zip(regions, plan.regions)):
        region.finalize_offload(selected, manager, model_plan=plan.as_dict())
        region.set_native_scope_metadata(index, plan.identity)
    return plan
