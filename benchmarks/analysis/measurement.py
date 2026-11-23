"""Synchronized timing, separate profiling, numerical and memory diagnostics."""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from hashlib import sha256
import math
from statistics import median, pstdev
from time import perf_counter
from copy import deepcopy


def stats(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    return {"samples": samples, "median": median(samples), "min": ordered[0], "max": ordered[-1],
            "p95": ordered[max(0, math.ceil(len(ordered) * .95) - 1)],
            "stdev": pstdev(samples), "relative_stdev": pstdev(samples) / max(median(samples), 1e-30)}


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(call, torch, device, warmup, iterations, repeats, *, sample_callback=None):
    from analysis.telemetry import quiet_timing_window

    with quiet_timing_window():
        return _measure(call, torch, device, warmup, iterations, repeats, sample_callback=sample_callback)


def _measure(call, torch, device, warmup, iterations, repeats, *, sample_callback=None):
    for _ in range(warmup):
        call()
    synchronize(torch, device)
    wall, stream, submit = [], [], []
    end_record, sync_wait, query = [], [], []
    events = None
    if device.type == "cuda":
        events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
        # Allocate/initialize timing handles outside all measured samples.
        for event in events:
            event.record()
        events[-1].synchronize()
    for index in range(repeats):
        synchronize(torch, device)
        if device.type == "cuda":
            first, last = events
            first.record()
        began = perf_counter()
        for _ in range(iterations):
            call()
        submitted = perf_counter()
        if device.type == "cuda":
            last.record()
            recorded = perf_counter()
            last.synchronize()
            returned = perf_counter()
            stream.append(first.elapsed_time(last) / iterations)
        finished = perf_counter()
        wall.append((finished - began) * 1000 / iterations)
        submit.append((submitted - began) * 1000 / iterations)
        if device.type == "cuda":
            end_record.append((recorded - submitted) * 1000 / iterations)
            sync_wait.append((returned - recorded) * 1000 / iterations)
            query.append((finished - returned) * 1000 / iterations)
        if sample_callback is not None:
            sample_callback(index, {"wall_ms": wall[-1], "host_submission_ms": submit[-1],
                "cuda_stream_ms": stream[-1] if stream else None,
                "host_end_event_record_ms": end_record[-1] if end_record else None,
                "host_event_synchronize_ms": sync_wait[-1] if sync_wait else None,
                "host_event_elapsed_query_ms": query[-1] if query else None})
    return {"wall_ms": stats(wall), "cuda_stream_ms": stats(stream), "host_submission_ms": stats(submit),
            "host_end_event_record_ms": stats(end_record), "host_event_synchronize_ms": stats(sync_wait),
            "host_event_elapsed_query_ms": stats(query),
            "wall_minus_cuda_stream_ms": stats([w - s for w, s in zip(wall, stream)]),
            "note": "CUDA event intervals include submission gaps; host submission may include implicit waits. These are not additive. "
                    "Host synchronization includes outstanding GPU work and host return delays; wall-minus-event is a boundary difference, not pure CPU overhead."}


def tensor_leaves(value):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensor_leaves(child)


def dense(value):
    from deepopt.opt.quantization.activation_tensor import dequantize_activation

    return dequantize_activation(value)


def state_fingerprint(model, inputs, kwargs):
    """Hash logical initial values, including integer buffers and input strides separately in metadata."""
    digest = sha256()
    for name, value in list(model.state_dict().items()) + list(enumerate(tensor_leaves((inputs, kwargs)))):
        digest.update(str(name).encode())
        if not hasattr(value, "shape"):
            import json
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
            continue
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        raw = dense(value).detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8)
        digest.update(memoryview(raw.numpy()))
    return digest.hexdigest()


def cpu_state(state):
    import torch

    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {key: cpu_state(value) for key, value in state.items()}
    if isinstance(state, list):
        return [cpu_state(value) for value in state]
    if isinstance(state, tuple):
        return tuple(cpu_state(value) for value in state)
    return deepcopy(state)


@contextmanager
def memory_fill_scope(enabled):
    """Change allocation filling for performance diagnostics, restoring it on error."""
    import torch

    previous = torch.utils.deterministic.fill_uninitialized_memory
    if enabled is not None:
        torch.utils.deterministic.fill_uninitialized_memory = enabled
    try:
        yield torch.utils.deterministic.fill_uninitialized_memory
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


def sampled_indices(numel, count, device):
    """One deterministic sample per bin, avoiding fixed-stride feature aliasing."""
    import torch

    bins = torch.arange(count, device=device, dtype=torch.int64)
    first = bins * numel // count
    last = (bins + 1) * numel // count
    hashed = bins + 0x9E3779B9
    for _ in range(2):
        hashed = ((hashed ^ (hashed >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    hashed = hashed ^ (hashed >> 16)
    selected = first + hashed % (last - first)
    selected[0], selected[-1] = 0, numel - 1
    return selected


def snapshot(session):
    import torch
    from deepopt.opt.quantization.activation_tensor import FP8Activation

    session.reset()
    session.clear()
    output, loss = session.forward()
    probe = session.runtime.get("numerical_probe", "loss")
    if probe == "random_upstream":
        generator = torch.Generator(device=session.device).manual_seed(session.seed + 104729)
        values = [dense(value).float() for value in tensor_leaves(output) if value.is_floating_point()]
        loss = sum((value * torch.randn(value.shape, device=value.device, dtype=torch.float32,
                                       generator=generator)).sum() / max(1, value.numel()) ** .5 for value in values)
    session.backward(loss)
    metadata = {"original_shapes": {}, "representations": {}, "full_tensor_finite": True,
                "max_sampled_elements_per_float_tensor": session.max_numerical_elements,
                "sampling": "deterministic stratified bins with hashed within-bin offsets and endpoint coverage",
                "numerical_probe": probe, "throughput_loss_unchanged": True,
                "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
                "optimizer_warmup_steps": session.runtime.get("optimizer_warmup_steps", 0)}

    def capture(value, name):
        if value is None:
            return None
        metadata["representations"][name] = {
            "type": type(value).__name__, "logical_dtype": str(value.dtype),
            "storage_dtype": str(value._data.dtype if isinstance(value, FP8Activation) else value.dtype)}
        value = dense(value).detach()
        metadata["original_shapes"][name] = list(value.shape)
        if value.is_floating_point():
            metadata["full_tensor_finite"] &= bool(torch.isfinite(value).all())
            if session.max_numerical_elements and value.numel() > session.max_numerical_elements:
                indices = sampled_indices(value.numel(), session.max_numerical_elements, value.device)
                value = value.reshape(-1)[indices]
            value = value.float()
        return value.cpu().clone()

    values = {f"output.{index}": capture(value, f"output.{index}")
              for index, value in enumerate(tensor_leaves(output))}
    values["loss"] = capture(loss, "loss")
    for index, value in enumerate(tensor_leaves(session.inputs)):
        if value.requires_grad:
            values[f"input_grad.{index}"] = capture(value.grad, f"input_grad.{index}")
    for name, parameter in session.model.named_parameters():
        if parameter.requires_grad:
            values[f"parameter_grad.{name}"] = capture(parameter.grad, f"parameter_grad.{name}")
    if session.optimizer is not None:
        before = {name: capture(p, f"parameter_update.{name}") for name, p in session.model.named_parameters()}
        session.optimizer.step()
        for name, parameter in session.model.named_parameters():
            values[f"parameter_update.{name}"] = capture(parameter, f"parameter_update.{name}") - before[name]
    if session.after_step is not None:
        session.after_step()
    session.snapshot_metadata = metadata
    session.reset()
    return values


def errors(actual, expected, rtol, atol, *, max_relative_l2=None, min_cosine=None,
           gradient_policy=None, update_policy=None, chunk_elements=65536):
    import torch

    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("Numerical comparison chunk size must be a positive integer")

    def chunk(value, start, stop):
        if value.is_contiguous():
            return value.view(-1)[start:stop]
        # Avoid flattening an entire strided tensor into a dense temporary.
        linear = torch.arange(start, stop, device=value.device)
        indices = []
        for size in reversed(value.shape):
            indices.append(linear % size)
            linear = linear // size
        return value[tuple(reversed(indices))]

    result, passed, finite = {}, True, True
    for key in sorted(set(actual) | set(expected)):
        policy = {"rtol": rtol, "atol": atol, "max_relative_l2": max_relative_l2, "min_cosine": min_cosine}
        selected = gradient_policy if key.startswith(("input_grad.", "parameter_grad.")) else (
            update_policy if key.startswith("parameter_update.") else None)
        if selected:
            policy.update(selected)
        x, y = actual.get(key), expected.get(key)
        if x is None or y is None:
            same = x is None and y is None and key in actual and key in expected
            result[key] = {"passed": same, "missing_or_unused": True}
            passed &= same
            continue
        if x.shape != y.shape:
            result[key] = {"passed": False, "shape_mismatch": [list(x.shape), list(y.shape)]}
            passed = False
            continue
        if x.dtype != y.dtype:
            result[key] = {"passed": False, "dtype_mismatch": [str(x.dtype), str(y.dtype)]}
            passed = False
            continue
        exact = not x.is_floating_point() and not y.is_floating_point()
        count = x.numel()
        is_finite, close, norm_x, norm_y, norm_delta, dot, max_abs, signs = True, True, 0., 0., 0., 0., 0., 0
        for start in range(0, count, chunk_elements):
            xc = chunk(x, start, min(count, start + chunk_elements))
            yc = chunk(y, start, min(count, start + chunk_elements))
            if exact:
                close &= bool(torch.equal(xc, yc))
            xc, yc = xc.double(), yc.double()
            if not bool(torch.isfinite(xc).all() and torch.isfinite(yc).all()):
                is_finite = False
                break
            delta = xc - yc
            norm_x, norm_y = math.hypot(norm_x, xc.norm().item()), math.hypot(norm_y, yc.norm().item())
            norm_delta = math.hypot(norm_delta, delta.norm().item())
            dot += (xc @ yc).item()
            max_abs = max(max_abs, delta.abs().max().item())
            signs += ((xc * yc) < 0).sum().item()
            if not exact:
                close &= bool(torch.allclose(xc, yc, rtol=policy["rtol"], atol=policy["atol"]))
        finite &= is_finite
        if not is_finite:
            result[key] = {"passed": False, "finite": False}
            passed = False
            continue
        cosine = (dot / (norm_x * norm_y)) if norm_x and norm_y else (1.0 if not norm_x and not norm_y else 0.0)
        cosine = max(-1.0, min(1.0, cosine))
        relative = norm_delta / max(norm_y, 1e-30)
        parallel = dot / (norm_y * norm_y) if norm_y else None
        orthogonal = math.sqrt(max(0., norm_x * norm_x - (dot / norm_y) ** 2)) / norm_y if norm_y else None
        accepted = close and (exact or (
            (policy["max_relative_l2"] is None or relative <= policy["max_relative_l2"])
            and (policy["min_cosine"] is None or cosine >= policy["min_cosine"])))
        rounding = None
        ulps = policy.get("parameter_rounding_ulps")
        if ulps is not None and key.startswith("parameter_update."):
            if type(ulps) is not int or not 1 <= ulps <= 2:
                raise ValueError("Update rounding budget must be one or two FP32 ULPs")
            parameter_key = "parameter." + key.removeprefix("parameter_update.")
            px, py = actual.get(parameter_key), expected.get(parameter_key)
            if (px is not None and py is not None and px.dtype == py.dtype == torch.float32
                    and px.shape == py.shape == x.shape):
                absolute_l2, absolute_max, parameter_close, rounding_close = 0., 0., True, True
                for start in range(0, count, chunk_elements):
                    a = chunk(px, start, min(count, start + chunk_elements))
                    b = chunk(py, start, min(count, start + chunk_elements))
                    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                        parameter_close = False
                        continue
                    parameter_close &= bool(torch.allclose(a, b, rtol=rtol, atol=atol))
                    scale = torch.maximum(a.abs(), b.abs())
                    spacing = torch.nextafter(scale, torch.full_like(scale, float("inf"))) - scale
                    spacing = torch.where(torch.isfinite(spacing), spacing,
                                          scale - torch.nextafter(scale, torch.zeros_like(scale))).double()
                    bound = spacing * ulps
                    delta_chunk = (chunk(x, start, min(count, start + chunk_elements)).double()
                                   - chunk(y, start, min(count, start + chunk_elements)).double()).abs()
                    rounding_close &= bool((delta_chunk <= bound).all())
                    absolute_l2 = math.hypot(absolute_l2, bound.norm().item())
                    absolute_max = max(absolute_max, bound.max().item())
                floor = absolute_l2 / max(policy.get("max_relative_l2") or 1e-3, 1e-30)
                near_zero = max(norm_x, norm_y) <= floor
                rounding = {"parameter_rounding_ulps": ulps, "absolute_l2_budget": absolute_l2,
                            "absolute_max_budget": absolute_max, "direction_norm_floor": floor,
                            "post_parameters_close": parameter_close, "per_element_rounding_passed": rounding_close,
                            "near_zero": near_zero}
                if near_zero:
                    accepted = parameter_close and rounding_close and norm_delta <= absolute_l2 and max_abs <= absolute_max
        result[key] = {"passed": accepted, "allclose_passed": close, "finite": True, "max_abs": max_abs,
                       "relative_l2": relative, "cosine": cosine,
                       "reference_l2": norm_y, "actual_l2": norm_x,
                       "reference_rms": norm_y / math.sqrt(max(1, count)),
                       "absolute_l2": norm_delta,
                       "direction_defined": bool(norm_x and norm_y),
                       "angle_degrees": math.degrees(math.acos(cosine)) if norm_x and norm_y else None,
                       "parallel_scale": parallel, "orthogonal_relative_l2": orthogonal,
                       "sign_mismatch_fraction": signs / count if count else 0.,
                       "exact": exact}
        if rounding is not None:
            result[key]["update_rounding"] = rounding
            if rounding["near_zero"]:
                result[key].update(direction_defined=False, angle_degrees=None,
                                   raw_cosine=cosine, cosine=None)
        passed &= accepted
    components = {}
    for key in actual.keys() & expected.keys():
        x, y = actual[key], expected[key]
        if (key.startswith(("parameter_grad.", "parameter_update."))
                and key.endswith(("in_proj_bias", "qkv.bias"))
                and x is not None and y is not None and x.ndim == y.ndim == 1
                and x.shape == y.shape and x.numel() % 3 == 0):
            a = {f"{key}.{name}": value for name, value in zip(("Q", "K", "V"), x.chunk(3))}
            b = {f"{key}.{name}": value for name, value in zip(("Q", "K", "V"), y.chunk(3))}
            components[key] = errors(a, b, rtol, atol, max_relative_l2=max_relative_l2, min_cosine=min_cosine,
                                     gradient_policy=gradient_policy, update_policy=update_policy,
                                     chunk_elements=chunk_elements)["tensors"]
    return {"passed": passed, "finite": finite, "rtol": rtol, "atol": atol,
            "max_relative_l2": max_relative_l2, "min_cosine": min_cosine,
            "gradient_policy": gradient_policy, "update_policy": update_policy, "tensors": result,
            "qkv_bias_components": components, "component_diagnostics_change_acceptance": False,
            "comparison_chunk_elements": chunk_elements}


def latency_diagnostics(session, count):
    """Serialized per-update diagnostic with GC events; never throughput timing."""
    import gc
    import resource
    import sys

    session.reset()
    graph_mode = session.runtime.get("execution") == "graph"
    if graph_mode and session.graph is None:
        # Capture/setup must not masquerade as a slow steady-state update.
        session.graph_step()
        synchronize(session.torch, session.device)
        session.reset()
    call = session.graph_step if graph_mode else session.step
    replay_before = getattr(session, "graph_replay_count", 0)
    events, samples, current = [], [], {"step": -1}
    began = perf_counter()
    def callback(phase, info):
        events.append({"step": current["step"], "phase": phase,
                       "elapsed_ms": (perf_counter() - began) * 1000, **info})
    gc.callbacks.append(callback)
    def update():
        current["step"] += 1
        call()
    def collect(index, timing):
        item = {"step": index, **timing, "gc_counts": list(gc.get_count()),
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
        if session.device.type == "cuda":
            item.update(allocated_bytes=session.torch.cuda.memory_allocated(session.device),
                        reserved_bytes=session.torch.cuda.memory_reserved(session.device))
        samples.append(item)
    try:
        measure(update, session.torch, session.device, 0, 1, count, sample_callback=collect)
    finally:
        gc.callbacks.remove(callback)
        session.reset()
    return {"updates": samples, "gc_events": events, "execution": "graph" if graph_mode else "eager",
            "diagnostic_graph_replays": getattr(session, "graph_replay_count", 0) - replay_before,
            "note": "Each update synchronizes and collects telemetry; compare with throughput only as a diagnostic."}


def quality_diagnostics(session, count, interval):
    """Matched training curve; validation semantics/data belong to the factory."""
    session.reset()
    curve, validation = [], []
    def validate(step):
        if session.validate is None:
            return
        training = session.model.training
        try:
            session.model.eval()
            with session.torch.no_grad(), session.context():
                metrics = session.validate(session.model)
            if not isinstance(metrics, dict):
                raise ValueError("validate(model) must return a dict of finite scalar metrics")
            normalized = {key: float(value) for key, value in metrics.items()}
            if not normalized or any(not isinstance(key, str) or not math.isfinite(value) for key, value in normalized.items()):
                raise ValueError("Validation metrics must have string names and finite scalar values")
            validation.append({"step": step, "metrics": normalized})
        finally:
            session.model.train(training)
    try:
        validate(0)
        for index in range(1, count + 1):
            _, loss = session.step(return_loss=True)
            value = float(loss)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite training loss at quality step {index}")
            curve.append({"step": index, "training_loss": value})
            if index % interval == 0 or index == count:
                validate(index)
    finally:
        session.reset()
    return {"execution": "eager diagnostic, outside throughput timing", "seed": session.seed,
            "curve": curve, "validation": validation,
            "validation_status": "factory_callback" if session.validate is not None else "not_provided",
            "note": "Training loss on synthetic/resident data does not establish task quality; compare real validation metrics over multiple seeds."}


class SavedStorageLifetimes:
    """Saved-hook ownership backed by the invocation's weak storage ledger."""

    class Token:
        __slots__ = ("value", "key", "owner")

        def __init__(self, value, key, owner):
            self.value, self.key, self.owner = value.detach(), key, owner

        def __del__(self):
            self.owner.release(self.key)

    def __init__(self, persistent=(), *, ledger=None):
        from deepopt.memory.ledger import StorageLedger

        self.ledger = ledger if ledger is not None else StorageLedger()
        self.ledger.observe(persistent, owner="model_parameter_or_buffer")

    def pack(self, value):
        return self.Token(value, self.ledger.save(value), self)

    @staticmethod
    def unpack(token):
        token.owner.ledger.unpack(token.key)
        return token.value

    def release(self, key):
        self.ledger.release_saved(key)

    def snapshot(self):
        return self.ledger.saved_snapshot()


def memory_diagnostics(session):
    import torch
    from deepopt.memory.ledger import StorageLedger, StorageObserver

    session.reset()
    synchronize(torch, session.device)
    cuda = session.device.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
    # Populate optimizer state and reusable buffers before measuring a steady update.
    for _ in range(3):
        session.step()
    session.last_output = None
    session.clear()
    synchronize(torch, session.device)
    baseline = torch.cuda.memory_allocated(session.device) if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats(session.device)
    rows = []
    ledger = StorageLedger(max_records=session.runtime.get("memory_ledger_max_records", 50000))
    ledger.observe(tuple(session.model.parameters()), owner="model_parameter")
    ledger.observe(tuple(session.model.buffers()), owner="model_buffer")
    if session.optimizer is not None:
        ledger.observe(tuple(tensor_leaves(session.optimizer.state)), owner="optimizer_state")
    ledger.observe((session.inputs, getattr(session, "kwargs", {})), owner="caller_input")
    lifetimes = SavedStorageLifetimes(ledger=ledger)
    after_forward_saved = []

    def save(value):
        token = lifetimes.pack(value)
        from deepopt.memory.ledger import tensor_components

        components = list(tensor_components(value))
        rows.append({"shape": list(value.shape), "dtype": str(value.dtype), "strides": list(value.stride()),
                     "storage_offset": value.storage_offset(),
                     "logical_bytes": value.numel() * value.element_size(),
                     "storage_ids": list(token.key),
                     "component_bytes": [child.untyped_storage().nbytes() for _, child in components],
                     "storage_bytes": sum({(str(child.device), child.untyped_storage()._cdata):
                         child.untyped_storage().nbytes() for _, child in components}.values())})
        return token

    forward_allocated = None
    # A non-infrastructure TorchDispatchMode changes Dynamo guards and blocks
    # fullgraph capture, even after warmup. Keep saved hooks, boundary owners
    # and CUDA allocator peaks without observing every dispatched intermediate.
    dispatch_enabled = session.runtime.get("execution") != "compile"
    observer = StorageObserver(ledger) if dispatch_enabled else nullcontext()
    with observer, torch.autograd.graph.saved_tensors_hooks(save, lifetimes.unpack):
        accumulation = session.runtime["accumulation_steps"]
        for index in range(accumulation):
            with ledger.scope("forward_and_loss"):
                output, loss = session.forward()
                ledger.observe((output, loss), owner="escaped_output")
            after_forward_saved.append(lifetimes.snapshot())
            if index == 0:
                synchronize(torch, session.device)
                forward_allocated = torch.cuda.memory_allocated(session.device) if cuda else None
            with ledger.scope("backward"):
                session.backward(loss / accumulation)
                ledger.observe(tuple(parameter.grad for parameter in session.model.parameters()), owner="leaf_gradient")
        if session.optimizer is not None:
            with ledger.scope("optimizer"):
                session.optimizer.step()
                ledger.observe(tuple(tensor_leaves(session.optimizer.state)), owner="optimizer_state")
        if session.after_step is not None:
            session.after_step()
    synchronize(torch, session.device)
    allocator_stats = torch.cuda.memory_stats(session.device) if cuda else {}
    result = {"observed_saved_logical_bytes": sum(row["logical_bytes"] for row in rows),
              "dispatch_observation_enabled": dispatch_enabled,
              "all_intermediate_storage_observed": dispatch_enabled,
              "dispatch_observation_reason": ("Python dispatch observation enabled; opaque native workspace remains unknown"
                  if dispatch_enabled else "Disabled for fullgraph compilation; saved hooks, boundary owners and allocator peaks remain measured"),
              "observed_saved_unique_storage_bytes": lifetimes.snapshot()["cumulative_unique_storage_bytes"],
              "saved_tensors": rows, "invocation_storage_ledger": ledger.snapshot(),
              "saved_storage_lifetimes": {**lifetimes.snapshot(), "after_forward": after_forward_saved,
                  "note": "Saved-object lifetime, including repeated unpack; nonpersistent totals exclude known parameters/buffers/optimizer storage. Private caches and inner hooks can limit classification/visibility."},
              "baseline_allocated_bytes": baseline, "after_forward_allocated_bytes": forward_allocated,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(session.device) if cuda else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(session.device) if cuda else None,
              "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(session.device) - baseline if cuda else None,
              "optimizer_state_warmed": session.optimizer is not None,
              "warmup_updates": 3,
              "allocator_stats_after_diagnostic": {key: allocator_stats.get(key) for key in (
                  "num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free",
                  "inactive_split_bytes.all.current", "inactive_split_bytes.all.peak",
                  "allocated_bytes.all.current", "reserved_bytes.all.current")} if cuda else None,
              "coverage": "Outer saved-tensor hooks cannot observe payloads owned by nested hooks (attention/checkpointing). "
                          "Storage includes referenced parameter/input storage; this is not incremental activation memory. "
                          "Saved-hook bytes sum calls across accumulated microbatches, not simultaneous retention. "
                          "Allocator peaks include warmed optimizer state, gradients, codec scratch, and workspaces. CPU allocator peaks are unavailable."}
    session.reset()
    return result


def phase_timings(session, count):
    """Separate diagnostic pass; no phase synchronization until the complete step ends."""
    import torch

    session.reset()
    result = defaultdict(lambda: {"host_submission_ms": [], "cuda_stream_ms": []})
    for _ in range(count):
        session.clear()
        marks, host = [], []
        synchronize(torch, session.device)
        for phase in range(3):
            if session.device.type == "cuda":
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                marks.append(event)
            host.append(perf_counter())
            if phase == 0:
                _, loss = session.forward()
            elif phase == 1:
                session.backward(loss)
        synchronize(torch, session.device)
        for index, name in enumerate(("forward_and_loss", "backward")):
            result[name]["host_submission_ms"].append((host[index + 1] - host[index]) * 1000)
            if marks:
                result[name]["cuda_stream_ms"].append(marks[index].elapsed_time(marks[index + 1]))
    session.reset()
    return {key: {name: stats(samples) for name, samples in value.items()} for key, value in result.items()}


def classify_kernel(name):
    name = name.lower()
    stem = name.split("<", 1)[0]
    if name == "_bf16_prepare":
        return "pack_or_fused_cast_pack"
    if "fillfunctor" in name:
        return "fill"
    if any(word in name for word in ("sdpa", "flash", "fmha")):
        return "attention"
    if (any(word in name for word in ("layer_norm", "batch_norm", "group_norm", "normalization"))
            or name in ("_apply", "_partial_stats", "_merge_stats", "_finish_stats", "_partial_backward",
                        "_partial_affine", "_finish_affine", "_row_forward", "_tiled_backward", "_input_gradient")):
        return "normalization"
    if name in ("_channels_last_stats", "_channels_last_apply", "_channels_last_sums", "_channels_last_dx"):
        return "normalization"
    if any(word in name for word in ("convolution", "depthwise", "cudnn", "conv2d", "conv3d")):
        return "convolution"
    if "pool" in stem:
        return "pooling"
    if any(word in name for word in ("unpack", "decode", "dequant")):
        return "decode_or_dequantize"
    # Template argument types such as GenericPackedTensorAccessor are not codecs.
    if "pack" in stem:
        return "pack_or_fused_cast_pack"
    if any(word in name for word in ("amax", "min_max", "scale_update")):
        return "scaling_or_range_reduction"
    if "weight_gradient" in name:
        return "weight_gradient"
    if "input_gradient" in name:
        return "input_gradient"
    if name.startswith(("_activation_", "_channels_last_activation")) or any(word in name for word in ("gelu", "silu", "relu")):
        return "activation"
    if any(word in name for word in ("gemm", "matmul", "mma", "xmma", "nvjet")):
        return "gemm"
    if any(word in name for word in ("quant", "cast")):
        return "cast_or_quantize"
    if any(word in name for word in ("transpose", "copy", "contiguous")):
        return "copy_or_layout"
    return "other"


@contextmanager
def module_annotations(model):
    """Only used in the profiler pass; never changes throughput timing."""
    import torch

    handles, stacks = [], defaultdict(list)
    for name, module in model.named_modules():
        label = f"module::{name or '<root>'}::{type(module).__name__}"

        def before(current, args, label=label):
            context = torch.profiler.record_function(label)
            context.__enter__()
            stacks[id(current)].append(context)

        def after(current, args, output):
            if stacks[id(current)]:
                stacks[id(current)].pop().__exit__(None, None, None)

        handles.extend((module.register_forward_pre_hook(before),
                        module.register_forward_hook(after, always_call=True)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _chrome_compiler_launch_evidence(trace, steps):
    """Join same-thread host launches to CUDA kernels by CUPTI correlation.

    Kineto's FunctionEvent tree can attach Triton driver launches to an outer
    autograd operator, losing the nested record_function ancestry. Chrome
    retains the host API's thread and correlation even in that case. GPU
    timestamps alone never establish candidate ownership here.
    """
    if not isinstance(trace, dict):
        import gzip
        import json
        from pathlib import Path

        path = Path(trace)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            trace = json.load(stream)
    events = trace.get("traceEvents", ())
    ranges, launches, kernels, correlation_processes = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(set)
    result = {}

    def interval(event):
        start, duration = event.get("ts"), event.get("dur")
        if (isinstance(start, bool) or isinstance(duration, bool)
                or not isinstance(start, (int, float)) or not isinstance(duration, (int, float))
                or not math.isfinite(start) or not math.isfinite(duration) or duration < 0):
            return None
        return start, start + duration

    for index, event in enumerate(events):
        if event.get("ph") != "X" or interval(event) is None:
            continue
        category, name = event.get("cat"), event.get("name", "")
        args = event.get("args", {})
        correlation = args.get("correlation")
        correlated = isinstance(correlation, int) and not isinstance(correlation, bool) and correlation >= 0
        if event.get("pid") is None or event.get("tid") is None:
            continue
        if category == "user_annotation" and name.startswith(("deepopt.compiler.", "deepopt.fusion::", "deepopt.pack::")):
            item = result.setdefault(name, {"range_calls": 0, "kernels": {}, "range_event_ids": set(), "correlation_ids": set()})
            item["range_calls"] += 1
            event_id = args.get("External id")
            if isinstance(event_id, int) and not isinstance(event_id, bool):
                item["range_event_ids"].add(event_id)
            ranges[event.get("pid"), event.get("tid")].append((*interval(event), name))
        elif category in ("cuda_runtime", "cuda_driver") and correlated:
            launches[event.get("pid"), event.get("tid")].append((*interval(event), correlation))
            correlation_processes[correlation].add(event.get("pid"))
        elif category == "kernel" and correlated:
            kernels[correlation].append((index, name))
    for thread, thread_ranges in ranges.items():
        pending = iter(sorted(thread_ranges))
        upcoming = next(pending, None)
        active = []
        for start, end, correlation in sorted(launches.get(thread, ())):
            while upcoming is not None and upcoming[0] <= start:
                active.append(upcoming)
                upcoming = next(pending, None)
            active = [row for row in active if row[1] >= start]
            # A merged multi-process trace can reuse correlation IDs. Do not
            # guess which process owns the device activity in that case.
            if len(correlation_processes[correlation]) != 1 or not kernels.get(correlation):
                continue
            for _, range_end, name in active:
                if end > range_end:
                    continue
                result[name]["correlation_ids"].add(correlation)
                result[name]["kernels"].update(kernels[correlation])
    formatted = {}
    for name, item in result.items():
        counts = defaultdict(int)
        for symbol in item["kernels"].values():
            counts[symbol] += 1
        formatted[name] = {"range_calls_per_step": item["range_calls"] / steps,
                           "linked_cuda_launches_per_step": len(item["kernels"]) / steps if counts else None,
                           "kernel_names": sorted(counts),
                           "kernel_launches_per_step": {symbol: count / steps for symbol, count in sorted(counts.items())},
                           "range_event_ids": sorted(item["range_event_ids"]),
                           "correlation_ids": sorted(item["correlation_ids"]),
                           "attribution": "chrome_same_thread_host_launch_correlation"}
    return formatted


def compiler_launch_evidence(events, steps, *, trace=None):
    """Attribute CUDA kernels through CPU ancestry or correlated host launches."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Profile steps must be a positive integer")
    result = {}
    for event in events:
        if not event.name.startswith(("deepopt.compiler.", "deepopt.fusion::", "deepopt.pack::")):
            continue
        item = result.setdefault(event.name, {"range_calls": 0, "linked_cuda_launches": 0, "kernel_names": set(),
                                             "kernel_launches": defaultdict(int), "range_event_ids": set(), "correlation_ids": set()})
        item["range_calls"] += 1
        event_id = getattr(event, "id", None)
        if isinstance(event_id, int):
            item["range_event_ids"].add(event_id)
        pending, seen = [event], set()
        while pending:
            current = pending.pop()
            for key in ("correlation_id", "linked_correlation_id"):
                correlation = getattr(current, key, None)
                if isinstance(correlation, int):
                    item["correlation_ids"].add(correlation)
            for kernel in getattr(current, "kernels", ()):
                if id(kernel) not in seen:
                    seen.add(id(kernel))
                    item["linked_cuda_launches"] += 1
                    item["kernel_names"].add(kernel.name)
                    item["kernel_launches"][kernel.name] += 1
            pending.extend(getattr(current, "cpu_children", ()))
    result = {name: {"range_calls_per_step": item["range_calls"] / steps,
                   "linked_cuda_launches_per_step": item["linked_cuda_launches"] / steps if item["kernel_names"] else None,
                   "kernel_names": sorted(item["kernel_names"]),
                   "kernel_launches_per_step": {symbol: count / steps for symbol, count in sorted(item["kernel_launches"].items())},
                   "range_event_ids": sorted(item["range_event_ids"]),
                   "correlation_ids": sorted(item["correlation_ids"])} for name, item in result.items()}
    if trace is not None:
        # Use one attribution source per range, never add duplicate launches
        # observed in both the FunctionEvent tree and the Chrome trace.
        result.update(_chrome_compiler_launch_evidence(trace, steps))
    return result


def fusion_candidate_evidence(report, profile):
    """Join selected implementations to their own CUDA range and kernel roles.

    A compiler range with an unrelated launch is insufficient. Every declared
    role must have a matching symbol in the candidate's exact range. Symbols
    within a role are alternatives; all roles are required. Missing attribution
    or an incomplete implementation contract remains unknown. Optional role
    launch limits apply per invocation of the candidate range, so repeated
    forwards in one measured update do not masquerade as duplicate encodings.
    """
    decisions = report.get("fusion_candidates", ())
    if not decisions:
        decisions = [{"graph_index": index, **candidate}
                     for index, graph in enumerate(report.get("graphs", ()))
                     for candidate in graph.get("fusion_candidates", ())]
    regions = (profile or {}).get("compiler_regions", {})
    result, selected_ids = [], set()
    for candidate in decisions:
        selected = candidate.get("selected") is True
        item = {key: candidate.get(key) for key in (
            "candidate_id", "profile_range", "phase", "execution_phase", "implementation_fingerprint",
            "pattern", "module_path", "node_name")}
        item.update(selected=selected, observation="not_selected", reason="not_selected",
                    observed_cuda_launches=None, kernel_names=[], matched_kernel_roles={},
                    missing_kernel_roles=[], forbidden_kernel_symbols=[], remaining_profile_ranges=[],
                    range_event_ids=[], correlation_ids=[], unattributed_profile_ranges=[],
                    matched_kernel_role_launches={}, matched_kernel_role_launches_per_call={}, excess_kernel_roles=[])
        result.append(item)
        if not selected:
            continue
        item.update(observation="unknown", reason="incomplete_candidate_contract")
        contract = candidate.get("kernel_evidence") or {}
        roles = contract.get("required_kernel_roles")
        forbidden = contract.get("forbidden_kernel_symbols", [])
        removed_ranges = contract.get("removed_profile_ranges", [])
        role_limits = contract.get("maximum_kernel_role_launches", {})
        valid_patterns = lambda values: isinstance(values, (list, tuple)) and all(
            isinstance(value, str) and bool(value) for value in values)
        if (not all(isinstance(item[key], str) and item[key] for key in (
                "candidate_id", "profile_range", "implementation_fingerprint", "execution_phase"))
                or not isinstance(roles, dict) or not roles
                or any(not isinstance(role, str) or not role or not valid_patterns(patterns) or not patterns
                       for role, patterns in roles.items())
                or not valid_patterns(forbidden) or not valid_patterns(removed_ranges)
                or not isinstance(role_limits, dict)
                or any(role not in roles or isinstance(limit, bool) or not isinstance(limit, (int, float))
                       or not math.isfinite(limit) or limit <= 0 for role, limit in role_limits.items())):
            continue
        if item["candidate_id"] in selected_ids:
            item["reason"] = "duplicate_candidate_id"
            for previous in result[:-1]:
                if previous["selected"] and previous["candidate_id"] == item["candidate_id"]:
                    previous.update(observation="unknown", reason="duplicate_candidate_id")
            continue
        selected_ids.add(item["candidate_id"])
        if profile is None:
            item["reason"] = "profile_missing"
            continue
        region = regions.get(item["profile_range"])
        if not isinstance(region, dict):
            item["reason"] = "candidate_range_missing"
            continue
        launches = region.get("linked_cuda_launches_per_step")
        names = region.get("kernel_names")
        if (isinstance(launches, bool) or not isinstance(launches, (int, float))
                or not math.isfinite(launches) or launches <= 0
                or not valid_patterns(names) or not names):
            item["reason"] = "candidate_cuda_attribution_missing"
            continue
        item["observed_cuda_launches"] = launches
        item["kernel_names"] = names
        item["range_event_ids"] = region.get("range_event_ids", [])
        item["correlation_ids"] = region.get("correlation_ids", [])
        item["matched_kernel_roles"] = {
            role: [name for name in names if any(pattern in name for pattern in patterns)]
            for role, patterns in roles.items()}
        item["missing_kernel_roles"] = [role for role, matches in item["matched_kernel_roles"].items() if not matches]
        kernel_counts = region.get("kernel_launches_per_step") or {}
        calls = region.get("range_calls_per_step")
        calls_observed = (not isinstance(calls, bool) and isinstance(calls, (int, float))
                          and math.isfinite(calls) and calls > 0)
        for role, symbols in item["matched_kernel_roles"].items():
            counts = [kernel_counts.get(symbol) for symbol in symbols]
            launches_per_step = (sum(counts) if all(
                not isinstance(count, bool) and isinstance(count, (int, float)) and math.isfinite(count) and count > 0
                for count in counts) else None)
            item["matched_kernel_role_launches"][role] = launches_per_step
            item["matched_kernel_role_launches_per_call"][role] = (
                launches_per_step / calls if calls_observed and launches_per_step is not None else None)
        item["excess_kernel_roles"] = [role for role, limit in role_limits.items()
            if item["matched_kernel_role_launches_per_call"][role] is not None
            and item["matched_kernel_role_launches_per_call"][role] > limit]
        item["forbidden_kernel_symbols"] = [name for name in names if any(pattern in name for pattern in forbidden)]
        for name in removed_ranges:
            if name not in regions:
                continue
            standalone_launches = (regions[name] or {}).get("linked_cuda_launches_per_step")
            if (isinstance(standalone_launches, bool) or not isinstance(standalone_launches, (int, float))
                    or not math.isfinite(standalone_launches) or standalone_launches < 0):
                item["unattributed_profile_ranges"].append(name)
            elif standalone_launches > 0:
                item["remaining_profile_ranges"].append(name)
        if item["missing_kernel_roles"]:
            item.update(observation="not_observed", reason="required_kernel_roles_missing")
        elif item["forbidden_kernel_symbols"] or item["remaining_profile_ranges"]:
            item.update(observation="not_observed", reason="standalone_work_remaining")
        elif item["excess_kernel_roles"]:
            item.update(observation="not_observed", reason="kernel_role_launch_limit_exceeded")
        elif any(item["matched_kernel_role_launches_per_call"][role] is None for role in role_limits):
            item.update(observation="unknown", reason="kernel_launch_count_attribution_missing")
        elif item["unattributed_profile_ranges"]:
            item.update(observation="unknown", reason="standalone_work_attribution_missing")
        else:
            item.update(observation="observed", reason=None)
    selected = [item for item in result if item["selected"]]
    observed = sum(item["observation"] == "observed" for item in selected)
    status = ("unfused" if not selected else "observed" if observed == len(selected)
              else "not_observed" if any(item["observation"] == "not_observed" for item in selected)
              else "unknown")
    return {"status": status, "selected_candidates": len(selected), "observed_candidates": observed,
            "all_selected_observed": bool(selected) and observed == len(selected), "candidates": result}


def pack_sharing_evidence(report, profile):
    """Check one produced encoding separately from numerical fusion counts."""
    decisions = report.get("pack_sharing_decisions", ())
    if not decisions:
        decisions = [candidate for graph in report.get("graphs", ())
                     for native in graph.get("native_autodiff_graphs", ())
                     for candidate in native.get("pack_sharing_decisions", ())]
    evidence = fusion_candidate_evidence({"fusion_candidates": decisions}, profile)
    if evidence["status"] == "unfused":
        evidence["status"] = "unshared"
    return evidence


def profile(session, directory, steps, rows=60, *, replay=False, detail="trace", compress=False):
    import torch

    session.reset()
    call = session.graph_step if replay else session.step
    call()
    synchronize(torch, session.device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if session.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    from contextlib import nullcontext
    # Adding module hooks after compile warmup changes Dynamo guards and can
    # pull record_function.__enter__ into fullgraph capture. Compiled executions
    # already expose phase ranges and ranges from their generated executables.
    module_annotations_enabled = not (replay or detail == "summary" or session.runtime["execution"] == "compile")
    with (module_annotations(session.model) if module_annotations_enabled else nullcontext()), torch.profiler.profile(
            activities=activities, record_shapes=detail != "summary", profile_memory=detail == "stacks",
            with_stack=detail == "stacks") as capture:
        for _ in range(steps):
            call() if replay else session.step(annotate=True)
            capture.step()
        synchronize(torch, session.device)
    captured_events = capture.events()
    chrome_regions = None
    trace_path = directory / ("replay-trace.json" if replay else "trace.json")
    export_error = None
    if detail != "summary":
        temporary = trace_path.with_suffix(".tmp.json")
        compressed = trace_path.with_suffix(".json.gz")
        compressed_tmp = compressed.with_suffix(".tmp.gz")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            capture.export_chrome_trace(str(temporary))
            chrome_regions = _chrome_compiler_launch_evidence(temporary, steps)
            if compress:
                import gzip
                import shutil
                with temporary.open("rb") as src, gzip.open(compressed_tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                compressed_tmp.replace(compressed)
                trace_path = compressed
            else:
                temporary.replace(trace_path)
        except OSError as error:
            export_error = f"{type(error).__name__}: {error}"
            trace_path = None
        finally:
            for leftover in (temporary, compressed_tmp):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError as error:
                    export_error = (export_error or "") + f"; trace cleanup: {error}"
    else:
        trace_path = None
        if session.device.type == "cuda" and any(event.name.startswith(
                ("deepopt.compiler.", "deepopt.fusion::", "deepopt.pack::")) for event in captured_events):
            # Summary profiles still need driver-launch correlation for fusion
            # qualification. Export once transiently, without retaining a trace.
            from pathlib import Path
            from tempfile import TemporaryDirectory

            try:
                with TemporaryDirectory(prefix="deepopt-profile-attribution-") as temporary_directory:
                    transient = Path(temporary_directory) / "trace.json"
                    capture.export_chrome_trace(str(transient))
                    chrome_regions = _chrome_compiler_launch_evidence(transient, steps)
            except OSError as error:
                export_error = f"{type(error).__name__}: {error}"
    kernels, buckets = {}, defaultdict(lambda: {"launches": 0, "kernel_ms_per_step": 0.0})
    cpu, sync_ops, operators = [], [], []
    for event in captured_events:
        if event.device_type != torch.autograd.DeviceType.CUDA or getattr(event, "is_user_annotation", False):
            continue
        if event.name.startswith(("Memcpy", "Memset", "[memory]")):
            continue
        row = kernels.setdefault(event.name, {"name": event.name, "launches": 0, "kernel_ms_per_step": 0.0})
        duration = event.device_time_total / 1000 / steps
        row["launches"] += 1
        row["kernel_ms_per_step"] += duration
        bucket = buckets[classify_kernel(event.name)]
        bucket["launches"] += 1
        bucket["kernel_ms_per_step"] += duration
    for event in capture.key_averages(group_by_input_shape=True):
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        row = {"name": event.key, "count_per_step": event.count / steps,
               "self_cpu_ms_per_step": event.self_cpu_time_total / 1000 / steps,
               "input_shapes": event.input_shapes}
        cpu.append(row)
        if event.key.startswith(("module::", "analysis.", "deepopt.compiler.", "deepopt.fusion::", "deepopt.pack::")):
            operators.append({**row, "inclusive_device_ms_per_step": event.device_time_total / 1000 / steps})
        if any(word in event.key.lower() for word in ("synchronize", "_local_scalar_dense", "aten::item")):
            sync_ops.append(row)
    compiler_regions = compiler_launch_evidence(captured_events, steps)
    if chrome_regions is not None:
        compiler_regions.update(chrome_regions)
    compiler_report = getattr(session.model, "_deepopt_compile_report", None)
    compiler_payload = compiler_report.as_dict() if compiler_report is not None else {}
    candidate_evidence = fusion_candidate_evidence(
        compiler_payload, {"compiler_regions": compiler_regions})
    sharing_evidence = pack_sharing_evidence(compiler_payload, {"compiler_regions": compiler_regions})
    session.reset()
    return {"trace": str(trace_path) if trace_path is not None else None, "steps": steps, "detail": detail,
            "trace_export_error": export_error,
            "execution": "graph_replay_with_eager_optimizer" if replay else session.runtime["execution"],
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "fill_kernel_activity_sum_ms_per_step": buckets.get("fill", {}).get("kernel_ms_per_step", 0.),
            "fill_note": "FillFunctor kernels include allocation initialization and explicit fills; inspect CPU ancestry for allocation attribution.",
            "kernel_launches_per_step": sum(r["launches"] for r in kernels.values()) / steps,
            "kernel_activity_sum_ms_per_step": sum(r["kernel_ms_per_step"] for r in kernels.values()),
            "top_kernels": sorted(kernels.values(), key=lambda r: -r["kernel_ms_per_step"])[:rows],
            "kernel_buckets": dict(buckets),
            "top_cpu_ops": sorted(cpu, key=lambda r: -r["self_cpu_ms_per_step"])[:rows],
            "operator_ranges": sorted(operators, key=lambda r: -r["inclusive_device_ms_per_step"]),
            "compiler_regions": compiler_regions, "fusion_qualification": candidate_evidence,
            "pack_sharing_qualification": sharing_evidence,
            "module_annotations_enabled": module_annotations_enabled,
            "operator_ranges_note": "Inclusive nested ranges overlap. Module hooks annotate eager profiles only; "
                                    "compiled profiles retain phase and compiler ranges. Replay launches do not reexecute "
                                    "Python module annotations. Inspect the paired traces.",
            "synchronization_ops": sync_ops,
            "note": "Kernel buckets are name-based hints. Kernel activity sums can overlap and exclude host gaps; "
                    "module ranges and nested CPU totals must not be added to kernel time. Inspect the trace for causality."}
