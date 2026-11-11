"""Synchronized timing, separate profiling, numerical and memory diagnostics."""

from collections import defaultdict
from contextlib import contextmanager
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


def measure(call, torch, device, warmup, iterations, repeats):
    for _ in range(warmup):
        call()
    synchronize(torch, device)
    wall, stream, submit = [], [], []
    for _ in range(repeats):
        synchronize(torch, device)
        if device.type == "cuda":
            first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            first.record()
        began = perf_counter()
        for _ in range(iterations):
            call()
        submitted = perf_counter()
        if device.type == "cuda":
            last.record()
            last.synchronize()
            stream.append(first.elapsed_time(last) / iterations)
        wall.append((perf_counter() - began) * 1000 / iterations)
        submit.append((submitted - began) * 1000 / iterations)
    return {"wall_ms": stats(wall), "cuda_stream_ms": stats(stream), "host_submission_ms": stats(submit),
            "note": "CUDA event intervals include submission gaps; host submission may include implicit waits. These are not additive."}


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

    return {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            for key, value in state.items()}


def snapshot(session):
    import torch

    session.reset()
    session.clear()
    output, loss = session.forward()
    session.backward(loss)
    metadata = {"original_shapes": {}, "full_tensor_finite": True,
                "max_sampled_elements_per_float_tensor": session.max_numerical_elements}

    def capture(value, name):
        if value is None:
            return None
        value = dense(value).detach()
        metadata["original_shapes"][name] = list(value.shape)
        if value.is_floating_point():
            metadata["full_tensor_finite"] &= bool(torch.isfinite(value).all())
            if session.max_numerical_elements and value.numel() > session.max_numerical_elements:
                step = math.ceil(value.numel() / session.max_numerical_elements)
                value = value.reshape(-1)[::step]
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
    session.snapshot_metadata = metadata
    session.reset()
    return values


def errors(actual, expected, rtol, atol):
    import torch

    result, passed, finite = {}, True, True
    for key in sorted(set(actual) | set(expected)):
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
        exact = not x.is_floating_point() and not y.is_floating_point()
        exact_match = bool(torch.equal(x, y)) if exact else None
        x, y = x.double().flatten(), y.double().flatten()
        is_finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        finite &= is_finite
        if not is_finite:
            result[key] = {"passed": False, "finite": False}
            passed = False
            continue
        delta = x - y
        norm_x, norm_y = x.norm().item(), y.norm().item()
        cosine = ((x @ y).item() / (norm_x * norm_y)) if norm_x and norm_y else (1.0 if not norm_x and not norm_y else 0.0)
        close = exact_match if exact else bool(torch.allclose(x, y, rtol=rtol, atol=atol))
        result[key] = {"passed": close, "finite": True, "max_abs": delta.abs().max().item() if x.numel() else 0,
                       "relative_l2": delta.norm().item() / max(norm_y, 1e-30), "cosine": cosine,
                       "exact": exact}
        passed &= close
    return {"passed": passed, "finite": finite, "rtol": rtol, "atol": atol, "tensors": result}


def memory_diagnostics(session):
    import torch

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
    rows, storages = [], {}

    def save(value):
        storage = value.untyped_storage()
        key = (str(value.device), storage._cdata)
        storages[key] = storage.nbytes()
        rows.append({"shape": list(value.shape), "dtype": str(value.dtype), "strides": list(value.stride()),
                     "logical_bytes": value.numel() * value.element_size(), "storage_bytes": storage.nbytes()})
        return value

    forward_allocated = None
    with torch.autograd.graph.saved_tensors_hooks(save, lambda value: value):
        accumulation = session.runtime["accumulation_steps"]
        for index in range(accumulation):
            _, loss = session.forward()
            if index == 0:
                synchronize(torch, session.device)
                forward_allocated = torch.cuda.memory_allocated(session.device) if cuda else None
            session.backward(loss / accumulation)
        if session.optimizer is not None:
            session.optimizer.step()
    synchronize(torch, session.device)
    result = {"observed_saved_logical_bytes": sum(row["logical_bytes"] for row in rows),
              "observed_saved_unique_storage_bytes": sum(storages.values()), "saved_tensors": rows,
              "baseline_allocated_bytes": baseline, "after_forward_allocated_bytes": forward_allocated,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(session.device) if cuda else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(session.device) if cuda else None,
              "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(session.device) - baseline if cuda else None,
              "optimizer_state_warmed": session.optimizer is not None,
              "warmup_updates": 3,
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
    if any(word in name for word in ("unpack", "decode", "dequant")):
        return "decode_or_dequantize"
    if "pack" in name:
        return "pack_or_fused_cast_pack"
    if any(word in name for word in ("amax", "min_max", "scale_update")):
        return "scaling_or_range_reduction"
    if "weight_gradient" in name:
        return "weight_gradient"
    if "input_gradient" in name:
        return "input_gradient"
    if any(word in name for word in ("gemm", "matmul", "mma", "xmma")):
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


def profile(session, directory, steps, rows=60):
    import torch

    session.reset()
    session.step()
    synchronize(torch, session.device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if session.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with module_annotations(session.model), torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as capture:
        for _ in range(steps):
            session.step(annotate=True)
            capture.step()
        synchronize(torch, session.device)
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / "trace.json"
    capture.export_chrome_trace(str(trace_path))
    kernels, buckets = {}, defaultdict(lambda: {"launches": 0, "kernel_ms_per_step": 0.0})
    cpu, sync_ops = [], []
    for event in capture.events():
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
        if any(word in event.key.lower() for word in ("synchronize", "_local_scalar_dense", "aten::item")):
            sync_ops.append(row)
    session.reset()
    return {"trace": str(trace_path), "steps": steps,
            "kernel_launches_per_step": sum(r["launches"] for r in kernels.values()) / steps,
            "kernel_activity_sum_ms_per_step": sum(r["kernel_ms_per_step"] for r in kernels.values()),
            "top_kernels": sorted(kernels.values(), key=lambda r: -r["kernel_ms_per_step"])[:rows],
            "kernel_buckets": dict(buckets),
            "top_cpu_ops": sorted(cpu, key=lambda r: -r["self_cpu_ms_per_step"])[:rows],
            "synchronization_ops": sync_ops,
            "note": "Kernel buckets are name-based hints. Kernel activity sums can overlap and exclude host gaps; "
                    "module ranges and nested CPU totals must not be added to kernel time. Inspect the trace for causality."}
