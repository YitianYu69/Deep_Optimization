"""Installed-wheel CUDA calibration for bounded 2-bit/group1024 triplets.

Example (budgets are bytes, including validation/compute tensor storage)::

    python benchmarks/compiler/benchmark_offload_transport.py --groups 262144 \
      --host-bytes 268435456 --gpu-bytes 536870912 --steps 10 \
      --output work/transport-new/result.json --profile

The six cases copy D2H/H2D/both, alone and with independent FP32 pointwise
compute. GPU-event intervals, host enqueue and whole-case wall time are
different measurements. Stream-span intersection is not proof of continuous
kernel contention; inspect the optional CUDA trace. There is no VAE, native
AutoDiff, training throughput or offload-policy qualification in this program.
Outputs are metadata only, use new paths, and never contain tensor payloads.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--groups", type=_positive, required=True, help="Number of affine groups; 264 bytes per triplet group")
    result.add_argument("--host-bytes", type=_positive, required=True, help="Admission limit: two pinned triplets plus one pageable validation triplet")
    result.add_argument("--gpu-bytes", type=_positive, required=True, help="Admission limit: two triplets plus two FP32 compute buffers; allocator reservation is separate")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--compute-elements", type=_positive, default=4 * 1024 * 1024)
    result.add_argument("--compute-iterations", type=_positive, default=32)
    result.add_argument("--copy-chunk-bytes", type=_positive, default=16 * 1024 * 1024)
    result.add_argument("--warmup", type=_positive, default=2)
    result.add_argument("--steps", type=_positive, default=10)
    result.add_argument("--profile", action="store_true", help="Capture separate, untimed CUDA profiler iterations")
    result.add_argument("--preflight-only", action="store_true", help="Validate installation/device/capacity without allocating fixture tensors")
    result.add_argument("--output", type=Path, required=True, help="New immutable scalar JSON path; sibling artifact directory must also be absent")
    return result


def calibration_plan(args):
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        raise ValueError("Transport calibration requires an explicit cuda:N device")
    # The arange byte fixture uses signed int32 words. Total byte arithmetic
    # remains Python int and supports triplets larger than 2^31 bytes.
    if args.groups * 64 > 2**31 - 1:
        raise ValueError("Fixture payload word count exceeds its signed-int32 arange domain")
    triplet = args.groups * 264
    compute = args.compute_elements * 2 * 4
    required_host, required_gpu = triplet * 3, triplet * 2 + compute
    if args.host_bytes < required_host:
        raise ValueError(f"Host budget requires {required_host} bytes; allowed {args.host_bytes}")
    if args.gpu_bytes < required_gpu:
        raise ValueError(f"GPU tensor budget requires {required_gpu} bytes; allowed {args.gpu_bytes}")
    return dict(bits=2, group_size=1024, groups=args.groups, triplet_bytes=triplet,
                component_shapes=((args.groups, 64), (args.groups,), (args.groups,)),
                component_dtypes=("torch.int32", "torch.float32", "torch.float32"),
                pinned_bytes=triplet * 2, validation_pageable_allowance_bytes=triplet,
                host_required_bytes=required_host, host_budget_bytes=args.host_bytes,
                GPU_tensor_required_bytes=required_gpu, GPU_tensor_budget_bytes=args.gpu_bytes,
                compute_buffer_bytes=compute, compute_elements=args.compute_elements,
                compute_iterations=args.compute_iterations, copy_chunk_bytes=args.copy_chunk_bytes,
                budget_scope="explicit fixture tensors; CUDA allocator/context/library reservation reported separately",
                cases=tuple(dict(direction=direction, independent_compute=load)
                            for load in (False, True) for direction in ("d2h", "h2d", "bidirectional")))


def cuda_preflight(torch, args, plan):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; calibration did not allocate fixture tensors")
    device = torch.device(args.device)
    if device.index >= torch.cuda.device_count():
        raise RuntimeError("Requested CUDA device does not exist")
    with torch.cuda.device(device):
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
    if plan["GPU_tensor_required_bytes"] > free:
        raise RuntimeError(f"GPU fixture needs {plan['GPU_tensor_required_bytes']} bytes; free {free}")
    return dict(device=str(device), name=properties.name, uuid=str(getattr(properties, "uuid", None)),
                capability=(properties.major, properties.minor), total_bytes=total, free_before_bytes=free,
                torch=str(torch.__version__), cuda=torch.version.cuda, python=platform.python_version())


@contextmanager
def _installed_runtime():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from benchmarks.compiler.installed_runtime import identity_session

    with identity_session(ROOT) as session:
        import torch

        yield SimpleNamespace(torch=torch, identity=session.identity)


def _summary(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError("Nonfinite or missing calibration timing")
    return dict(samples=values, mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), population_sd=statistics.pstdev(values))


def _with_owned_stream_cleanup(operation, evidence):
    """Drain every owned stream even when one drain fails; preserve cause."""
    owned, primary = {}, None
    try:
        return operation(owned)
    except BaseException as error:
        primary = error
        raise
    finally:
        errors, attempted = [], []
        for name, stream in owned.items():
            attempted.append(name)
            try:
                stream.synchronize()
            except BaseException as error:
                errors.append((name, error))
        evidence["owned_stream_cleanup"] = dict(attempted=attempted, succeeded=not errors,
            errors=[dict(stream=name, type=type(error).__name__, message=str(error)) for name, error in errors])
        if errors:
            detail = "; ".join(f"{name}: {type(error).__name__}: {error}" for name, error in errors)
            if primary is not None:
                primary.add_note("Calibration owned-stream cleanup also failed: " + detail)
            else:
                raise RuntimeError("Calibration owned-stream cleanup failed: " + detail) from errors[0][1]


def _observe_cost(model, copy, *, direction, contended):
    """Admit measured spans without claiming proven kernel contention."""
    if contended and copy["compute_stream_span_intersection_ms"] <= 0:
        return "excluded_zero_compute_stream_span_overlap"
    # The production schema has no bidirectional-idle domain. Keep that case's
    # complete raw measurements without silently merging it into plain idle.
    if direction == "bidirectional" and not contended:
        return "excluded_unmodelled_bidirectional_idle_domain"
    load = "bidirectional_compute" if contended and direction == "bidirectional" else "compute" if contended else "idle"
    model.observe(copy["direction"], copy["bytes"], copy["GPU_copy_ms"], load=load)
    return "retained_positive_stream_span_contention_unverified" if contended else "retained_idle_copy_span"


def _execute(torch, args, plan, artifacts, evidence):
    def operation(owned):
        result = _execute_fixture(torch, args, plan, artifacts, evidence, owned)
        # Retain completed calibration/trace hashes if final cleanup fails.
        evidence.update(result)
        return result

    return _with_owned_stream_cleanup(operation, evidence)


def _execute_fixture(torch, args, plan, artifacts, evidence, owned):
    from deepopt.compiler.offload_plan import PackedOffloadComponent
    from deepopt.compiler.runtime.packed_offload import _CudaBackend
    from deepopt.compiler.runtime.offload_cost import TransferCostModel

    device = torch.device(args.device)
    backend = _CudaBackend(device)
    owned.update(D2H=backend.d2h, H2D=backend.h2d)
    shapes = plan["component_shapes"]
    descriptors = tuple(PackedOffloadComponent(name, index, shape, dtype,
        (64, 1) if index == 0 else (1,), str(device), math.prod(shape) * 4,
        ("payload", "codec_step", "codec_minimum")[index])
        for index, (name, shape, dtype) in enumerate(zip(("data", "step", "minimum"), shapes, plan["component_dtypes"])))
    torch.cuda.reset_peak_memory_stats(device)
    baseline = dict(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device))
    source_host = tuple(backend.host(component, pinned=True) for component in descriptors)
    target_host = tuple(backend.host(component, pinned=True) for component in descriptors)
    torch.arange(args.groups * 64, dtype=torch.int32, out=source_host[0].view(-1))
    # Include signed zero and NaN payloads; validation compares bytes, not FP equality.
    pattern = torch.tensor([-2147483648, 2143363909, 1, 1051372203], dtype=torch.int32)
    for index in (1, 2):
        source_host[index].view(torch.int32).copy_(pattern.repeat((args.groups + 3) // 4)[:args.groups])
    del pattern  # Do not retain fixture-initialization scratch during validation.
    source_gpu = tuple(backend.allocate(component) for component in descriptors)
    target_gpu = tuple(backend.allocate(component) for component in descriptors)
    backend.copy(source_gpu, source_host, direction="h2d", chunk_bytes=args.copy_chunk_bytes).synchronize()
    backend.copy(target_host, source_gpu, direction="d2h", chunk_bytes=args.copy_chunk_bytes).synchronize()
    evidence["CUDA_fixture_execution_observed"] = True
    compute_stream = torch.cuda.Stream(device=device)
    owned["compute"] = compute_stream
    with torch.cuda.device(device), torch.cuda.stream(compute_stream):
        left = torch.full((args.compute_elements,), .5, dtype=torch.float32, device=device)
        right = torch.empty_like(left)
    compute_stream.synchronize()
    model = TransferCostModel()

    def compute_once():
        torch.add(left, .001, out=right)
        torch.mul(right, .999, out=left)

    def iteration(direction, contended):
        wall = time.perf_counter()
        gate = backend.event(timing=True)
        compute_start = compute_end = None
        enqueue_compute = 0.0
        if contended:
            start = time.perf_counter()
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(gate.event)
                compute_start = backend.event(compute_stream, timing=True)
                compute_once()
            enqueue_compute += (time.perf_counter() - start) * 1000
        copy_start = time.perf_counter()
        transfers = []
        if direction in ("d2h", "bidirectional"):
            transfers.append(("D2H", backend.copy(target_host, source_gpu, direction="d2h", dependency=gate,
                                                 chunk_bytes=args.copy_chunk_bytes, measure=True)))
        if direction in ("h2d", "bidirectional"):
            transfers.append(("H2D", backend.copy(target_gpu, source_host, direction="h2d", dependency=gate,
                                                 chunk_bytes=args.copy_chunk_bytes, measure=True)))
        enqueue_copy = (time.perf_counter() - copy_start) * 1000
        if contended:
            start = time.perf_counter()
            with torch.cuda.stream(compute_stream):
                for _ in range(args.compute_iterations - 1):
                    compute_once()
                compute_end = backend.event(compute_stream, timing=True)
            enqueue_compute += (time.perf_counter() - start) * 1000
        for _, event in transfers:
            event.synchronize()
        if compute_end is not None:
            compute_end.synchronize()
        wall_ms = (time.perf_counter() - wall) * 1000
        compute_interval = ((gate.event.elapsed_time(compute_start.event), gate.event.elapsed_time(compute_end.event))
                            if compute_end is not None else None)
        copies = []
        for name, event in transfers:
            first, last = gate.event.elapsed_time(event.start.event), gate.event.elapsed_time(event.event)
            intersection = (max(0., min(last, compute_interval[1]) - max(first, compute_interval[0]))
                            if compute_interval is not None else 0.)
            copies.append(dict(direction=name, bytes=plan["triplet_bytes"], GPU_copy_ms=event.elapsed_ms(),
                               GPU_start_from_gate_ms=first, GPU_end_from_gate_ms=last,
                               compute_stream_span_intersection_ms=intersection))
        return dict(wall_ms=wall_ms, CPU_copy_enqueue_ms=enqueue_copy, CPU_compute_enqueue_ms=enqueue_compute,
                    GPU_compute_stream_span_ms=(compute_interval[1] - compute_interval[0]) if compute_interval else 0., copies=copies)

    def validate(direction):
        for original, host, gpu in zip(source_host, target_host, target_gpu):
            if direction in ("d2h", "bidirectional") and not torch.equal(original.view(torch.int32), host.view(torch.int32)):
                raise AssertionError("D2H encoded-byte validation failed")
            if direction in ("h2d", "bidirectional") and not torch.equal(original.view(torch.int32), gpu.cpu().view(torch.int32)):
                raise AssertionError("H2D encoded-byte validation failed")
        if not all(math.isfinite(float(left[index].item())) for index in (0, args.compute_elements - 1)):
            raise AssertionError("Independent compute produced nonfinite values")

    rows = []
    evidence["cases"] = rows
    for case in plan["cases"]:
        evidence["active_case"] = dict(case)
        direction, contended = case["direction"], case["independent_compute"]
        for _ in range(args.warmup):
            iteration(direction, contended)
        for tensor in target_host:
            tensor.fill_(-1)
        with torch.cuda.stream(backend.h2d):
            for tensor in target_gpu:
                tensor.zero_()
        backend.event(backend.h2d).synchronize()
        samples = [iteration(direction, contended) for _ in range(args.steps)]
        validate(direction)  # Tensor scans and validation CPU copies are outside timing.
        row = dict(**case, samples=samples, byte_validation_passed=True,
                   actual_kernel_contention_verified=False,
                   wall_ms=_summary([sample["wall_ms"] for sample in samples]),
                   CPU_copy_enqueue_ms=_summary([sample["CPU_copy_enqueue_ms"] for sample in samples]),
                   CPU_compute_enqueue_ms=_summary([sample["CPU_compute_enqueue_ms"] for sample in samples]))
        row["GPU_copy_ms"] = {name: _summary([copy["GPU_copy_ms"] for sample in samples
            for copy in sample["copies"] if copy["direction"] == name])
            for name in ("D2H", "H2D") if any(copy["direction"] == name for copy in samples[0]["copies"])}
        admission = []
        for sample in samples:
            for copy in sample["copies"]:
                admission.append(_observe_cost(model, copy, direction=direction, contended=contended))
        row["cost_profile_sample_admission"] = admission
        rows.append(row)
    profile = model.freeze(profile_identity=dict(device=str(device), torch=str(torch.__version__),
        cuda=torch.version.cuda, triplet_groups=args.groups, independent_compute="FP32 add/mul out buffers",
        compute_elements=args.compute_elements, compute_iterations=args.compute_iterations))
    memory = dict(baseline=baseline, allocated_peak=torch.cuda.max_memory_allocated(device),
                  reserved_peak=torch.cuda.max_memory_reserved(device), allocated_end=torch.cuda.memory_allocated(device),
                  reserved_end=torch.cuda.memory_reserved(device), scope="fixture setup, timing and validation; separate profiler excluded")
    trace = None
    if args.profile:
        path = artifacts / "transport-trace.json"
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    profile_memory=True, record_shapes=False) as profiler:
            for case in plan["cases"]:
                with torch.profiler.record_function(f"transport::{case['direction']}::compute={case['independent_compute']}"):
                    iteration(case["direction"], case["independent_compute"])
        profiler.export_chrome_trace(str(path))
        trace = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    return dict(cases=rows, transfer_cost_profile=profile.as_dict(), profiler_trace=trace,
                profile_enqueue_and_queue_costs_attributed_per_direction=False,
                transfer_cost_profile_qualifies_training_policy=False,
                contention_profile_verified=False,
                contention_profile_scope="positive stream-span overlap only; actual kernel overlap/continuous load unverified, including when a trace is saved",
                CUDA_fixture_execution_observed=True, native_AutoDiff_qualified=False, VAE_execution_qualified=False,
                tensor_payload_files=0, memory=memory,
                overlap_scope="GPU event stream-span intersection includes host submission gaps; use trace for actual kernel overlap",
                compute_scope="independent FP32 add/mul kernels with disjoint buffers; not the model's compute distribution",
                compute_validation="finite first/last elements outside timing; not a model numerical oracle",
                wall_scope="copy and independent compute submission through owned event completion; validation/profiling excluded")


def run(args, *, runtime_context=_installed_runtime):
    output = args.output.resolve()
    artifacts = output.with_name(output.stem + "-artifacts")
    if output.exists() or artifacts.exists():
        raise FileExistsError("Use a new calibration output and artifact path; historical results are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()  # Also reserves this result name against another worker.
    report = dict(schema_version=1, status="failed", stage="planning", created_at=datetime.now(timezone.utc).isoformat(),
                  native_AutoDiff_qualified=False, VAE_execution_qualified=False, tensor_payload_files=0,
                  CUDA_fixture_execution_observed=False,
                  runner=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        plan = report["plan"] = calibration_plan(args)
        report["stage"] = "installed_CUDA_preflight"
        with runtime_context() as runtime:
            report["installed"] = runtime.identity
            report["hardware"] = cuda_preflight(runtime.torch, args, plan)
            # Importing the installed classes validates their presence without
            # constructing streams or allocating any fixture tensors.
            from deepopt.compiler.runtime.packed_offload import _CudaBackend
            from deepopt.compiler.runtime.offload_cost import TransferCostModel

            if not callable(_CudaBackend) or not callable(TransferCostModel):
                raise RuntimeError("Installed wheel lacks the packed transport calibration APIs")
            if not args.preflight_only:
                report["stage"] = "CUDA_transport_calibration"
                report.update(_execute(runtime.torch, args, plan, artifacts, report))
            report["status"] = "preflight_passed" if args.preflight_only else "completed"
            report["stage"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=dict(type=type(error).__name__, message=str(error),
                                                   traceback=traceback.format_exc()))
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(encoded)
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = run(args)
    except FileExistsError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(dict(status=report["status"], output=str(args.output), stage=report["stage"])))
    return 0 if report["status"] in ("completed", "preflight_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
