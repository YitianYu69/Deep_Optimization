#!/usr/bin/env python3
"""Resumable Q/K/V direction and actual optimizer-update comparisons.

Use --snapshot for real-model replay and extraction of actual attention
operands/upstream gradients. Without it, inputs are synthetic, explicitly
labelled as such. Timing and profiles run separately from alignment collection.
"""

import argparse
from argparse import Namespace
from copy import deepcopy
import csv
import gc
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import statistics
import sys
import traceback
from unittest.mock import patch


def _helpers():
    if __package__:
        from . import attention_comparison as cases, diagnose_training_stability as diagnostic
    else:
        import attention_comparison as cases
        import diagnose_training_stability as diagnostic
    return cases, diagnostic


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _specs(args):
    specs = []
    for backend in args.backends:
        for mode in args.modes:
            for bit in args.bits:
                layouts = args.layouts if mode == "shared" and bit else ["bshd"]
                for layout in layouts:
                    for codec in args.codecs if mode == "shared" and bit else ["current"]:
                        specs.append(dict(backend=backend, bits=[bit] * 3, mode=mode, layout=layout,
                                          codec=codec, group_size=args.group_size))
        if args.individual:
            for bits in ([8, 4, 4], [4, 8, 4], [4, 4, 8]):
                specs.append(dict(backend=backend, bits=bits, mode="shared", layout="bshd",
                                  codec="current", group_size=args.group_size))
    return specs


def _case_id(spec):
    return "-".join((spec["backend"], spec["mode"], "".join(map(str, spec["bits"])), spec["layout"], spec["codec"]))


def _step(workload, spec, device, routing=None):
    import torch
    cases, _ = _helpers()
    qkv, upstream = cases.restore_workload(workload, device)
    from deepopt.opt.nn.attention_backend import observe_attention
    with observe_attention(lambda inputs, out, info: routing.append(info) if routing is not None else None):
        output = cases.attention_case(*qkv, **spec)
    output.backward(upstream)
    if not all(bool(torch.isfinite(x).all()) for x in (output, *(x.grad for x in qkv))):
        raise FloatingPointError("Nonfinite attention output/gradient")
    return {"output": output.detach().cpu(), **{name: x.grad.detach().cpu()
            for name, x in zip(("dq", "dk", "dv"), qkv)}}


def _compare(actual, reference):
    _, diagnostic = _helpers()
    return {name: {"error": diagnostic.tensor_error(actual[name], value),
                   "alignment": diagnostic.gradient_alignment(actual[name], value,
                                                               head_axis=1 if name != "output" else None)}
            for name, value in reference.items()}


def _timing(workload, spec, args, output, *, existing=None, on_progress=None):
    import torch
    cases, _ = _helpers()
    if args.iterations == 0:
        return {"status": "disabled"}
    if not str(args.device).startswith("cuda"):
        return {"status": "unavailable", "reason": "CPU runs qualify diagnostic logic only; CUDA timings required"}
    qkv, dy = cases.restore_workload(workload, args.device)
    def forward():
        return cases.attention_case(*qkv, **spec)
    def backward(y):
        for x in qkv:
            x.grad = None
        y.backward(dy)
    def step():
        backward(forward())
    result = deepcopy(existing or {})
    result.update(status="running", units="milliseconds",
                  includes="production codec or diagnostic reconstruction as selected",
                  source_layout=cases.describe_layout(qkv), consumer="independent_qkv_leaf_gradients")
    stages = result.setdefault("stages", {})
    def save():
        if on_progress is not None:
            on_progress(result)
    def failed(name, error):
        stages[name] = {"status": "error", "reason": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc()}
        save()
    stream = torch.cuda.Stream(device=args.device)
    stream.wait_stream(torch.cuda.current_stream(args.device))
    # Keep every autograd node on one stream, including shared-codec warmup.
    # Reusing default-stream AccumulateGrad nodes in capture fails on PT 2.15.
    with torch.cuda.stream(stream):
        for value in (*qkv, dy):
            value.record_stream(stream)
        for _ in range(max(1, args.warmup)):
            step()
        stream.synchronize()
        for name in ("forward", "backward", "step"):
            if stages.get(name, {}).get("status") == "ok":
                continue
            stages[name] = {"status": "running"}; save()
            try:
                samples = []
                for _ in range(args.repeats):
                    elapsed = 0.
                    for _ in range(args.iterations):
                        y = forward() if name == "backward" else None
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        if name == "forward":
                            y = forward()
                        elif name == "backward":
                            backward(y)
                        else:
                            step()
                        end.record(); end.synchronize()
                        elapsed += start.elapsed_time(end)
                        del y
                    samples.append(elapsed / args.iterations)
                result[name] = {"median": statistics.median(samples), "repeats": samples}
                stages[name] = {"status": "ok"}; save()
            except Exception as error:
                failed(name, error)
        if stages.get("memory", {}).get("status") != "ok":
            try:
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                step(); stream.synchronize()
                result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                result["step_peak_increment_bytes"] = max(0, torch.cuda.max_memory_allocated() - before)
                stages["memory"] = {"status": "ok"}; save()
            except Exception as error:
                failed("memory", error)
        graph = None
        if args.graph:
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    step()
                if stages.get("graph", {}).get("status") != "ok":
                    samples = []
                    for _ in range(args.repeats):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.iterations):
                            graph.replay()
                        end.record(); end.synchronize()
                        samples.append(start.elapsed_time(end) / args.iterations)
                    result["graph_step_ms"] = statistics.median(samples)
                    result["graph_repeats_ms"] = samples
                    stages["graph"] = {"status": "ok"}; save()
            except Exception as error:
                graph = None
                failed("graph", error)
        if args.profile and stages.get("profile", {}).get("status") != "ok":
            if args.graph and graph is None:
                stages["profile"] = {"status": "blocked", "reason": "Graph capture failed"}; save()
            else:
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        graph.replay() if args.graph else step()
                        stream.synchronize()
                    profile.export_chrome_trace(str(output / "trace.json"))
                    result["profile_scope"] = "graph_replay" if args.graph else "eager_step"
                    result["profile_cuda_kernels"] = sorted({event.name for event in profile.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA})
                    stages["profile"] = {"status": "ok"}; save()
                except Exception as error:
                    failed("profile", error)
    torch.cuda.current_stream(args.device).wait_stream(stream)
    result["status"] = "ok" if all(s["status"] == "ok" for s in stages.values()) else "error"
    save()
    return result


def _snapshot_workloads(args, report):
    """Replay actual captured optimizer state, then extract matched attention data."""
    import torch
    from deepopt.opt.nn.attention_backend import observe_attention
    cases, diagnostic = _helpers()
    packet = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    if packet.get("format") != "deepopt-stability-v1" or packet.get("unreconstructed_forward_overrides"):
        raise ValueError("Use a stability snapshot with declared/reconstructable forward overrides")
    if packet["config"]["runtime"].get("fp8", {}).get("enabled", False):
        raise ValueError("First isolate BF16+compression: capture with FP8 disabled")
    torch.use_deterministic_algorithms(packet["deterministic"])
    torch.utils.deterministic.fill_uninitialized_memory = packet["fill_uninitialized_memory"]
    report["captured_environment"] = packet["environment"]
    report["cross_build_snapshot"] = packet["environment"]["deepopt_source_sha256"] != diagnostic._library_hash()
    report["optimizer"] = packet["optimizer"]["factory"]
    folder = args.output / "model-replays"
    folder.mkdir(exist_ok=True)
    workloads = []
    model_cases = [{"name": "all_dense", "spec": dict(backend="torch_flash" if "torch_flash" in args.backends else args.backends[0], bits=[0]*3, mode="production",
                    layout="bshd", codec="current", group_size=args.group_size)}]
    for backend in args.backends:
        for bit in args.bits:
            model_cases.append({"name": f"{backend}-qkv{bit}", "spec": dict(backend=backend, bits=[bit]*3,
                               mode="production", layout="bshd", codec="current", group_size=args.group_size)})
    report.setdefault("model_replays", {})
    for case in model_cases:
        name, spec = case["name"], case["spec"]
        result_file = folder / (name + ".json")
        tensor_file = folder / (name + ".pt")
        if args.resume and result_file.exists() and tensor_file.exists():
            result = json.loads(result_file.read_text())
            if result["status"] == "ok":
                report["model_replays"][name] = result
                if name == "all_dense":
                    workloads = [Path(p) for p in result["workloads"]]
                continue
        cloned = deepcopy(packet)
        cloned["config"]["runtime"]["attention"] = {"backend": "flash_attn_2" if spec["backend"] == "flash_attn_2" else "native"}
        trace, budget, skipped = [], [0], []
        captured = []
        def observe(inputs, out, info):
            index = len(trace)
            trace.append({**info, "qkv_shapes": [list(x.shape) for x in inputs],
                          "qkv_dtypes": [str(x.dtype) for x in inputs]})
            if name != "all_dense":
                return
            size = sum(x.numel()*x.element_size() for x in (*inputs, out))
            if budget[0] + size > args.max_workload_bytes:
                skipped.append(index)
                return
            budget[0] += size
            data = {"qkv": tuple(x.detach().cpu().clone() for x in inputs),
                    "layout": {"qkv": cases.describe_layout(inputs)},
                    "source": {"kind": "real_snapshot", "attention_call": index, "routing": trace[-1]}}
            path = args.output / "workloads" / f"attention-{index:03d}.pt"
            path.parent.mkdir(exist_ok=True)
            def save_upstream(gradient):
                data["grad_output"] = gradient.detach().cpu().clone()
                data["layout"]["grad_output"] = cases.describe_layout((gradient,))[0]
                temporary = path.with_suffix(".tmp")
                torch.save(data, temporary); temporary.replace(path)
                captured.append(path)
            if not out.requires_grad:
                raise ValueError("Attention output has no gradient; snapshot cannot qualify this comparison")
            out.register_hook(save_upstream)
        case_args = Namespace(device=args.device, model_factory=args.model_factory, model_kwargs=args.model_kwargs,
                              optimizer=args.optimizer, probe_modules=[], max_probe_bytes=0,
                              audit_codec=False, rtol=.03, atol=1e-5)
        original_factory = diagnostic._factory_model
        routing_changes = []
        def model_factory(*factory_args, **factory_kwargs):
            model = original_factory(*factory_args, **factory_kwargs)
            # Every algorithm receives the same SDPA entry point. A timm
            # manual branch otherwise bypasses the selectable backend entirely.
            for module_name, module in model.named_modules():
                if type(module).__module__.startswith("timm.") and hasattr(module, "fused_attn"):
                    routing_changes.append({"module": module_name, "original_fused_attn": bool(module.fused_attn)})
                    module.fused_attn = True
            return model
        try:
            with observe_attention(observe), cases.model_attention_case(spec), \
                    patch.object(diagnostic, "_factory_model", side_effect=model_factory):
                result = diagnostic._case("native" if name == "all_dense" else "production", cloned,
                                          case_args, folder, {}, [])
            generated = folder / (("native" if name == "all_dense" else "production") + ".pt")
            generated.replace(tensor_file)
            result.update(trace=trace, workloads=[str(p) for p in sorted(captured)], skipped_attention_calls=skipped,
                          forced_sdpa_modules=routing_changes)
            if not trace:
                raise ValueError("No attention routed through SDPA. Ensure timm fused_attn=True before capture")
            if name == "all_dense":
                workloads = sorted(captured)
            else:
                a = torch.load(tensor_file, map_location="cpu", weights_only=False)
                b = torch.load(folder / "all_dense.pt", map_location="cpu", weights_only=False)
                result["parameter_gradients"] = diagnostic._compare(a["gradients"], b["gradients"], rtol=.03, atol=1e-5)
                result["optimizer_updates"] = (diagnostic._compare(a["updates"], b["updates"], rtol=.03, atol=1e-5)
                    if a["updates"] and b["updates"] else {"status": "unavailable", "reason": result.get("optimizer_error")})
                result["qkv_parameter_heads"] = {}
                for key in a["gradients"]:
                    if key.endswith("attn.qkv.weight") and a["gradients"][key].shape[0] % (3*args.heads) == 0:
                        shape = (3, args.heads, -1, a["gradients"][key].shape[-1])
                        result["qkv_parameter_heads"][key] = diagnostic.gradient_alignment(
                            a["gradients"][key].reshape(shape), b["gradients"][key].reshape(shape), head_axis=1)
                del a, b
        except Exception as error:
            result = {"status": "error", "reason": str(error), "traceback": traceback.format_exc()}
        _atomic_json(result_file, result)
        report["model_replays"][name] = result
        _atomic_json(args.output / "report.json", report)
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        if name == "all_dense" and result["status"] != "ok":
            raise RuntimeError("Dense snapshot replay failed: " + result["reason"])
    return workloads


def run(args):
    import torch
    _, diagnostic = _helpers()
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    args.output.mkdir(parents=True, exist_ok=True)
    options = json.loads(json.dumps({k: v for k, v in vars(args).items() if k not in ("output", "resume")}, default=str))
    sources = {Path(p).name: _file_hash(p) for p in (__file__, diagnostic.__file__, _helpers()[0].__file__)}
    try:
        flash_version = metadata.version("flash-attn")
    except metadata.PackageNotFoundError:
        flash_version = None
    identity = {"options": options, "sources": sources, "deepopt_source_sha256": diagnostic._library_hash(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda, "flash_attn": flash_version,
                "matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "snapshot_sha256": _file_hash(args.snapshot) if args.snapshot else None,
                "workload_sha256": {str(p): _file_hash(p) for p in args.workloads},
                "device": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else "cpu"}
    fingerprint = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report_path = args.output / "report.json"
    report = {"fingerprint": fingerprint, "identity": identity, "results": {},
              "scope": "real matched-batch snapshot" if args.snapshot else "synthetic attention diagnostic",
              "notes": ["Alignment uses a common dense manual reference and each backend's own dense reference.",
                        "Shared mode fixes reconstructed Q/K/V; production mode preserves actual save layouts/dtypes.",
                        "legacy_rules is an unfused Torch reference, not bitwise verification of the old Triton codec.",
                        "Local attention workloads share upstream gradients. Model replays include propagation and actual optimizer updates.",
                        "A one-step alignment improvement does not establish long-run convergence."]}
    if report_path.exists():
        if not args.resume:
            raise ValueError("Output already exists; use --resume or a new output directory")
        report = json.loads(report_path.read_text())
        if report["fingerprint"] != fingerprint:
            raise ValueError("Resume fingerprint mismatch: source, environment, snapshot or settings changed")
    else:
        _atomic_json(report_path, report)
    if args.snapshot:
        workloads = _snapshot_workloads(args, report)
    elif args.workloads:
        workloads = args.workloads
    else:
        torch.manual_seed(args.seed)
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        workload = _helpers()[0].synthetic_workload(args.batch_size, args.heads, args.sequence_length,
                                                   args.head_dim, dtype=dtype, layout=args.source_layout)
        workload["source"]["seed"] = args.seed
        workloads = [workload]
    specs = _specs(args)
    expected = len(workloads) * len(specs)
    for index, source in enumerate(workloads):
        workload = torch.load(source, map_location="cpu", weights_only=True) if isinstance(source, Path) else source
        if any(x.ndim != 4 for x in workload["qkv"]) or workload["grad_output"].shape != workload["qkv"][0].shape:
            raise ValueError("Workloads must contain BHSD Q/K/V and a matching grad_output")
        reference_dir = args.output / "references" / str(index)
        reference_dir.mkdir(parents=True, exist_ok=True)
        references = {}
        for backend in dict.fromkeys(["manual", *args.backends]):
            path = reference_dir / (backend + ".pt")
            try:
                if args.resume and path.exists():
                    references[backend] = torch.load(path, map_location="cpu", weights_only=True)
                else:
                    dense = dict(backend=backend, bits=[0]*3, group_size=args.group_size, mode="production")
                    references[backend] = _step(workload, dense, args.device)
                    torch.save(references[backend], path)
            except Exception as error:
                references[backend] = {"error": str(error)}
        for spec in specs:
            key = f"workload-{index:03d}/" + _case_id(spec)
            if args.resume and report["results"].get(key, {}).get("status") == "ok":
                continue
            row = deepcopy(report["results"].get(key, {})) if args.resume else {}
            row.update(spec=spec, source=workload.get("source"), shape=list(workload["qkv"][0].shape),
                       source_layout=workload.get("layout"), status="running")
            report["results"][key] = row
            def timing_progress(timing):
                row["timing"] = deepcopy(timing)
                row["timing_status"] = timing["status"]
                _atomic_json(report_path, report)
            try:
                if "error" in references[spec["backend"]]:
                    raise RuntimeError(references[spec["backend"]]["error"])
                if row.get("alignment_status") != "ok":
                    routing = []
                    actual = _step(workload, spec, args.device, routing)
                    row.update(alignment_status="ok", versus_dense=_compare(actual, references["manual"]),
                               versus_own_dense=_compare(actual, references[spec["backend"]]), executed=routing)
                    del actual
                    _atomic_json(report_path, report)
                case_folder = args.output / key
                case_folder.mkdir(parents=True, exist_ok=True)
                row["timing"] = _timing(workload, spec, args, case_folder,
                                         existing=row.get("timing"), on_progress=timing_progress)
                row["timing_status"] = row["timing"]["status"]
                row["status"] = "error" if row["timing_status"] == "error" else "ok"
                if row["status"] == "ok":
                    row.pop("reason", None); row.pop("traceback", None)
            except Exception as error:
                row.update(status="error", reason=str(error), traceback=traceback.format_exc())
                if row.get("alignment_status") != "ok":
                    row["alignment_status"] = "error"
            report["results"][key] = row
            report["manifest"] = {"expected": expected, "completed": len(report["results"]),
                                  "ok": sum(r["status"] == "ok" for r in report["results"].values())}
            _atomic_json(report_path, report)
            print(json.dumps({"case": key, "status": row["status"]}), flush=True)
            gc.collect()
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    models_ok = all(r["status"] == "ok" and r.get("optimizer_update") == "matched_captured_state"
                    and not r.get("skipped_attention_calls") for r in report.get("model_replays", {}).values())
    report["status"] = "complete" if expected and report["manifest"]["ok"] == expected and models_ok else "partial"
    _atomic_json(report_path, report)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "status", "dq_cosine", "dq_angle_degrees", "dk_cosine", "dv_cosine", "step_ms"])
        for key, row in report["results"].items():
            alignment = row.get("versus_dense", {})
            metric = lambda name, key: alignment.get(name, {}).get("alignment", {}).get(key)
            writer.writerow([key, row["status"], metric("dq", "cosine"), metric("dq", "angle_degrees"),
                             metric("dk", "cosine"), metric("dv", "cosine"),
                             row.get("timing", {}).get("step", {}).get("median")])
    return report


def main(argv=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("attention-alignment-results"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backends", nargs="+", choices=("manual", "torch_flash", "flash_attn_2"),
                        default=["manual", "torch_flash", "flash_attn_2"])
    parser.add_argument("--bits", nargs="+", type=int, choices=(0, 1, 2, 4, 8), default=[0, 4, 8])
    parser.add_argument("--modes", nargs="+", choices=("production", "shared"), default=["production", "shared"])
    parser.add_argument("--layouts", nargs="+", choices=("bshd", "bhsd", "saved"), default=["bshd", "saved"])
    parser.add_argument("--codecs", nargs="+", choices=("current", "legacy_rules"), default=["current"])
    parser.add_argument("--individual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=197)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--source-layout", choices=("bhsd", "bshd", "interleaved"), default="interleaved",
                        help="Physical input layout; interleaved reproduces Timm fused QKV projection storage")
    parser.add_argument("--attention-pack", choices=("separate", "joint"), default="separate")
    parser.add_argument("--pack-indexing", choices=("wide", "bounded"), default="wide")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graph", action="store_true", help="Time graph replay separately; requires CUDA")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--workloads", nargs="*", type=Path, default=[])
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--model-factory", default="timm:create_model")
    parser.add_argument("--model-kwargs", type=json.loads,
                        default={"model_name": "vit_base_patch16_224", "pretrained": False, "num_classes": 10})
    parser.add_argument("--optimizer", help="Importable module:Class for the actual captured optimizer")
    parser.add_argument("--max-workload-bytes", type=int, default=8 * 2**30)
    args = parser.parse_args(argv)
    if any(getattr(args, field) < 1 for field in ("batch_size", "heads", "sequence_length", "head_dim", "group_size", "repeats")):
        parser.error("shape, group size and repeats must be positive")
    if min(args.warmup, args.iterations, args.max_workload_bytes) < 0:
        parser.error("warmup, iterations and byte budget must be nonnegative")
    if args.model_kwargs.get("pretrained") or args.snapshot and args.workloads:
        parser.error("Use pretrained=False and select either --snapshot or --workloads")
    # Keep custom optimizer modules in the launch directory importable.
    sys.path.append(str(Path.cwd()))
    if __name__ == "__main__" and str(args.device).startswith("cuda"):
        import runpy
        bootstrap = Path(__file__).with_name("_gpu_lease.py")
        if not bootstrap.is_file():
            raise RuntimeError("Keep _gpu_lease.py beside the benchmark")
        code = runpy.run_path(str(bootstrap))["ensure"](__file__, argv, args.device)
        if code is not None:
            return code
    from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
    with activation_codec_scope(attention_pack=args.attention_pack, strided_index=args.pack_indexing):
        report = run(args)
    print("Report: " + str(args.output / "report.json"))
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
