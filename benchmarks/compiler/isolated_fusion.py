"""Fresh-trial, same-recipe controls for isolated native-derived fusions.

Workers use the installed DeepOpt distribution. This module deliberately imports
Torch only in workers. Dry runs are portable and describe proposed experiments;
they never claim kernel support or execution.
Strategy isolation starts every control in a fresh process. Trial isolation
amortizes imports and CUDA initialization across sequential controls, retaining
an independent fresh process for each fixture/backend/trial and reporting this
different isolation boundary explicitly.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import random
import threading
import subprocess
import sys
from time import perf_counter
import uuid


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ("gemm-activation", "gemm-residual", "layernorm-activation", "groupnorm-activation",
            "packed-multiply", "shared-packs")
PHASES = ("forward", "backward", "both")


class WorkerProgress:
    """Publish setup and compilation work separately from GPU measurements."""

    def __init__(self, job, directory, shared_path=None):
        self.job = job
        self.paths = {Path(directory) / "worker-phase.json"}
        if shared_path is not None:
            self.paths.add(Path(shared_path))
        self.started = self.phase_started = perf_counter()
        self.phase = None
        self.durations = {}

    def start(self, phase, *, status="running"):
        now = perf_counter()
        if self.phase is not None:
            self.durations[self.phase] = (now - self.phase_started) * 1000
        self.phase, self.phase_started = phase, now
        value = {"job_id": self.job["id"], "strategy": self.job["strategy"]["name"],
                 "phase": phase, "status": status, "process_id": os.getpid(),
                 "elapsed_seconds": now - self.started,
                 "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                 "completed_phase_ms": dict(self.durations)}
        from compiler.benchmark_bf16 import write_json

        for path in self.paths:
            if path.parent.is_dir():
                write_json(path, value)
        print(f"[fusion {self.job['id']}] {phase} ({value['elapsed_seconds']:.1f}s elapsed)", flush=True)

    def finish(self):
        self.start("completed", status="completed")
        return {**self.durations, "total": (perf_counter() - self.started) * 1000}


@dataclass(frozen=True)
class Fixture:
    kind: str
    phase: str = "both"
    activation: str = "silu"
    activation_state: str = "recipe"
    batch: int = 2
    tokens: int = 17
    width: int = 64
    hidden: int = 96
    spatial: tuple = (8, 8)
    groups: int = 8
    dtype: str = "bf16"
    layout: str = "contiguous"
    gradient_mask: str = "all"
    compression: bool = False
    bits: int = 4
    group_size: int = 1024
    backward_order: str = "dx_first"
    autocast: bool = False
    bf16_policy: str = "torch_amp"
    qualify_native_group_fusion: bool = False

    def __post_init__(self):
        if self.bf16_policy not in ("torch_amp", "bf16_storage"):
            raise ValueError("Unknown BF16 policy")
        if self.bf16_policy == "bf16_storage" and self.dtype != "bf16" and not self.autocast:
            raise ValueError("Storage fixtures require BF16 inputs or BF16 autocast")
        if self.compression and self.group_size != 1024:
            raise ValueError("Fusion compression qualification requires group_size=1024")
        if self.kind not in FIXTURES or self.phase not in PHASES:
            raise ValueError("Unknown fixture kind or fusion phase")
        if self.activation not in ("silu", "gelu", "gelu-tanh"):
            raise ValueError("Activation must be silu, gelu, or gelu-tanh")
        if self.activation_state not in ("recipe", "exact"):
            raise ValueError("Activation state must follow the recipe or remain exact")
        if min(self.batch, self.tokens, self.width, self.hidden, self.groups,
               self.group_size, *self.spatial) < 1:
            raise ValueError("Fixture dimensions and codec group size must be positive")
        if len(self.spatial) != 2 or self.bits not in (1, 2, 4, 8):
            raise ValueError("Expected two spatial dimensions and 1/2/4/8 codec bits")
        if self.kind == "groupnorm-activation" and self.width % self.groups:
            raise ValueError("GroupNorm channels must be divisible by groups")
        if self.dtype not in ("bf16", "fp32"):
            raise ValueError("Dtype must be bf16 or fp32")
        if self.layout not in ("contiguous", "strided", "channels-last"):
            raise ValueError("Unsupported fixture layout")
        if self.layout == "channels-last" and self.kind != "groupnorm-activation":
            raise ValueError("Channels-last is a spatial GroupNorm fixture layout")
        if self.gradient_mask not in ("all", "input", "weight", "bias", "none"):
            raise ValueError("Unsupported gradient mask")
        if self.backward_order not in ("dx_first", "dw_first"):
            raise ValueError("Unsupported requested backward order")

    @property
    def identity(self):
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"{self.kind}-{self.phase}-{sha256(raw.encode()).hexdigest()[:16]}"


def source_identity():
    """Freeze measurement helpers without importing the installed workload.

    Installed DeepOpt bytes are frozen separately once execution is requested.
    Consequently a dry run needs neither a package checkout nor Torch.
    """
    files = []
    for directory in (ROOT / "benchmarks/compiler", ROOT / "benchmarks/analysis"):
        files.extend(directory.rglob("*.py"))
    digest = sha256()
    for path in sorted(set(files)):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_runtime():
    path = str(ROOT / "benchmarks")
    if path not in sys.path:
        sys.path.insert(0, path)
    from compiler import installed_runtime

    installed_runtime.prepare_imports(ROOT)
    return installed_runtime


def strategies(backend, fixture):
    """Only the candidate toggle differs inside the generated pair."""
    common = {"kernel_backend": backend, "require_full_aot": True,
              "bf16_gemm_backend": "generated", "bf16_tile": 64, "bf16_num_stages": 3,
              "bf16_decode": "materialized", "bf16_backward_order": fixture.backward_order,
              "fusion_phases": fixture.phase,
              "share_compatible_packs": False, "fuse_packed_multiply": False}
    if fixture.qualify_native_group_fusion:
        common["qualify_native_group_fusion"] = True
    pair = [
        {"name": "generated-unfused", "options": {**common, "fuse_deepopt_activations": False}},
        {"name": "generated-fused", "options": {**common, "fuse_deepopt_activations": True}},
    ]
    if fixture.kind == "packed-multiply":
        pair[1]["options"].update(fuse_deepopt_activations=False, fuse_packed_multiply=True)
    elif fixture.kind == "shared-packs":
        pair.extend({"name": f"{item['name']}-cse", "options": {**item["options"], "share_compatible_packs": True}}
                    for item in list(pair))
    if fixture.kind in ("gemm-activation", "gemm-residual", "shared-packs"):
        pair.append({"name": "library-control", "options": {**common, "bf16_gemm_backend": "native",
                                                             "fuse_deepopt_activations": False}})
    return pair


def make_plan(args):
    if len(set(args.backends)) != len(args.backends):
        raise ValueError("Backends must not contain duplicates")
    source = source_identity()
    jobs = []
    fixtures = [Fixture(kind=kind, phase=phase, activation=args.activation, activation_state=args.activation_state, batch=args.batch,
                        tokens=args.tokens, width=args.width, hidden=args.hidden,
                        spatial=tuple(args.spatial), groups=args.groups, dtype=args.dtype,
                        layout=args.layout, gradient_mask=args.gradient_mask,
                        compression=args.compression == "on", bits=bits, group_size=args.group_size,
                        backward_order=args.backward_order, autocast=args.autocast,
                        bf16_policy=getattr(args, "bf16_policy", "torch_amp"),
                        qualify_native_group_fusion=getattr(args, "qualify_native_group_fusion", False))
                for kind in args.fixtures for phase in args.phases
                for bits in (args.bits if args.compression == "on" else [args.bits[0]])]
    if len({fixture.identity for fixture in fixtures}) != len(fixtures):
        raise ValueError("Fixture matrix contains duplicates")
    for trial in range(args.trials):
        for fixture in fixtures:
            for backend in args.backends:
                order = strategies(backend, fixture)
                # Paired strategies share a seed but their process order is randomized.
                random.Random(args.seed + trial).shuffle(order)
                for strategy in order:
                    jobs.append({"id": f"{fixture.identity}-{backend}-t{trial}-{strategy['name']}",
                                 "fixture": asdict(fixture), "fixture_id": fixture.identity,
                                 "trial": trial, "seed": args.seed + trial, "backend": backend,
                                 "strategy": strategy, "device": args.device,
                                 "worker_isolation": args.worker_isolation,
                                 "source_sha256": source,
                                 "measurement": {"warmup": args.warmup, "iterations": args.iterations,
                                                 "repeats": args.repeats, "profile_steps": args.profile_steps,
                                                 "phase_iterations": args.phase_iterations,
                                                 "phase_repeats": args.phase_repeats}})
    diagnostics = []
    if getattr(args, "counterbalanced_norms", True):
        for group in comparison_groups({"jobs": jobs}):
            if group[0]["fixture"]["kind"] not in ("layernorm-activation", "groupnorm-activation"):
                continue
            by_name = {job["strategy"]["name"]: job for job in group}
            for label, names in (("AB", ("generated-unfused", "generated-fused")),
                                 ("BA", ("generated-fused", "generated-unfused"))):
                for name in names:
                    job = deepcopy(by_name[name])
                    job.update(id=job["id"] + "-order-" + label, diagnostic_order=label,
                               diagnostic="counterbalanced_norm_order")
                    diagnostics.append(job)
    return {"schema_version": 1, "run_id": uuid.uuid4().hex, "source_sha256": source,
            "worker_isolation": args.worker_isolation,
            "fresh_process_per_strategy": args.worker_isolation == "strategy",
            "fresh_process_per_trial": True, "jobs": jobs, "diagnostic_jobs": diagnostics,
            "qualification_policy": {"minimum_fresh_process_paired_trials": 3,
                                     "numerical_oracle": "matching eager recipe",
                                     "positive_cuda_evidence": "every selected candidate and removed standalone work",
                                     "performance_promotion": "manual review of repeated paired timing and drift",
                                     "gpu_performance_qualified": False}}


def worker_groups(plan):
    """Keep a trial's randomized controls sequential on one exclusive GPU."""
    if plan.get("worker_isolation", "strategy") == "strategy":
        return [[job] for job in plan["jobs"]]
    groups = {}
    for job in plan["jobs"]:
        key = job["fixture_id"], job["backend"], job["trial"], job.get("diagnostic_order")
        groups.setdefault(key, []).append(job)
    return list(groups.values())


def comparison_groups(plan):
    """Bound private numerical inputs to one fixture/backend/trial at a time."""
    groups = {}
    for job in plan["jobs"]:
        key = job["fixture_id"], job["backend"], job["trial"], job.get("diagnostic_order")
        groups.setdefault(key, []).append(job)
    return list(groups.values())


def build_fixture(fixture, device, seed):
    """Construct ordinary semantic models; the compiler derives every VJP."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            if fixture.kind == "gemm-activation":
                self.first = nn.Linear(fixture.width, fixture.hidden)
                self.last = nn.Linear(fixture.hidden if fixture.phase == "both" else fixture.width,
                                      fixture.width)
            elif fixture.kind == "gemm-residual":
                self.projection = nn.Linear(fixture.width, fixture.width)
            elif fixture.kind == "layernorm-activation":
                self.norm = nn.LayerNorm(fixture.width)
            elif fixture.kind == "groupnorm-activation":
                self.norm = nn.GroupNorm(fixture.groups, fixture.width)
            elif fixture.kind == "packed-multiply":
                self.scale = nn.Parameter(torch.randn(1, 1, fixture.width))
            elif fixture.kind == "shared-packs":
                self.gate, self.up = nn.Linear(fixture.width, fixture.hidden), nn.Linear(fixture.width, fixture.hidden)

        def activation(self, x):
            if fixture.activation_state == "exact":
                # Direct native ATen calls keep the activation and its VJP
                # exact in both controls. Prepared norms/linears still retain
                # their declared codec recipe; this does not change xhat saves.
                return (torch.ops.aten.silu.default(x) if fixture.activation == "silu" else
                        torch.ops.aten.gelu.default(x, approximate="tanh" if fixture.activation == "gelu-tanh" else "none"))
            return F.silu(x) if fixture.activation == "silu" else F.gelu(
                x, approximate="tanh" if fixture.activation == "gelu-tanh" else "none")

        def forward(self, x):
            if fixture.kind == "gemm-activation":
                if fixture.phase == "forward":
                    return self.activation(self.first(x))
                if fixture.phase == "backward":
                    return self.last(self.activation(x))
                return self.last(self.activation(self.first(x)))
            if fixture.kind == "gemm-residual":
                # One SSA input is both the GEMM source and exact residual.
                # Its gradient must accumulate both native-derived paths.
                return self.projection(x) + x
            if fixture.kind in ("layernorm-activation", "groupnorm-activation"):
                return self.activation(self.norm(x))
            if fixture.kind == "packed-multiply":
                return x * self.scale
            source = self.activation(x)
            return self.gate(source) + self.up(source)

    dtype = torch.bfloat16 if fixture.dtype == "bf16" else torch.float32
    # AMP receives FP32 values; native GroupNorm's CUDA FP32 output policy is preserved.
    model = Model().to(device=device, dtype=torch.float32 if fixture.autocast else dtype)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(fixture.gradient_mask == "all"
                                 or fixture.gradient_mask == "weight" and not name.endswith("bias")
                                 or fixture.gradient_mask == "bias" and name.endswith("bias"))
    shape = ((fixture.batch, fixture.width, *fixture.spatial)
             if fixture.kind == "groupnorm-activation" else (fixture.batch, fixture.tokens, fixture.width))
    if fixture.layout == "strided":
        x = torch.randn((*shape[:-1], shape[-1] * 2), device=device, dtype=next(model.parameters()).dtype)[..., ::2]
    else:
        x = torch.randn(shape, device=device, dtype=next(model.parameters()).dtype)
        if fixture.layout == "channels-last":
            x = x.contiguous(memory_format=torch.channels_last)
    x.requires_grad_(fixture.gradient_mask in ("all", "input"))
    return model, x


def _snapshot(model, x, output, baseline):
    values = {"output": output.detach().to(device="cpu", copy=True),
              "input_grad.0": x.grad.detach().to(device="cpu", copy=True) if x.grad is not None else None}
    for name, parameter in model.named_parameters():
        values[f"parameter_grad.{name}"] = parameter.grad.detach().to(device="cpu", copy=True) if parameter.grad is not None else None
        values[f"parameter_update.{name}"] = parameter.detach().cpu() - baseline[name].cpu()
    return values


def _snapshot_shapes(job, *, include_absent=False):
    fixture = Fixture(**job["fixture"])
    width, hidden = fixture.width, fixture.hidden
    shape = ((fixture.batch, width, *fixture.spatial) if fixture.kind == "groupnorm-activation"
             else (fixture.batch, fixture.tokens, width))
    output_shape = shape
    if fixture.kind == "gemm-activation":
        parameters = {"first.weight": (hidden, width), "first.bias": (hidden,),
                      "last.weight": (width, hidden if fixture.phase == "both" else width), "last.bias": (width,)}
        if fixture.phase == "forward":
            output_shape = (*shape[:-1], hidden)
    elif fixture.kind == "gemm-residual":
        parameters = {"projection.weight": (width, width), "projection.bias": (width,)}
    elif fixture.kind in ("layernorm-activation", "groupnorm-activation"):
        parameters = {"norm.weight": (width,), "norm.bias": (width,)}
    elif fixture.kind == "packed-multiply":
        parameters = {"scale": (1, 1, width)}
    else:
        parameters = {f"{name}.{role}": dimensions for name in ("gate", "up")
                      for role, dimensions in (("weight", (hidden, width)), ("bias", (hidden,)))}
        output_shape = (*shape[:-1], hidden)
    shapes = {"output": output_shape, "input_grad.0": shape,
              **{f"{kind}.{name}": dimensions for kind in ("parameter_grad", "parameter_update")
                 for name, dimensions in parameters.items()}}
    if include_absent:
        mask = fixture.gradient_mask
        if mask not in ("all", "input"):
            shapes["input_grad.0"] = None
        for name in parameters:
            requested = (mask == "all" or mask == "weight" and not name.endswith("bias")
                         or mask == "bias" and name.endswith("bias"))
            unused = fixture.kind == "gemm-activation" and (
                fixture.phase == "forward" and name.startswith("last.") or
                fixture.phase == "backward" and name.startswith("first."))
            if not requested or unused:
                shapes[f"parameter_grad.{name}"] = None
    return shapes


def snapshot_descriptor(job, snapshot):
    """Validate private full numerical inputs; expose scalar metadata only."""
    import torch

    expected = _snapshot_shapes(job, include_absent=True)
    if not isinstance(snapshot, dict) or snapshot.keys() != expected.keys():
        raise ValueError("Numerical snapshot has missing or unexpected tensor names")
    for name in sorted(expected):
        tensor = snapshot[name]
        if tensor is None:
            if expected[name] is not None:
                raise ValueError(f"Required numerical tensor {name} is absent")
            continue
        if (expected[name] is None or not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(expected[name])
                or tensor.device.type != "cpu" or tensor.layout != torch.strided
                or tensor.dtype not in (torch.bfloat16, torch.float32) or tensor.requires_grad):
            raise ValueError(f"Malformed full numerical tensor {name}")
    from analysis.numerical_ipc import snapshot_descriptor as describe

    return describe(_numerical_identity(job), snapshot)


def _numerical_identity(job):
    return {"job_id": job["id"], "fixture_id": job["fixture_id"], "backend": job["backend"],
            "trial": job["trial"], "strategy": job["strategy"]["name"], "source_sha256": job["source_sha256"]}


def _same_job(actual, expected):
    # Planned tuples become lists across the scalar JSON worker boundary.
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def _packed_multiply_oracle(model, x, gradient, fixture):
    """Expose the intended codec recipe independently of unsupported lowering."""
    import torch

    with torch.no_grad():
        decoded_x, packed, reason = _multiplication_coefficient(x, fixture)
        input_gradient = gradient * model.scale.detach()
        # Multiplication rounds in the operand dtype before broadcast reduction.
        product = gradient * decoded_x
        scale_gradient = product.sum(dim=(0, 1), keepdim=True)
    return {"status": "oracle_only", "product_dtype": str(product.dtype),
            "feature_encoded": packed is not None, "feature_rejection_reason": reason,
            "model_owned_scale": "exact",
            "unexpanded_coefficient_shape": list(model.scale.shape),
            "codec_components": [{"shape": list(value.shape), "dtype": str(value.dtype),
                                  "bytes": value.numel() * value.element_size()}
                                 for value in ((packed.data, packed.step, packed.minimum) if packed is not None else ())],
            "input_gradient_shape": list(input_gradient.shape), "scale_gradient_shape": list(scale_gradient.shape),
            "reconstruction_max_abs": float((x.detach() - decoded_x).abs().max()),
            "note": "Diagnostic oracle uses matching codec and product rounding. It is not a compiler implementation."}


def _multiplication_coefficient(x, fixture):
    from deepopt.compiler.autodiff.activation_saving import SavedActivationPolicy, SavedUse
    from deepopt.compiler.ir import TensorMeta, TensorValue
    from deepopt.opt.quantization.activation import _quantize_pack_strided_activation, dequantize_unpack_activation

    policy = SavedActivationPolicy(fixture.compression, fixture.bits, fixture.group_size)
    source = TensorValue("fixture_input", TensorMeta(tuple(x.shape), x.dtype, x.device, tuple(x.stride()), x.requires_grad))
    codec, reason = policy.select(source, SavedUse(role="mul.grad_other_coefficient", allow_approximation=True))
    if codec is None:
        return x.detach(), None, reason
    packed = _quantize_pack_strided_activation(x.detach(), fixture.group_size, bits=fixture.bits)
    reconstructed = dequantize_unpack_activation(packed, dtype=x.dtype).reshape_as(x)
    return reconstructed, packed, None


def _matching_multiply_snapshot(snapshot, model, x, gradient, fixture, baseline):
    """Use the permitted compressed saved operand, retaining exact small state."""
    import torch

    if not model.scale.requires_grad:
        return {"encoded": False, "reason": "coefficient_gradient_not_requested"}
    reconstructed, packed, reason = _multiplication_coefficient(x, fixture)
    if packed is None:
        return {"encoded": False, "reason": reason}
    with torch.no_grad():
        # Model-owned scale is exact. Only the feature coefficient used by
        # its gradient is encoded; preserve BF16 product before sum-to-size.
        expected_gradient = (gradient * reconstructed).sum(dim=(0, 1), keepdim=True).to(model.scale.dtype)
        updated = baseline["scale"].to(x.device).clone()
        updated.add_(expected_gradient, alpha=-0.001)
        snapshot["parameter_grad.scale"] = expected_gradient.cpu()
        snapshot["parameter_update.scale"] = updated.cpu() - baseline["scale"]
    return {"encoded": True, "representation": "packed_saved_operand",
            "recipe": {"bits": fixture.bits, "group_size": fixture.group_size},
            "coefficient_source": "input", "model_owned_scale": "exact", "product_dtype": str(x.dtype),
            "reference": "explicit matching-codec product before broadcast reduction"}


def _phase_timings(model, x, gradient, optimizer, scope, settings, device):
    """Time fresh ordinary forward/backward invocations with explicit boundaries.

    These synchronized diagnostics change host submission overlap. Normal full
    update timing is measured separately without synchronization between phases.
    """
    import torch
    from analysis.measurement import stats, synchronize

    iterations = settings.get("phase_iterations", min(settings["iterations"], 3))
    repeats = settings.get("phase_repeats", 1)
    if not iterations:
        return {phase: {"wall_ms": None, "cuda_stream_ms": None, "mode": "disabled",
                        "iterations": 0, "repeats": repeats}
                for phase in ("forward", "backward")}
    wall = {"forward": [], "backward": []}
    stream = {"forward": [], "backward": []}
    cuda = device.type == "cuda"
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)] if cuda else None
    if cuda:
        for event in events:
            event.record()
        events[-1].synchronize()
    backward_requested = None
    for _ in range(repeats):
        forward_wall = backward_wall = forward_stream = backward_stream = 0.0
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            x.grad = None
            synchronize(torch, device)
            if cuda:
                events[0].record()
            began = perf_counter()
            with scope():
                output = model(x)
            if cuda:
                events[1].record()
            synchronize(torch, device)
            forward_wall += (perf_counter() - began) * 1000
            if cuda:
                forward_stream += events[0].elapsed_time(events[1])
            backward_requested = output.requires_grad
            if backward_requested:
                began = perf_counter()
                if cuda:
                    events[1].record()
                output.backward(gradient)
                if cuda:
                    events[2].record()
                synchronize(torch, device)
                backward_wall += (perf_counter() - began) * 1000
                if cuda:
                    backward_stream += events[1].elapsed_time(events[2])
                optimizer.step()
        count = iterations
        wall["forward"].append(forward_wall / count)
        if backward_requested:
            wall["backward"].append(backward_wall / count)
        if cuda:
            stream["forward"].append(forward_stream / count)
            if backward_requested:
                stream["backward"].append(backward_stream / count)
    return {phase: {"wall_ms": stats(wall[phase]), "cuda_stream_ms": stats(stream[phase]),
                    "mode": "synchronized_fresh_graph_diagnostic",
                    "iterations": iterations, "repeats": repeats,
                    "note": "Phase synchronization changes overlap; use full_update for dispatch performance."}
            for phase in ("forward", "backward")}


def run_worker(job, directory, *, progress_path=None, snapshot_sink=None):
    progress = WorkerProgress(job, directory, progress_path)
    progress.start("identity_and_import")
    source_before = source_identity()
    if source_before != job["source_sha256"]:
        raise RuntimeError("Source changed after the experiment plan was frozen")
    runtime = _installed_runtime()
    installed_before = runtime.installed_identity()
    if job.get("installed_deepopt", installed_before) != installed_before:
        raise RuntimeError("Installed DeepOpt changed after the experiment plan was frozen")
    import torch
    import deepopt
    from deepopt import autocast
    from deepopt.compiler import compilation_report
    from deepopt.config import CompileConfig
    from deepopt.engine.execution.compilation import compile_model
    from deepopt.kernel_policy import kernel_policy_scope
    from deepopt.opt.policy import prepare_fp8_model
    from analysis.measurement import (SavedStorageLifetimes, errors, measure, state_fingerprint,
                                      synchronize, compiler_launch_evidence)

    fixture = Fixture(**{**job["fixture"], "spatial": tuple(job["fixture"]["spatial"])})
    device = torch.device(job["device"])
    torch.compiler.reset()
    compression = {"enabled": fixture.compression, "bits": fixture.bits, "group_size": fixture.group_size}
    policy_config = ({"runtime": {"bf16_policy": fixture.bf16_policy}}
                     if fixture.bf16_policy != "torch_amp" else {})
    progress.start("setup_and_eager_reference")
    with kernel_policy_scope(job["backend"]):
        model, x = build_fixture(fixture, device, job["seed"])
        model = prepare_fp8_model(model, False, activation_compression=compression, config=policy_config)[0]
        if fixture.kind == "packed-multiply":
            # Pure multiplication has no adapter to carry the recipe into
            # capture. Bind the explicit forward-time policy to this model.
            model._deepopt_activation_compression_config = compression
        if fixture.kind in ("layernorm-activation", "groupnorm-activation") and not fixture.compression:
            # Choose the shared normalization implementation for both paired
            # routes even with compression off. Ordinary unprepared LayerNorm
            # selects a different native-library implementation.
            from deepopt.opt.nn.normalization.modules import DOLayerNorm, DOGroupNorm

            adapter = DOLayerNorm if fixture.kind == "layernorm-activation" else DOGroupNorm
            norm_plan = {"fp8": False, "activation_compression": compression}
            if fixture.bf16_policy != "torch_amp":
                norm_plan['bf16_policy'] = {"name": fixture.bf16_policy, "version": 1}
            model.norm = adapter(model.norm, plan=norm_plan)
        reference = deepcopy(model)
        xr = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(x.requires_grad)
        initial_state = state_fingerprint(model, (x,), {})
        rng_state = (torch.get_rng_state(), *((torch.cuda.get_rng_state(device),) if device.type == "cuda" else ()))
        rng_identity = state_fingerprint(torch.nn.Module(), rng_state, {})
        baseline = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                           options=job["strategy"]["options"]))

        def scope():
            policy_options = {"bf16_policy": fixture.bf16_policy} if fixture.bf16_policy != "torch_amp" else {}
            return autocast(device.type, enabled=fixture.autocast or fixture.bf16_policy == "bf16_storage",
                            dtype=torch.bfloat16, fp8=False, activation_compression=compression,
                            config=policy_config, **policy_options)

        with scope():
            expected = reference(xr)
        backward_required = expected.requires_grad
        torch.manual_seed(job["seed"] + 1000)
        gradient = torch.randn_like(expected)
        if expected.requires_grad:
            expected.backward(gradient)
            reference_optimizer.step()
        expected_snapshot = _snapshot(reference, xr, expected, baseline)
        multiplication_recipe = (_matching_multiply_snapshot(expected_snapshot, reference, xr, gradient,
                                 fixture, baseline) if fixture.kind == "packed-multiply" else None)
        del expected

        progress.start("cold_compile_and_first_update")
        synchronize(torch, device)
        began = perf_counter()
        with scope():
            actual = model(x)
        if actual.requires_grad:
            # Deliberately outside the originating precision scope.
            actual.backward(gradient)
            optimizer.step()
        synchronize(torch, device)
        cold_ms = (perf_counter() - began) * 1000
        progress.start("numerical_validation")
        actual_snapshot = _snapshot(model, x, actual, baseline)
        numerical = errors(actual_snapshot, expected_snapshot, 0.02 if fixture.dtype == "bf16" else 0.0001,
                           0.02 if fixture.dtype == "bf16" else 0.0001)
        numerical["matching_dtypes"] = all(
            value is None or expected_snapshot.get(name) is not None and value.dtype == expected_snapshot[name].dtype
            for name, value in actual_snapshot.items())
        numerical["passed"] &= numerical["matching_dtypes"]
        del actual
        report = compilation_report(model).as_dict()
        native_ownership = bool((report["native_autodiff_operations"] or not backward_required)
                                and not report["legacy_backward_operations"]
                                and not report["runtime_fallbacks"] and all(
                                    not graph["eager_autograd_operations"] for graph in report["graphs"]))
        # Remove eager-oracle GPU state before timing or allocator diagnostics.
        del reference, reference_optimizer, xr, expected_snapshot

        def reset():
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(baseline[name].to(device))
            optimizer.zero_grad(set_to_none=True)
            x.grad = None

        def update():
            optimizer.zero_grad(set_to_none=True)
            x.grad = None
            with scope():
                output = model(x)
            if output.requires_grad:
                output.backward(gradient)
                optimizer.step()
            return output

        reset()
        settings = job["measurement"]
        progress.start("full_update_timing")
        timing = measure(update, torch, device, settings["warmup"], settings["iterations"], settings["repeats"])
        reset()
        progress.start("phase_timing")
        phase_timing = _phase_timings(model, x, gradient, optimizer, scope, settings, device)
        reset()
        progress.start("memory_diagnostics")
        persistent = tuple(model.parameters()) + tuple(model.buffers())
        lifetimes = SavedStorageLifetimes(persistent)
        saved = []

        def save(value):
            saved.append({"shape": list(value.shape), "strides": list(value.stride()), "dtype": str(value.dtype),
                          "logical_bytes": value.numel() * value.element_size()})
            return lifetimes.pack(value)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        with torch.autograd.graph.saved_tensors_hooks(save, lifetimes.unpack):
            with scope():
                output = model(x)
            after_forward = lifetimes.snapshot()
            if output.requires_grad:
                output.backward(gradient)
                optimizer.step()
        del output
        synchronize(torch, device)
        memory = {"saved_tensors": saved, "after_forward": after_forward, "after_backward": lifetimes.snapshot(),
                  "baseline_allocated_bytes": allocated,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if allocated is not None else None,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if allocated is not None else None,
                  "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - allocated if allocated is not None else None,
                  "decoded_scratch_intervals": None,
                  "scratch_note": "Allocator peak and saved-storage lifetimes are measured; internal scratch intervals are not instrumented."}

        reset()
        progress.start("profiling")
        profile = None
        if settings["profile_steps"]:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as capture:
                for _ in range(settings["profile_steps"]):
                    update()
                    capture.step()
                synchronize(torch, device)
            trace = directory / "trace.json"
            capture.export_chrome_trace(str(trace))
            profile = {"trace": str(trace), "compiler_regions": compiler_launch_evidence(
                capture.events(), settings["profile_steps"], trace=trace)}
        # The shared evidence gate matches candidate ranges and symbols. CPU
        # records or a bias-reduction launch can never qualify a CUDA fusion.
        from analysis.measurement import fusion_candidate_evidence
        progress.start("final_validation_and_identity")
        evidence = fusion_candidate_evidence(report, profile)
        finite_after_timing = all(bool(torch.isfinite(value).all()) for value in model.parameters())
        reset()
        oracle = _packed_multiply_oracle(model, x, gradient, fixture) if fixture.kind == "packed-multiply" and fixture.compression else None
        source_after = source_identity()
        installed_after = runtime.installed_identity()
        result = {"status": "ok", "job": job, "source_before_sha256": source_before,
                  "source_after_sha256": source_after,
                  "source_unchanged": source_before == source_after and installed_before == installed_after,
                  "installed_deepopt": installed_before, "installed_deepopt_after": installed_after,
                  "initial_state_sha256": initial_state,
                  "rng_baseline_sha256": rng_identity,
                  "gradient_sha256": state_fingerprint(torch.nn.Module(), (gradient,), {}),
                  "numerical": numerical, "native_autodiff_ownership": native_ownership,
                  "comparison_input": snapshot_descriptor(job, actual_snapshot),
                  "native_autodiff_required": backward_required,
                  "finite_after_timing": finite_after_timing,
                  "output_dtype": str(actual_snapshot["output"].dtype),
                  "input_signature": {"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype)},
                  "cold_compile_and_first_update_ms": cold_ms,
                  "timing": {"full_update": timing, **phase_timing},
                  "memory": memory, "compiler_report": report, "profile": profile,
                  "candidate_evidence": evidence, "packed_multiply_oracle": oracle,
                  "multiplication_recipe": multiplication_recipe,
                  "fixture_contract": {"activation_state": fixture.activation_state,
                      "shared_projection_cse": "unsupported_no_explicit_pack_operations" if fixture.kind == "shared-packs" else None,
                      "note": "Exact activation keeps normalization/linear compression unchanged; recipe activation includes its configured coefficient codec."},
                  "optimizer": {"name": "SGD", "lr": 0.001, "momentum": 0, "weight_decay": 0},
                  "gpu_performance_qualified": False,
                  "worker_process_id": os.getpid(),
                  "worker_isolation": job.get("worker_isolation", "strategy"),
                  "runtime": {"torch": torch.__version__, "device": str(device), "deepopt_source": deepopt.__file__,
                              "cuda": torch.version.cuda,
                              "capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    torch.compiler.reset()
    if snapshot_sink is not None:
        progress.start("numerical_comparison_transfer")
        snapshot_sink(job, actual_snapshot)
    result["lifecycle_ms"] = progress.finish()
    return result


def _validate_success_result(row):
    if not isinstance(row, dict) or row.get("status") != "ok":
        raise ValueError("Numerical comparison requires a successful strategy result")
    numerical = row.get("numerical")
    if not isinstance(numerical, dict) or type(numerical.get("passed")) is not bool:
        raise ValueError("Strategy numerical result must contain a boolean pass/fail decision")
    for name in ("source_unchanged", "native_autodiff_ownership", "finite_after_timing"):
        if type(row.get(name)) is not bool:
            raise ValueError(f"Strategy result {name} must be boolean")
    for name in ("initial_state_sha256", "rng_baseline_sha256", "gradient_sha256",
                 "source_before_sha256", "source_after_sha256"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f"Strategy identity {name} is missing or malformed")
    for name in ("installed_deepopt", "installed_deepopt_after", "comparison_input", "compiler_report"):
        if not isinstance(row.get(name), dict):
            raise ValueError(f"Strategy result {name} must be a scalar descriptor mapping")
    reports = (row["compiler_report"], *row["compiler_report"].get("graphs", ())) if isinstance(
        row["compiler_report"].get("graphs", ()), (list, tuple)) else (None,)
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("Strategy compiler graph reports must be mappings")
        for name in ("fusion_candidates", "pack_sharing_decisions"):
            items = report.get(name, ())
            if not isinstance(items, (list, tuple)) or any(not isinstance(item, dict) for item in items):
                raise ValueError(f"Strategy compiler collection {name} is malformed")
    evidence = row.get("candidate_evidence")
    if (not isinstance(evidence, dict) or type(evidence.get("selected_candidates")) is not int
            or evidence["selected_candidates"] < 0 or type(evidence.get("all_selected_observed")) is not bool):
        raise ValueError("Strategy candidate evidence is malformed")
    try:
        median = row["timing"]["full_update"]["wall_ms"]["median"]
    except (KeyError, TypeError) as error:
        raise ValueError("Strategy throughput timing is missing") from error
    if type(median) not in (int, float) or not isfinite(median) or median <= 0:
        raise ValueError("Strategy throughput timing must be finite and positive")


def compare_pair(unfused, fused, *, snapshot_inputs=None, toggle="fuse_deepopt_activations", transformation="kernel_fusion"):
    """Correctness, physical observation and performance are separate decisions."""
    import torch
    from analysis.measurement import errors, fusion_candidate_evidence

    a, b = unfused["job"], fused["job"]
    try:
        _validate_success_result(unfused)
        _validate_success_result(fused)
    except ValueError as error:
        return {"fixture_id": a["fixture_id"], "backend": a["backend"], "trial": a["trial"],
                "candidate_toggle": toggle, "transformation": transformation, "status": "failed",
                "correctness_passed": False, "reason": "invalid_strategy_result_schema",
                "error": str(error), "cuda_fusion_observed": False, "gpu_performance_qualified": False}
    matching = (a["fixture"] == b["fixture"] and a["backend"] == b["backend"]
                and a["seed"] == b["seed"] and a["measurement"] == b["measurement"])
    ao, bo = dict(a["strategy"]["options"]), dict(b["strategy"]["options"])
    disabled, enabled = ao.pop(toggle, None), bo.pop(toggle, None)
    same_backend_control = matching and ao == bo and disabled is False and enabled is True
    identities = all(unfused.get(key) and unfused.get(key) == fused.get(key)
                     for key in ("initial_state_sha256", "rng_baseline_sha256", "gradient_sha256",
                                 "source_before_sha256", "source_after_sha256",
                                 "installed_deepopt", "installed_deepopt_after"))
    try:
        if not isinstance(snapshot_inputs, (tuple, list)) or len(snapshot_inputs) != 2:
            raise ValueError("Full in-memory fused/unfused comparison inputs are required")
        first, second = snapshot_inputs
        for row, snapshot in ((unfused, first), (fused, second)):
            if row.get("comparison_input") != snapshot_descriptor(row["job"], snapshot):
                raise ValueError("Comparison input does not match its strategy result")
        numerical = errors(second, first, 0.02, 0.02)
        numerical["matching_dtypes"] = all(first[name] is None and second[name] is None or
            first[name] is not None and second[name] is not None and first[name].dtype == second[name].dtype
            for name in first)
        numerical["passed"] &= numerical["matching_dtypes"]
    except (ValueError, TypeError, KeyError) as error:
        numerical = {"passed": False, "reason": "missing_or_invalid_comparison_input",
                     "error": f"{type(error).__name__}: {error}"}
    correct = bool(same_backend_control and identities and all(row["numerical"]["passed"] is True
                   and row["source_unchanged"] is True and row["native_autodiff_ownership"] is True
                   and row["finite_after_timing"] is True for row in (unfused, fused))
                   and numerical["passed"])
    evidence = fused["candidate_evidence"]
    selected = evidence.get("selected_candidates", 0)
    observed = evidence.get("all_selected_observed") is True and bool(selected)
    if transformation == "pack_cse":
        decisions = fused["compiler_report"].get("pack_sharing_decisions", [])
        if not decisions:
            decisions = [item for graph in fused["compiler_report"].get("graphs", [])
                         for item in graph.get("pack_sharing_decisions", [])]
        encodings = {item["encoding_id"]: {**item, "candidate_id": item["encoding_id"]}
                     for item in decisions if item.get("selected") is True and item.get("encoding_id")}
        evidence = fusion_candidate_evidence({"fusion_candidates": list(encodings.values())}, fused.get("profile"))
        selected = evidence["selected_candidates"]
        observed = evidence["all_selected_observed"] is True
        evidence["decisions"] = decisions
        evidence["transformation"] = "pack_cse"
        evidence["kernel_fusion"] = False
    observed &= torch.device(a["device"]).type == "cuda" and torch.device(b["device"]).type == "cuda"
    status = "observed" if observed and correct else "unqualified" if selected else "unsupported"
    reason = None if observed and correct else "selected_candidate_without_cuda_evidence" if selected else "no_physical_candidate_selected"
    control_selected = (unfused["candidate_evidence"].get("selected_candidates", 0)
                        if transformation == "kernel_fusion" else 0)
    # An independent producer fusion may remain enabled in a CSE comparison,
    # and a CSE pass may be enabled in a producer-fusion comparison.
    if transformation == "pack_cse":
        reason = (None if observed else "shared_pack_without_encode_launch_attribution"
                  if selected else "no_compatible_explicit_pack_operations")
    if control_selected:
        correct, status, reason = False, "invalid_control", "unfused_control_selected_a_candidate"
    if not correct:
        status, reason = "failed", "numerical_ownership_or_pair_identity_failure" if not control_selected else reason
        if numerical.get("reason") == "missing_or_invalid_comparison_input":
            reason = numerical["reason"]
    return {"fixture_id": a["fixture_id"], "backend": a["backend"], "trial": a["trial"],
            "transformation": transformation, "candidate_toggle": toggle,
            "pair_id": sha256(json.dumps({"fixture": a["fixture_id"], "options": ao,
                                           "transformation": transformation, "toggle": toggle},
                                          sort_keys=True).encode()).hexdigest()[:24],
            "control_options": ao,
            "status": status, "reason": reason, "correctness_passed": correct,
            "same_backend_control": same_backend_control, "same_frozen_identity": bool(identities),
            "rejection_reasons": sorted({item["reason"] for item in fused["compiler_report"].get("fusion_candidates", [])
                                          if not item.get("selected") and isinstance(item.get("reason"), str)}),
            "numerical_fused_vs_unfused": numerical, "cuda_fusion_observed": observed,
            "fused_candidate_evidence": evidence,
            "wall_speedup_generated_pair": unfused["timing"]["full_update"]["wall_ms"]["median"] /
                                           fused["timing"]["full_update"]["wall_ms"]["median"],
            "gpu_performance_qualified": False}


def _pair_specs(kind):
    toggle = "fuse_packed_multiply" if kind == "packed-multiply" else "fuse_deepopt_activations"
    pairs = [("generated-unfused", "generated-fused", toggle, "kernel_fusion")]
    if kind == "shared-packs":
        pairs += [("generated-unfused", "generated-unfused-cse", "share_compatible_packs", "pack_cse"),
                  ("generated-fused", "generated-fused-cse", "share_compatible_packs", "pack_cse"),
                  ("generated-unfused-cse", "generated-fused-cse", "fuse_deepopt_activations", "kernel_fusion")]
    return tuple(pairs)


class NumericalComparisons:
    """Keep full inputs privately for one fixture/trial and release at last use."""

    def __init__(self, entries):
        self.entries = entries
        first = entries[0]["job"]
        self.identity = first["fixture_id"], first["backend"], first["trial"]
        self.jobs = {entry["job"]["strategy"]["name"]: entry["job"] for entry in entries}
        expected = {strategy["name"] for strategy in strategies(first["backend"], Fixture(**first["fixture"]))}
        if len(self.jobs) != len(entries) or self.jobs.keys() != expected or any(
                (job["fixture_id"], job["backend"], job["trial"]) != self.identity for job in self.jobs.values()):
            raise ValueError("Comparison worker requires every strategy for exactly one fixture/backend/trial")
        self.specs = _pair_specs(first["fixture"]["kind"])
        self.snapshots, self.descriptors, self.rows, self.comparisons = {}, {}, {}, {}
        self.snapshot_bytes, self.peak_snapshot_bytes = 0, 0

    def add_snapshot(self, job, snapshot):
        name = job["strategy"]["name"]
        if name not in self.jobs or not _same_job(self.jobs[name], job) or name in self.snapshots or name in self.rows:
            raise ValueError("Unexpected, duplicate or late numerical comparison input")
        descriptor = snapshot_descriptor(job, snapshot)
        self.snapshots[name] = snapshot
        self.descriptors[name] = descriptor
        self.snapshot_bytes += descriptor["total_bytes"]
        self.peak_snapshot_bytes = max(self.peak_snapshot_bytes, self.snapshot_bytes)

    def add_result(self, row):
        job = row["job"]
        name = job["strategy"]["name"]
        if name not in self.jobs or not _same_job(self.jobs[name], job) or name in self.rows:
            raise ValueError("Unexpected or duplicate comparison strategy result")
        if row.get("status") == "ok":
            try:
                _validate_success_result(row)
                if name not in self.snapshots or row.get("comparison_input") != self.descriptors[name]:
                    raise ValueError("Comparison input descriptor differs from its strategy result")
            except (ValueError, TypeError, KeyError) as error:
                row.clear()
                row.update(status="error", job=job, reason=f"missing_or_invalid_comparison_input: {error}")
        self.rows[name] = row
        for spec in self.specs:
            first, second, toggle, transformation = spec
            if spec in self.comparisons or first not in self.rows or second not in self.rows:
                continue
            if any(self.rows[name].get("status") != "ok" for name in (first, second)):
                comparison = {"fixture_id": self.identity[0], "backend": self.identity[1], "trial": self.identity[2],
                              "transformation": transformation, "candidate_toggle": toggle,
                              "status": "failed", "correctness_passed": False, "reason": "worker_failed"}
            else:
                try:
                    comparison = compare_pair(self.rows[first], self.rows[second],
                        snapshot_inputs=(self.snapshots.get(first), self.snapshots.get(second)),
                        toggle=toggle, transformation=transformation)
                except Exception as error:
                    comparison = {"fixture_id": self.identity[0], "backend": self.identity[1], "trial": self.identity[2],
                                  "transformation": transformation, "candidate_toggle": toggle,
                                  "status": "failed", "correctness_passed": False,
                                  "reason": "numerical_comparison_failed", "error": f"{type(error).__name__}: {error}"}
            comparison["comparison_jobs"] = [self.jobs[first]["id"], self.jobs[second]["id"]]
            if job.get("diagnostic_order"):
                comparison["diagnostic_order"] = job["diagnostic_order"]
            self.comparisons[spec] = comparison
        # Library route snapshots need no cross-route comparison; every route
        # has already compared all outputs/gradients/updates with its eager oracle.
        for name in list(self.snapshots):
            needed = any(name in spec[:2] and spec not in self.comparisons for spec in self.specs)
            if name in self.rows and (not needed or self.rows[name].get("status") != "ok"):
                self.snapshot_bytes -= self.descriptors[name]["total_bytes"]
                del self.snapshots[name]
                del self.descriptors[name]

    def finish(self):
        for name, job in self.jobs.items():
            if name not in self.rows:
                self.add_result({"status": "error", "job": job, "reason": "missing_strategy_result"})
        for spec in self.specs:
            if spec not in self.comparisons:
                raise ValueError("A required direct numerical comparison was not completed")
        library = []
        if "library-control" in self.rows and all(self.rows[name].get("status") == "ok"
                                                 for name in ("generated-unfused", "library-control")):
            generated, native = self.rows["generated-unfused"], self.rows["library-control"]
            library.append({"fixture_id": self.identity[0], "backend": self.identity[1], "trial": self.identity[2],
                "comparison": "generated_gemm_vs_library_gemm", "isolates_fusion": False,
                "wall_speedup_generated_vs_library": native["timing"]["full_update"]["wall_ms"]["median"] /
                                                      generated["timing"]["full_update"]["wall_ms"]["median"],
                "gpu_performance_qualified": False})
        self.snapshots.clear()
        self.descriptors.clear()
        self.snapshot_bytes = 0
        return {"comparisons": [self.comparisons[spec] for spec in self.specs],
                "library_route_comparisons": library,
                "numerical_storage": {"mode": "private_group_ram", "persisted_tensor_bytes": 0,
                    "peak_snapshot_bytes": self.peak_snapshot_bytes, "remaining_snapshot_bytes": 0,
                    "scope": "one fixture/backend/trial", "full_tensor_comparisons": True}}


def _validate_group_report(jobs, rows, report):
    """Never substitute missing scalar pair evidence for a full comparison."""
    if not isinstance(report, dict) or not isinstance(report.get("comparisons"), list):
        raise ValueError("Comparison worker returned no scalar pair reports")
    planned = {job["strategy"]["name"]: job for job in jobs}
    completed = {row["job"]["id"]: row for row in rows}
    if len(completed) != len(jobs) or completed.keys() != {job["id"] for job in jobs}:
        raise ValueError("Comparison worker strategy identities are incomplete")
    first = jobs[0]
    expected = {(planned[a]["id"], planned[b]["id"], toggle, transformation)
                for a, b, toggle, transformation in _pair_specs(first["fixture"]["kind"])}
    seen = set()
    for comparison in report["comparisons"]:
        if (not isinstance(comparison, dict) or type(comparison.get("correctness_passed")) is not bool
                or not isinstance(comparison.get("comparison_jobs"), list) or len(comparison["comparison_jobs"]) != 2
                or comparison.get("fixture_id") != first["fixture_id"] or comparison.get("backend") != first["backend"]
                or comparison.get("trial") != first["trial"]
                or comparison.get("diagnostic_order") != first.get("diagnostic_order")):
            raise ValueError("Malformed or misattributed scalar pair report")
        key = (*comparison["comparison_jobs"], comparison.get("candidate_toggle"), comparison.get("transformation"))
        if key not in expected or key in seen:
            raise ValueError("Missing, duplicate or unexpected direct comparison")
        seen.add(key)
        if comparison["correctness_passed"] and (
                comparison.get("numerical_fused_vs_unfused", {}).get("passed") is not True
                or comparison.get("same_backend_control") is not True or comparison.get("same_frozen_identity") is not True
                or type(comparison.get("cuda_fusion_observed")) is not bool
                or type(comparison.get("wall_speedup_generated_pair")) not in (int, float)
                or not isfinite(comparison["wall_speedup_generated_pair"])
                or comparison["wall_speedup_generated_pair"] <= 0
                or any(completed[job_id].get("status") != "ok" for job_id in key[:2])):
            raise ValueError("Successful pair report lacks complete numerical/strategy evidence")
    storage = report.get("numerical_storage", {})
    if (seen != expected or storage.get("persisted_tensor_bytes") != 0
            or storage.get("remaining_snapshot_bytes") != 0 or storage.get("full_tensor_comparisons") is not True):
        raise ValueError("Direct comparisons are missing or their tensor lifetime contract failed")
    return report["comparisons"]


def paired_trial_summary(comparisons):
    from analysis.measurement import stats

    groups = {}
    for row in comparisons:
        if not row.get("correctness_passed"):
            continue
        key = row["fixture_id"], row["backend"], row["transformation"], row["candidate_toggle"], row["pair_id"]
        groups.setdefault(key, []).append(row)
    return [{"fixture_id": key[0], "backend": key[1], "transformation": key[2], "candidate_toggle": key[3],
             "pair_id": key[4], "control_options": rows[0]["control_options"],
             "fresh_process_paired_trials": len({row["trial"] for row in rows}),
             "minimum_three_trials_met": len({row["trial"] for row in rows}) >= 3,
             "wall_speedup_generated_pair": stats([row["wall_speedup_generated_pair"] for row in rows]),
             "all_trials_observed": all(row["cuda_fusion_observed"] for row in rows),
             "gpu_performance_qualified": False,
             "note": "Paired ratios retain trial variability. Near-drift differences require more trials and review."}
            for key, rows in groups.items()]


def counterbalanced_summary(jobs, rows, comparisons):
    """Do not count AB/BA repetitions as additional independent trials."""
    from analysis.measurement import stats

    completed = {row["job"]["id"]: row for row in rows}
    observed = {(row["fixture_id"], row["backend"], row["trial"], row.get("diagnostic_order")): row
                for row in comparisons}
    groups = {}
    for job in jobs:
        key = job["fixture_id"], job["backend"]
        groups.setdefault(key, {}).setdefault(job["trial"], []).append(job)
    summaries = []
    for (fixture, backend), trials in groups.items():
        records = []
        for trial, planned in trials.items():
            orders = {order: observed.get((fixture, backend, trial, order)) for order in ("AB", "BA")}
            matching = len(planned) == 4
            for strategy in ("generated-unfused", "generated-fused"):
                pair = [completed.get(job["id"], {}) for job in planned if job["strategy"]["name"] == strategy]
                matching &= len(pair) == 2 and all(row.get("status") == "ok" for row in pair)
                if len(pair) == 2:
                    matching &= all(pair[0].get(name) is not None and pair[0].get(name) == pair[1].get(name)
                                    for name in ("initial_state_sha256", "rng_baseline_sha256", "gradient_sha256",
                                                 "source_before_sha256", "source_after_sha256", "installed_deepopt"))
            correct = matching and all(row and row.get("correctness_passed") is True for row in orders.values())
            cuda = correct and all(row.get("cuda_fusion_observed") is True for row in orders.values())
            ratios = {order: row.get("wall_speedup_generated_pair") if row else None for order, row in orders.items()}
            records.append({"trial": trial, "matching_state_across_orders": bool(matching),
                "correctness_passed": bool(correct), "cuda_observed": bool(cuda), "ratios": ratios,
                "both_orders_faster": bool(cuda and all(value > 1 for value in ratios.values()))})
        valid = len(records) >= 3 and all(row["cuda_observed"] for row in records)
        summaries.append({"fixture_id": fixture, "backend": backend, "trials": records,
            "minimum_three_trials_met": len(records) >= 3, "qualified_execution": bool(valid),
            "order_stable_latency_gain": bool(valid and all(row["both_orders_faster"] for row in records)),
            "by_order": {order: stats([row["ratios"][order] for row in records
                                       if row["correctness_passed"] and row["ratios"][order] is not None])
                         for order in ("AB", "BA")}, "gpu_performance_qualified": False})
    return summaries


def measured_cost_decisions(rows, comparisons, order_diagnostics, *, objective="latency", capacity_bytes=None):
    """Recommend a legal measured route; never change force controls or promote it automatically."""
    from statistics import median

    if objective not in ("latency", "capacity"):
        raise ValueError("Unknown fusion cost objective")
    groups = {}
    for row in rows:
        job = row["job"]
        groups.setdefault((job["fixture_id"], job["backend"]), []).append(row)
    decisions = []
    for (fixture_id, backend), group in groups.items():
        fixture = group[0]["job"]["fixture"]
        pairs = [p for p in comparisons if p["fixture_id"] == fixture_id and p["backend"] == backend
                 and p.get("transformation") == "kernel_fusion"]
        reasons = []
        if len({p["trial"] for p in pairs}) < 3:
            reasons.append("insufficient_independent_trials")
        if not pairs or any(p.get("correctness_passed") is not True or p.get("cuda_fusion_observed") is not True for p in pairs):
            reasons.append("missing_matching_recipe_or_candidate_cuda_evidence")
        order = next((d for d in order_diagnostics if d["fixture_id"] == fixture_id and d["backend"] == backend), None)
        if fixture["kind"] in ("layernorm-activation", "groupnorm-activation") and (
                not order or order["qualified_execution"] is not True or
                objective == "latency" and order["order_stable_latency_gain"] is not True):
            reasons.append("counterbalanced_order_evidence_missing_or_unstable")
        routes = {}
        for name in ("generated-fused", "generated-unfused", "library-control"):
            selected = [row for row in group if row["job"]["strategy"]["name"] == name]
            if not selected:
                continue
            valid = len({row["job"]["trial"] for row in selected}) >= 3 and all(
                row.get("status") == "ok" and row.get("numerical", {}).get("passed") is True and
                row.get("source_unchanged") is True and row.get("native_autodiff_ownership") is True and
                row.get("finite_after_timing") is True for row in selected)
            for row in selected:
                control = next((item for item in group if item["job"]["trial"] == row["job"]["trial"] and
                                item["job"]["strategy"]["name"] == "generated-unfused"), None)
                valid &= control is not None and all(row.get(field) is not None and row.get(field) == control.get(field)
                    for field in ("initial_state_sha256", "rng_baseline_sha256", "gradient_sha256",
                                  "installed_deepopt", "installed_deepopt_after", "source_before_sha256", "source_after_sha256"))
            peaks = [row.get("memory", {}).get("peak_extra_allocated_bytes") for row in selected]
            saved = [row.get("memory", {}).get("after_forward", {}).get("live_nonpersistent_bytes") for row in selected]
            wall = [row.get("timing", {}).get("full_update", {}).get("wall_ms", {}).get("median") for row in selected]
            finite_wall = all(type(value) in (int, float) and isfinite(value) and value > 0 for value in wall)
            valid &= finite_wall
            peak = max(peaks) if peaks and all(type(value) is int and value >= 0 for value in peaks) else None
            routes[name] = {"legal_numerical_evidence": bool(valid), "median_wall_ms": median(wall) if finite_wall else None,
                            "maximum_extra_allocated_bytes": peak,
                            "maximum_nonpersistent_saved_bytes": max(saved) if saved and all(
                                type(value) is int and value >= 0 for value in saved) else None,
                            "scratch_intervals": "unknown_uninstrumented",
                            "timing_scope": "complete_update_including_codec_and_layout_work"}
        if objective == "capacity" and (type(capacity_bytes) is not int or capacity_bytes < 1):
            reasons.append("capacity_budget_required")
        legal = {name: evidence for name, evidence in routes.items() if evidence["legal_numerical_evidence"] and
                 (objective != "capacity" or evidence["maximum_extra_allocated_bytes"] is not None and
                  capacity_bytes is not None and evidence["maximum_extra_allocated_bytes"] <= capacity_bytes)}
        if not legal:
            reasons.append("no_measured_route_meets_objective")
        selected = min(legal, key=lambda name: legal[name]["median_wall_ms"]) if legal and not reasons else None
        decisions.append({"fixture_id": fixture_id, "backend": backend, "objective": objective,
            "capacity_budget_bytes": capacity_bytes, "recommendation": selected, "rejection_reasons": reasons,
            "routes": routes, "signature": {"fixture": fixture, "device": group[0]["job"]["device"],
                "runtime": group[0].get("runtime"),
                "installed_deepopt": group[0].get("installed_deepopt"),
                "control_options": pairs[0].get("control_options") if pairs else None},
            "force_controls_preserved": True, "automatic_default_changed": False, "gpu_performance_qualified": False})
    return decisions


def _cleanup_strategy(device):
    """Release the previous strategy outside every timing interval."""
    import gc

    torch = sys.modules.get("torch")
    if torch is not None:
        torch.compiler.reset()
    gc.collect()
    if torch is not None and torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def execute_worker_jobs(entries, *, progress_path=None, snapshot_sink=None, result_sink=None):
    """Run controls sequentially; persist failures without dropping other controls."""
    from compiler.benchmark_bf16 import write_json

    results = []
    cleanup_failure = None
    for entry in entries:
        job, result_path = entry["job"], Path(entry["result_path"])
        directory = result_path.parent
        began = perf_counter()
        if cleanup_failure is not None:
            result = {"status": "error", "job": job, "reason": "not_run_after_strategy_cleanup_failure",
                      "blocking_job_id": cleanup_failure}
        else:
            try:
                result = run_worker(job, directory, progress_path=progress_path, snapshot_sink=snapshot_sink)
            except Exception as error:
                result = {"status": "error", "job": job, "reason": f"{type(error).__name__}: {error}"}
            began = perf_counter()
            try:
                _cleanup_strategy(job["device"])
            except Exception as error:
                result.update(status="error", cleanup_error=f"{type(error).__name__}: {error}")
                cleanup_failure = job["id"]
        result.update(directory=str(directory), worker_process_id=os.getpid(),
                      worker_isolation=job.get("worker_isolation", "strategy"),
                      cleanup_after_strategy_ms=(perf_counter() - began) * 1000)
        if result_sink is not None:
            result_sink(result)
        write_json(result_path, result)
        if result["status"] != "ok":
            state = {"phase": "failed", "job_id": job["id"], "status": "error",
                     "elapsed_seconds": result.get("lifecycle_ms", {}).get("total", 0) / 1000,
                     "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                     "reason": result.get("reason", result.get("cleanup_error"))}
            write_json(directory / "worker-phase.json", state)
            if progress_path is not None:
                write_json(progress_path, state)
        results.append(result)
    return results


def _isolated_strategy(entry, comparisons, *, progress_path=None):
    """Fresh strategy process, with bounded raw IPC outside GPU measurement."""
    from math import prod
    from analysis.numerical_ipc import IdleTimeout, NumericalSnapshotBroker
    from compiler.benchmark_bf16 import write_json

    job, result_path = entry["job"], Path(entry["result_path"])
    directory = result_path.parent
    budget = sum(prod(shape) * 4 for shape in _snapshot_shapes(job).values())
    received, failures = [], []
    stopped = threading.Event()
    with NumericalSnapshotBroker(max_bytes=max(1, budget), idle_timeout=1) as broker:
        def receive():
            while not stopped.is_set():
                try:
                    received.append(broker.receive(expected_identity=_numerical_identity(job),
                                                   expected_shapes=_snapshot_shapes(job, include_absent=True)))
                    return
                except IdleTimeout:
                    continue
                except Exception as error:
                    failures.append(f"{type(error).__name__}: {error}")
                    return

        environment = _installed_runtime().child_environment(ROOT, identity=job["installed_deepopt"])
        environment.update(broker.environment)
        command = [sys.executable, str(Path(__file__).resolve()), "--fusion-worker",
                   str(directory / "job.json"), str(result_path)]
        if progress_path is not None:
            command += ["--worker-progress", str(progress_path)]
        receiver = threading.Thread(target=receive, name="fusion-numerical-receiver", daemon=True)
        receiver.start()
        returncode = 1
        launch_error = None
        try:
            with (directory / "worker.log").open("w") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
                returncode = completed.returncode
        except OSError as error:
            launch_error = f"{type(error).__name__}: {error}"
        finally:
            stopped.set()
            broker.close()
            receiver.join(timeout=2)
        try:
            result = json.loads(result_path.read_text())
            if not isinstance(result, dict) or not _same_job(result.get("job"), job):
                raise ValueError("Strategy result identity does not match its planned job")
        except (OSError, ValueError, TypeError) as error:
            result = {"status": "error", "job": job, "reason": f"missing_or_invalid_strategy_result: {error}"}
        if launch_error is not None:
            result.update(status="error", reason=f"strategy_launch_failed: {launch_error}")
        elif returncode and result.get("status") == "ok":
            result.update(status="error", reason=f"worker_exit_{returncode}_after_result")
        if result.get("status") == "ok":
            if receiver.is_alive() or failures or len(received) != 1:
                result.update(status="error", reason="missing_or_invalid_comparison_input",
                              numerical_transfer_errors=failures)
            else:
                try:
                    payload = received.pop()
                    if payload["descriptor"] != result.get("comparison_input"):
                        raise ValueError("Numerical IPC descriptor differs from the strategy result")
                    comparisons.add_snapshot(job, payload["tensors"])
                except (ValueError, TypeError, KeyError) as error:
                    result.update(status="error", reason=f"missing_or_invalid_comparison_input: {error}")
        comparisons.add_result(result)
        write_json(result_path, result)
        return result


def execute_comparison_group(entries, *, isolation, progress_path=None):
    runtime = _installed_runtime()
    with runtime.identity_session(ROOT):
        return _execute_comparison_group(entries, isolation=isolation, progress_path=progress_path)


def _execute_comparison_group(entries, *, isolation, progress_path=None):
    comparisons = NumericalComparisons(entries)
    if isolation == "trial":
        results = execute_worker_jobs(entries, progress_path=progress_path,
            snapshot_sink=comparisons.add_snapshot, result_sink=comparisons.add_result)
    else:
        results = [_isolated_strategy(entry, comparisons, progress_path=progress_path) for entry in entries]
    return {"results": results, **comparisons.finish(), "comparison_worker_process_id": os.getpid()}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fixtures", nargs="+", choices=FIXTURES, required=True)
    result.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    result.add_argument("--activation", choices=("silu", "gelu", "gelu-tanh"), default="silu")
    result.add_argument("--activation-state", choices=("recipe", "exact"), default="recipe",
                        help="Follow the activation codec recipe, or use exact native ATen activation/VJP while retaining prepared norm/linear compression")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--backends", nargs="+", choices=("triton", "tilelang"), default=["triton"])
    result.add_argument("--batch", type=int, default=2)
    result.add_argument("--tokens", type=int, default=17)
    result.add_argument("--width", type=int, default=64)
    result.add_argument("--hidden", type=int, default=96)
    result.add_argument("--spatial", type=int, nargs=2, default=[8, 8])
    result.add_argument("--groups", type=int, default=8)
    result.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    result.add_argument("--autocast", action="store_true")
    result.add_argument("--bf16-policy", choices=("torch_amp", "bf16_storage"), default="torch_amp")
    result.add_argument("--qualify-native-group-fusion", action="store_true",
                        help="Explicit experimental native-statistics GN replacement; requires CUDA launch/numerical proof")
    result.add_argument("--layout", choices=("contiguous", "strided", "channels-last"), default="contiguous")
    result.add_argument("--gradient-mask", choices=("all", "input", "weight", "bias", "none"), default="all")
    result.add_argument("--compression", choices=("off", "on"), default="off")
    result.add_argument("--bits", nargs="+", type=int, choices=(1, 2, 4, 8), default=[4])
    result.add_argument("--group-size", type=int, default=1024)
    result.add_argument("--counterbalanced-norms", action=argparse.BooleanOptionalAction, default=True,
                        help="Add separate AB and BA norm diagnostics; original cases/trials/samples remain unchanged")
    result.add_argument("--fusion-objective", choices=("latency", "capacity"), default="latency")
    result.add_argument("--max-extra-allocated-bytes", type=int,
                        help="Required physical extra-allocation budget for a capacity-oriented recommendation")
    result.add_argument("--backward-order", choices=("dx_first", "dw_first"), default="dx_first")
    result.add_argument("--seed", type=int, default=614)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--trials", type=int, default=3)
    result.add_argument("--profile-steps", type=int, default=1)
    result.add_argument("--phase-iterations", type=int, default=3,
                        help="Updates per synchronized phase diagnostic; 0 disables it without changing throughput samples")
    result.add_argument("--phase-repeats", type=int, default=1)
    result.add_argument("--worker-isolation", choices=("strategy", "trial"), default="strategy",
                        help="Fresh process per strategy, or reuse one process for a trial's sequential controls with compiler/cache cleanup")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    runtime = _installed_runtime()
    from compiler.benchmark_bf16 import write_json

    argv = list(sys.argv[1:] if argv is None else argv)
    if "--fusion-worker" in argv:
        position = argv.index("--fusion-worker")
        job_path, result_path = map(Path, argv[position + 1:position + 3])
        from analysis.numerical_ipc import CHANNEL_ENV, send_snapshot

        progress_path = Path(argv[argv.index("--worker-progress") + 1]) if "--worker-progress" in argv else None
        sink = (lambda job, snapshot: send_snapshot(_numerical_identity(job), snapshot)) if CHANNEL_ENV in os.environ else None
        results = execute_worker_jobs([{"job": json.loads(job_path.read_text()), "result_path": str(result_path)}],
                                      progress_path=progress_path, snapshot_sink=sink)
        return 0 if results[0]["status"] == "ok" else 1
    if "--fusion-trial-worker" in argv or "--fusion-strategy-group-worker" in argv:
        mode = "--fusion-trial-worker" if "--fusion-trial-worker" in argv else "--fusion-strategy-group-worker"
        position = argv.index(mode)
        payload_path, result_path = map(Path, argv[position + 1:position + 3])
        payload = json.loads(payload_path.read_text())
        result = execute_comparison_group(payload["entries"], progress_path=payload["progress_path"],
                                          isolation="trial" if mode == "--fusion-trial-worker" else "strategy")
        write_json(result_path, result)
        return 0 if all(row["status"] == "ok" for row in result["results"]) else 1
    arguments = parser()
    args = arguments.parse_args(argv)
    if min(args.iterations, args.repeats, args.trials, args.phase_repeats) < 1 or min(args.warmup, args.profile_steps, args.phase_iterations) < 0:
        arguments.error("Iterations, repeats and trials must be positive; warmup/profile/phase steps must be nonnegative")
    try:
        plan = make_plan(args)
    except ValueError as error:
        arguments.error(str(error))
    from analysis.scheduler import output_lock

    with output_lock(args.output.resolve()):
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "fusion-plan.json", plan)
        if args.dry_run:
            return 0
        installed = runtime.installed_identity()
        plan["installed_deepopt"] = installed
        execution_jobs = plan["jobs"] + plan["diagnostic_jobs"]
        for job in execution_jobs:
            job["installed_deepopt"] = installed
        write_json(args.output / "fusion-plan.json", plan)
        results, comparisons, library_comparisons, numerical_groups = [], [], [], []
        groups = comparison_groups({"jobs": execution_jobs})
        for index, jobs in enumerate(groups):
            entries = []
            for job in jobs:
                directory = args.output / "runs" / plan["run_id"] / job["id"]
                directory.mkdir(parents=True)
                write_json(directory / "job.json", job)
                entries.append({"job": job, "result_path": str(directory / "result.json")})
            directory = args.output / "runs" / plan["run_id"] / "comparison-workers" / f"group-{index}"
            directory.mkdir(parents=True)
            phase_path = directory / "worker-phase.json"
            payload_path = directory / "jobs.json"
            group_result_path = directory / "results.json"
            write_json(payload_path, {"entries": entries, "progress_path": str(phase_path)})
            worker_arguments = ["--fusion-trial-worker" if args.worker_isolation == "trial" else
                                "--fusion-strategy-group-worker", str(payload_path), str(group_result_path)]
            active = {"job_ids": [job["id"] for job in jobs], "worker_phase_path": str(phase_path),
                      "worker_isolation": args.worker_isolation, "worker_index": index + 1,
                      "worker_count": len(groups)}
            # Keep heartbeat I/O independent of compiler-report/trace size.
            # Complete results are already persisted per strategy and retained
            # in the final summary; rewriting their growing history is O(n^2).
            write_json(args.output / "fusion-progress.json", {"run_id": plan["run_id"], "active": active,
                "completed_strategies": len(results), "total_strategies": len(execution_jobs),
                "failed_strategies": sum(row.get("status") != "ok" for row in results)})
            print(f"[fusion] starting worker {index + 1}/{len(groups)}: "
                  f"{jobs[0]['fixture_id']} trial {jobs[0]['trial']} ({len(jobs)} sequential strategies)", flush=True)
            with (directory / "worker.log").open("w") as stream:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), *worker_arguments],
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           env=runtime.child_environment(ROOT, identity=installed))
            for entry in entries:
                result_path = Path(entry["result_path"])
                try:
                    row = json.loads(result_path.read_text())
                    if not isinstance(row, dict) or not _same_job(row.get("job"), entry["job"]):
                        raise ValueError("Strategy result does not match its planned identity")
                    if row.get("status") == "ok":
                        _validate_success_result(row)
                except (OSError, ValueError, TypeError) as error:
                    row = {"status": "error", "job": entry["job"],
                           "reason": f"missing_or_invalid_strategy_result: {type(error).__name__}: {error}"}
                if completed.returncode and row.get("status") == "ok":
                    row.update(status="error", reason=f"worker_exit_{completed.returncode}_after_result")
                    write_json(result_path, row)
                results.append(row)
            group_rows = results[-len(entries):]
            try:
                group_report = json.loads(group_result_path.read_text())
                group_comparisons = _validate_group_report(jobs, group_rows, group_report)
                comparisons.extend(group_comparisons)
                library_comparisons.extend(group_report.get("library_route_comparisons", []))
                numerical_groups.append(group_report["numerical_storage"])
            except (OSError, ValueError, TypeError, KeyError) as error:
                routes = {job["strategy"]["name"]: job for job in jobs}
                for first, second, toggle, transformation in _pair_specs(jobs[0]["fixture"]["kind"]):
                    comparisons.append({"fixture_id": jobs[0]["fixture_id"], "backend": jobs[0]["backend"],
                        "trial": jobs[0]["trial"], "status": "failed", "correctness_passed": False,
                        "diagnostic_order": jobs[0].get("diagnostic_order"),
                        "candidate_toggle": toggle, "transformation": transformation,
                        "comparison_jobs": [routes[first]["id"], routes[second]["id"]],
                        "reason": "missing_or_invalid_group_comparison", "error": f"{type(error).__name__}: {error}"})
            print(f"[fusion] completed worker {index + 1}/{len(groups)} (exit {completed.returncode})", flush=True)
            write_json(args.output / "fusion-progress.json", {"run_id": plan["run_id"], "active": None,
                "completed_strategies": len(results), "total_strategies": len(execution_jobs),
                "failed_strategies": sum(row.get("status") != "ok" for row in results)})
        unchanged = (source_identity() == plan["source_sha256"]
                     and runtime.installed_identity() == plan["installed_deepopt"])
        all_routes_correct = all(row.get("status") == "ok" and row["numerical"]["passed"] is True
                                 and row["native_autodiff_ownership"] is True and row["source_unchanged"] is True
                                 and row["finite_after_timing"] is True for row in results)
        diagnostic_rows = [row for row in results if row["job"].get("diagnostic_order")]
        main_rows = [row for row in results if not row["job"].get("diagnostic_order")]
        diagnostic_pairs = [row for row in comparisons if row.get("diagnostic_order")]
        main_pairs = [row for row in comparisons if not row.get("diagnostic_order")]
        balanced = counterbalanced_summary(plan["diagnostic_jobs"], diagnostic_rows, diagnostic_pairs)
        summary = {"run_id": plan["run_id"], "source_sha256": plan["source_sha256"], "source_unchanged": unchanged,
                   "installed_deepopt": plan["installed_deepopt"],
                   "worker_isolation": plan["worker_isolation"],
                   "fresh_process_per_strategy": plan["fresh_process_per_strategy"],
                   "fresh_process_per_trial": plan["fresh_process_per_trial"],
                   "worker_processes": len(execution_jobs) if args.worker_isolation == "strategy" else len(groups),
                   "comparison_worker_processes": len(groups),
                   "numerical_storage": {"mode": "private_group_ram", "persisted_tensor_bytes": 0,
                       "strategy_transport": "raw_socket_ipc" if args.worker_isolation == "strategy" else "in_process",
                       "groups": numerical_groups},
                   "results": main_rows, "comparisons": main_pairs, "library_route_comparisons": library_comparisons,
                   "paired_trials": paired_trial_summary(main_pairs),
                   "counterbalanced_diagnostics": {"results": diagnostic_rows, "comparisons": diagnostic_pairs,
                       "paired_orders": balanced, "original_strategy_count": len(plan["jobs"]),
                       "diagnostic_strategy_count": len(plan["diagnostic_jobs"])},
                   "measured_cost_decisions": measured_cost_decisions(main_rows, main_pairs, balanced,
                       objective=args.fusion_objective, capacity_bytes=args.max_extra_allocated_bytes),
                   "correctness_passed": unchanged and all_routes_correct and all(row["correctness_passed"] for row in comparisons),
                   "gpu_performance_qualified": False,
                   "note": "Observed CUDA candidates require paired traces. Promotion requires review of repeated timings, drift and complete-update memory costs; CPU execution cannot qualify GPU kernels."}
        write_json(args.output / "fusion-summary.json", summary)
        return 0 if summary["correctness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
