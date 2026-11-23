"""Compare eager/compiled PyTorch BF16 and maximum-speed DeepOpt BF16 training.

Run with Python, not torchrun; --gpus launches one worker per GPU for each mode.
Requires installed DeepOpt, CUDA PyTorch and matching torchvision. See README.md.
"""

import argparse
from contextlib import ExitStack
import gc
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import os
from pathlib import Path
import subprocess
import statistics
import sys
import tempfile
import time
import tomllib


MODES = {
    "torch-bf16": dict(managed=False, compressed=False, compiled=False, native=False),
    "torch-bf16-compiled": dict(managed=False, compressed=False, compiled=True, native=False),
    "deepopt-bf16": dict(managed=True, compressed=False, compiled=False, native=False),
    "deepopt-bf16-compressed": dict(managed=True, compressed=True, compiled=False, native=False),
    "deepopt-bf16-native": dict(managed=True, compressed=False, compiled=True, native=True),
    "deepopt-bf16-native-compressed": dict(managed=True, compressed=True, compiled=True, native=True),
}
CUSTOM_OPTIMIZER = "source.training.geo_optimizer:SGD_NS_Overshoot"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path(__file__).with_name("benchmark_imagenet.toml"))
    result.add_argument("--train-dir", type=Path, help="Override application.data.train_dir")
    result.add_argument("--batch-size", type=int, help="Full batch size PER GPU")
    result.add_argument("--batches", type=int, default=10, help="Measured batches per rank (default: 10)")
    result.add_argument("--warmup", type=int, default=10, help="Untimed optimizer updates per trial (default: 10)")
    result.add_argument("--repeats", type=int, default=3, help="Independent fresh-process trials per mode (default: 3)")
    result.add_argument("--gpus", type=int, default=1, help="GPUs on this node; launches DDP when > 1")
    result.add_argument("--workers", type=int, help="DataLoader workers per GPU; defaults to the TOML")
    result.add_argument("--input-mode", choices=("cached", "loader"), default="cached",
                        help="cached: preload real CPU batches; loader: include loading in wall time")
    result.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    result.add_argument("--bf16-policy", choices=("torch_amp", "bf16_storage"), help="DeepOpt forward storage policy; default from runtime config")
    result.add_argument("--inductor-preset", choices=("default", "cast-preserving"), default="default")
    result.add_argument("--optimizer", choices=("custom", "sgd", "adamw"), help="Explicit TOML override for ALL modes")
    result.add_argument("--optimizer-factory", default=CUSTOM_OPTIMIZER, help="module:callable for optimizer=custom")
    result.add_argument("--model-factory", help="Optional module:callable replacing the adjacent MBConvVAE_V2")
    result.add_argument("--compile-mode", choices=("default", "max-autotune-no-cudagraphs"),
                        default="max-autotune-no-cudagraphs", help="Inductor mode for torch-bf16-compiled")
    result.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable deterministic algorithms instead of the speed preset (default: off)")
    result.add_argument("--fill-uninitialized-memory", action=argparse.BooleanOptionalAction, default=False,
                        help="Fill unused allocations in deterministic mode (default: off)")
    result.add_argument("--output", type=Path, default=Path("work/vae_benchmark.json"))
    result.add_argument("--diagnose", action="store_true", help="Untimed eager/native numerical controls and isolated VJP replays")
    result.add_argument("--diagnostic-from-result", type=Path, help="Reuse workload settings and optimizer from an earlier result JSON")
    result.add_argument("--diagnostic-device", default="cuda:0")
    result.add_argument("--diagnostic-batch-size", type=int, default=1)
    result.add_argument("--diagnostic-cases", nargs="+", help="Select named diagnostic cases; default runs the complete matrix")
    result.add_argument("--diagnostic-deterministic", action=argparse.BooleanOptionalAction, default=None,
                        help="Explicit deterministic-algorithm control; otherwise preserve recorded settings")
    result.add_argument("--diagnostic-max-bytes", type=int, default=1024 * 2**20,
                        help="Fixed CPU oracle RAM budget (default: 1 GiB; separate from CUDA memory)")
    result.add_argument("--diagnostic-trace-max-bytes", type=int, default=128 * 2**20)
    result.add_argument("--diagnostic-trace-modules", nargs="+", help="Exact leaf module paths for bounded boundary comparisons")
    result.add_argument("--diagnostic-project-root", type=Path, help="Project containing the external custom optimizer package")
    result.add_argument("--diagnostic-synthetic", action="store_true", help="Explicit seeded synthetic inputs; not dataset qualification")
    result.add_argument("--diagnostic-large-operators", action="store_true", help="Opt into isolated original batch128 operator geometries")
    result.add_argument("--diagnostic-dry-run", action="store_true", help="Print the diagnostic plan without importing Torch or executing it")
    result.add_argument("--_diagnostic-task", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--qualify", action="store_true", help="Untimed checkpoint-free original-recipe eager/native trajectory and capacity gate")
    result.add_argument("--qualification-capacity-only", action="store_true",
                        help="Independent fresh-process candidate allocation probe; never numerical qualification")
    result.add_argument("--qualification-device", default="cuda:0")
    result.add_argument("--qualification-updates", type=int, default=50)
    result.add_argument("--qualification-compressed", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--qualification-capacity-batches", type=int, nargs="+", default=[1])
    result.add_argument("--qualification-max-bytes", type=int, default=512 * 2**20,
                        help="Maximum retained CPU oracle tensor bytes per update; exceeding it fails closed")
    result.add_argument("--_case", choices=tuple(MODES), help=argparse.SUPPRESS)
    result.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--_snapshot", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--_repeat", type=int, default=0, help=argparse.SUPPRESS)
    return result


def settings(args):
    """Validate the workload without importing torch or touching ImageNet."""
    with args.config.open("rb") as stream:
        raw = tomllib.load(stream)
    training, data = raw["training"], raw["application"]["data"]
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.train_dir is not None:
        data["train_dir"] = str(args.train_dir)
    if args.optimizer is not None:
        training["optimizer"] = args.optimizer
    if args.workers is not None:
        raw.setdefault("loader", {})["num_workers"] = args.workers
    if training["batch_size"] < 1 or raw.get("loader", {}).get("num_workers", 0) < 0:
        raise ValueError("batch_size must be positive and workers must be nonnegative")
    if training.get("accumulation_steps", 1) != 1:
        raise ValueError("This benchmark requires accumulation_steps=1: every batch is an optimizer update")
    if data["image_size"] < 16 or data["resize_size"] < data["image_size"]:
        raise ValueError("image_size must be >= 16 and resize_size must be >= image_size")
    if "beta" not in raw.get("schedules", {}):
        raise ValueError("Configure the KL coefficient in schedules.beta")
    # Match throughput controls, preserving requested native kernel options.
    # Each explicitly labeled case selects its own compiler in mode_settings.
    runtime = raw.setdefault("runtime", {})
    runtime.update(device="cuda", precision="bf16", checks=False, execution_mode="max_speed",
                   diagnostics=None, mode_schedule=None, non_blocking=True,
                   deterministic=args.deterministic,
                   fill_uninitialized_memory=args.fill_uninitialized_memory)
    runtime["bf16_policy"] = getattr(args, "bf16_policy", None) or runtime.get("bf16_policy", "torch_amp")
    runtime["fp8"] = {"enabled": False}
    runtime.setdefault("activation_compression", {}).update(group_size=1024, norm_xhat_recenter=True)
    runtime["compile"] = {**runtime.get("compile", {}), "enabled": False,
                          "mode": args.compile_mode}
    runtime["cuda_graph"] = {}
    training["ema"] = {"enabled": False}
    raw.setdefault("loader", {})["drop_last"] = True
    return raw


def mode_settings(raw, mode):
    value = deepcopy(raw)
    case = MODES[mode]
    runtime = value["runtime"]
    if not case["managed"]:
        # Native Torch controls remain independent of DeepOpt storage policy.
        runtime["bf16_policy"] = "torch_amp"
    runtime.setdefault("activation_compression", {}).update(enabled=case["compressed"])
    runtime["fp8"] = {"enabled": False}
    compilation = runtime.setdefault("compile", {})
    native_options = dict(compilation.get("options", {}))
    compilation.update(enabled=case["compiled"], backend="deepopt" if case["native"] else "inductor",
                       fullgraph=case["native"], dynamic=False)
    if case["native"]:
        compilation.update(mode=None, options={**native_options, "use_native_autodiff": True,
                                               "require_full_aot": True})
    else:
        # DeepOpt backend options do not belong to the Inductor control.
        compilation["options"] = {}
    return value


def native_report_error(report):
    """Validate serialized ownership evidence only after a completed backward."""
    if (not isinstance(report, dict) or not isinstance(report.get("graphs"), list)
            or not report["graphs"] or any(not isinstance(graph, dict) for graph in report["graphs"])):
        return "native compiler produced no graph evidence"
    if report.get("runtime_fallbacks") != 0:
        return "native compiler used a runtime fallback"
    if not report.get("native_autodiff_operations") or not report.get("native_autodiff_regions"):
        return "native compiler produced no differentiated region"
    if "legacy_backward_operations" not in report or report["legacy_backward_operations"]:
        return "native compiler retained a legacy backward owner"
    if any("eager_autograd_operations" not in graph or "legacy_backward_operations" not in graph
           or graph["eager_autograd_operations"] or graph["legacy_backward_operations"]
           for graph in report["graphs"]):
        return "native compiler retained an eager or legacy derivative owner"
    return None


def native_compilation_evidence(model):
    from deepopt.compiler import compilation_report

    report = compilation_report(model)
    evidence = report.as_dict() if report is not None else None
    error = native_report_error(evidence)
    if error is not None:
        raise RuntimeError(error)
    return evidence


def json_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_provenance(deepopt, factory, optimizer):
    root = Path(deepopt.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())

    def source_hash(value):
        for candidate in (value, getattr(value, "func", None), type(value)):
            if candidate is None:
                continue
            try:
                path = inspect.getsourcefile(candidate)
                if path:
                    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except (TypeError, OSError):
                continue
        return None

    identity = {
        "deepopt_source_sha256": digest.hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualification_source_sha256": hashlib.sha256(Path(__file__).with_name("qualification.py").read_bytes()).hexdigest(),
        "model_source_sha256": source_hash(factory),
        "optimizer_source_sha256": source_hash(type(optimizer)),
    }
    try:
        package_version = version("deep-optimization")
    except PackageNotFoundError:
        package_version = None
    return {**identity, "source_sha256": json_fingerprint(identity),
            "deepopt_version": package_version, "deepopt_path": str(root), "python": sys.version}


def inductor_artifact_task(raw, preset="default"):
    """Native Torch dump identity for this exact VAE geometry and recipe."""
    runtime = deepcopy(raw["runtime"])
    runtime.update(execution="compile", bf16_policy="torch_amp")
    runtime["compile"] = {**runtime.get("compile", {}), "backend": "inductor",
        "options": {"emulate_precision_casts": preset == "cast-preserving"}}
    runtime["activation_compression"] = {"enabled": False, "group_size": 1024}
    image_size = raw["application"]["data"]["image_size"]
    return {"case": "torch_bf16", "settings": {"runtime": runtime},
            "workload": {"name": "MBConvVAE_V2", "gradient_mode": "all",
                "shape": [raw["training"]["batch_size"], 3, image_size, image_size],
                "model_options": deepcopy(raw["application"]["model"])},
            "optimizer": deepcopy(raw["training"]), "beta_schedule": deepcopy(raw["schedules"]["beta"]),
            "timing_scope": {"implementation": "vae_wall_clock_and_cuda_events",
                "wall_includes": ["batch_iteration", "H2D", "forward", "loss", "backward", "optimizer", "scheduling", "DDP"],
                "cuda_events_exclude": ["H2D"],
                "excludes": ["dataset_preparation", "eager_oracle", "compile_warmup", "diagnostics", "artifact_copy_hash"]}}


def compile_torch_model(model, options):
    import torch

    return torch.compile(model, backend="inductor", mode=options["mode"],
                         fullgraph=False, dynamic=False)


def import_factory(spec):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("Factory must have the form module:callable")
    try:
        return getattr(import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise ImportError(
            f"Cannot import {spec}. Add your custom project's root to PYTHONPATH, "
            "or explicitly choose --optimizer sgd/adamw for a different benchmark."
        ) from error


def make_optimizer(model, training, factory=CUSTOM_OPTIMIZER):
    import torch

    name = training["optimizer"]
    options = {"lr": training["learning_rate"], "weight_decay": training.get("weight_decay", 0.0)}
    if name in ("sgd", "custom"):
        options["momentum"] = training.get("momentum", 0.9)
    if name == "custom":
        return import_factory(factory)(model.parameters(), **options)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **options)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **options)
    raise ValueError(f"Unsupported optimizer: {name}")


@dataclass
class ImageTransform:
    """Same resize/crop/flip/normalization as training, keyed by sample index."""

    image_size: int
    resize_size: int
    seed: int

    def __call__(self, image, index):
        import torch
        from torchvision.transforms import InterpolationMode, functional as TF

        generator = torch.Generator().manual_seed(self.seed + index)
        image = TF.resize(image, self.resize_size, InterpolationMode.BILINEAR, antialias=True)
        width, height = image.size
        top = int(torch.randint(height - self.image_size + 1, (), generator=generator))
        left = int(torch.randint(width - self.image_size + 1, (), generator=generator))
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        if torch.rand((), generator=generator) < 0.5:
            image = TF.hflip(image)
        return TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)


class Images:
    def __init__(self, data, seed):
        from torchvision.datasets import ImageFolder

        self.folder = ImageFolder(data["train_dir"])
        self.transform = ImageTransform(data["image_size"], data["resize_size"], seed)

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        image, _label = self.folder[index]
        return self.transform(image, index)


def build_loader(raw, args, context):
    import torch

    dataset = Images(raw["application"]["data"], raw.get("seed", 42))
    batch_size = raw["training"]["batch_size"]
    count = args.batches * batch_size * context.world_size
    if len(dataset) < count:
        raise ValueError(f"Need {count} distinct images for full batches, but ImageFolder contains {len(dataset)}")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(raw.get("seed", 42)))
    indices = order[:count].tolist()[context.global_rank::context.world_size]
    options = raw.get("loader", {})
    workers = options.get("num_workers", 0)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=indices, drop_last=True,
        num_workers=workers, pin_memory=options.get("pin_memory", True),
        persistent_workers=workers > 0 and options.get("persistent_workers", True),
        generator=torch.Generator().manual_seed(raw.get("seed", 42) + context.global_rank),
    )


def fingerprint(tensors):
    import torch

    digest = hashlib.sha256()
    for value in tensors:
        digest.update(f"{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def objective(model, images, beta):
    import torch
    from torch.nn import functional as F

    from deepopt._bf16_policy import bf16_policy_scope

    reconstruction, mu, logvar = model(images)
    with bf16_policy_scope("torch_amp", images.device.type), torch.autocast(images.device.type, enabled=False):
        mu, logvar = mu.float(), logvar.float()
        l1 = F.l1_loss(reconstruction.float(), images.float())
        kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=(1, 2, 3)).mean()
        return l1 + beta * kl


def train_step(model, optimizer, runtime, images, beta, device_type="cuda", grad_clip_norm=None, progress=None):
    import torch

    def phase(name):
        if progress is not None:
            progress({"phase": name})

    phase("forward_and_loss")
    if runtime is None:
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            loss = objective(model, images, beta)
        phase("backward")
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        phase("optimizer")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    else:
        loss = runtime.compute(images, lambda batch: objective(model, batch, beta))
        phase("backward")
        runtime.backward(loss)
        phase("optimizer")
        runtime.step()
    return loss.detach()


def cuda_failure_diagnostics(device=None):
    """Read allocator counters only after failure; never synchronize or reset them."""
    result = {"device": None if device is None else str(device),
              "allocator_environment": {name: os.environ.get(name) for name in
                  ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_MODULE_LOADING",
                   "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT")},
              "synchronized": False, "errors": {},
              "counter_scope": "Collected after exception stack unwind, not at the failing allocation; "
                               "peaks retain their last-reset scope. No synchronization or counter reset performed."}

    def query(name, call):
        try:
            result[name] = call()
        except Exception as error:
            result["errors"][name] = f"{type(error).__name__}: {error}"

    try:
        import torch

        query("cuda_available", torch.cuda.is_available)
        if not result.get("cuda_available") or device is not None and torch.device(device).type != "cuda":
            return result
        result["torch_version"], result["cuda_version"] = str(torch.__version__), torch.version.cuda
        query("cudnn_version", torch.backends.cudnn.version)
        query("gpu_name", lambda: torch.cuda.get_device_name(device))
        query("gpu_capability", lambda: list(torch.cuda.get_device_capability(device)))

        def device_memory():
            free, total = torch.cuda.mem_get_info(device)
            return {"free_bytes": int(free), "total_bytes": int(total)}

        query("device_memory", device_memory)
        query("torch_flags", lambda: {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
        for name, function in (("allocated_bytes", torch.cuda.memory_allocated),
                               ("reserved_bytes", torch.cuda.memory_reserved),
                               ("peak_allocated_bytes", torch.cuda.max_memory_allocated),
                               ("peak_reserved_bytes", torch.cuda.max_memory_reserved)):
            query(name, lambda function=function: int(function(device)))
        try:
            stats = torch.cuda.memory_stats(device)
            result["allocator_stats"] = {key: stats.get(key) for key in (
                "allocated_bytes.all.current", "allocated_bytes.all.peak",
                "reserved_bytes.all.current", "reserved_bytes.all.peak",
                "inactive_split_bytes.all.current", "inactive_split_bytes.all.peak",
                "num_alloc_retries", "num_ooms")}
        except Exception as error:
            result["errors"]["allocator_stats"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        result["errors"]["cuda_diagnostics"] = f"{type(error).__name__}: {error}"
    return result


def gate_failure_status(error, context):
    """An eager oracle OOM is reference unavailability, never candidate failure."""
    if type(error).__name__ == "MemoryError":
        return "unavailable_reference_byte_budget"
    if type(error).__name__ == "OutOfMemoryError":
        if str(context.get("route", "")).startswith("eager"):
            return "unavailable_eager_oom"
        return "candidate_oom" if context.get("route") == "native" else "unavailable_oom"
    return "failed_numerical_or_execution"


def failure_error_details(error):
    """Retain the original Python stack/notes, without tensor snapshots."""
    import traceback

    details = {"error": f"{type(error).__name__}: {error}",
            "exception_notes": [str(note) for note in getattr(error, "__notes__", ())],
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        contract = getattr(current, "dtype_contract", None)
        if contract is not None:
            details["dtype_contract"] = contract
        for attribute in ("qualification_evidence", "oracle_report", "comparison", "boundary_localization"):
            value = getattr(current, attribute, None)
            if value is not None and attribute not in details:
                try:
                    json.dumps(value, allow_nan=False)
                    details[attribute] = value
                except (TypeError, ValueError):
                    details[attribute + "_collection_error"] = "non_serializable_metadata"
        current = current.__cause__ or current.__context__
    return details


def failure_candidate_diagnostics(candidate, dtype_edges=None):
    """Read partial candidate evidence without applying the success-only gate."""
    result = {"compiler_report_scope": "failed_candidate",
              "compiler_report": None, "eager_dtype_edges": dtype_edges}
    if candidate is None:
        result["compiler_report_status"] = "candidate_unavailable"
        return result
    try:
        from deepopt.compiler import compilation_report

        report = compilation_report(candidate)
        result["compiler_report"] = report.as_dict() if report is not None else None
        # Reject non-metadata values instead of formatting tensor payloads.
        json.dumps(result, allow_nan=False)
        result["compiler_report_status"] = "available" if report is not None else "not_captured"
    except Exception as diagnostic_error:
        result["compiler_report"] = None
        result["compiler_report_status"] = "collection_failed"
        result["compiler_report_collection_error"] = type(diagnostic_error).__name__
    return result


def worker(args):
    import torch
    import deepopt
    from deepopt.runtime import seed_everything

    snapshot = json.loads(args._snapshot.read_text(encoding="utf-8"))
    raw = mode_settings(snapshot, args._case)
    case = MODES[args._case]
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark needs CUDA GPUs; this Python environment has no CUDA device")
    from benchmarks.compiler.installed_runtime import installed_identity
    installed = installed_identity()
    context = deepopt.initialize_distributed("cuda")
    provenance = None
    gate_provenance = None
    optimizer = None
    model = gate_candidate = None
    dtype_edges = None
    device = context.device
    failure_context = {"stage": "setup", "phase": "initialization", "route": args._case, "update": None}

    def gate_progress(event):
        failure_context.update(stage="matching_recipe_gate", **event)
        if context.is_main_process:
            try:
                print(f"{args._case}: matching-recipe gate route={event.get('route')} "
                      f"update={event.get('update')} phase={event['phase']}", flush=True)
            except Exception:
                pass

    execution_stack = ExitStack()
    inductor_controls = None
    inductor_artifacts = None
    artifact_directory = None
    try:
        from deepopt._bf16_policy import resolve_bf16_policy
        policy = resolve_bf16_policy(raw["runtime"]["bf16_policy"])
        if case["compiled"] and not case["native"]:
            from benchmarks.compiler.inductor_benchmark_artifacts import InductorBenchmarkArtifacts
            artifact_directory = args.output.resolve().parent / (args.output.stem + "-artifacts") / (
                f"{args._case}-trial{args._repeat}-rank{context.global_rank}")
            inductor_artifacts = execution_stack.enter_context(InductorBenchmarkArtifacts(
                artifact_directory, inductor_artifact_task(raw, args.inductor_preset), enabled=True))
            inductor_controls = inductor_artifacts.configuration["numerical_controls"]
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")
        seed = raw.get("seed", 42)
        seed_everything(seed, raw["runtime"].get("deterministic", True))
        config = deepopt.load_config(raw)
        torch.utils.deterministic.fill_uninitialized_memory = config.runtime.fill_uninitialized_memory
        failure_context["phase"] = "input_loading"
        loader = build_loader(raw, args, context)
        if args.input_mode == "cached":
            batches = list(loader)  # Real ImageNet decode and augmentation are untimed.
            warmup_batch = batches[0]
        else:
            warmup_batch = next(iter(loader))  # Start workers and exercise the real input path.
            batches = None

        if args.model_factory:
            factory = import_factory(args.model_factory)
        elif __package__:
            from .mbconv_vae import MBConvVAE_V2 as factory
        else:
            from mbconv_vae import MBConvVAE_V2 as factory
        # Identify source before the throwaway eager gate can fail. Optimizer
        # provenance is added once that optimizer has actually been constructed.
        provenance = source_provenance(deepopt, factory, None)
        failure_context["phase"] = "model_initialization"
        model = factory(**raw["application"]["model"])
        initial_hash = fingerprint(model.state_dict().values())
        same_recipe_oracle = None
        if case["compiled"]:
            failure_context.update(stage="matching_recipe_gate", phase="initialization", route=None, update=None)
            # Gate the requested geometry/optimizer before admitting timing.
            # These throwaway models are CPU-resident between paired updates;
            # the benchmark's initial model and optimizer remain untouched.
            from deepopt.config import CompileConfig
            from deepopt.engine.execution.compilation import compile_model
            from deepopt.opt.policy import prepare_fp8_model
            from examples.vae.qualification import paired_trajectory
            from examples.vae.qualification import preserve_rng
            with preserve_rng():
                compression = raw["runtime"]["activation_compression"]
                eager = (prepare_fp8_model(deepcopy(model), False,
                    activation_compression=compression, config=raw)[0] if case["managed"] else deepcopy(model)).train()
                from examples.vae.qualification import eager_dtype_diagnostic, assert_dtype_edge_contracts
                failure_context.update(phase="eager_dtype_diagnostic", route="eager", update=1)
                dtype_edges = eager_dtype_diagnostic(eager, warmup_batch, compression,
                    bf16_policy=policy, managed=case["managed"], device=device,
                    max_bytes=args.qualification_max_bytes)
                assert_dtype_edge_contracts(dtype_edges)
                candidate = deepcopy(eager)
                gate_candidate = candidate
                if case["native"]:
                    compile_model(candidate, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                        dynamic=False, options=raw["runtime"]["compile"]["options"]), config=raw)
                else:
                    candidate = compile_torch_model(candidate, raw["runtime"]["compile"])
                    gate_candidate = candidate

                reference_optimizer = make_optimizer(eager, raw["training"], args.optimizer_factory)
                candidate_optimizer = make_optimizer(candidate, raw["training"], args.optimizer_factory)
                getattr(candidate, "_orig_mod", candidate).load_state_dict(eager.state_dict(), strict=True)
                candidate_optimizer.load_state_dict(deepcopy(reference_optimizer.state_dict()))
                gate_provenance = source_provenance(deepopt, factory, candidate_optimizer)
                gate_schedule = deepopt.ScheduleController.from_config(config.schedules)
                gate_schedule.prepare_update(1, epoch=1)
                same_recipe_oracle = paired_trajectory(eager, candidate, reference_optimizer, candidate_optimizer,
                    warmup_batch, [gate_schedule["beta"]], compression, device=device,
                    max_bytes=args.qualification_max_bytes, bf16_policy=policy, managed=case["managed"],
                    grad_clip_norm=config.training.grad_clip_norm,
                    progress=gate_progress)
                failure_context.update(phase="native_ownership_check", route="native", update=1)
                if case["native"]:
                    native_compilation_evidence(candidate)
                if gate_provenance != source_provenance(deepopt, factory, candidate_optimizer):
                    raise RuntimeError("VAE source changed during matching-recipe gate")
                failure_context["phase"] = "cleanup"
                del eager, candidate, reference_optimizer, candidate_optimizer
                gate_candidate = None
                # Release the throwaway oracle's Dynamo guards and device constants
                # before measuring the fresh benchmark model.
                torch.compiler.reset()
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
        failure_context.update(stage="benchmark_setup", phase="device_transfer", route=args._case, update=None)
        model = model.to(device).train()
        failure_context["phase"] = "optimizer_initialization"
        optimizer = make_optimizer(model, raw["training"], args.optimizer_factory)
        provenance = source_provenance(deepopt, factory, optimizer)
        runtime = None
        report = None
        if case["managed"]:
            runtime = deepopt.TrainingRuntime(model, optimizer, device=device, precision="bf16", config=config)
            model = runtime.model.train()
            report = runtime.transformation_report
            if case["compressed"] and (report is None or not report.converted_modules):
                raise RuntimeError("DeepOpt converted no modules; refusing to label this an optimized run")
        elif context.is_distributed:
            ddp = dict(raw["runtime"].get("ddp", {}))
            ddp.pop("validation_sync_buffers", None)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], **ddp)
        if case["compiled"] and not case["native"]:
            # Compile the DDP wrapper when present so Dynamo can respect its
            # communication buckets. The optimizer and loss remain eager.
            model = compile_torch_model(model, raw["runtime"]["compile"])
        optimizer.zero_grad(set_to_none=True)
        # DDP initialization may consume RNG; reset before the first latent draw.
        seed_everything(seed + context.global_rank, raw["runtime"].get("deterministic", True))
        schedules = deepopt.ScheduleController.from_config(config.schedules)
        non_blocking = raw["runtime"].get("non_blocking", True)

        def update(cpu_images, number):
            failure_context.update(stage="warmup", phase="input_transfer", route=args._case, update=number)
            schedules.prepare_update(number, epoch=1)
            images = cpu_images.to(device, non_blocking=non_blocking)
            failure_context["phase"] = "update"
            loss = train_step(model, optimizer, runtime, images, schedules["beta"],
                              grad_clip_norm=config.training.grad_clip_norm, progress=failure_context.update)
            schedules.complete_update(number)
            return loss

        if context.is_main_process:
            print(f"{args._case}, trial {args._repeat + 1}/{args.repeats}: batch/GPU={raw['training']['batch_size']}, "
                  f"{args.warmup} warmup + {args.batches} measured batches", flush=True)
        failure_context.update(stage="warmup", phase="start", route=args._case, update=None)
        warmup_start = time.perf_counter()
        losses = [update(warmup_batch, step + 1) for step in range(args.warmup)]
        torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start
        if case["native"]:
            # Compilation is lazy. Check ownership after a real optimizer update,
            # before admitting its timings as a native compiler measurement.
            native_compilation_evidence(model)
        measured = iter(batches) if batches is not None else iter(loader)
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(args.batches)]
        # Instantiate event handles before measurement as well as compiling kernels.
        for start, end in events:
            start.record()
            end.record()
        torch.cuda.synchronize(device)
        if context.is_distributed:
            torch.distributed.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        seen = []
        betas = []
        wall_start = time.perf_counter()
        for step, (start, end) in enumerate(events, args.warmup + 1):
            failure_context.update(stage="measurement", phase="input_transfer", route=args._case, update=step)
            cpu_images = next(measured)
            seen.append(cpu_images)
            schedules.prepare_update(step, epoch=1)
            beta = schedules["beta"]
            betas.append(beta)
            images = cpu_images.to(device, non_blocking=non_blocking)
            start.record()  # Model time excludes the preceding host-to-device copy.
            failure_context["phase"] = "update"
            losses.append(train_step(model, optimizer, runtime, images, beta,
                                     grad_clip_norm=config.training.grad_clip_norm, progress=failure_context.update))
            end.record()
            schedules.complete_update(step)
        torch.cuda.synchronize(device)
        failure_context.update(stage="post_measurement", phase="validation", update=None)
        wall_seconds = time.perf_counter() - wall_start
        cuda_seconds = sum(start.elapsed_time(end) for start, end in events) / 1000
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        finite = bool(torch.isfinite(torch.stack(losses)).all())
        finite = finite and all(bool(torch.isfinite(p).all()) for p in model.parameters())
        compiler_report = native_compilation_evidence(model) if case["native"] else None
        provenance_after = source_provenance(deepopt, factory, optimizer)
        source_stable = (provenance["source_sha256"] == provenance_after["source_sha256"]
                         and installed == installed_identity())
        artifact_report = None
        if inductor_artifacts is not None:
            from torch._dynamo.utils import counters
            artifact_report = inductor_artifacts.finish(row={
                "status": "ok" if finite and source_stable else "failed",
                "timing": {"wall_seconds": wall_seconds, "cuda_model_seconds": cuda_seconds,
                           "measured_updates": args.batches},
                "finite_after_timing": finite,
                "environment": {"device": str(device)},
                "execution": {"compile_counters": {key: dict(value) for key, value in counters.items()}},
            })
            if not artifact_report or not artifact_report.get("reference_valid"):
                raise RuntimeError("Raw Inductor artifacts lack native-only forward/backward provenance")
        local = {
            "rank": context.global_rank, "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "wall_seconds": wall_seconds, "cuda_model_seconds": cuda_seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "initial_state_sha256": initial_hash, "input_sha256": fingerprint(seen),
            "finite": finite, "last_loss": float(losses[-1]) if finite else None,
            "source_sha256": provenance["source_sha256"], "source_stable": source_stable,
            "provenance": provenance, "provenance_after": provenance_after,
            "compiler_report": compiler_report,
            "same_recipe_oracle": same_recipe_oracle, "eager_dtype_edges": dtype_edges,
                "bf16_policy": policy.as_dict(), "inductor_controls": inductor_controls,
                "inductor_artifacts": artifact_report,
                "inductor_artifact_directory": None if artifact_directory is None else str(artifact_directory),
        }
        ranks = [None] * context.world_size
        if context.is_distributed:
            torch.distributed.all_gather_object(ranks, local)
        else:
            ranks[0] = local
        executions = {}
        for name, module in model.named_modules():
            info = getattr(module, "execution_info", None)
            if info is not None:
                executions[name] = info
        if context.is_main_process:
            wall = max(rank["wall_seconds"] for rank in ranks)
            cuda = max(rank["cuda_model_seconds"] for rank in ranks)
            batch_size = raw["training"]["batch_size"]
            result = {
                "mode": args._case, "repeat": args._repeat,
                "installed": installed,
                "status": ("invalid_changed_source" if not all(rank["source_stable"] for rank in ranks)
                           else "ok" if all(rank["finite"] for rank in ranks) else "nonfinite"),
                "batches_per_rank": args.batches, "batch_size_per_gpu": batch_size,
                "image_size": raw["application"]["data"]["image_size"],
                "workload_sha256": json_fingerprint(snapshot),
                "source_sha256": provenance["source_sha256"], "provenance": provenance,
                "provenance_after": provenance_after,
                "source_stable": all(rank["source_stable"] for rank in ranks),
                "native_ownership_verified": case["native"],
                "autodiff_owner": "deepopt_native" if case["native"] else "pytorch_aotautograd" if case["compiled"] else "pytorch_autograd",
                "same_recipe_oracle": same_recipe_oracle, "eager_dtype_edges": dtype_edges,
                "bf16_policy": policy.as_dict(), "inductor_controls": inductor_controls,
                "inductor_artifacts": artifact_report,
                "inductor_artifact_directory": None if artifact_directory is None else str(artifact_directory),
                "compiler_report": compiler_report,
                "world_size": context.world_size, "effective_batch_size": batch_size * context.world_size,
                "total_images": args.batches * batch_size * context.world_size,
                "wall_seconds": wall, "wall_ms_per_batch": wall * 1000 / args.batches,
                "cuda_model_ms_per_batch": cuda * 1000 / args.batches,
                "images_per_second": args.batches * batch_size * context.world_size / wall,
                "peak_allocated_gib": max(rank["peak_allocated_bytes"] for rank in ranks) / 2**30,
                "peak_reserved_gib": max(rank["peak_reserved_bytes"] for rank in ranks) / 2**30,
                "betas": betas, "ranks": ranks, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "warmup_seconds": warmup_seconds,
                "torch_flags": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
                                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                "cudnn_deterministic": torch.backends.cudnn.deterministic},
                "deepopt_path": str(Path(deepopt.__file__).resolve()),
                "transformation_report": asdict(report) if report is not None else None,
                "module_execution": executions, "runtime": asdict(config.runtime),
            }
            write_json(args._result, result)
    except Exception as error:
        artifact_error = None
        if inductor_artifacts is not None:
            try:
                inductor_artifacts.finish(error=error)
            except Exception as diagnostic_error:
                artifact_error = type(diagnostic_error).__name__
        if context.is_main_process:
            try:
                failure = {"mode": args._case, "repeat": args._repeat, "status": "failed",
                           "workload_sha256": json_fingerprint(snapshot),
                           "failure_context": dict(failure_context),
                           "gate_status": gate_failure_status(error, failure_context),
                           "candidate_capacity_status": "not_measured",
                           **failure_error_details(error)}
                failure.update(failure_candidate_diagnostics(
                    gate_candidate if gate_candidate is not None else model, dtype_edges))
                try:
                    failure["cuda_failure_diagnostics"] = cuda_failure_diagnostics(device)
                except Exception as diagnostic_error:
                    failure["cuda_diagnostics_collection_error"] = type(diagnostic_error).__name__
                if artifact_error is not None:
                    failure["inductor_artifact_collection_error"] = artifact_error
                if inductor_artifacts is not None:
                    failure["inductor_artifacts"] = inductor_artifacts.report
                    failure["inductor_artifact_directory"] = str(artifact_directory)
                if gate_provenance is not None:
                    failure["matching_recipe_gate_provenance"] = gate_provenance
                if provenance is not None:
                    failure.update(provenance=provenance, source_sha256=provenance["source_sha256"])
                    try:
                        after = source_provenance(deepopt, factory, optimizer)
                        failure.update(provenance_after=after,
                                       source_stable=provenance["source_sha256"] == after["source_sha256"])
                    except Exception as diagnostic_error:
                        failure["provenance_collection_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                write_json(args._result, failure)
            except Exception as diagnostic_error:
                try:
                    print(f"VAE failure reporting also failed: {type(diagnostic_error).__name__}: {diagnostic_error}",
                          file=sys.stderr, flush=True)
                except Exception:
                    pass
        raise
    finally:
        execution_stack.close()
        deepopt.shutdown_distributed()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def workload_identity(row):
    fields = ("workload_sha256", "batch_size_per_gpu", "world_size", "batches_per_rank", "total_images", "image_size")
    values = tuple(row.get(key) for key in fields)
    ranks = row.get("ranks", [])
    size = row.get("world_size")
    if (any(value is None for value in values) or not isinstance(ranks, list) or not ranks
            or any(not isinstance(rank, dict) for rank in ranks) or not row.get("source_sha256")
            or row.get("source_stable") is False
            or MODES.get(row.get("mode"), {}).get("native") and row.get("native_ownership_verified") is not True
            or type(size) is not int or size < 1 or len(ranks) != size
            or [rank.get("rank") for rank in ranks] != list(range(size))):
        return None
    if any(rank.get("source_stable") is False or rank.get("source_sha256", row["source_sha256"]) != row["source_sha256"]
           for rank in ranks):
        return None
    hashes = tuple((rank.get("initial_state_sha256"), rank.get("input_sha256")) for rank in ranks)
    if any(not value for pair in hashes for value in pair):
        return None
    policy = row.get("bf16_policy", {"name": "torch_amp", "version": 1})
    if (not isinstance(policy, dict) or set(policy) != {"name", "version"}
            or policy["name"] not in ("torch_amp", "bf16_storage")
            or type(policy["version"]) is not int or policy["version"] != 1):
        return None
    return values + (hashes, row.get("source_sha256"), policy["name"], policy["version"])


def worker_result_error(row, mode, repeat, workload):
    """Reject stale/malformed receipts without discarding the requested case."""
    if not isinstance(row, dict) or row.get("mode") != mode or row.get("repeat") != repeat:
        return "worker result mode/repeat does not match the requested trial"
    if row.get("workload_sha256") != workload:
        return "worker result does not match the frozen workload"
    if row.get("status") not in {"ok", "failed", "nonfinite", "invalid_changed_source"}:
        return "worker result has no recognized status"
    if row.get("status") != "ok":
        return None
    if workload_identity(row) is None:
        return "worker result has incomplete or inconsistent workload/source identity"
    for rank in (row, *row["ranks"]):
        before, after = rank.get("provenance", {}), rank.get("provenance_after", {})
        if (not isinstance(before, dict) or not isinstance(after, dict)
                or rank.get("source_stable") is not True or before.get("source_sha256") != row["source_sha256"]
                or before.get("source_sha256") != after.get("source_sha256")):
            return "worker source was not verified stable on every rank"
        if MODES[mode]["native"]:
            if (rank.get("same_recipe_oracle") or {}).get("passed") is not True:
                return "native worker has no passing matching-recipe eager oracle"
            error = native_report_error(rank.get("compiler_report"))
            if error is not None:
                return error
    return None


def aggregate_runs(runs, modes, repeats, *, running=False):
    rows = []
    metrics = ("wall_seconds", "wall_ms_per_batch", "cuda_model_ms_per_batch", "images_per_second",
               "peak_allocated_gib", "peak_reserved_gib")
    for mode in modes:
        trials = [run for run in runs if run["mode"] == mode]
        successful = [run for run in trials if run["status"] == "ok"]
        complete = (len(trials) == repeats and {run["repeat"] for run in trials} == set(range(repeats))
                    and len(successful) == repeats)
        identities = [workload_identity(run) for run in successful]
        consistent = bool(identities) and identities[0] is not None and all(item == identities[0] for item in identities)
        if MODES[mode]["native"]:
            consistent &= all(run.get("native_ownership_verified") is True
                              and (run.get("same_recipe_oracle") or {}).get("passed") is True
                              and native_report_error(run.get("compiler_report")) is None
                              and all((rank.get("same_recipe_oracle") or {}).get("passed") is True
                                      and native_report_error(rank.get("compiler_report")) is None
                                      for rank in run.get("ranks", [])) for run in successful)
        if not complete or not consistent:
            pending = (running and len(trials) < repeats and len(successful) == len(trials)
                       and len({trial.get("repeat") for trial in trials}) == len(trials)
                       and (not trials or consistent))
            rows.append(dict(mode=mode, status=("pending" if not trials else "incomplete") if pending
                             else "failed" if not complete else "mismatch",
                             successful_repeats=len(successful), requested_repeats=repeats))
            continue
        row = {key: trials[0][key] for key in ("mode", "batch_size_per_gpu", "world_size", "batches_per_rank",
                                              "total_images", "image_size", "workload_sha256", "ranks", "source_sha256")}
        row.update(status="ok", successful_repeats=repeats, requested_repeats=repeats)
        row.update({key: trials[0][key] for key in ("source_stable", "native_ownership_verified", "bf16_policy") if key in trials[0]})
        # Summary timings are medians; keep rank identity separate from the
        # individual trial's timings, which remain in the raw runs array.
        row["ranks"] = [{key: rank[key] for key in ("rank", "gpu", "capability", "initial_state_sha256", "input_sha256")
                         if key in rank} for rank in row["ranks"]]
        row["effective_batch_size"] = row["batch_size_per_gpu"] * row["world_size"]
        for metric in metrics:
            samples = [trial[metric] for trial in trials]
            row[metric] = statistics.median(samples)
            row[metric + "_samples"] = samples
            row[metric + "_min"], row[metric + "_max"] = min(samples), max(samples)
        rows.append(row)
    return rows


def comparisons(rows):
    baseline = next((row for row in rows if row["mode"] == "torch-bf16" and row["status"] == "ok"), None)
    compiled = next((row for row in rows if row["mode"] == "torch-bf16-compiled" and row["status"] == "ok"), None)
    for row in rows:
        row["speedup_vs_torch"] = None
        row["speedup_vs_torch_compiled"] = None
        identity = workload_identity(row) if row["status"] == "ok" else None
        if compiled is not None and identity is not None and identity == workload_identity(compiled):
            row["speedup_vs_torch_compiled"] = compiled["wall_seconds"] / row["wall_seconds"]
        if row["status"] != "ok" or baseline is None:
            continue
        same = identity is not None and identity == workload_identity(baseline)
        row["same_initial_weights_and_inputs"] = same
        if same:
            row["speedup_vs_torch"] = baseline["wall_seconds"] / row["wall_seconds"]
    for row in rows:
        row["speedup_vs_matching_eager"] = None
        case = MODES[row["mode"]]
        if row["status"] != "ok" or not case["native"]:
            continue
        eager_mode = "deepopt-bf16-compressed" if case["compressed"] else "deepopt-bf16"
        eager = next((item for item in rows if item["mode"] == eager_mode and item["status"] == "ok"), None)
        if eager is not None and workload_identity(eager) == workload_identity(row):
            row["speedup_vs_matching_eager"] = eager["wall_seconds"] / row["wall_seconds"]
    return rows


def suite(args, argv):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Run this supervisor with Python, not torchrun; use --gpus 4 for DDP")
    raw = settings(args)
    runs = []
    result = {
        "schema_version": 2, "status": "running",
        "settings": raw, "input_mode": args.input_mode, "warmup_updates": args.warmup,
        "repeats": args.repeats, "batches_per_rank": args.batches,
        "trial_scope": "initial_one_trial" if args.repeats == 1 else "promotion_trials" if args.repeats >= 3 else "diagnostic_two_trials",
        "workload_sha256": json_fingerprint(raw),
        "model_factory": args.model_factory or "mbconv_vae:MBConvVAE_V2",
        "optimizer_factory": args.optimizer_factory if raw["training"]["optimizer"] == "custom" else None,
        "notes": [
            "Every mode/trial starts a fresh process with the same seeded initial weights and rank-local real images.",
            "The original TOML is read once; every worker receives the same immutable resolved JSON snapshot.",
            "All modes use BF16 autocast and FP32 master weights and L1/KL reductions. FP8 is explicitly disabled.",
            "All modes share deterministic/fill/cuDNN flags; the default disables determinism/filling and enables cuDNN benchmarking.",
            "Wall time includes H2D, forward, loss, backward, optimizer, scheduling and DDP communication.",
            "cached preloads actual CPU batches; loader additionally includes input waiting/decoding in wall time.",
            "CUDA model events exclude H2D but include host submission gaps; event times are not kernel-only time.",
            "Warmup trains on the first batch and advances weights, optimizer and beta before measurement.",
            "torch-bf16-compiled uses Inductor; deepopt-bf16-native[-compressed] requires fullgraph native AutoDiff and checks every rank after backward. Loss and optimizer remain eager.",
            "Compiler warmup is outside timing. Imported library/model/optimizer/benchmark sources are hashed before and after each trial; changed sources invalidate results.",
            "DeepOpt uses max_speed. EMA, per-update diagnostics, metrics, validation and checkpoints are disabled.",
            "Finite losses/parameters and input/weight fingerprints are checked outside timing; this is not a quality evaluation.",
            "Full batches are required; batch size is per GPU. Memory is the maximum per-rank allocator peak during timing.",
            "Results report median/min/max across independent trials; case order rotates between trials.",
            "Summary ranks contain identity metadata; full per-rank measurements are retained in runs.",
            "Nondeterministic kernels and compiled RNG may produce different training trajectories despite equal starting inputs/weights.",
        ],
        "runs": runs, "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="vae-benchmark-") as directory:
        snapshot = Path(directory) / "config.json"
        write_json(snapshot, raw)
        for repeat in range(args.repeats):
            offset = repeat % len(args.modes)
            order = args.modes[offset:] + args.modes[:offset]
            for mode in order:
                destination = Path(directory) / f"{mode}-{repeat}.json"
                command = [sys.executable]
                if args.gpus > 1:
                    command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={args.gpus}"]
                command += [str(Path(__file__).resolve()), *argv, "--_case", mode, "--_result", str(destination),
                            "--_snapshot", str(snapshot), "--_repeat", str(repeat)]
                print(f"\nRunning {mode}, trial {repeat + 1}/{args.repeats}, on {args.gpus} GPU(s)", flush=True)
                completed = subprocess.run(command, check=False)
                failure = {"mode": mode, "repeat": repeat, "status": "failed", "returncode": completed.returncode,
                           "workload_sha256": result["workload_sha256"]}
                if destination.is_file():
                    try:
                        receipt = json.loads(destination.read_text(encoding="utf-8"))
                        error = worker_result_error(receipt, mode, repeat, result["workload_sha256"])
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        failure["error"] = f"Invalid worker result: {error}"
                        runs.append(failure)
                    else:
                        if error:
                            runs.append({**failure, "error": error, "worker_result": receipt})
                        elif completed.returncode != 0:
                            runs.append({**receipt, "status": "failed", "returncode": completed.returncode,
                                         "error": receipt.get("error", "worker exited unsuccessfully after writing its result")})
                        else:
                            runs.append(receipt)
                else:
                    failure["error"] = "worker failed or did not produce its result"
                    runs.append(failure)
                result["results"] = comparisons(aggregate_runs(runs, args.modes, args.repeats, running=True))
                write_json(args.output, result)
    rows = result["results"] = comparisons(aggregate_runs(runs, args.modes, args.repeats))
    failed = any(row["status"] != "ok" or row.get("same_initial_weights_and_inputs") is False for row in rows)
    result["status"] = "failed" if failed else "ok"
    write_json(args.output, result)
    print("\nMode                       batch/GPU GPUs trials  median ms/batch [min, max]     images/s peak GiB vs eager vs compiled")
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['mode']:27} {row['status']}: {row['successful_repeats']}/{args.repeats} successful trials")
            continue
        speed = row["speedup_vs_torch"]
        compiled_speed = row["speedup_vs_torch_compiled"]
        print(f"{row['mode']:27} {row['batch_size_per_gpu']:9} {row['world_size']:4} {row['successful_repeats']:6} "
              f"{row['wall_ms_per_batch']:10.2f} [{row['wall_ms_per_batch_min']:.2f}, {row['wall_ms_per_batch_max']:.2f}] "
              f"{row['images_per_second']:10.1f} {row['peak_allocated_gib']:8.2f} "
              f"{f'{speed:.2f}x' if speed is not None else 'n/a':>9} "
              f"{f'{compiled_speed:.2f}x' if compiled_speed is not None else 'n/a':>11}")
        if row.get("same_initial_weights_and_inputs") is False:
            print("  Workload/source/initialization mismatch: speedup comparison is invalid.")
    print(f"Results: {args.output.resolve()}")
    return int(failed)


def measure_candidate_capacity(model, optimizer, images, compression, policy, schedules, batches,
                               device, *, start_update=1, progress=None, grad_clip_norm=None):
    """Candidate-only complete updates; no inference of eager-gate availability."""
    import torch
    from examples.vae.qualification import emit_update, move_optimizer

    model.to(device)
    move_optimizer(optimizer, device)
    measurements = []
    context = {"stage": "capacity", "route": "native", "phase": "input_transfer"}
    for index, batch in enumerate(batches):
        number = start_update + index
        context.update(physical_batch=batch, update=number, phase="input_transfer")
        def phase(event):
            context.update(event)
            if progress is not None:
                progress(dict(context))
        try:
            torch.cuda.empty_cache()
            free_before, total = torch.cuda.mem_get_info(device)
            reserved_before = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            batch_images = images.expand(batch, -1, -1, -1).contiguous().to(device)
            schedules.prepare_update(number, epoch=1)
            beta = schedules["beta"]
            emit_update(model, optimizer, batch_images, beta, compression, lambda *_: None,
                        bf16_policy=policy, grad_clip_norm=grad_clip_norm, progress=phase)
            schedules.complete_update(number)
            torch.cuda.synchronize(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            allocated, reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
            margin = max(256 * 2**20, int(total * .05))
            headroom = free_before + reserved_before - reserved
            measurements.append({"physical_batch": batch, "measured": True,
                "schedule_update": number, "beta": beta,
                "admitted": min(free_after, headroom) >= margin,
                "peak_allocated": allocated, "peak_reserved": reserved,
                "peak_allocated_target_bytes": 87990000000,
                "peak_allocated_target_gib": 87990000000 / 2**30,
                "peak_allocated_target_met": allocated <= 87990000000,
                "batch128_fits": batch == 128, "allocator_peak_headroom": headroom,
                "free_before": free_before, "free_after": free_after, "total": total,
                "safety_margin_bytes": margin,
                "opaque_workspace": "allocator_managed_included;nonallocator_transient_peak_unknown"})
            del batch_images
            if not measurements[-1]["admitted"]:
                break
        except torch.cuda.OutOfMemoryError:
            measurements.append({"physical_batch": batch, "measured": True, "admitted": False,
                "batch128_fits": False if batch == 128 else None, "peak_allocated_target_met": False,
                "reason": "cuda_oom", "failure_context": dict(context),
                "cuda_failure_diagnostics": cuda_failure_diagnostics(device)})
            break
    batch128 = next((row for row in measurements if row["physical_batch"] == 128), None)
    return {"batch1_admitted": bool(measurements and measurements[0]["admitted"]),
            "batch128_fits": None if batch128 is None else batch128["batch128_fits"],
            "peak_allocated_target_met": None if batch128 is None else batch128["peak_allocated_target_met"],
            "target_physical_batch": 128, "target_allocated_bytes": 87990000000,
            "target_allocated_gib": 87990000000 / 2**30,
            "target_measurement_scope": "completed_physical_batch128_update;not_all_process_memory",
            "peak_reserved_bytes_batch128": None if batch128 is None else batch128.get("peak_reserved"),
            "measurements": measurements, "batch128_metadata_is_capacity_evidence": False,
            "numerical_trajectory_physical_batches": [], "same_batch_eager_gate": "not_measured_by_capacity_probe",
            "admission_scope": "observed_complete_update_allocator_peak_with_explicit_workspace_unknowns"}


def qualify(args):
    """Installed-only target gate; synthetic seeded images remove dataset/IPC files."""
    # Identity checks run before importing DeepOpt or any VAE helper importing it.
    from benchmarks.compiler.installed_runtime import installed_identity

    receipt = {"qualified": False, "cuda_verified": False, "checkpoint_free": True,
               "status": "unqualified", "geometry": [1, 3, 224, 224]}
    device = None
    native = None
    diagnostic_candidate = None
    repeat_a = repeat_b = None
    failure_context = {"stage": "qualification_setup", "phase": "initialization", "route": None, "update": None}

    def gate_progress(event):
        nonlocal diagnostic_candidate
        event = dict(event)
        if failure_context["stage"] == "native_repeat_gate" and event.get("route") is not None:
            diagnostic_candidate = repeat_a if event["route"] == "eager" else repeat_b
        if event.get("route") is not None and failure_context["stage"] in ("eager_repeat_gate", "native_repeat_gate"):
            prefix = "eager" if failure_context["stage"] == "eager_repeat_gate" else "native"
            event["route"] = prefix + ("_repeat_a" if event["route"] == "eager" else "_repeat_b")
        failure_context.update(event)
        try:
            print(f"VAE qualification: stage={failure_context['stage']} route={event.get('route')} "
                  f"update={event.get('update')} phase={event['phase']}", flush=True)
        except Exception:
            pass

    try:
        installed = installed_identity()
        import torch
        import deepopt
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model
        from examples.vae.mbconv_vae import MBConvVAE_V2
        from examples.vae.qualification import emit_update, paired_trajectory, rng_state, restore_rng

        receipt["installed"] = installed
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: target VAE remains unqualified")
        device = torch.device(args.qualification_device)
        if device.type != "cuda":
            raise ValueError("Target qualification requires a CUDA device")
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Target GPU does not support BF16")
        raw = settings(args)
        from deepopt._bf16_policy import resolve_bf16_policy
        policy = resolve_bf16_policy(raw["runtime"]["bf16_policy"])
        receipt["bf16_policy"] = policy.as_dict()
        original = {"base_channels": 32, "max_channels": 512, "latent_channels": 4, "blocks_per_stage": 2}
        if (raw["application"]["model"] != original or raw["application"]["data"]["image_size"] != 224
                or raw["training"]["batch_size"] != 1
                or raw["training"]["optimizer"] != "custom" or args.optimizer_factory != CUSTOM_OPTIMIZER
                or args.model_factory or args.qualification_updates < 50):
            raise ValueError("Original target requires original architecture, 224-square, configured custom optimizer and >=50 updates")
        expected_beta = dict(type="linear_warmup", start=0., end=1e-6, duration=10000, interval="update")
        if (raw["schedules"]["beta"] != expected_beta or raw["training"]["learning_rate"] != .001
                or raw["training"].get("weight_decay") != 1e-5 or raw["training"].get("momentum") != .9):
            raise ValueError("Original target requires the configured beta schedule and custom optimizer hyperparameters")
        batches = args.qualification_capacity_batches
        if not batches or batches[0] != 1 or batches != sorted(set(batches)) or any(batch < 1 for batch in batches):
            raise ValueError("Capacity batches must be strictly increasing, starting at physical batch 1")
        compression = dict(raw["runtime"]["activation_compression"], enabled=args.qualification_compressed)
        from deepopt.runtime import seed_everything
        seed_everything(raw.get("seed", 42), raw["runtime"]["deterministic"])
        generator = torch.Generator().manual_seed(raw.get("seed", 42) + 1)
        images = torch.rand((1, 3, 224, 224), generator=generator) * 2 - 1

        def make_pair():
            eager = prepare_fp8_model(MBConvVAE_V2(**original), False,
                                      activation_compression=compression, config=raw)[0].train()
            native = deepcopy(eager)
            compile_model(native, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                dynamic=False, options={"require_full_aot": True, "use_native_autodiff": True,
                                        "share_compatible_packs": True}), config=raw)
            eager_optimizer = make_optimizer(eager, raw["training"], args.optimizer_factory)
            after_reference_optimizer = rng_state()
            native_optimizer = make_optimizer(native, raw["training"], args.optimizer_factory)
            native.load_state_dict(eager.state_dict(), strict=True)
            native_optimizer.load_state_dict(deepcopy(eager_optimizer.state_dict()))
            restore_rng(after_reference_optimizer)
            if type(eager_optimizer).__module__.startswith("torch.optim"):
                raise ValueError("A stock optimizer cannot qualify the custom optimizer recipe")
            return eager, native, eager_optimizer, native_optimizer

        receipt["provenance_before_gate"] = source_provenance(deepopt, MBConvVAE_V2, None)
        initialization_start = time.perf_counter()
        eager, native, eager_optimizer, native_optimizer = make_pair()
        receipt["model_optimizer_initialization_seconds"] = time.perf_counter() - initialization_start
        provenance = source_provenance(deepopt, MBConvVAE_V2, native_optimizer)
        schedules = deepopt.ScheduleController.from_config(deepopt.load_config(raw).schedules)
        betas = []
        for update in range(1, args.qualification_updates + 1):
            schedules.prepare_update(update, epoch=1)
            betas.append(schedules["beta"])
            schedules.complete_update(update)
        receipt.update(original_recipe=True, compression=compression, betas=betas,
                       input_sha256=fingerprint([images]), provenance=provenance,
                       input_policy="seeded_synthetic_uniform_minus_one_to_one;not_dataset_quality_evidence",
                       device={"name": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
                               "cuda": torch.version.cuda, "capability": list(torch.cuda.get_device_capability(device))})
        if getattr(args, "qualification_capacity_only", False):
            del eager, eager_optimizer
            receipt["capacity"] = measure_candidate_capacity(native, native_optimizer, images, compression,
                policy, deepopt.ScheduleController.from_config(deepopt.load_config(raw).schedules), batches,
                device, progress=gate_progress, grad_clip_norm=raw["training"].get("grad_clip_norm"))
            receipt.update(status="capacity_measured", candidate_capacity_status="measured",
                           gate_status="not_requested_capacity_only",
                           cuda_verified=receipt["capacity"]["batch1_admitted"],
                           qualification_scope="candidate_allocation_probe_only", qualified=False)
            receipt["compiler_report"] = native_compilation_evidence(native)
            receipt["source_stable"] = provenance == source_provenance(deepopt, MBConvVAE_V2, native_optimizer)
            write_json(args.output, receipt)
            return 0 if receipt["source_stable"] and receipt["capacity"]["batch1_admitted"] else 1
        from examples.vae.qualification import eager_dtype_diagnostic, assert_dtype_edge_contracts
        failure_context.update(stage="eager_dtype_diagnostic", route="eager", phase="forward", update=1)
        receipt["eager_dtype_edges"] = eager_dtype_diagnostic(eager, images, compression,
            bf16_policy=policy, device=device, max_bytes=args.qualification_max_bytes)
        assert_dtype_edge_contracts(receipt["eager_dtype_edges"])
        # A fresh eager/eager control separates repeat spread from route error.
        before_control = rng_state()
        control = deepcopy(eager)
        repeat_eager = deepcopy(eager)
        control_optimizer = make_optimizer(control, raw["training"], args.optimizer_factory)
        reference_control_optimizer = make_optimizer(repeat_eager, raw["training"], args.optimizer_factory)
        control.load_state_dict(repeat_eager.state_dict(), strict=True)
        control_optimizer.load_state_dict(deepcopy(reference_control_optimizer.state_dict()))
        failure_context.update(stage="eager_repeat_gate", phase="initialization", route=None, update=None)
        receipt["same_route_control"] = paired_trajectory(repeat_eager, control,
            reference_control_optimizer, control_optimizer,
            images, betas[:1], compression, device=device, max_bytes=args.qualification_max_bytes, bf16_policy=policy,
            grad_clip_norm=raw["training"].get("grad_clip_norm"), progress=gate_progress)
        restore_rng(before_control)
        del control, repeat_eager, control_optimizer, reference_control_optimizer
        before_native_repeat = rng_state()
        repeat_a, repeat_b, optimizer_a, optimizer_b = make_pair()
        compile_model(repeat_a, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            dynamic=False, options={"require_full_aot": True, "use_native_autodiff": True,
                                    "share_compatible_packs": True}), config=raw)
        failure_context.update(stage="native_repeat_gate", phase="initialization", route=None, update=None)
        diagnostic_candidate = repeat_a
        receipt["native_repeat_control"] = paired_trajectory(repeat_a, repeat_b, optimizer_a, optimizer_b,
            images, betas[:1], compression, device=device, max_bytes=args.qualification_max_bytes, bf16_policy=policy,
            grad_clip_norm=raw["training"].get("grad_clip_norm"), progress=gate_progress)
        diagnostic_candidate = repeat_a
        native_compilation_evidence(repeat_a)
        diagnostic_candidate = repeat_b
        native_compilation_evidence(repeat_b)
        restore_rng(before_native_repeat)
        del optimizer_a, optimizer_b
        repeat_a = repeat_b = None
        diagnostic_candidate = native
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
        failure_context.update(stage="matching_recipe_trajectory", phase="initialization", route=None, update=None)
        receipt["trajectory"] = paired_trajectory(eager, native, eager_optimizer, native_optimizer, images,
            betas, compression, device=device, max_bytes=args.qualification_max_bytes, bf16_policy=policy,
            grad_clip_norm=raw["training"].get("grad_clip_norm"), progress=gate_progress)
        receipt["compiler_report"] = native_compilation_evidence(native)
        from examples.vae.qualification import saved_state_evidence
        receipt["saved_state"] = saved_state_evidence(receipt["compiler_report"], args.qualification_compressed)
        del eager, eager_optimizer
        native.to(device)
        from examples.vae.qualification import move_optimizer, named_tree
        move_optimizer(native_optimizer, device)
        from deepopt.compiler import compilation_report
        from resource import RUSAGE_SELF, getrusage
        # Separate the first resident-device update from fifty warmed updates.
        cold_start = time.perf_counter()
        failure_context.update(stage="residency_update", phase="update", route="native",
                               update=args.qualification_updates + 1)
        schedules.prepare_update(args.qualification_updates + 1, epoch=1)
        emit_update(native, native_optimizer, images.to(device), schedules["beta"], compression, lambda *_: None, bf16_policy=policy, progress=gate_progress)
        schedules.complete_update(args.qualification_updates + 1)
        torch.cuda.synchronize(device)
        receipt["post_trajectory_residency_update_seconds"] = time.perf_counter() - cold_start
        samples = []
        for update in range(50):
            number = args.qualification_updates + update + 2
            failure_context.update(stage="lifetime_updates", phase="update", route="native", update=number)
            schedules.prepare_update(number, epoch=1)
            emit_update(native, native_optimizer, images.to(device), schedules["beta"], compression, lambda *_: None, bf16_policy=policy, progress=gate_progress)
            schedules.complete_update(number)
            torch.cuda.synchronize(device)
            state_shapes = [(name, None if value is None else (str(value.dtype), list(value.shape)))
                            for name, value in named_tree(native_optimizer.state_dict(), "optimizer")]
            samples.append({"update": update, "graphs": len(compilation_report(native).graphs),
                            "optimizer_structure": json_fingerprint(state_shapes),
                            "allocated": torch.cuda.memory_allocated(device),
                            "reserved": torch.cuda.memory_reserved(device),
                            "process_peak_rss_bytes": getrusage(RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)})
        stable = (len({row["graphs"] for row in samples}) == 1
                  and len({row["optimizer_structure"] for row in samples}) == 1
                  and max(row["allocated"] for row in samples[-10:]) <= max(row["allocated"] for row in samples[:5]))
        receipt["lifetime"] = {"passed": stable, "warmed_updates": 50, "samples": samples,
                               "saved_nonpersistent_bytes_after_each_update": 0,
                               "process_memory_kind": "high_water_RSS_bytes",
                               "gate_scope": ["saved_reference_release", "specialization_count", "optimizer_structure", "allocated_bytes"],
                               "reserved_and_process_memory": "diagnostic_trends_only_no_leak_qualification"}
        receipt["capacity"] = measure_candidate_capacity(native, native_optimizer, images, compression,
            policy, schedules, batches, device, start_update=args.qualification_updates + 52,
            progress=gate_progress, grad_clip_norm=raw["training"].get("grad_clip_norm"))
        receipt["capacity"]["numerical_trajectory_physical_batches"] = [1]
        diagnostic_candidate = None
        native = native_optimizer = None
        failure_context = {"stage": "odd_geometry_gate", "phase": "initialization", "route": None, "update": None}
        eager, native, eager_optimizer, native_optimizer = make_pair()
        diagnostic_candidate = native
        odd_images = torch.rand((1, 3, 225, 227), generator=generator) * 2 - 1
        receipt["odd_geometry"] = paired_trajectory(eager, native, eager_optimizer, native_optimizer,
            odd_images, betas[:1], compression, device=device, max_bytes=args.qualification_max_bytes, bf16_policy=policy,
            grad_clip_norm=raw["training"].get("grad_clip_norm"), progress=gate_progress)
        receipt["odd_geometry"]["shape"] = [1, 3, 225, 227]
        receipt["source_stable"] = (provenance["source_sha256"] == source_provenance(deepopt, MBConvVAE_V2, native_optimizer)["source_sha256"]
                                    and installed == installed_identity())
        receipt["cuda_verified"] = True
        receipt["numerical_policy"] = "strict_storage_dtype_and_relative_l2_with_eager_and_native_repeat_controls"
        receipt["qualification_scope"] = "numerical_recipe_saved_reference_lifetime_and_observed_allocator_capacity"
        receipt["qualified"] = bool(stable and receipt["source_stable"] and receipt["capacity"]["batch1_admitted"])
        receipt["status"] = "qualified" if receipt["qualified"] else "failed"
    except Exception as error:
        receipt.update(failure_error_details(error))
        receipt.update(failure_candidate_diagnostics(
            diagnostic_candidate if diagnostic_candidate is not None else native,
            receipt.get("eager_dtype_edges")))
        receipt["failure_context"] = dict(failure_context)
        try:
            receipt["cuda_failure_diagnostics"] = cuda_failure_diagnostics(device)
        except Exception as diagnostic_error:
            receipt["cuda_diagnostics_collection_error"] = type(diagnostic_error).__name__
        receipt["gate_status"] = gate_failure_status(error, failure_context)
        receipt["candidate_capacity_status"] = "not_measured" if "capacity" not in receipt else "measured_separately"
    write_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}), flush=True)
    return 0 if receipt["qualified"] else 1


def main(argv=None):
    if not __package__:
        # Expose benchmark helpers, never the checkout's src/deepopt package.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = parser()
    args = cli.parse_args(argv)
    if args._diagnostic_task:
        from examples.vae.diagnostics import diagnostic_worker

        return diagnostic_worker(args)
    if args.diagnose:
        if args.qualify:
            cli.error("Choose --diagnose or --qualify")
        from examples.vae.diagnostics import run_diagnostics

        try:
            return run_diagnostics(args)
        except (ValueError, OSError, KeyError) as error:
            cli.error(str(error))
    if args.qualify:
        if "--config" not in argv and not any(value.startswith("--config=") for value in argv):
            args.config = Path(__file__).with_name("native_bf16.toml")
        return qualify(args)
    if args.batches < 1 or args.warmup < 0 or args.gpus < 1 or args.repeats < 1:
        cli.error("batches/gpus/repeats must be positive and warmup must be nonnegative")
    if len(set(args.modes)) != len(args.modes):
        cli.error("modes must be unique")
    if args.warmup == 0 and any(MODES[mode]["compiled"] for mode in args.modes):
        cli.error("compiled modes require warmup >= 1 so compilation is excluded from timing")
    if args._case:
        if args._snapshot is None or args._result is None:
            cli.error("workers require a frozen --_snapshot and a --_result destination")
        worker(args)
        return 0
    try:
        return suite(args, argv)
    except ValueError as error:
        cli.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
