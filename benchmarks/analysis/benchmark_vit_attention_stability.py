#!/usr/bin/env python3
"""Train the real CIFAR10/224 ViT recipe with an explicitly selected attention path.

No pretrained weights or dataset downloads. The optimizer must be supplied
explicitly; custom counters missing from state_dict are saved before each
epoch checkpoint. A completed run means the requested epochs finished, not an
automatic claim that the accuracy curve is stable.
"""

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys


def _modules():
    if __package__:
        from . import benchmark_attention_alignment as alignment, diagnose_training_stability as diagnostic
        from .attention_comparison import model_attention_case
    else:
        import benchmark_attention_alignment as alignment
        import diagnose_training_stability as diagnostic
        from attention_comparison import model_attention_case
    return alignment, diagnostic, model_attention_case


def optimizer_extra(optimizer, model):
    import torch
    _, diagnostic, _ = _modules()
    base = torch.optim.Optimizer([torch.nn.Parameter(torch.empty(0))], {})
    excluded = set(vars(base)) | {"state", "param_groups", "defaults"}
    names = {id(p): name for name, p in model.named_parameters()}
    extra = {}
    for key, value in vars(optimizer).items():
        if key in excluded:
            if "hook" in key and value:
                raise ValueError("Nonempty optimizer hooks cannot qualify resumable comparisons")
        else:
            extra[key] = diagnostic._freeze(value, names)
    return extra


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=("native", "manual", "flash_attn_2"), help="Otherwise inherit runtime.attention.backend")
    parser.add_argument("--attention-bits", type=int, choices=(0, 1, 2, 4, 8), help="Otherwise inherit config; 0 keeps only Q/K/V dense")
    parser.add_argument("--group-size", type=int, help="Explicit global compression group override")
    parser.add_argument("--compression-bits", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--compression", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--optimizer", required=True, help="Actual optimizer module:Class, e.g. test_optimizer:SGD_NS_Overshoot")
    parser.add_argument("--optimizer-kwargs", type=json.loads, help="Explicit constructor kwargs; lr otherwise inherits training.learning_rate")
    parser.add_argument("--learning-rate", type=float, help="Explicit LR override; must agree with an explicitly supplied optimizer lr")
    parser.add_argument("--initializer", help="Optional module:function applied using model.apply")
    parser.add_argument("--model-name", choices=("vit_base_patch16_224", "resnet18", "efficientnet_b0"), default="vit_base_patch16_224")
    parser.add_argument("--model-library", choices=("timm", "torchvision"), default="timm")
    parser.add_argument("--convolution-policy", choices=("strict", "native_spatial", "depthwise"))
    parser.add_argument("--producer-pack", choices=("off", "gelu", "layer_norm", "all"), default="off")
    parser.add_argument("--attention-grad-layout", choices=("auto", "preserve", "bshd", "bhsd", "interleaved"), default="auto")
    parser.add_argument("--activation-layout", choices=("auto", "baseline", "channels_last"), default="auto")
    parser.add_argument("--convolution-kernel-variant", choices=("auto", "baseline", "tuned"), default="auto")
    parser.add_argument("--profile-train-steps", type=int, default=0,
                        help="Diagnostic first-N-update trace; profiled epochs are not throughput comparisons")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path(os.environ.get("DEEPOPT_CIFAR10_ROOT", "data")))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--execution-mode", choices=("max_speed", "balanced", "analyze"))
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fill-uninitialized-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--attention-pack", choices=("auto", "separate", "joint"), default="auto",
                        help="Joint packing is an opt-in candidate pending H100 acceptance")
    parser.add_argument("--pack-indexing", choices=("wide", "bounded"), default="wide")
    parser.add_argument("--dense-out-proj", action="store_true", help="Use identically in every comparison if your recipe excludes attn.proj")
    parser.add_argument("--snapshot-before-update", type=int, default=0)
    return parser.parse_args(argv)


def resolve_run(args):
    """CLI > supplied config > benchmark defaults, before allocating any data/model.

    A learning rate must come from the caller. The benchmark never guesses a
    safe LR for an arbitrary optimizer or silently replaces its supplied recipe.
    """
    import math
    from deepopt._config_resolution import resolve_config
    if args.profile_train_steps < 0:
        raise ValueError("profile-train-steps must be nonnegative")
    if args.model_library == "torchvision" and args.model_name == "vit_base_patch16_224":
        raise ValueError("Use timm for vit_base_patch16_224; torchvision options are resnet18/efficientnet_b0")
    if args.model_name != "vit_base_patch16_224" and (args.backend == "manual" or args.dense_out_proj):
        raise ValueError("Manual attention and dense-out-proj controls require ViT")

    supplied = resolve_config(args.config, base=None) if args.config else {}
    if args.convolution_policy is not None:
        supplied.setdefault("runtime", {}).setdefault("fp8", {})["convolution_policy"] = args.convolution_policy
    defaults = {"seed": 14, "training": {"epochs": 32, "batch_size": 512, "accumulation_steps": 1},
        "loader": {"num_workers": 20, "prefetch_factor": 2, "persistent_workers": True,
                   "pin_memory": True, "drop_last": True},
        "runtime": {"device": "cuda", "precision": "bf16", "execution_mode": "max_speed",
            "deterministic": True, "fill_uninitialized_memory": False,
            "attention": {"backend": "flash_attn_2"},
            "activation_compression": {"enabled": True, "bits": 2, "attention_bits": 4,
                                       "group_size": 1024, "norm_xhat_recenter": True},
            "compile": {"enabled": False},
            "cuda_graph": {"warmup": 10, "autograd": True},
            "fp8": {"enabled": False, "backend": "triton", "trace_mode": "modules", "intercept_linear": False,
                    "activation_dtype": "e4m3fn", "weight_dtype": "e4m3fn", "grad_dtype": "e4m3fn",
                    "num_stages": 3, "scaling_recipe": "current", "backward_backend": "native_fast"}}}
    if supplied.get("loader", {}).get("num_workers") == 0:
        defaults["loader"].update(persistent_workers=False, prefetch_factor=None)
    snapshot = resolve_config(defaults, overrides=supplied, base=None)
    changes = {"runtime": {}, "training": {}, "loader": {}}
    for arg, section, key in (("batch_size", "training", "batch_size"), ("epochs", "training", "epochs"),
        ("workers", "loader", "num_workers"), ("drop_last", "loader", "drop_last"),
        ("execution_mode", "runtime", "execution_mode"),
        ("deterministic", "runtime", "deterministic"), ("fill_uninitialized_memory", "runtime", "fill_uninitialized_memory")):
        value = getattr(args, arg)
        if value is not None:
            changes[section][key] = value
    if args.seed is not None:
        changes["seed"] = args.seed
    if args.backend is not None:
        changes["runtime"]["attention"] = {"backend": "native" if args.backend == "manual" else args.backend}
    if args.fp8 is not None:
        changes["runtime"]["fp8"] = {"enabled": args.fp8}
    compression = {}
    for value, key in ((args.group_size, "group_size"), (args.compression_bits, "bits"), (args.compression, "enabled")):
        if value is not None:
            compression[key] = value
    # Zero is a benchmark-only Q/K/V exclusion, not a replacement with four bits.
    if args.attention_bits is not None and args.attention_bits != 0:
        compression["attention_bits"] = args.attention_bits
    if compression:
        changes["runtime"]["activation_compression"] = compression
    kwargs = deepcopy(args.optimizer_kwargs) if args.optimizer_kwargs is not None else {}
    if not isinstance(kwargs, dict):
        raise ValueError("--optimizer-kwargs must be a JSON object")
    if args.learning_rate is not None:
        if "lr" in kwargs and kwargs["lr"] != args.learning_rate:
            raise ValueError("--learning-rate and --optimizer-kwargs lr disagree")
        kwargs["lr"] = args.learning_rate
    if "lr" not in kwargs:
        if "learning_rate" not in supplied.get("training", {}):
            raise ValueError("Supply --optimizer-kwargs with lr, --learning-rate, or training.learning_rate in --config")
        kwargs["lr"] = supplied["training"]["learning_rate"]
    if isinstance(kwargs["lr"], bool) or not isinstance(kwargs["lr"], (int, float)) or not math.isfinite(kwargs["lr"]) or kwargs["lr"] <= 0:
        raise ValueError("The benchmark learning rate must be finite and positive")
    changes["training"]["learning_rate"] = kwargs["lr"]
    if changes["loader"].get("num_workers", snapshot["loader"]["num_workers"]) == 0:
        changes["loader"].update(persistent_workers=False, prefetch_factor=None)
    snapshot = resolve_config(snapshot, overrides=changes)
    effective = deepcopy(args)
    effective.optimizer_kwargs = kwargs
    for arg, section, key in (("batch_size", "training", "batch_size"), ("epochs", "training", "epochs"),
                             ("workers", "loader", "num_workers")):
        setattr(effective, arg, snapshot[section][key])
    effective.seed = snapshot["seed"]
    effective.backend = args.backend or snapshot["runtime"]["attention"]["backend"]
    effective.fp8 = snapshot["runtime"]["fp8"]["enabled"]
    policy = snapshot["runtime"]["activation_compression"]
    effective.group_size = policy["group_size"]
    effective.attention_bits = (0 if args.attention_bits == 0 or not policy["enabled"] else
                                policy["attention_bits"] if policy["attention_bits"] is not None else max(4, policy["bits"]))
    if effective.workers == 0:
        snapshot = resolve_config(snapshot, overrides={"loader": {"persistent_workers": False, "prefetch_factor": None}})
    if effective.snapshot_before_update < 0:
        raise ValueError("snapshot update must be nonnegative")
    if snapshot["training"]["checkpoint_every_steps"]:
        raise ValueError("This runner persists custom optimizer extras at epoch boundaries; set training.checkpoint_every_steps=0")
    if args.graph and not snapshot["loader"]["drop_last"]:
        # Partial batches have different graph signatures; require an explicit
        # eager experiment rather than silently producing graph fallback timings.
        raise ValueError("Graph benchmarking requires loader.drop_last=true; select --no-graph for partial batches")
    if args.graph and not snapshot["runtime"]["cuda_graph"]["autograd"]:
        raise ValueError("Training graph benchmarking requires runtime.cuda_graph.autograd=true")
    return effective, snapshot


def check_training_replay(captured, previous):
    """Check dispatch after an actual update, without device synchronization."""
    if captured is None:
        return None
    now = captured.statistics
    if now["forward_replays"] <= previous["forward_replays"]:
        raise RuntimeError("CUDA Graph requested but the training update did not replay; "
                           + json.dumps(captured.fallback_diagnostics, sort_keys=True))
    for key in ("eager_autocast_fallbacks", "eager_memory_policy_fallbacks"):
        if now[key] != previous[key]:
            raise RuntimeError("Unexpected training graph fallback: " + json.dumps(captured.fallback_diagnostics, sort_keys=True))
    return now


def training_device(name):
    import torch
    value = torch.device("cuda" if name == "auto" else name)
    if value.type != "cuda":
        raise ValueError("This benchmark requires runtime.device=cuda")
    return torch.device("cuda", torch.cuda.current_device()) if value.index is None else value


@contextmanager
def graph_launch_evidence(torch):
    """Observe real replay calls, including autograd's backward graph objects."""
    from unittest.mock import patch
    replay = torch.cuda.CUDAGraph.replay
    handles = {}
    def observed(handle):
        key = id(handle)
        handles[key] = handles.get(key, 0) + 1
        return replay(handle)
    with patch.object(torch.cuda.CUDAGraph, "replay", observed):
        yield lambda: {"total_replays": sum(handles.values()), "distinct_graph_handles": len(handles),
                       "replays_per_handle": sorted(handles.values()),
                       "note": "Includes forward and autograd backward launches; other captured work, if enabled, is included."}


def main(argv=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    requested = parse_args(argv)
    args, config = resolve_run(requested)
    sys.path.append(str(Path.cwd()))
    import numpy as np
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import v2
    from torchmetrics import Accuracy
    import deepopt
    from deepopt.data import build_loaders
    from deepopt.opt.nn import functional as doF
    from deepopt.opt.nn.attention_backend import observe_attention

    alignment, diagnostic, model_attention_case = _modules()
    if not torch.cuda.is_available():
        raise RuntimeError("The stability recipe requires an NVIDIA CUDA GPU")
    device = training_device(config["runtime"]["device"])
    torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.use_deterministic_algorithms(config["runtime"]["deterministic"])
    mean, std = [.485, .456, .406], [.229, .224, .225]
    train_transform = v2.Compose([v2.ToImage(), v2.Resize((224, 224), antialias=True),
        v2.RandomHorizontalFlip(), v2.RandAugment(num_ops=5, magnitude=9),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std),
        v2.RandomErasing(p=.5, scale=(.03, .15), ratio=(.3, 3.3), value=0)])
    valid_transform = v2.Compose([v2.ToImage(), v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])
    train = CIFAR10(str(args.dataset_root), train=True, download=False, transform=train_transform)
    valid = CIFAR10(str(args.dataset_root), train=False, download=False, transform=valid_transform)
    dataset_identity = {name: {"samples": len(dataset), "raw_shape": list(dataset.data.shape),
        "raw_pixels_sha256": sha256(dataset.data.tobytes(order='C')).hexdigest(),
        "targets_sha256": sha256(np.asarray(dataset.targets, dtype=np.int64).tobytes()).hexdigest()}
        for name, dataset in (("train", train), ("validation", valid))}
    train_loader, valid_loader, _ = build_loaders(train, valid, config=config, resumable=True)
    if args.model_library == "timm":
        import timm
        model = timm.create_model(args.model_name, pretrained=False, num_classes=10).to(device)
        model_factory = {"factory": "timm:create_model", "kwargs": {"model_name": args.model_name, "pretrained": False, "num_classes": 10}}
    else:
        from torchvision import models
        model = getattr(models, args.model_name)(weights=None, num_classes=10).to(device)
        model_factory = {"factory": "torchvision.models:" + args.model_name, "kwargs": {"weights": None, "num_classes": 10}}
    original_attention_flags = {}
    for name, module in model.named_modules():
        if type(module).__module__.startswith("timm.") and hasattr(module, "fused_attn"):
            original_attention_flags[name] = bool(module.fused_attn)
            # All selected algorithms enter the same interceptable SDPA call.
            # The manual control substitutes DeepOpt's explicit FP32 math there.
            module.fused_attn = True
    if args.initializer:
        model.apply(diagnostic._resolve(args.initializer))
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
    optimizer_class = diagnostic._resolve(args.optimizer)
    optimizer = optimizer_class(model.parameters(), **args.optimizer_kwargs)
    trainer = deepopt.Trainer(model, optimizer, device=device, config=config,
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": Accuracy(task="multiclass", num_classes=10, sync_on_compute=True)})
    # Process flags are normally owned by initialize_distributed. This standalone
    # runner owns them explicitly, before loader workers start or capture runs.
    from deepopt._memory_policy import resolve_fill
    torch.utils.deterministic.fill_uninitialized_memory = resolve_fill(
        trainer.execution_mode, config["runtime"]["fill_uninitialized_memory"])
    dense_names = []
    if args.dense_out_proj:
        for name, module in trainer.model.named_modules():
            if name.endswith("attn.proj"):
                original = module.forward
                def wrap(original):
                    @wraps(original)
                    def forward(*a, **kw):
                        with deepopt.autocast("cuda", activation_compression=False):
                            return original(*a, **kw)
                    return forward
                module.forward = wrap(original)
                dense_names.append(name)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("output", "resume")}
    identity = {"arguments": settings, "config": trainer.config, "deepopt_source_sha256": diagnostic._library_hash(),
                "actual_optimizer": {"factory": args.optimizer, "kwargs": args.optimizer_kwargs},
                "actual_model": model_factory,
                "dataset_identity": dataset_identity,
                "original_fused_attn": original_attention_flags,
                "optimizer_source_sha256": diagnostic._source_hash(optimizer_class),
                "runner_sha256": alignment._file_hash(__file__), "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                "execution_flags": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                "train_transform": str(train_transform), "validation_transform": str(valid_transform)}
    fingerprint = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report_path = args.output / "training-report.json"
    report = {"fingerprint": fingerprint, "identity": identity, "epochs": [], "status": "running",
              "requested_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(requested).items()}}
    print(json.dumps({"effective_config": trainer.config, "actual_optimizer": identity["actual_optimizer"],
                      "effective_attention_bits": args.attention_bits}, sort_keys=True), flush=True)
    checkpoint = deepopt.Checkpoint(args.output / "checkpoints", every_steps=0)
    if report_path.exists():
        if not args.resume:
            raise ValueError("Output already exists; use --resume or a new directory")
        report = json.loads(report_path.read_text())
        if report["fingerprint"] != fingerprint:
            raise ValueError("Resume fingerprint mismatch")
        checkpoint.restore(trainer, train_loader, valid_loader)
        extra_file = args.output / f"optimizer-extra-{trainer.state.optimizer_steps}.pt"
        extra = torch.load(extra_file, map_location="cpu", weights_only=False)
        for key, value in extra.items():
            setattr(optimizer, key, diagnostic._move(value, trainer.device, dict(trainer.model.named_parameters())))
        if trainer.state.completed_epochs >= args.epochs:
            print("Requested epochs are already complete")
            return 0
    elif args.resume:
        raise ValueError("--resume requires an existing training report/checkpoint")
    # Validate that every custom optimizer attribute can be persisted now.
    optimizer_extra(optimizer, trainer.model)
    alignment._atomic_json(report_path, report)

    class SaveEpoch:
        def on_fit_start(self, trainer, train_loader, valid_loader):
            checkpoint.on_fit_start(trainer, train_loader, valid_loader)
            self.previous_graph = captured.statistics if captured else {}
            self.previous_launches = replay_evidence()['total_replays']
        def __call__(self, trainer, row):
            path = args.output / f"optimizer-extra-{trainer.state.optimizer_steps}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(optimizer_extra(optimizer, trainer.model), temporary); temporary.replace(path)
            checkpoint(trainer, row)
            report["epochs"] = [r for r in report["epochs"] if r["epoch"] < row["epoch"]] + [row]
            report["graph_statistics"] = captured.statistics if captured else None
            row["graph_statistics"] = report["graph_statistics"]
            row["graph_launch_evidence"] = replay_evidence()
            row["epoch_graph_evidence"] = {
                "graph_replays": row["graph_launch_evidence"]['total_replays'] - self.previous_launches,
                "policy_counter_deltas": {k: v - self.previous_graph.get(k, 0)
                                          for k, v in (report["graph_statistics"] or {}).items()}}
            self.previous_graph = report["graph_statistics"] or {}
            self.previous_launches = row["graph_launch_evidence"]['total_replays']
            row["profiled_diagnostic_run"] = bool(args.profile_train_steps)
            alignment._atomic_json(report_path, report)

    with ExitStack() as stack:
        from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
        stack.enter_context(activation_codec_scope(attention_pack=args.attention_pack, strided_index=args.pack_indexing,
                            producer_pack=args.producer_pack, attention_grad_layout=args.attention_grad_layout,
                            activation_layout=args.activation_layout))
        from deepopt.opt.backends._shared.convolution import convolution_kernel_scope
        stack.enter_context(convolution_kernel_scope(args.convolution_kernel_variant))
        if args.backend == "manual":
            spec = dict(backend="manual", bits=[args.attention_bits]*3, group_size=args.group_size,
                        mode="production", layout="bshd", codec="current")
            stack.enter_context(model_attention_case(spec))
        elif not args.attention_bits:
            # Keep other operators compressed while attention saves stay dense.
            from unittest.mock import patch
            stack.enter_context(patch.object(doF, "_attention_compression_settings", return_value=None))
        routes = []
        def observe(inputs, out, info):
            if len(routes) < 24:
                routes.append({**info, "qkv_shapes": [list(x.shape) for x in inputs]})
        stack.enter_context(observe_attention(observe))
        if args.graph:
            # Capture preserves the starting parameters/buffers/RNG through the
            # existing CUDAGraphCore lifecycle, before fit consumes any batches.
            sample = torch.zeros(args.batch_size, 3, 224, 224, device=device)
            with trainer.runtime.autocast():
                captured = deepopt.CUDAGraphCore(device, config=trainer.config).capture_model(
                    trainer.model, (sample,))
            stack.callback(captured.close)
        else:
            captured = None
        replay_evidence = stack.enter_context(graph_launch_evidence(torch))
        report["capture_policy"] = captured.capture_policy if captured else None
        profiler = None
        profile_active = False
        profile_updates = 0
        if args.profile_train_steps:
            from unittest.mock import patch
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
            def annotate(function, name):
                @wraps(function)
                def call(*a, **kw):
                    with torch.profiler.record_function(name):
                        return function(*a, **kw)
                return call
            stack.enter_context(patch.object(optimizer, "step", annotate(optimizer.step, "training.optimizer")))
            if trainer.ema is not None:
                stack.enter_context(patch.object(trainer.ema, "update", annotate(trainer.ema.update, "training.ema")))
            profiler.start()
            profile_active = True
        def finish_profile():
            nonlocal profile_active
            if not profile_active:
                return
            profiler.stop()
            profile_active = False
            trace = args.output / "training-diagnostic-trace.json"
            try:
                profiler.export_chrome_trace(str(trace))
            except OSError as error:
                report["training_profile_error"] = f"{type(error).__name__}: {error}"
            ranges = [{"name": e.key, "count": e.count, "self_cpu_ms": e.self_cpu_time_total / 1000,
                       "inclusive_device_ms": e.device_time_total / 1000}
                      for e in profiler.key_averages() if e.key.startswith("training.") or
                      "DataLoader" in e.key or e.key in ("aten::to", "aten::_to_copy")]
            report["training_profile"] = {"trace": str(trace), "optimizer_updates": profile_updates,
                "ranges": ranges, "note": "Diagnostic overhead is present. Nested ranges are not additive; transfers include CPU casts."}
        if args.snapshot_before_update:
            train_loader = diagnostic.SnapshotLoader(train_loader, trainer, args.output / "stability-snapshot.pt",
                before_update=args.snapshot_before_update, dense_modules=dense_names)
        try:
            previous_graph = captured.statistics if captured else None
            def verify_route(*unused):
                nonlocal previous_graph, profile_updates
                previous_graph = check_training_replay(captured, previous_graph)
                expected = {"manual": "manual", "native": "native_sdpa", "flash_attn_2": "flash_attn_2"}[args.backend]
                if original_attention_flags and (not routes or any(row["backend"] != expected for row in routes)):
                    raise RuntimeError("The requested attention backend was not observed; inspect attention_routes")
                if profile_active:
                    profiler.step()
                    profile_updates += 1
                    if profile_updates >= args.profile_train_steps:
                        finish_profile()
            trainer.fit(train_loader, valid_loader, epochs=args.epochs, callbacks=[SaveEpoch()],
                        step_callbacks=[verify_route], progress_bar="epoch")
            report["status"] = "completed_epochs"
        except BaseException as error:
            report["status"], report["error"] = "interrupted", f"{type(error).__name__}: {error}"
            raise
        finally:
            finish_profile()
            report["graph_launch_evidence"] = replay_evidence()
            report["attention_routes"] = routes
            report["graph_statistics"] = captured.statistics if captured else None
            report["graph_fallback_diagnostics"] = captured.fallback_diagnostics if captured else None
            alignment._atomic_json(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
