"""Isolated full-model vision gates and the one-H200 cached ImageNet trial.

Use ``python -m benchmarks.compiler.benchmark_vision_compression`` from the
checkout/export root. Dry runs use only the standard library. Workers write
metadata/code diagnostics; model and input tensors are never serialized.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from benchmarks.compiler.vision_compression_support import MODELS, PROFILES, vision_cells
from benchmarks.compiler.inductor_reference_models import describe_reference_model


ROOT = Path(__file__).resolve().parents[2]


def source_identity(directory):
    digest = hashlib.sha256()
    count = 0
    for path in sorted(Path(directory).rglob("*.py")):
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0" + path.read_bytes())
        count += 1
    return dict(python_source_sha256=digest.hexdigest(), python_files=count)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    result.add_argument("--bf16-policies", nargs="+", choices=("torch_amp", "bf16_storage"), default=["torch_amp", "bf16_storage"])
    result.add_argument("--executions", nargs="+", choices=("eager", "native"), default=["eager", "native"])
    result.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--correctness-batch-size", type=int, default=1)
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--group-size", type=int, default=1024)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--batches", type=int, default=10)
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--updates", type=int, default=3)
    result.add_argument("--train-dir")
    result.add_argument("--input-mode", choices=("cached",), default="cached")
    result.add_argument("--require-native-autodiff", action="store_true")
    result.add_argument("--qualification-max-bytes", type=int, default=2 * 2**30)
    result.add_argument("--qualify", action="store_true", help="Synthetic correctness only, never ImageNet runtime evidence")
    result.add_argument("--preflight", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--source-runtime", action="store_true", help="Explicit CPU development correctness only")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    return result


def make_plan(args):
    if args.image_size != 224 or args.group_size != 1024 or args.batch_size < 1 or args.correctness_batch_size < 1:
        raise ValueError("Original vision models require image_size=224, group_size=1024 and positive batches")
    if (args.warmup, args.batches, args.repeats) != (5, 10, 1):
        raise ValueError("The requested runtime contract is exactly 5 warmup, 10 measured batches, repeats=1")
    if args.updates < 3 or args.workers < 0 or args.qualification_max_bytes < 1:
        raise ValueError("Correctness requires >=3 updates, nonnegative workers and a positive oracle bound")
    if args.source_runtime and (args.device != "cpu" or not args.qualify):
        raise ValueError("Source runtime is restricted to explicit CPU correctness diagnostics")
    cells = vision_cells(args.models, args.bf16_policies, args.executions, args.profiles)
    return dict(schema_version=1, cells=cells, unique_cells=len(cells),
        models={name: {key: value for key, value in describe_reference_model(name, batch_size=args.batch_size).items()
                      if key in ("name", "constructor", "architecture", "weights", "expected_parameter_count", "input_shape", "download_policy")}
                for name in args.models},
        runtime_request=dict(device=args.device, input_shape=[args.batch_size, 3, 224, 224],
            warmup=5, measured_batches=10, repeats=1, input_mode="cached_host", transfer_included=True),
        correctness_request=dict(input_shape=[args.correctness_batch_size, 3, 224, 224], updates=args.updates,
            input="seeded_synthetic_correctness_only", full_tensor_comparison=True, oracle_max_bytes=args.qualification_max_bytes),
        compilation_scope="network forward/native backward; eager FP32 cross-entropy and diagnostic SGD",
        cuda_executed=False, installed_wheel_required=not args.source_runtime,
        recipe_versions=dict(bf16_storage_primitives=2, attention_preparation=2, saved_state_schema=3),
        reference_comparison="pending; historical rejected/batch-1 Inductor runs are not runtime baselines",
        quality_qualification="pending; compiler agreement does not establish lossy-training quality",
        support_status="requested; execution and target receipts required")


def preflight(args, *, worker=False):
    try:
        return _preflight(args, worker=worker)
    except Exception as error:
        return dict(passed=False, blockers=[f"{type(error).__name__}: {error}"], cuda_verified=False,
                    model_allocated=False, tensor_files_written=0)


def _preflight(args, *, worker=False):
    # Do this before importing Torch/DeepOpt: inherited repro/trace settings
    # can otherwise dump real invocation tensors on compiler failure.
    for name in ("TORCHDYNAMO_REPRO_AFTER", "TORCH_TRACE"):
        os.environ.pop(name, None)
    os.environ.update(TORCHDYNAMO_REPRO_LEVEL="0", TORCH_COMPILE_DEBUG_SAVE_REAL="0",
                      TORCHINDUCTOR_SAVE_COMPILE_ARGS="0")
    missing = [name for name in ("torch", "torchvision") if importlib.util.find_spec(name) is None]
    blockers = [f"Missing dependency: {name}" for name in missing]
    if not args.qualify and (not args.train_dir or not Path(args.train_dir).is_dir()):
        blockers.append("A real ImageNet --train-dir is required for runtime measurement")
    elif not args.qualify and len([path for path in Path(args.train_dir).iterdir() if path.is_dir()]) != 1000:
        blockers.append("ImageNet --train-dir must contain 1000 class directories")
    evidence = dict(passed=False, blockers=blockers, cuda_verified=False, model_allocated=False,
                    tensor_files_written=0)
    if blockers:
        return evidence
    if args.source_runtime:
        import deepopt
        evidence["runtime"] = dict(kind="explicit_cpu_source_diagnostic", path=deepopt.__file__,
                                   **source_identity(Path(deepopt.__file__).parent))
    else:
        from benchmarks.compiler.installed_runtime import installed_identity
        evidence["runtime"] = installed_identity(ROOT, full=True)
    import torch
    import torchvision
    from benchmarks.compiler.save_inductor_references import disable_dynamo_reproduction

    evidence["compiler_diagnostics"] = disable_dynamo_reproduction()
    from deepopt.opt._recipe import bf16_recipe
    from deepopt.compiler.saved_state import SAVED_STATE_SCHEMA_VERSION
    if bf16_recipe("bf16_storage").get("primitive_recipe_version", 0) < 2 or SAVED_STATE_SCHEMA_VERSION < 3:
        blockers.append("Install a current wheel with vision primitive recipe 2 and saved-state schema 3")
    evidence["software"] = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
        cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        allocator=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    evidence["benchmark_code"] = source_identity(ROOT / "benchmarks")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            blockers.append("CUDA unavailable; no target execution evidence")
        elif importlib.util.find_spec("triton") is None:
            blockers.append("Triton is required for CUDA storage/compression execution")
        elif worker:
            device = torch.device(args.device)
            props = torch.cuda.get_device_properties(device)
            evidence["gpu"] = dict(name=props.name, uuid=str(props.uuid), total_memory=props.total_memory,
                                   capability=[props.major, props.minor])
            if torch.cuda.device_count() != 1:
                blockers.append("Worker must see exactly one leased CUDA device")
            if not args.qualify and "H200" not in props.name:
                blockers.append("The runtime workload requires an H200")
            if not torch.cuda.is_bf16_supported():
                blockers.append("The selected CUDA device does not support BF16")
    elif args.device != "cpu" or not args.qualify:
        blockers.append("CPU execution is correctness-only; runtime measurement requires H200 CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        blockers.append("This qualification requires one process/GPU and no distributed accumulation")
    evidence["passed"] = not blockers
    evidence["cuda_verified"] = worker and not blockers and args.device.startswith("cuda")
    return evidence


def cached_imagenet(args):
    import random
    import torch
    from torchvision import datasets, transforms

    preprocessing = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    dataset = datasets.ImageFolder(args.train_dir, transform=preprocessing)
    if len(dataset.classes) != 1000:
        raise ValueError("The ImageNet runtime contract requires 1000 class directories")
    count = args.batch_size * args.batches
    if len(dataset) < count:
        raise ValueError(f"ImageNet root has {len(dataset)} samples; {count} distinct samples are required")
    indices = random.Random(9521).sample(range(len(dataset)), count)
    ordering = [(str(Path(dataset.samples[index][0]).relative_to(args.train_dir)), dataset.samples[index][1],
                 hashlib.sha256(Path(dataset.samples[index][0]).read_bytes()).hexdigest()) for index in indices]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, drop_last=True, num_workers=args.workers, pin_memory=True,
        generator=torch.Generator().manual_seed(11521),
        **({"multiprocessing_context": "spawn"} if args.workers else {}))
    batches = list(loader)
    if len(batches) != 10:
        raise AssertionError("Exactly ten real CPU batches must be cached")
    return batches, dict(dataset_root=str(Path(args.train_dir).resolve()), samples=count,
        ordering_sha256=hashlib.sha256(json.dumps(ordering).encode()).hexdigest(),
        class_mapping_sha256=hashlib.sha256(json.dumps(dataset.class_to_idx, sort_keys=True).encode()).hexdigest(),
        preprocessing="Resize(256), CenterCrop(224), ToTensor, ImageNet normalization", gpu_cache=False)


def timing_trial(cell, args):
    import torch
    from deepopt import autocast
    from deepopt.compiler import compilation_report
    from deepopt.config import CompileConfig
    from deepopt.engine.execution.compilation import compile_model
    from benchmarks.compiler.vision_compression_support import construct_model, prepare_model, optimizer_for

    batches, inputs = cached_imagenet(args)
    torch.manual_seed(9521)
    model, config, compression = prepare_model(construct_model(cell["model"]), cell["bf16_policy"], cell["profile"])
    model.to(args.device)
    if cell["execution"] == "native":
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options=dict(require_full_aot=True, use_native_autodiff=True, share_compatible_packs=True)), config=config)
    optimizer = optimizer_for(model)
    def update(batch):
        optimizer.zero_grad(set_to_none=True)
        images, labels = (value.to(args.device, non_blocking=True) for value in batch)
        with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy=cell["bf16_policy"],
                      activation_compression=compression, config=config):
            logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.detach()
    cold_start = time.perf_counter()
    for _ in range(5):
        update(batches[0])
    torch.cuda.synchronize(args.device)
    cold = dict(seconds=time.perf_counter() - cold_start, peak_allocated=torch.cuda.max_memory_allocated(args.device),
                peak_reserved=torch.cuda.max_memory_reserved(args.device))
    torch.cuda.reset_peak_memory_stats(args.device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    samples, live = [], []
    for batch in batches:
        wall = time.perf_counter()
        start.record()
        loss = update(batch)
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter() - wall) * 1000
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            raise FloatingPointError("Nonfinite runtime loss")
        samples.append(dict(wall_ms=wall_ms, cuda_ms=start.elapsed_time(end), loss=loss_value))
        live.append(torch.cuda.memory_allocated(args.device))
    if max(live[-3:]) > max(live[:3]):
        raise AssertionError("Fixed-shape end-of-update allocated bytes keep growing after warmup")
    report = compilation_report(model)
    if cell["execution"] == "native" and (report is None or report.runtime_fallbacks or not report.as_dict()["native_autodiff_operations"]):
        raise AssertionError("Timed native network lost its required derivative ownership")
    free, total = torch.cuda.mem_get_info(args.device)
    wall = [sample["wall_ms"] for sample in samples]
    return dict(passed=True, warmup=5, measured_batches=10, repeats=1, samples=samples, inputs=inputs,
        median_ms=statistics.median(wall), mean_ms=statistics.mean(wall),
        images_per_second=args.batch_size * 1000 / statistics.mean(wall),
        cold_start=cold, peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device), end_of_update_live_bytes=live,
        device_used_bytes=total - free, process_device_bytes=None,
        process_device_memory_status="unknown; allocator counters are reported separately",
        transfer_included=True, preprocessing_included=False, instrumentation_in_timing=False,
        compiler_report=report.as_dict() if report is not None else None,
        memory_attribution="separate diagnostic phase; category maxima are not added to this peak")


def diagnostic_trial(cell, args):
    """One separately counted, untimed update; never contributes timing samples."""
    import torch
    from deepopt import autocast
    from deepopt.compiler import compilation_report
    from deepopt.config import CompileConfig
    from deepopt.engine.execution.compilation import compile_model
    from deepopt.memory.ledger import StorageLedger, StorageObserver
    from deepopt.opt.quantization.attention import observe_attention_codec
    from benchmarks.analysis.measurement import SavedStorageLifetimes
    from benchmarks.analysis.telemetry import UtilizationTelemetry
    from benchmarks.compiler.vision_compression_support import construct_model, prepare_model, optimizer_for
    from examples.vae.qualification import saved_state_evidence

    batches, inputs = cached_imagenet(args)
    torch.manual_seed(9521)
    model, config, compression = prepare_model(construct_model(cell["model"]), cell["bf16_policy"], cell["profile"])
    model.to(args.device)
    if cell["execution"] == "native":
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options=dict(require_full_aot=True, use_native_autodiff=True, share_compatible_packs=True,
                         collect_saved_state_metadata=True)), config=config)
    optimizer = optimizer_for(model)
    def forward(images):
        with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy=cell["bf16_policy"],
                      activation_compression=compression, config=config):
            return model(images)
    for _ in range(5):
        images, labels = (value.to(args.device, non_blocking=True) for value in batches[0])
        loss = torch.nn.functional.cross_entropy(forward(images).float(), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    del images, labels, loss
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    ledger = StorageLedger()
    ledger.observe(tuple(model.parameters()), owner="model_parameter")
    ledger.observe(tuple(model.buffers()), owner="model_buffer")
    ledger.observe(optimizer.state, owner="optimizer_state")
    tracker = SavedStorageLifetimes(ledger=ledger)
    stages = {}
    def mark(name):
        torch.cuda.synchronize(args.device)
        free, total = torch.cuda.mem_get_info(args.device)
        stages[name] = dict(allocated=torch.cuda.memory_allocated(args.device),
            reserved=torch.cuda.memory_reserved(args.device), stage_peak_allocated=torch.cuda.max_memory_allocated(args.device),
            stage_peak_reserved=torch.cuda.max_memory_reserved(args.device), device_used=total - free,
            saved=tracker.snapshot())
        torch.cuda.reset_peak_memory_stats(args.device)
    with StorageObserver(ledger), observe_attention_codec() as codec:
        mark("baseline")
        with ledger.scope("input_transfer"):
            images, labels = (value.to(args.device, non_blocking=True) for value in batches[0])
        mark("after_transfer")
        with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
            with ledger.scope("forward"):
                output = forward(images)
            mark("after_forward")
            with ledger.scope("loss"):
                loss = torch.nn.functional.cross_entropy(output.float(), labels)
        mark("after_loss")
        with ledger.scope("backward"):
            loss.backward()
        mark("after_backward")
        with ledger.scope("optimizer"):
            optimizer.step()
        mark("after_optimizer")
        optimizer.zero_grad(set_to_none=True)
        mark("after_zero_grad")
    if tracker.snapshot()["live_nonpersistent_bytes"]:
        raise AssertionError("Invocation saved state remains after its final backward consumer")
    telemetry = UtilizationTelemetry(args.device, interval=1)
    telemetry.sample()
    process = [row for sample in telemetry.report()["samples"] for row in sample["processes"] if row["pid"] == os.getpid()]
    if not process:
        raise RuntimeError("Process device-memory snapshot unavailable; allocator counters alone cannot complete this diagnostic")
    report = compilation_report(model)
    saved = saved_state_evidence(report.as_dict(), compression["enabled"]) if report is not None else None
    return dict(passed=True, untimed=True, warmup_updates=5, diagnostic_updates=1, inputs=inputs,
        stages=stages, storage_ledger=ledger.snapshot(), attention_codec=list(codec), native_saved_state=saved,
        peak_allocated_bytes=max(row["stage_peak_allocated"] for row in stages.values()),
        peak_reserved_bytes=max(row["stage_peak_reserved"] for row in stages.values()),
        sampled_process_device_bytes=max(row["used_memory_mib"] for row in process) * 2**20,
        process_device_memory=telemetry.report(), process_device_peak_status="unknown; external counters are snapshots",
        hook_coverage="Outer saved hooks exclude attention's nested hooks; codec telemetry and dispatch ledger supplement them",
        allocator_attribution="Visible storage intervals only; library workspaces and intra-kernel temporaries remain unknown",
        category_maxima_are_additive=False, tensor_files_written=0)


def worker(request):
    from types import SimpleNamespace

    task = json.loads(Path(request).read_text())
    args = SimpleNamespace(**task["arguments"])
    cell, phase = task["cell"], task["phase"]
    output = Path(task["result"])
    row = dict(cell=cell, phase=phase, passed=False, cuda_verified=False, tensor_files_written=0)
    try:
        row["preflight"] = preflight(args, worker=True)
        if not row["preflight"]["passed"]:
            raise RuntimeError("; ".join(row["preflight"]["blockers"]))
        import torch
        torch.set_num_threads(1)
        torch.compiler.reset()
        if phase == "correctness":
            from benchmarks.compiler.vision_compression_support import qualify_pair, qualify_eager

            def progress(event):
                row.update(route=event["route"], update=event["update"], last_progress=event)
            qualify = qualify_pair if cell["execution"] == "native" else qualify_eager
            row["qualification"] = qualify(cell["model"], cell["bf16_policy"], cell["profile"],
                device=args.device, batch_size=task["batch_size"], updates=args.updates,
                max_bytes=args.qualification_max_bytes, progress=progress)
        elif phase == "measurement":
            row["measurement"] = timing_trial(cell, args)
        elif phase == "diagnostic":
            row["diagnostic"] = diagnostic_trial(cell, args)
        elif phase == "preflight":
            row["model_allocated"] = False
        else:
            raise ValueError(f"Unknown worker phase: {phase}")
        row.update(passed=True, status="passed", cuda_verified=args.device.startswith("cuda"))
    except Exception as error:
        unavailable = row.get("route") == "eager" and isinstance(error, (MemoryError, torch.OutOfMemoryError))
        row.update(status="gate_unavailable" if unavailable else "failed",
                   error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    output.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    return 0 if row["passed"] else 1


def launch_stage(args, cell, phase, batch, folder, index, identity):
    result = folder / f"{index}-{phase}.json"
    request = folder / f"{index}-request.json"
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.device.startswith("cuda"):
        arguments["device"] = "cuda:0"  # The broker exposes one physical UUID.
    request.write_text(json.dumps(dict(arguments=arguments, cell=cell, phase=phase,
        batch_size=batch, result=str(result.resolve())), indent=2) + "\n")
    command = [sys.executable, "-m", "benchmarks.compiler.benchmark_vision_compression",
               "--output", str(folder), "--_worker", str(request.resolve())]
    log = folder / f"{index}-process.log"
    try:
        if not args.source_runtime:
            from benchmarks.compiler.installed_runtime import IDENTITY_ENV

            os.environ[IDENTITY_ENV] = json.dumps(identity, sort_keys=True)
        if args.device.startswith("cuda"):
            from benchmarks.analysis.scheduler import run_external, discover_devices, default_broker_path

            code = run_external(command, device=discover_devices(None, args.device)[0],
                broker_path=default_broker_path(), timeout=21600, cancel=folder / "cancel", log_path=log)
        else:
            with log.open("x") as stream:
                code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=21600).returncode
    except Exception as error:
        code = 1
        result.write_text(json.dumps(dict(passed=False, status="worker_launch_failed",
            error_type=type(error).__name__, error=str(error)), indent=2) + "\n")
    receipt = json.loads(result.read_text()) if result.exists() else dict(passed=False, status="missing_worker_result", exit_code=code)
    if code:
        receipt.update(passed=False, exit_code=code)
    return receipt


def main(argv=None):
    args = parser().parse_args(argv)
    if args._worker is not None:
        return worker(args._worker)
    try:
        plan = make_plan(args)
    except ValueError as error:
        parser().error(str(error))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    args.output = args.output.resolve()
    manifest = dict(plan, preflight=preflight(args), cells=[], qualified=False)
    path = args.output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["preflight"]["passed"]:
        return 1
    if args.device.startswith("cuda"):
        folder = args.output / "hardware-preflight"
        folder.mkdir()
        manifest["hardware_preflight"] = launch_stage(args, plan["cells"][0], "preflight", args.correctness_batch_size,
            folder, 0, manifest["preflight"]["runtime"])
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        if not manifest["hardware_preflight"]["passed"]:
            return 1
    if args.preflight:
        return 0
    for cell in plan["cells"]:
        folder = args.output / cell["id"]
        folder.mkdir()
        entry = dict(cell=cell, stages=[], passed=True)
        stages = [("correctness", args.correctness_batch_size)]
        if not args.qualify:
            if args.batch_size != args.correctness_batch_size:
                stages.append(("correctness", args.batch_size))
            stages.extend((("measurement", args.batch_size), ("diagnostic", args.batch_size)))
        for index, (phase, batch) in enumerate(stages):
            receipt = launch_stage(args, cell, phase, batch, folder, index, manifest["preflight"].get("runtime"))
            entry["stages"].append(receipt)
            if not receipt["passed"]:
                entry["passed"] = False
                break
        ordering = {receipt[key]["inputs"]["ordering_sha256"] for receipt in entry["stages"]
                    for key in ("measurement", "diagnostic") if key in receipt}
        if len(ordering) > 1:
            entry.update(passed=False, error="Cached ImageNet ordering/content differs between timing and diagnosis")
        manifest["cells"].append(entry)
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    manifest["qualified"] = bool(manifest["cells"]) and all(cell["passed"] for cell in manifest["cells"])
    manifest["cuda_verified"] = manifest["qualified"] and args.device.startswith("cuda")
    manifest["scope"] = "synthetic correctness only" if args.qualify else "H200 full-update ImageNet runtime"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(qualified=manifest["qualified"], output=str(path), cuda_verified=manifest["cuda_verified"])))
    return 0 if manifest["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
